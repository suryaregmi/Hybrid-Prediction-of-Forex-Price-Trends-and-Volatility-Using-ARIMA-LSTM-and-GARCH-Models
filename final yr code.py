import warnings
warnings.filterwarnings('ignore')   # Suppress non-critical warnings for cleaner output logs

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ============================================================
# SETTINGS
# ============================================================
# Central configuration block for file paths, column names,
# model hyperparameters, and train/validation/test split ratios.
# Keeping all settings here makes the experiment easier to
# reproduce and modify.
FILE_PATH = 'Forex_OHLC_50k_Dataset.xlsx'   # change if needed
OUTPUT_DIR = 'hybrid_outputs'

OPEN_COL = 'Open'
HIGH_COL = 'High'
LOW_COL = 'Low'
CLOSE_COL = 'Close'
VOLUME_COL = 'Volume'   # if missing, code creates a dummy volume column

WINDOW = 60
SEED = 42
MAX_EPOCHS = 13
BATCH_SIZE = 64
LEARNING_RATE = 0.001
GARCH_P = 1
GARCH_Q = 1

# Chronological data split:
# 80% for training, 20% for final testing.
# A validation slice is taken from within the training period
# so that model tuning remains fully out-of-sample.
TRAIN_RATIO = 0.80
TEST_RATIO = 0.20
VAL_RATIO_WITHIN_TRAIN = 0.125   # 12.5% of 80% = 10% of total

# ============================================================
# REPRODUCIBILITY
# ============================================================
# Fix random seeds so that results are as consistent as possible
# across repeated executions.
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Create output folder if it does not already exist.
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# UTILITIES
# ============================================================
def rmse(y_true, y_pred):
    # Compute Root Mean Squared Error, which penalises larger
    # forecasting errors more heavily than MAE.
    return np.sqrt(mean_squared_error(y_true, y_pred))


def print_metrics(name, y_true, y_pred):
    # Helper function to print and return the main regression
    # metrics used throughout the project.
    mae_ = mean_absolute_error(y_true, y_pred)
    rmse_ = rmse(y_true, y_pred)
    print(f'{name:<10} | MAE: {mae_:.6f} | RMSE: {rmse_:.6f}')
    return mae_, rmse_


def load_data(path):
    # Load the dataset from CSV or Excel and validate that the
    # required OHLC columns are present.
    if path.lower().endswith('.csv'):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    needed = [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found. Available columns: {list(df.columns)}")

    # If volume is not available, create a dummy column so the
    # pipeline can still run without breaking feature engineering.
    if VOLUME_COL not in df.columns:
        print(f"[INFO] '{VOLUME_COL}' column not found. Creating dummy volume = 0.")
        df[VOLUME_COL] = 0.0

    # Convert all numeric columns safely and coerce invalid entries to NaN.
    for col in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove rows where core OHLC values are missing.
    df = df.dropna(subset=[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]).reset_index(drop=True)
    return df


def adf_stationarity(series):
    # Perform the Augmented Dickey-Fuller test to determine
    # whether the series is stationary.
    result = adfuller(series.dropna(), autolag='AIC')
    out = {
        'adf_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'n_obs': result[3]
    }
    return out


def choose_d_via_adf(close_series, max_d=2, alpha=0.05):
    # Repeatedly difference the close-price series until the ADF
    # test suggests stationarity or until the maximum differencing
    # order is reached.
    current = pd.Series(close_series).astype(float).copy()
    for d in range(max_d + 1):
        test = adf_stationarity(current)
        print(f"ADF check for d={d}: p-value={test['p_value']:.6f}")
        if test['p_value'] < alpha:
            return d
        current = current.diff().dropna()
    return max_d


def auto_select_arima(train_series, d, p_range=range(0, 4), q_range=range(0, 4)):
    # Select the ARIMA order that best fits the training series.
    # First try auto_arima for convenience; if unavailable, fall
    # back to a manual AIC-based grid search.
    best_order = None
    best_aic = np.inf
    best_model = None

    try:
        import pmdarima as pm
        print('[ARIMA] Trying pmdarima auto_arima...')
        auto_model = pm.auto_arima(
            train_series,
            d=d,
            start_p=0,
            start_q=0,
            max_p=5,
            max_q=5,
            seasonal=False,
            information_criterion='aic',
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        order = auto_model.order
        model = ARIMA(train_series, order=order).fit()
        return order, model
    except Exception as e:
        print(f'[ARIMA] pmdarima failed, using manual AIC grid search. Reason: {e}')

    for p in p_range:
        for q in q_range:
            try:
                model = ARIMA(train_series, order=(p, d, q)).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_order = (p, d, q)
                    best_model = model
            except Exception:
                continue

    if best_model is None:
        raise RuntimeError('ARIMA selection failed for all candidate orders.')

    return best_order, best_model


def save_acf_pacf_plot(series, file_path, lags=40):
    # Save ACF and PACF plots to support ARIMA order selection.
    # These plots help identify potential autoregressive and
    # moving-average structure in the differenced series.
    series = pd.Series(series).dropna()
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title('ACF Plot')
    ax2 = fig.add_subplot(1, 2, 2)
    plot_pacf(series, lags=lags, ax=ax2, method='ywm')
    ax2.set_title('PACF Plot')
    plt.tight_layout()
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()


def engineer_features(df):
    # Create additional time-series features to improve LSTM
    # learning beyond raw OHLC inputs alone.
    out = df.copy()
    out['log_return'] = np.log(out[CLOSE_COL]).diff()                # standard financial return transformation
    out['rolling_mean_10'] = out[CLOSE_COL].rolling(10).mean()       # short-term trend proxy
    out['rolling_std_10'] = out[CLOSE_COL].rolling(10).std()         # local volatility proxy
    out['high_low_range'] = out[HIGH_COL] - out[LOW_COL]             # intrabar price range
    out['volume_change'] = out[VOLUME_COL].pct_change().replace([np.inf, -np.inf], np.nan)
    return out


def build_feature_matrix(df):
    # Keep only the required model features and remove rows with
    # missing values caused by differencing, rolling windows, or
    # invalid percentage changes.
    feature_cols = [
        OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL,
        'log_return', 'rolling_mean_10', 'rolling_std_10',
        'high_low_range', 'volume_change', 'arima_residual'
    ]
    out = df.copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    return out, feature_cols


def make_multifeature_sequences(features_scaled, close_scaled, window):
    # Convert the tabular data into sliding-window sequences for
    # LSTM input. Each sample uses the previous `window` time steps
    # to predict the next close price.
    X, y = [], []
    for i in range(window, len(features_scaled)):
        X.append(features_scaled[i - window:i, :])
        y.append(close_scaled[i])
    return np.array(X), np.array(y)


def inverse_close_transform(close_scaler, arr_1d):
    # Convert scaled predictions back into the original close-price scale.
    return close_scaler.inverse_transform(np.array(arr_1d).reshape(-1, 1)).flatten()


def best_weight_by_validation_mae(y_true, pred_a, pred_b, grid_step=0.01):
    # Find the best linear combination of ARIMA and LSTM forecasts
    # by minimising validation MAE.
    best_w = 0.5
    best_mae = np.inf
    for w in np.arange(0.0, 1.0 + grid_step, grid_step):
        pred = w * pred_a + (1 - w) * pred_b
        score = mean_absolute_error(y_true, pred)
        if score < best_mae:
            best_mae = score
            best_w = w
    return best_w, 1 - best_w, best_mae

# ============================================================
# 1. LOAD DATA
# ============================================================
# Load and clean the raw Forex dataset before any modelling.
df = load_data(FILE_PATH)
print(f'Total rows after loading/cleaning: {len(df)}')

# Visualise the raw close-price series to inspect overall market behaviour.
plt.figure(figsize=(12, 5))
plt.plot(df[CLOSE_COL].values)
plt.title('Close Price Over Time')
plt.xlabel('Time Index')
plt.ylabel('Close Price')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_3_1_close_price_over_time.png'), bbox_inches='tight')
plt.close()

# ============================================================
# 2. TRAIN / TEST SPLIT (80 / 20) + validation slice inside train
# ============================================================
# Split the raw data chronologically so the model is always trained
# on past observations and evaluated on future observations.
n = len(df)
train_end = int(n * TRAIN_RATIO)

df_train_full = df.iloc[:train_end].copy().reset_index(drop=True)
df_test = df.iloc[train_end:].copy().reset_index(drop=True)

val_size = int(len(df_train_full) * VAL_RATIO_WITHIN_TRAIN)
subtrain_end = len(df_train_full) - val_size

df_subtrain = df_train_full.iloc[:subtrain_end].copy().reset_index(drop=True)
df_val = df_train_full.iloc[subtrain_end:].copy().reset_index(drop=True)

print(f'Subtrain rows: {len(df_subtrain)}')
print(f'Validation rows: {len(df_val)}')
print(f'Train-full rows: {len(df_train_full)}')
print(f'Test rows: {len(df_test)}')

# ============================================================
# 3. ARIMA ANALYSIS
# ============================================================
# Use the subtraining close-price series to determine the
# required differencing order and select the best ARIMA model.
subtrain_close = df_subtrain[CLOSE_COL].astype(float)
d_selected = choose_d_via_adf(subtrain_close, max_d=2, alpha=0.05)
print(f'Selected differencing order d = {d_selected}')

# Save ACF/PACF of the differenced series for model-order support.
series_for_acf = subtrain_close.copy()
for _ in range(d_selected):
    series_for_acf = series_for_acf.diff().dropna()
save_acf_pacf_plot(series_for_acf, os.path.join(OUTPUT_DIR, 'figure_3_3_acf_pacf.png'))

# Fit ARIMA on the subtraining set and record the selected order.
best_order, arima_subtrain_fit = auto_select_arima(subtrain_close, d_selected)
print(f'Selected ARIMA order: {best_order}')

# Run residual diagnostics to check whether the ARIMA model has
# captured the main linear structure in the series.
lb = acorr_ljungbox(arima_subtrain_fit.resid.dropna(), lags=[10], return_df=True)
lb.to_csv(os.path.join(OUTPUT_DIR, 'ljung_box_diagnostic.csv'), index=True)
print('Ljung-Box diagnostic:')
print(lb)

# Generate validation forecasts for hybrid weight calibration.
arima_val_pred = arima_subtrain_fit.forecast(steps=len(df_val))

# Refit ARIMA on the full training set for final test forecasting.
arima_train_fit = ARIMA(df_train_full[CLOSE_COL].astype(float), order=best_order).fit()
arima_test_pred = arima_train_fit.forecast(steps=len(df_test))

# Fit ARIMA across the full dataset to obtain fitted values and residuals.
# These residuals are later used as an additional feature for the LSTM.
arima_all_fit = ARIMA(df[CLOSE_COL].astype(float), order=best_order).fit()
df['arima_fitted'] = pd.Series(arima_all_fit.predict(start=0, end=len(df)-1), index=df.index)
df['arima_residual'] = df[CLOSE_COL] - df['arima_fitted']

# ============================================================
# 4. FEATURE ENGINEERING FOR LSTM
# ============================================================
# Build the multivariate feature set for the LSTM, including the
# ARIMA residuals as a nonlinear-correction feature.
df_feat = engineer_features(df)
df_feat['arima_residual'] = df['arima_residual']

# Save the return series plot for dissertation reporting.
plt.figure(figsize=(12, 5))
plt.plot(df_feat['log_return'].values)
plt.title('Log Returns Plot')
plt.xlabel('Time Index')
plt.ylabel('Log Return')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_3_2_log_returns_plot.png'), bbox_inches='tight')
plt.close()

# Keep only fully valid rows after all engineered features are created.
model_df, feature_cols = build_feature_matrix(df_feat)
print(f'Rows after feature engineering and NaN removal: {len(model_df)}')

# Re-split the aligned modelling dataset chronologically.
n2 = len(model_df)
train_end2 = int(n2 * TRAIN_RATIO)
model_train_full = model_df.iloc[:train_end2].copy().reset_index(drop=True)
model_test = model_df.iloc[train_end2:].copy().reset_index(drop=True)
val_size2 = int(len(model_train_full) * VAL_RATIO_WITHIN_TRAIN)
subtrain_end2 = len(model_train_full) - val_size2
model_subtrain = model_train_full.iloc[:subtrain_end2].copy().reset_index(drop=True)
model_val = model_train_full.iloc[subtrain_end2:].copy().reset_index(drop=True)

# ============================================================
# 4A. SCALING
# ============================================================
# Normalise the input features and target close price to improve
# LSTM training stability and convergence.
feature_scaler = MinMaxScaler()
close_scaler = MinMaxScaler()

X_subtrain_raw = model_subtrain[feature_cols].values
X_val_raw = model_val[feature_cols].values
X_trainfull_raw = model_train_full[feature_cols].values
X_test_raw = model_test[feature_cols].values

y_subtrain_close = model_subtrain[[CLOSE_COL]].values
y_val_close = model_val[[CLOSE_COL]].values
y_trainfull_close = model_train_full[[CLOSE_COL]].values
y_test_close = model_test[[CLOSE_COL]].values

X_subtrain_scaled = feature_scaler.fit_transform(X_subtrain_raw)
X_val_scaled = feature_scaler.transform(X_val_raw)
X_trainfull_scaled = feature_scaler.fit_transform(X_trainfull_raw)
X_test_scaled = feature_scaler.transform(X_test_raw)

y_subtrain_scaled = close_scaler.fit_transform(y_subtrain_close).flatten()
y_val_scaled = close_scaler.transform(y_val_close).flatten()
y_trainfull_scaled = close_scaler.fit_transform(y_trainfull_close).flatten()
y_test_scaled = close_scaler.transform(y_test_close).flatten()

# ============================================================
# 4B. SEQUENCE CONSTRUCTION
# ============================================================
# Transform the scaled data into fixed-length sliding windows
# for supervised LSTM learning.
seq_subtrain_input = X_subtrain_scaled
seq_subtrain_target = y_subtrain_scaled
X_lstm_subtrain, y_lstm_subtrain = make_multifeature_sequences(seq_subtrain_input, seq_subtrain_target, WINDOW)

val_context_features = np.vstack([X_subtrain_scaled[-WINDOW:], X_val_scaled])
val_context_close = np.concatenate([y_subtrain_scaled[-WINDOW:], y_val_scaled])
X_lstm_val, y_lstm_val = make_multifeature_sequences(val_context_features, val_context_close, WINDOW)

# ============================================================
# 4C. LSTM MODEL TRAINING
# ============================================================
# Define the stacked LSTM architecture used to capture nonlinear
# sequential dependencies in the Forex data.
lstm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW, len(feature_cols))),
    Dropout(0.20),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the network using Adam optimiser and MSE loss.
lstm_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

# Early stopping prevents unnecessary training once validation
# performance stops improving.
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = lstm_model.fit(
    X_lstm_subtrain,
    y_lstm_subtrain,
    validation_data=(X_lstm_val, y_lstm_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1
)

# Generate validation predictions for hybrid weight selection.
lstm_val_pred_scaled = lstm_model.predict(X_lstm_val, verbose=0).flatten()
lstm_val_pred = inverse_close_transform(close_scaler, lstm_val_pred_scaled)
actual_val = inverse_close_transform(close_scaler, y_lstm_val)

# Retrain the final LSTM model on the full training set before
# evaluating on the test set.
feature_scaler_full = MinMaxScaler()
close_scaler_full = MinMaxScaler()

X_trainfull_scaled = feature_scaler_full.fit_transform(X_trainfull_raw)
X_test_scaled = feature_scaler_full.transform(X_test_raw)
y_trainfull_scaled = close_scaler_full.fit_transform(y_trainfull_close).flatten()
y_test_scaled = close_scaler_full.transform(y_test_close).flatten()

X_lstm_trainfull, y_lstm_trainfull = make_multifeature_sequences(X_trainfull_scaled, y_trainfull_scaled, WINDOW)

test_context_features = np.vstack([X_trainfull_scaled[-WINDOW:], X_test_scaled])
test_context_close = np.concatenate([y_trainfull_scaled[-WINDOW:], y_test_scaled])
X_lstm_test, y_lstm_test = make_multifeature_sequences(test_context_features, test_context_close, WINDOW)

lstm_model_final = Sequential([
    LSTM(128, return_sequences=True, input_shape=(WINDOW, len(feature_cols))),
    Dropout(0.20),
    LSTM(64, return_sequences=False),
    Dense(32, activation='relu'),
    Dense(1)
])

lstm_model_final.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse')

early_stop_final = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)

lstm_model_final.fit(
    X_lstm_trainfull,
    y_lstm_trainfull,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop_final],
    verbose=1
)

# Convert LSTM test predictions back into the original price scale.
lstm_test_pred_scaled = lstm_model_final.predict(X_lstm_test, verbose=0).flatten()
lstm_test_pred = inverse_close_transform(close_scaler_full, lstm_test_pred_scaled)
actual_test_close = inverse_close_transform(close_scaler_full, y_lstm_test)

# Align ARIMA and LSTM predictions so that both models are
# evaluated over the exact same test horizon.
arima_test_pred = np.array(arima_test_pred)[-len(actual_test_close):]
model_test_close_aligned = actual_test_close.copy()

# ============================================================
# 5. WEIGHTED HYBRID USING VALIDATION MAE
# ============================================================
# Align the validation ARIMA forecast with the LSTM validation
# output, then choose the best blending weights using MAE.
arima_val_pred = np.array(arima_val_pred)[-len(actual_val):]
actual_val_aligned = actual_val.copy()

w_arima, w_lstm, best_val_mae = best_weight_by_validation_mae(
    actual_val_aligned,
    arima_val_pred,
    lstm_val_pred,
    grid_step=0.01
)

print(f'Best validation weights by MAE -> ARIMA: {w_arima:.2f}, LSTM: {w_lstm:.2f}')
print(f'Best validation hybrid MAE: {best_val_mae:.6f}')

# Compute the final hybrid point forecast as the weighted
# combination of ARIMA and LSTM predictions.
hybrid_test_pred = w_arima * arima_test_pred + w_lstm * lstm_test_pred

# ============================================================
# 6. GARCH VOLATILITY MODEL
# ============================================================
# Fit GARCH(1,1) on training log returns to model volatility
# clustering and estimate time-varying uncertainty.
train_returns = np.log(model_train_full[CLOSE_COL].astype(float)).diff()
train_returns = pd.Series(train_returns).replace([np.inf, -np.inf], np.nan).dropna()

if len(train_returns) < 50:
    raise ValueError('Not enough clean returns for GARCH fitting.')

garch = arch_model(train_returns * 100, p=GARCH_P, q=GARCH_Q, vol='Garch', dist='normal')
garch_fit = garch.fit(disp='off')
print(garch_fit.summary())

# Forecast conditional variance across the test horizon.
garch_forecast = garch_fit.forecast(horizon=len(hybrid_test_pred))
variance = garch_forecast.variance.values[-1, :]
sigma = np.sqrt(variance) / 100.0

# Build 95% volatility-aware confidence intervals around the
# hybrid point forecast.
k = 1.96
upper_band = hybrid_test_pred * (1 + k * sigma)
lower_band = hybrid_test_pred * (1 - k * sigma)

# Save volatility plot for reporting.
plt.figure(figsize=(12, 5))
plt.plot(sigma)
plt.title('GARCH(1,1) Conditional Volatility Over Test Period')
plt.xlabel('Test Index')
plt.ylabel('Conditional Volatility')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_4_3_garch_volatility.png'), bbox_inches='tight')
plt.close()

# ============================================================
# 7. EVALUATION
# ============================================================
# Compare standalone ARIMA, standalone LSTM, and the hybrid
# model using MAE and RMSE on the held-out test set.
print('\n================ TEST SET RESULTS ================')
arima_mae, arima_rmse = print_metrics('ARIMA', model_test_close_aligned, arima_test_pred)
lstm_mae, lstm_rmse = print_metrics('LSTM', model_test_close_aligned, lstm_test_pred)
hybrid_mae, hybrid_rmse = print_metrics('HYBRID', model_test_close_aligned, hybrid_test_pred)

performance_df = pd.DataFrame({
    'Model': ['ARIMA', 'LSTM', 'Hybrid ARIMA-LSTM-GARCH'],
    'MAE': [arima_mae, lstm_mae, hybrid_mae],
    'RMSE': [arima_rmse, lstm_rmse, hybrid_rmse]
})
performance_df.to_csv(os.path.join(OUTPUT_DIR, 'table_4_1_performance_metrics.csv'), index=False)

# ============================================================
# 8. VISUALS FOR REPORT
# ============================================================
# Generate final comparison plots for standalone and hybrid models.
plt.figure(figsize=(12, 5))
plt.plot(model_test_close_aligned, label='Actual')
plt.plot(arima_test_pred, label='ARIMA')
plt.title('ARIMA Standalone Predictions vs Actual Close Price')
plt.xlabel('Test Index')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_4_1_arima_vs_actual.png'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(model_test_close_aligned, label='Actual')
plt.plot(lstm_test_pred, label='LSTM')
plt.title('LSTM Standalone Predictions vs Actual Close Price')
plt.xlabel('Test Index')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_4_2_lstm_vs_actual.png'), bbox_inches='tight')
plt.close()

plt.figure(figsize=(12, 5))
plt.plot(model_test_close_aligned, label='Actual')
plt.plot(hybrid_test_pred, label='Hybrid')
plt.fill_between(np.arange(len(hybrid_test_pred)), lower_band, upper_band, alpha=0.2, label='95% GARCH Band')
plt.title('Hybrid ARIMA-LSTM-GARCH Predictions with Volatility Confidence Bands')
plt.xlabel('Test Index')
plt.ylabel('Close Price')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'figure_4_4_hybrid_with_bands.png'), bbox_inches='tight')
plt.close()

# ============================================================
# 9. SAVE OUTPUTS
# ============================================================
# Save the final prediction table, training history, and run
# summary so the experiment can be reviewed and reported easily.
results_df = pd.DataFrame({
    'Actual': model_test_close_aligned,
    'ARIMA_Pred': arima_test_pred,
    'LSTM_Pred': lstm_test_pred,
    'Hybrid_Pred': hybrid_test_pred,
    'Sigma': sigma,
    'Lower_Band_95': lower_band,
    'Upper_Band_95': upper_band
})
results_df.to_csv(os.path.join(OUTPUT_DIR, 'hybrid_forex_forecast_outputs.csv'), index=False)

history_df = pd.DataFrame(history.history)
history_df.to_csv(os.path.join(OUTPUT_DIR, 'lstm_training_history.csv'), index=False)

summary_path = os.path.join(OUTPUT_DIR, 'run_summary.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('Hybrid ARIMA-LSTM-GARCH Run Summary\n')
    f.write('===================================\n')
    f.write(f'Dataset path: {FILE_PATH}\n')
    f.write(f'Rows loaded: {len(df)}\n')
    f.write(f'Rows after feature engineering: {len(model_df)}\n')
    f.write(f'Selected ARIMA order: {best_order}\n')
    f.write(f'Validation weights -> ARIMA: {w_arima:.2f}, LSTM: {w_lstm:.2f}\n')
    f.write('\nPerformance:\n')
    f.write(performance_df.to_string(index=False))

print(f'\nSaved all outputs to: {OUTPUT_DIR}')
print('Main result file: hybrid_outputs/hybrid_forex_forecast_outputs.csv')
