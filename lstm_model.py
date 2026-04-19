"""
lstm_model.py
------------------------------------------------------------
Purpose:
    Build and train the LSTM component of the hybrid Forex
    prediction pipeline using OHLC features and ARIMA residuals.

This script performs:
    - feature engineering
    - data scaling
    - sliding-window sequence creation
    - LSTM training and validation
    - test-set prediction and evaluation
------------------------------------------------------------
"""

import warnings
warnings.filterwarnings("ignore")

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ============================================================
# SETTINGS
# ============================================================
FILE_PATH = "Forex_OHLC_50k_Dataset.xlsx"
ARIMA_RESIDUALS_PATH = "hybrid_outputs/arima_residuals_full.csv"
OUTPUT_DIR = "hybrid_outputs"

OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOLUME_COL = "Volume"

WINDOW = 60
SEED = 42
MAX_EPOCHS = 13
BATCH_SIZE = 64
LEARNING_RATE = 0.001

TRAIN_RATIO = 0.80
VAL_RATIO_WITHIN_TRAIN = 0.125   # 12.5% of 80% = 10% of total

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_data(path: str) -> pd.DataFrame:
    """
    Load the Forex OHLC dataset and ensure the core columns exist.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    needed = [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found.")

    if VOLUME_COL not in df.columns:
        print(f"[INFO] '{VOLUME_COL}' not found. Creating dummy volume column.")
        df[VOLUME_COL] = 0.0

    for col in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional explanatory features for LSTM training.
    """
    out = df.copy()
    out["log_return"] = np.log(out[CLOSE_COL]).diff()
    out["rolling_mean_10"] = out[CLOSE_COL].rolling(10).mean()
    out["rolling_std_10"] = out[CLOSE_COL].rolling(10).std()
    out["high_low_range"] = out[HIGH_COL] - out[LOW_COL]
    out["volume_change"] = out[VOLUME_COL].pct_change().replace([np.inf, -np.inf], np.nan)
    return out


def build_feature_matrix(df: pd.DataFrame):
    """
    Keep only the columns required for LSTM input and remove invalid rows.
    """
    feature_cols = [
        OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL,
        "log_return", "rolling_mean_10", "rolling_std_10",
        "high_low_range", "volume_change", "arima_residual"
    ]
    out = df.copy().replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feature_cols).reset_index(drop=True)
    return out, feature_cols


def make_multifeature_sequences(features_scaled, close_scaled, window):
    """
    Convert tabular time-series data into sliding-window sequences
    suitable for LSTM input.
    """
    X, y = [], []
    for i in range(window, len(features_scaled)):
        X.append(features_scaled[i - window:i, :])
        y.append(close_scaled[i])
    return np.array(X), np.array(y)


def inverse_close_transform(close_scaler, arr_1d):
    """
    Convert scaled close-price predictions back to original price space.
    """
    return close_scaler.inverse_transform(np.array(arr_1d).reshape(-1, 1)).flatten()


# ============================================================
# MAIN EXECUTION
# ============================================================
def main() -> None:
    # --------------------------------------------------------
    # STEP 1: Load raw OHLC data and ARIMA residuals
    # The LSTM uses ARIMA residuals as an extra feature so it
    # can learn nonlinear corrections to the linear baseline.
    # --------------------------------------------------------
    df = load_data(FILE_PATH)

    if not os.path.exists(ARIMA_RESIDUALS_PATH):
        raise FileNotFoundError(
            f"ARIMA residual file not found: {ARIMA_RESIDUALS_PATH}\n"
            "Run arima_analysis.py first."
        )

    residual_df = pd.read_csv(ARIMA_RESIDUALS_PATH)
    if "ARIMA_Residual" not in residual_df.columns:
        raise ValueError("ARIMA residual file must contain 'ARIMA_Residual' column.")

    if len(residual_df) != len(df):
        raise ValueError("ARIMA residual file length does not match raw dataset length.")

    # --------------------------------------------------------
    # STEP 2: Feature engineering
    # These features give the LSTM richer market context than
    # raw OHLC values alone.
    # --------------------------------------------------------
    df_feat = engineer_features(df)
    df_feat["arima_residual"] = residual_df["ARIMA_Residual"].values

    # Plot log returns for reporting.
    plt.figure(figsize=(12, 5))
    plt.plot(df_feat["log_return"].values)
    plt.title("Log Returns Plot")
    plt.xlabel("Time Index")
    plt.ylabel("Log Return")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_3_2_log_returns_plot.png"), bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # STEP 3: Build feature matrix and remove invalid rows
    # Rolling statistics and returns create NaNs at the start,
    # which must be removed before sequence construction.
    # --------------------------------------------------------
    model_df, feature_cols = build_feature_matrix(df_feat)
    print(f"Rows after feature engineering: {len(model_df)}")

    # --------------------------------------------------------
    # STEP 4: Chronological train/validation/test split
    # This preserves temporal order and avoids leakage.
    # --------------------------------------------------------
    n = len(model_df)
    train_end = int(n * TRAIN_RATIO)

    model_train_full = model_df.iloc[:train_end].copy().reset_index(drop=True)
    model_test = model_df.iloc[train_end:].copy().reset_index(drop=True)

    val_size = int(len(model_train_full) * VAL_RATIO_WITHIN_TRAIN)
    subtrain_end = len(model_train_full) - val_size

    model_subtrain = model_train_full.iloc[:subtrain_end].copy().reset_index(drop=True)
    model_val = model_train_full.iloc[subtrain_end:].copy().reset_index(drop=True)

    # --------------------------------------------------------
    # STEP 5: Scale features and close target
    # Scaling improves neural-network optimisation stability.
    # --------------------------------------------------------
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

    y_subtrain_scaled = close_scaler.fit_transform(y_subtrain_close).flatten()
    y_val_scaled = close_scaler.transform(y_val_close).flatten()

    # --------------------------------------------------------
    # STEP 6: Build sliding-window sequences
    # Each sample contains WINDOW previous time steps used to
    # predict the next close price.
    # --------------------------------------------------------
    X_lstm_subtrain, y_lstm_subtrain = make_multifeature_sequences(
        X_subtrain_scaled, y_subtrain_scaled, WINDOW
    )

    val_context_features = np.vstack([X_subtrain_scaled[-WINDOW:], X_val_scaled])
    val_context_close = np.concatenate([y_subtrain_scaled[-WINDOW:], y_val_scaled])
    X_lstm_val, y_lstm_val = make_multifeature_sequences(
        val_context_features, val_context_close, WINDOW
    )

    # --------------------------------------------------------
    # STEP 7: Define LSTM architecture
    # A stacked LSTM is used to capture both shorter and longer
    # dependencies in the Forex sequence data.
    # --------------------------------------------------------
    lstm_model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(WINDOW, len(feature_cols))),
        Dropout(0.20),
        LSTM(64, return_sequences=False),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    # --------------------------------------------------------
    # STEP 8: Compile and train the model
    # Adam is used for optimisation, and MSE is used because
    # this is a regression problem.
    # --------------------------------------------------------
    lstm_model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = lstm_model.fit(
        X_lstm_subtrain,
        y_lstm_subtrain,
        validation_data=(X_lstm_val, y_lstm_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )

    # --------------------------------------------------------
    # STEP 9: Validation prediction
    # This is useful later for hybrid weighting.
    # --------------------------------------------------------
    lstm_val_pred_scaled = lstm_model.predict(X_lstm_val, verbose=0).flatten()
    lstm_val_pred = inverse_close_transform(close_scaler, lstm_val_pred_scaled)
    actual_val = inverse_close_transform(close_scaler, y_lstm_val)

    pd.DataFrame({
        "Actual_Val": actual_val,
        "LSTM_Val_Pred": lstm_val_pred
    }).to_csv(os.path.join(OUTPUT_DIR, "lstm_validation_predictions.csv"), index=False)

    # --------------------------------------------------------
    # STEP 10: Retrain on full training data for final testing
    # Once validation has been used for model selection, the
    # final LSTM is trained on the full training portion.
    # --------------------------------------------------------
    feature_scaler_full = MinMaxScaler()
    close_scaler_full = MinMaxScaler()

    X_trainfull_scaled = feature_scaler_full.fit_transform(X_trainfull_raw)
    X_test_scaled = feature_scaler_full.transform(X_test_raw)
    y_trainfull_scaled = close_scaler_full.fit_transform(y_trainfull_close).flatten()
    y_test_scaled = close_scaler_full.transform(y_test_close).flatten()

    X_lstm_trainfull, y_lstm_trainfull = make_multifeature_sequences(
        X_trainfull_scaled, y_trainfull_scaled, WINDOW
    )

    test_context_features = np.vstack([X_trainfull_scaled[-WINDOW:], X_test_scaled])
    test_context_close = np.concatenate([y_trainfull_scaled[-WINDOW:], y_test_scaled])
    X_lstm_test, y_lstm_test = make_multifeature_sequences(
        test_context_features, test_context_close, WINDOW
    )

    lstm_model_final = Sequential([
        LSTM(128, return_sequences=True, input_shape=(WINDOW, len(feature_cols))),
        Dropout(0.20),
        LSTM(64, return_sequences=False),
        Dense(32, activation="relu"),
        Dense(1)
    ])

    lstm_model_final.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="mse")

    early_stop_final = EarlyStopping(
        monitor="loss",
        patience=10,
        restore_best_weights=True
    )

    lstm_model_final.fit(
        X_lstm_trainfull,
        y_lstm_trainfull,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop_final],
        verbose=1
    )

    # --------------------------------------------------------
    # STEP 11: Test prediction and inverse scaling
    # Convert predictions back to the original price scale.
    # --------------------------------------------------------
    lstm_test_pred_scaled = lstm_model_final.predict(X_lstm_test, verbose=0).flatten()
    lstm_test_pred = inverse_close_transform(close_scaler_full, lstm_test_pred_scaled)
    actual_test_close = inverse_close_transform(close_scaler_full, y_lstm_test)

    # --------------------------------------------------------
    # STEP 12: Evaluate final LSTM accuracy
    # --------------------------------------------------------
    mae = mean_absolute_error(actual_test_close, lstm_test_pred)
    score_rmse = rmse(actual_test_close, lstm_test_pred)
    print(f"\nLSTM Test MAE: {mae:.6f}")
    print(f"LSTM Test RMSE: {score_rmse:.6f}")

    pd.DataFrame({
        "Model": ["LSTM"],
        "MAE": [mae],
        "RMSE": [score_rmse]
    }).to_csv(os.path.join(OUTPUT_DIR, "lstm_performance_metrics.csv"), index=False)

    # --------------------------------------------------------
    # STEP 13: Save predictions and training history
    # These are needed for final comparison and hybridisation.
    # --------------------------------------------------------
    pd.DataFrame({
        "Actual": actual_test_close,
        "LSTM_Pred": lstm_test_pred
    }).to_csv(os.path.join(OUTPUT_DIR, "lstm_test_predictions.csv"), index=False)

    pd.DataFrame(history.history).to_csv(
        os.path.join(OUTPUT_DIR, "lstm_training_history.csv"),
        index=False
    )

    # --------------------------------------------------------
    # STEP 14: Save LSTM prediction plot
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(actual_test_close, label="Actual")
    plt.plot(lstm_test_pred, label="LSTM")
    plt.title("LSTM Standalone Predictions vs Actual Close Price")
    plt.xlabel("Test Index")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_4_2_lstm_vs_actual.png"), bbox_inches="tight")
    plt.close()

    print(f"\nLSTM outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
