"""
arima_analysis.py
------------------------------------------------------------
Purpose:
    Fit the ARIMA baseline model for Forex close price prediction.
    This file performs:
    - stationarity testing (ADF)
    - ACF/PACF plotting
    - ARIMA order selection
    - residual diagnostics
    - test-set forecasting

Outputs:
    - ARIMA predictions
    - ARIMA residuals
    - diagnostic tables and plots
------------------------------------------------------------
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ============================================================
# SETTINGS
# ============================================================
FILE_PATH = "Forex_OHLC_50k_Dataset.xlsx"
OUTPUT_DIR = "hybrid_outputs"

CLOSE_COL = "Close"
TRAIN_RATIO = 0.80

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def load_close_series(path: str) -> pd.Series:
    """
    Load the dataset and return the close price series.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if CLOSE_COL not in df.columns:
        raise ValueError(f"'{CLOSE_COL}' column not found in dataset.")

    df[CLOSE_COL] = pd.to_numeric(df[CLOSE_COL], errors="coerce")
    df = df.dropna(subset=[CLOSE_COL]).reset_index(drop=True)
    return df[CLOSE_COL].astype(float)


def adf_stationarity(series: pd.Series) -> dict:
    """
    Run Augmented Dickey-Fuller test to assess stationarity.
    """
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "adf_statistic": result[0],
        "p_value": result[1],
        "lags_used": result[2],
        "n_obs": result[3]
    }


def choose_d_via_adf(close_series: pd.Series, max_d: int = 2, alpha: float = 0.05) -> int:
    """
    Select differencing order d based on repeated ADF testing.
    """
    current = pd.Series(close_series).astype(float).copy()
    for d in range(max_d + 1):
        test = adf_stationarity(current)
        print(f"ADF check for d={d}: p-value={test['p_value']:.6f}")
        if test["p_value"] < alpha:
            return d
        current = current.diff().dropna()
    return max_d


def auto_select_arima(train_series: pd.Series, d: int, p_range=range(0, 4), q_range=range(0, 4)):
    """
    Try auto_arima first; if unavailable, fall back to manual AIC search.
    """
    best_order = None
    best_aic = np.inf
    best_model = None

    try:
        import pmdarima as pm
        print("[ARIMA] Trying pmdarima auto_arima...")
        auto_model = pm.auto_arima(
            train_series,
            d=d,
            start_p=0,
            start_q=0,
            max_p=5,
            max_q=5,
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False
        )
        order = auto_model.order
        model = ARIMA(train_series, order=order).fit()
        return order, model
    except Exception as e:
        print(f"[ARIMA] auto_arima unavailable. Using manual search instead. Reason: {e}")

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
        raise RuntimeError("ARIMA order selection failed.")

    return best_order, best_model


def save_acf_pacf_plot(series: pd.Series, file_path: str, lags: int = 40) -> None:
    """
    Save ACF and PACF plots to support ARIMA order selection.
    """
    series = pd.Series(series).dropna()
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    plot_acf(series, lags=lags, ax=ax1)
    ax1.set_title("ACF Plot")

    ax2 = fig.add_subplot(1, 2, 2)
    plot_pacf(series, lags=lags, ax=ax2, method="ywm")
    ax2.set_title("PACF Plot")

    plt.tight_layout()
    plt.savefig(file_path, bbox_inches="tight")
    plt.close()


# ============================================================
# MAIN EXECUTION
# ============================================================
def main() -> None:
    # --------------------------------------------------------
    # STEP 1: Load close price series
    # The ARIMA model is fitted only on the close series
    # because it serves as the linear benchmark model.
    # --------------------------------------------------------
    close_series = load_close_series(FILE_PATH)
    print(f"Rows loaded for ARIMA: {len(close_series)}")

    # --------------------------------------------------------
    # STEP 2: Split data chronologically
    # A time-based split avoids look-ahead bias.
    # --------------------------------------------------------
    train_end = int(len(close_series) * TRAIN_RATIO)
    train_series = close_series.iloc[:train_end].reset_index(drop=True)
    test_series = close_series.iloc[train_end:].reset_index(drop=True)

    # --------------------------------------------------------
    # STEP 3: Determine differencing order d using ADF test
    # ARIMA requires stationarity in the input series.
    # --------------------------------------------------------
    d_selected = choose_d_via_adf(train_series, max_d=2, alpha=0.05)
    print(f"Selected differencing order d = {d_selected}")

    # --------------------------------------------------------
    # STEP 4: Plot ACF/PACF of differenced training series
    # These plots help justify the AR and MA orders.
    # --------------------------------------------------------
    series_for_acf = train_series.copy()
    for _ in range(d_selected):
        series_for_acf = series_for_acf.diff().dropna()
    save_acf_pacf_plot(series_for_acf, os.path.join(OUTPUT_DIR, "figure_3_3_acf_pacf.png"))

    # --------------------------------------------------------
    # STEP 5: Select ARIMA order using AIC
    # Lower AIC generally indicates a better trade-off between
    # goodness-of-fit and model complexity.
    # --------------------------------------------------------
    best_order, arima_fit = auto_select_arima(train_series, d_selected)
    print(f"Selected ARIMA order: {best_order}")

    # --------------------------------------------------------
    # STEP 6: Run residual diagnostics
    # Ljung-Box checks whether residual autocorrelation remains.
    # --------------------------------------------------------
    lb = acorr_ljungbox(arima_fit.resid.dropna(), lags=[10], return_df=True)
    lb.to_csv(os.path.join(OUTPUT_DIR, "ljung_box_diagnostic.csv"), index=True)
    print("\nLjung-Box diagnostic:")
    print(lb)

    # --------------------------------------------------------
    # STEP 7: Forecast on the held-out test set
    # This gives out-of-sample ARIMA predictions.
    # --------------------------------------------------------
    arima_test_pred = arima_fit.forecast(steps=len(test_series))

    # --------------------------------------------------------
    # STEP 8: Refit on full series to produce fitted values and
    # residuals for downstream use in the hybrid/LSTM pipeline.
    # --------------------------------------------------------
    arima_all_fit = ARIMA(close_series, order=best_order).fit()
    fitted = pd.Series(arima_all_fit.predict(start=0, end=len(close_series)-1))
    residuals = close_series - fitted

    # --------------------------------------------------------
    # STEP 9: Evaluate ARIMA forecast accuracy on test set
    # --------------------------------------------------------
    mae = mean_absolute_error(test_series, arima_test_pred)
    score_rmse = rmse(test_series, arima_test_pred)
    print(f"\nARIMA Test MAE: {mae:.6f}")
    print(f"ARIMA Test RMSE: {score_rmse:.6f}")

    metrics_df = pd.DataFrame({
        "Model": ["ARIMA"],
        "MAE": [mae],
        "RMSE": [score_rmse]
    })
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "arima_performance_metrics.csv"), index=False)

    # --------------------------------------------------------
    # STEP 10: Save fitted values and residuals
    # Residuals are important because the LSTM later uses them
    # as an additional feature representing ARIMA's errors.
    # --------------------------------------------------------
    residual_df = pd.DataFrame({
        "Close": close_series.values,
        "ARIMA_Fitted": fitted.values,
        "ARIMA_Residual": residuals.values
    })
    residual_df.to_csv(os.path.join(OUTPUT_DIR, "arima_residuals_full.csv"), index=False)

    pred_df = pd.DataFrame({
        "Actual": test_series.values,
        "ARIMA_Pred": np.array(arima_test_pred)
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "arima_test_predictions.csv"), index=False)

    # --------------------------------------------------------
    # STEP 11: Plot ARIMA predictions against actual values
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(test_series.values, label="Actual")
    plt.plot(np.array(arima_test_pred), label="ARIMA")
    plt.title("ARIMA Standalone Predictions vs Actual Close Price")
    plt.xlabel("Test Index")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_4_1_arima_vs_actual.png"), bbox_inches="tight")
    plt.close()

    print(f"\nARIMA outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
