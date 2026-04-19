"""
garch_model.py
------------------------------------------------------------
Purpose:
    Estimate conditional volatility using a GARCH(1,1) model
    fitted on the log-return series of the close price.

This script performs:
    - log-return computation
    - GARCH(1,1) fitting
    - volatility forecasting
    - volatility plot generation
------------------------------------------------------------
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model

# ============================================================
# SETTINGS
# ============================================================
FILE_PATH = "Forex_OHLC_50k_Dataset.xlsx"
OUTPUT_DIR = "hybrid_outputs"

CLOSE_COL = "Close"
TRAIN_RATIO = 0.80
GARCH_P = 1
GARCH_Q = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_close_series(path: str) -> pd.Series:
    """
    Load the close price series from CSV or Excel.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    if CLOSE_COL not in df.columns:
        raise ValueError(f"'{CLOSE_COL}' column not found.")

    df[CLOSE_COL] = pd.to_numeric(df[CLOSE_COL], errors="coerce")
    df = df.dropna(subset=[CLOSE_COL]).reset_index(drop=True)
    return df[CLOSE_COL].astype(float)


# ============================================================
# MAIN EXECUTION
# ============================================================
def main() -> None:
    # --------------------------------------------------------
    # STEP 1: Load close series
    # GARCH is applied to returns, not directly to prices.
    # --------------------------------------------------------
    close_series = load_close_series(FILE_PATH)

    # --------------------------------------------------------
    # STEP 2: Create train/test split for horizon definition
    # The test length determines how far ahead volatility is
    # forecast for the final reporting stage.
    # --------------------------------------------------------
    train_end = int(len(close_series) * TRAIN_RATIO)
    train_close = close_series.iloc[:train_end].reset_index(drop=True)
    test_close = close_series.iloc[train_end:].reset_index(drop=True)

    # --------------------------------------------------------
    # STEP 3: Compute log returns
    # Returns are standard input for volatility modelling.
    # --------------------------------------------------------
    train_returns = np.log(train_close).diff()
    train_returns = pd.Series(train_returns).replace([np.inf, -np.inf], np.nan).dropna()

    if len(train_returns) < 50:
        raise ValueError("Not enough clean returns for GARCH fitting.")

    # --------------------------------------------------------
    # STEP 4: Fit GARCH(1,1)
    # GARCH models volatility clustering in financial returns.
    # --------------------------------------------------------
    garch = arch_model(
        train_returns * 100,
        p=GARCH_P,
        q=GARCH_Q,
        vol="Garch",
        dist="normal"
    )
    garch_fit = garch.fit(disp="off")
    print(garch_fit.summary())

    # --------------------------------------------------------
    # STEP 5: Forecast conditional variance over test horizon
    # These forecasts estimate time-varying uncertainty.
    # --------------------------------------------------------
    horizon = len(test_close)
    garch_forecast = garch_fit.forecast(horizon=horizon)
    variance = garch_forecast.variance.values[-1, :]
    sigma = np.sqrt(variance) / 100.0

    # --------------------------------------------------------
    # STEP 6: Save parameter estimates and volatility forecasts
    # --------------------------------------------------------
    params_df = pd.DataFrame({
        "Parameter": garch_fit.params.index,
        "Value": garch_fit.params.values
    })
    params_df.to_csv(os.path.join(OUTPUT_DIR, "garch_parameter_estimates.csv"), index=False)

    pd.DataFrame({
        "Test_Index": np.arange(len(sigma)),
        "Sigma": sigma
    }).to_csv(os.path.join(OUTPUT_DIR, "garch_volatility_forecast.csv"), index=False)

    # --------------------------------------------------------
    # STEP 7: Plot conditional volatility
    # This visual supports the discussion of volatility regimes.
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(sigma)
    plt.title("GARCH(1,1) Conditional Volatility Over Test Period")
    plt.xlabel("Test Index")
    plt.ylabel("Conditional Volatility")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_4_3_garch_volatility.png"), bbox_inches="tight")
    plt.close()

    print(f"\nGARCH outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
