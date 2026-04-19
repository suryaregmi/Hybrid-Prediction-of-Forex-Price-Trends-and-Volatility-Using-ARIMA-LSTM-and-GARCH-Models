"""
visualise_data.py
------------------------------------------------------------
Purpose:
    Load the Forex OHLC dataset, inspect its structure, compute
    basic descriptive statistics, and generate exploratory plots
    used in the dissertation.

Run this file first to create the visual outputs for the dataset.
------------------------------------------------------------
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================
FILE_PATH = "Forex_OHLC_50k_Dataset.xlsx"   # Change if needed
OUTPUT_DIR = "hybrid_outputs"

OPEN_COL = "Open"
HIGH_COL = "High"
LOW_COL = "Low"
CLOSE_COL = "Close"
VOLUME_COL = "Volume"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_data(path: str) -> pd.DataFrame:
    """
    Load the dataset from CSV or Excel and validate required columns.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    required_cols = [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}. Found: {list(df.columns)}")

    # If volume is missing, create a dummy volume column so the rest
    # of the pipeline can still run consistently.
    if VOLUME_COL not in df.columns:
        print(f"[INFO] '{VOLUME_COL}' not found. Creating dummy volume column.")
        df[VOLUME_COL] = 0.0

    # Convert numerical columns safely.
    for col in [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where core OHLC values are missing.
    df = df.dropna(subset=[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL]).reset_index(drop=True)
    return df


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """
    Compute log returns from the close price series.
    """
    return np.log(df[CLOSE_COL]).diff()


# ============================================================
# MAIN EXECUTION
# ============================================================
def main() -> None:
    # --------------------------------------------------------
    # STEP 1: Load and inspect dataset
    # --------------------------------------------------------
    df = load_data(FILE_PATH)
    print(f"Rows loaded: {len(df)}")
    print("\nFirst 5 rows:")
    print(df.head())

    # --------------------------------------------------------
    # STEP 2: Produce descriptive statistics for reporting
    # --------------------------------------------------------
    summary = df[[OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]].describe()
    summary.to_csv(os.path.join(OUTPUT_DIR, "dataset_statistical_summary.csv"))
    print("\nStatistical summary:")
    print(summary)

    # --------------------------------------------------------
    # STEP 3: Plot close price time series
    # This visual shows overall market movement across time.
    # --------------------------------------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(df[CLOSE_COL].values)
    plt.title("Close Price Over Time")
    plt.xlabel("Time Index")
    plt.ylabel("Close Price")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "figure_3_1_close_price_over_time.png"), bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # STEP 4: Compute log returns
    # Returns are more suitable than raw prices for volatility
    # analysis because they are closer to stationarity.
    # --------------------------------------------------------
    df["log_return"] = compute_log_returns(df)

    # --------------------------------------------------------
    # STEP 5: Plot return distribution
    # This helps examine skewness, heavy tails, and volatility.
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.hist(df["log_return"].dropna(), bins=60)
    plt.title("Distribution of Log Returns")
    plt.xlabel("Log Return")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "return_distribution.png"), bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # STEP 6: Plot volume distribution
    # Volume can provide a rough signal of market activity.
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))
    plt.hist(df[VOLUME_COL].dropna(), bins=60)
    plt.title("Distribution of Volume")
    plt.xlabel("Volume")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "volume_distribution.png"), bbox_inches="tight")
    plt.close()

    # --------------------------------------------------------
    # STEP 7: Annotate volatility regimes using rolling std
    # This is a simple volatility-regime visual for discussion.
    # --------------------------------------------------------
    rolling_vol = df["log_return"].rolling(100).std()

    plt.figure(figsize=(12, 5))
    plt.plot(rolling_vol.values)
    plt.title("Rolling Volatility (100-step std of log returns)")
    plt.xlabel("Time Index")
    plt.ylabel("Volatility")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "volatility_regime_annotation.png"), bbox_inches="tight")
    plt.close()

    print(f"\nAll visualisation outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
