"""
Step 2: Feature Engineering ("Market DNA")
--------------------------------------------
Raw price alone is not enough for KNN, since Bitcoin's price scale has
changed by orders of magnitude over the years. Instead we engineer
stationary, normalized indicators that describe momentum, volatility,
and trend, so that similar market *behavior* -- not similar price level
-- is what drives the distance calculation.

- Momentum:   Daily_Return, RSI_14
- Volatility: ATR_14, NATR_14 (normalized to be price-scale independent)
- Trend:      SMA_50, Dist_to_SMA_50
"""

import pandas as pd
import pandas_ta as ta


FEATURE_COLUMNS = ["Daily_Return", "RSI_14", "NATR_14", "Dist_to_SMA_50"]


def add_market_dna_features(df):
    """Compute momentum, volatility, and trend indicators."""
    print("\nCalculating Market DNA Features...")

    # Momentum
    df["Daily_Return"] = df["Close"].pct_change()
    df["RSI_14"] = df.ta.rsi(length=14)

    # -1. VOLATILITY INDICATORS -
    df["ATR_14"] = df.ta.atr(length=14)
    df["NATR_14"] = (df["ATR_14"] / df["Close"]) * 100

    # -2. TREND INDICATORS -
    df["SMA_50"] = df.ta.sma(length=50)
    df["Dist_to_SMA_50"] = (df["Close"] - df["SMA_50"]) / df["SMA_50"]

    # - 3. FINAL CLEANUP -
    df.dropna(inplace=True)

    print("Feature Engineering Complete! Here are the first 5 rows of our new features:")
    print(df[FEATURE_COLUMNS].head())

    return df


if __name__ == "__main__":
    from preprocessing import load_clean_data

    df = load_clean_data()
    df = add_market_dna_features(df)
    print(df[FEATURE_COLUMNS].describe())
