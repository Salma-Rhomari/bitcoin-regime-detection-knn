"""
Step 3: Target Variable Formulation (Regime Labeling)
--------------------------------------------------------
Bitcoin is highly volatile, so regimes are labeled using a 5-day
forward-looking return, split into three classes:

    Bull (+1): future 5-day return > +3%
    Bear (-1): future 5-day return < -3%
    Range (0): everything in between (choppy / sideways market)
"""

HORIZON = 5
BULL_THRESHOLD = 0.03
BEAR_THRESHOLD = -0.03


def get_regime(ret):
    if ret > BULL_THRESHOLD:
        return 1   # Bull Market
    if ret < BEAR_THRESHOLD:
        return -1  # Bear Market
    return 0       # Range / Choppy Market


def add_regime_labels(df, horizon=HORIZON):
    print("\nCréation des labels de régimes...")
    df["Future_Return"] = df["Close"].pct_change(periods=horizon).shift(-horizon)
    df["Regime"] = df["Future_Return"].apply(get_regime)
    df.dropna(inplace=True)

    print("Distribution des régimes dans l'historique :")
    print(df["Regime"].value_counts())

    return df


if __name__ == "__main__":
    from preprocessing import load_clean_data
    from features import add_market_dna_features

    df = load_clean_data()
    df = add_market_dna_features(df)
    df = add_regime_labels(df)
