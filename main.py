"""
SAFE entry point for the full Bitcoin Regime Detection (KNN) pipeline.

Run with:  python main.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import load_raw_data, export_clean_data, load_clean_data, check_missing_values
from features import add_market_dna_features
from labeling import add_regime_labels
from model import split_and_scale, train_knn, evaluate
from visualize import plot_predictions


def main():
    # Step 1: Load & clean raw data
    raw = load_raw_data("data/bitcoin_history.csv")
    export_clean_data(raw, "data/bitcoin_clean.csv")
    df = load_clean_data("data/bitcoin_clean.csv")
    df = check_missing_values(df)

    # Step 2: Feature engineering (Market DNA)
    df = add_market_dna_features(df)

    # Step 3: Regime labeling
    df = add_regime_labels(df)

    # Step 4 & 5: Split, scale, train, evaluate
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(df)
    knn = train_knn(X_train_scaled, y_train)
    y_pred = evaluate(knn, X_test_scaled, y_test)

    # Step 6: Visualize
    plot_predictions(df, y_test, y_pred)


if __name__ == "__main__":
    main()
