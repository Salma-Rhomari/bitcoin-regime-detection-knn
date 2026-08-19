"""
Step 1: Data Loading and Preprocessing
----------------------------------------
Loads the raw Bitcoin CSV, strips whitespace from headers, converts
string-formatted numbers (with comma thousand separators) into proper
floats/ints, builds a chronological DatetimeIndex, and checks for
missing values (with a Forward Fill safety net).
"""

import pandas as pd


def load_raw_data(path="data/bitcoin_history.csv"):
    """Load the raw Bitcoin dataset and standardize column formatting."""
    df = pd.read_csv(path, sep=",", quotechar='"', thousands=",")
    df.columns = df.columns.str.strip()
    print("Types de données après correction :")
    print(df.dtypes)
    return df


def export_clean_data(df, path="data/bitcoin_clean.csv"):
    """Save a standardized, comma-delimited checkpoint of the dataset."""
    df.to_csv(path, sep=",", index=False)
    print(f"\nLe fichier '{path}' a été mis à jour avec les corrections.")


def load_clean_data(path="data/bitcoin_clean.csv"):
    """Load the cleaned dataset, parse dates, and set the DatetimeIndex."""
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)
    return df


def check_missing_values(df):
    """
    Diagnostic test for missing values (NaN). Applies a Forward Fill
    if any are found, to avoid distorting momentum / volatility metrics.
    """
    print("\n--- Test des valeurs manquantes AVANT correction ---")
    missing_before = df.isnull().sum()
    print(missing_before)
    total_missing = missing_before.sum()
    print(f"\nTotal des valeurs manquantes dans tout le dataset : {total_missing}")

    if total_missing > 0:
        print("\n[!] Valeurs manquantes détectées. Application de la correction (Forward Fill)...")
        df.ffill(inplace=True)
        print("\n--- Test des valeurs manquantes APRÈS correction ---")
        missing_after = df.isnull().sum()
        print(missing_after)
        print(f"\nTotal des valeurs manquantes restantes : {missing_after.sum()}")
    else:
        print("\n[OK] Aucune valeur manquante détectée. Le dataset est parfait, aucune correction nécessaire !")

    return df


if __name__ == "__main__":
    raw = load_raw_data()
    export_clean_data(raw)
    clean = load_clean_data()
    check_missing_values(clean)
