"""
Step 4 & 5: Data Splitting, Scaling, and KNN Model Training
----------------------------------------------------------------
Chronological 80/20 split (shuffle=False) to avoid future data leakage.
StandardScaler is fit ONLY on the training data, then applied to the
test set, so the model evaluates on genuinely unseen "future" data.

The K-Nearest Neighbors classifier looks at the 15 closest historical
days (n_neighbors=15), using Euclidean distance with distance-based
weighting, to find days with similar "Market DNA".
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

from features import FEATURE_COLUMNS as features


def split_and_scale(df, test_size=0.2):
    X = df[features]
    y = df["Regime"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_knn(X_train_scaled, y_train, n_neighbors=15):
    print("\nRecherche des k-plus proches voisins...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="euclidean", weights="distance")
    knn.fit(X_train_scaled, y_train)
    return knn


def evaluate(knn, X_test_scaled, y_test):
    y_pred = knn.predict(X_test_scaled)
    print("\n--- Rapport de Performance du KNN ---")
    print(classification_report(
        y_test,
        y_pred,
        labels=[-1, 0, 1],
        target_names=["Bear (-1)", "Range (0)", "Bull (1)"]
    ))
    return y_pred


if __name__ == "__main__":
    from preprocessing import load_clean_data
    from features import add_market_dna_features
    from labeling import add_regime_labels

    df = load_clean_data()
    df = add_market_dna_features(df)
    df = add_regime_labels(df)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(df)
    knn = train_knn(X_train_scaled, y_train)
    evaluate(knn, X_test_scaled, y_test)
