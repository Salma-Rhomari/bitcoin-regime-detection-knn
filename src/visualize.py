"""
Step 6: Visual Interpretation of the Strategy
--------------------------------------------------
Maps the KNN's predicted regimes directly onto the Bitcoin price chart:
green triangles for predicted Bull days, red triangles for predicted
Bear days, plotted over the true closing price.
"""

import matplotlib.pyplot as plt


def plot_predictions(df, y_test, y_pred, output_path="outputs/regime_predictions.png"):
    print("\nGénération du graphique visuel...")
    df_test = df.iloc[-len(y_test):].copy().reset_index(drop=True)
    df_test["Prediction_IA"] = y_pred

    plt.figure(figsize=(14, 7))
    plt.plot(df_test.index, df_test["Close"], color="gray", label="Vrai Prix du Bitcoin", alpha=0.6, linewidth=2)

    jours_bull = df_test[df_test["Prediction_IA"] == 1]
    plt.scatter(jours_bull.index, jours_bull["Close"], color="green", marker="^", s=100, label="IA prédit : HAUSSE (Bull)")

    jours_bear = df_test[df_test["Prediction_IA"] == -1]
    plt.scatter(jours_bear.index, jours_bear["Close"], color="red", marker="v", s=100, label="IA prédit : BAISSE (Bear)")

    plt.title("Analyse de l'IA : Prédictions sur la courbe du Bitcoin", fontsize=14, fontweight="bold")
    plt.xlabel("Jours récents (Période d'examen de l'IA)")
    plt.ylabel("Prix du Bitcoin ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Graphique sauvegardé : {output_path}")
    plt.show()


if __name__ == "__main__":
    from preprocessing import load_clean_data
    from features import add_market_dna_features
    from labeling import add_regime_labels
    from model import split_and_scale, train_knn, evaluate

    df = load_clean_data()
    df = add_market_dna_features(df)
    df = add_regime_labels(df)

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(df)
    knn = train_knn(X_train_scaled, y_train)
    y_pred = evaluate(knn, X_test_scaled, y_test)

    plot_predictions(df, y_test, y_pred)
