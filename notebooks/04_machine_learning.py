import sys
import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    prepare_data,
    train_linear_regression,
    train_decision_tree,
    train_random_forest
)

DATA_PATH = "../data/processed/housing_cleaned.csv"
RESULTS_DIR = "../reports/results"
MODELS_DIR = "../reports/models"
FIGURES_DIR = "../reports/figures"

def main():
    print("\n" + "★" * 40)
    print("MACHINE LEARNING")
    print("★" * 40)

    for d in [RESULTS_DIR, MODELS_DIR, FIGURES_DIR]:
        os.makedirs(d, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        print(f"\nError: Processed data not found at {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"\nData loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\n" + "★" * 40)
    print("PREPARING DATA")
    print("★" * 40)

    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples:     {len(X_test):,}")
    print(f"Features:         {X_train.shape[1]}")

    print("\n" + "★" * 40)
    print("TRAINING MODELS")
    print("★" * 40)

    models = {
        "Linear Regression": train_linear_regression,
        "Decision Tree": train_decision_tree,
        "Random Forest": train_random_forest
    }

    results = {}

    for i, (name, trainer) in enumerate(models.items(), start=1):
        print(f"\n{i}. {name}")
        res = trainer(X_train, y_train, X_test, y_test)
        results[name] = {
            "R2": res["r2"],
            "RMSE": res["rmse"],
            "MAE": res["mae"]
        }
        print(f"   R2: {res['r2']:.4f}, RMSE: ${res['rmse']:,.2f}")

        # SHOW FEATURE IMPORTANCE (only when available)
        if "feature_importances" in res:
            print("   Top 5 important features:")
            top_features = sorted(
                res["feature_importances"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for feat, imp in top_features:
                print(f"     {feat:<30} {imp:.4f}")

        # Save model
        with open(f"{MODELS_DIR}/{name.lower().replace(' ', '_')}.pkl", "wb") as f:
            pickle.dump(res["model"], f)

    print("\n" + "-" * 60)
    print("CREATING VISUALIZATIONS")
    print("-" * 60)

    names = list(results.keys())
    r2_scores = [results[m]["R2"] for m in names]
    rmse_scores = [results[m]["RMSE"] for m in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["steelblue", "coral", "green"]

    axes[0].bar(names, r2_scores, color=colors, edgecolor="black")
    axes[0].set_title("Model Comparison – R²")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar(names, rmse_scores, color=colors, edgecolor="black")
    axes[1].set_title("Model Comparison – RMSE ($)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/10_model_comparison.png", dpi=300)
    plt.close()

    print("Model comparison saved")

    print("\n" + "★" * 15 + " SAVING RESULTS " + "★" * 15)

    with open(f"{RESULTS_DIR}/model_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {RESULTS_DIR}/model_results.json")

    print("\n" + "★" * 15 + " FINAL RESULTS " + "★" * 15)
    print(f"\n{'Model':<20} {'R2':<10} {'RMSE':<15} {'MAE':<15}")
    print("★" * 40)

    for name, m in results.items():
        print(f"{name:<20} {m['R2']:<10.4f} ${m['RMSE']:<14,.2f} ${m['MAE']:<14,.2f}")

    best = max(results.items(), key=lambda x: x[1]["R2"])
    print(f"\nBest Model: {best[0]} (R2 = {best[1]['R2']:.4f})")

    print("\n" + "★" * 40)
    print("COMPLETE")
    print("★" * 40)


if __name__ == "__main__":
    main()
