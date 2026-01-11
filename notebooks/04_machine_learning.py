import sys
import os
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Add project root directory to Python path
# This allows importing custom modules from the src/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model-related helper functions
from src.models import (
    prepare_data,
    train_linear_regression,
    train_decision_tree,
    train_random_forest
)

# Paths to input data and output directories
data_path = "../data/processed/housing_cleaned.csv"
results_direction = "../reports/results"
models_direction = "../reports/models"
figures_direction = "../reports/figures"


def main():
    # Header for machine learning stage
    print("\n" + "★" * 40)
    print("Machine Learning")
    print("★" * 40)

    # Ensure all required output directories exist
    for d in [results_direction, models_direction, figures_direction]:
        os.makedirs(d, exist_ok=True)

    # Check that processed data exists
    if not os.path.exists(data_path):
        print(f"\nError: Processed data not found at {data_path}")
        return

    # Load cleaned dataset
    df = pd.read_csv(data_path)
    print(f"\nData loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Prepare data (train-test split and feature/target separation)
    print("\n" + "★" * 40)
    print("Preparing Data")
    print("★" * 40)

    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples:     {len(X_test):,}")
    print(f"Features:         {X_train.shape[1]}")

    # Train multiple models
    print("\n" + "★" * 40)
    print("Training Models")
    print("★" * 40)

    models = {
        "Linear Regression": train_linear_regression,
        "Decision Tree": train_decision_tree,
        "Random Forest": train_random_forest
    }

    results = {}

    # Train and evaluate each model
    for i, (name, trainer) in enumerate(models.items(), start=1):
        print(f"\n{i}. {name}")

        # Train model and evaluate performance
        res = trainer(X_train, y_train, X_test, y_test)

        # Store evaluation metrics
        results[name] = {
            "R2": res["r2"],
            "RMSE": res["rmse"],
            "MAE": res["mae"]
        }

        print(f"   R2: {res['r2']:.4f}, RMSE: ${res['rmse']:,.2f}")

        # Display feature importance only for models that support it
        # (Decision Tree and Random Forest)
        if "feature_importances" in res:
            print("   Top 5 important features:")
            top_features = sorted(
                res["feature_importances"].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            for feat, imp in top_features:
                print(f"     {feat:<30} {imp:.4f}")

        # Save trained model
        with open(f"{models_direction}/{name.lower().replace(' ', '_')}.pkl", "wb") as f:
            pickle.dump(res["model"], f)

        # Create Actual vs Predicted visualization
        y_pred = res["model"].predict(X_test)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.4)

        ax.set_title(f"{name}: Actual vs Predicted Prices")
        ax.set_xlabel("Actual Median House Value")
        ax.set_ylabel("Predicted Median House Value")
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Numbered figure filenames (08, 09, 10)
        figure_number = 7 + i  # i starts at 1 → 08, 09, 10

        plt.savefig(
            f"{figures_direction}/0{+figure_number}_{name.lower().replace(' ', '_')}_prediction.png",
            dpi=300
        )
        plt.close()

    # Create model comparison visualizations
    print("\n" + "-" * 60)
    print("Creating Visualisation")
    print("-" * 60)

    names = list(results.keys())
    r2_scores = [results[m]["R2"] for m in names]
    rmse_scores = [results[m]["RMSE"] for m in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["steelblue", "coral", "green"]

    # R² comparison
    axes[0].bar(names, r2_scores, color=colors, edgecolor="black")
    axes[0].set_title("Model Comparison – R²")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.3)

    # RMSE comparison
    axes[1].bar(names, rmse_scores, color=colors, edgecolor="black")
    axes[1].set_title("Model Comparison – RMSE ($)")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{figures_direction}/07_model_comparison.png", dpi=300)
    plt.close()

    print("Model comparison saved")

    # Save results to JSON
    print("\n" + "★" * 15 + " SAVING RESULTS " + "★" * 15)
    with open(f"{results_direction}/model_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_direction}/model_results.json")

    # Print final results table
    print("\n" + "★" * 15 + " FINAL RESULTS " + "★" * 15)
    print(f"\n{'Model':<20} {'R2':<10} {'RMSE':<15} {'MAE':<15}")
    print("★" * 40)

    for name, m in results.items():
        print(f"{name:<20} {m['R2']:<10.4f} ${m['RMSE']:<14,.2f} ${m['MAE']:<14,.2f}")

    # Identify best model
    best = max(results.items(), key=lambda x: x[1]["R2"])
    print(f"\nBest Model: {best[0]} (R2 = {best[1]['R2']:.4f})")

    # Completion message
    print("\n" + "★" * 40)
    print("Complete")
    print("★" * 40)


if __name__ == "__main__":
    main()
