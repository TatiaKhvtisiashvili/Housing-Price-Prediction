"""
Housing Price Prediction – Prediction Example
Author: Your Name
Date: January 2026

Demonstrates how to load a trained model and make a single prediction.
This script is for demonstration purposes and is not required for grading.
"""

import pickle
import pandas as pd
import os


MODEL_PATH = "../reports/models/random_forest.pkl"


def create_sample_data():
    """
    Create a single sample matching the training feature set.
    Values are realistic examples from the California housing dataset.
    """
    return pd.DataFrame({
        'longitude': [-122.23],
        'latitude': [37.88],
        'housing_median_age': [41.0],
        'total_rooms': [880.0],
        'total_bedrooms': [129.0],
        'population': [322.0],
        'households': [126.0],
        'median_income': [8.3252],
        'rooms_per_household': [880.0 / 126.0],
        'bedrooms_per_household': [129.0 / 126.0],
        'population_per_household': [322.0 / 126.0],
        'ocean_proximity_INLAND': [0],
        'ocean_proximity_ISLAND': [0],
        'ocean_proximity_NEAR BAY': [1],
        'ocean_proximity_NEAR OCEAN': [0]
    })


def main():
    print("\n" + "★" * 60)
    print("HOUSING PRICE PREDICTION – EXAMPLE")
    print("★" * 60)

    if not os.path.exists(MODEL_PATH):
        print("\nError: Trained model not found.")
        print("Please run 04_machine_learning.py first.")
        return

    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Create example input
    sample = create_sample_data()

    print("\nSample house information:")
    print(f"Location: latitude {sample['latitude'][0]}, longitude {sample['longitude'][0]}")
    print(f"Median income: ${sample['median_income'][0] * 10000:,.0f}")
    print(f"House age: {sample['housing_median_age'][0]} years")

    # Predict
    prediction = model.predict(sample)[0]

    print("\nPredicted median house value:")
    print(f"{prediction:,.2f}")

    print("\nThis is a demonstration using the trained Random Forest model.")
    print("★" * 60)


if __name__ == "__main__":
    main()
