import sys
import os

# Adding the project root to the Python path as it allows importing modules from the src/ directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing preprocessing function
from src.data_processing import preprocess_data

# File paths
Raw_path = "../data/raw/housing.csv"
Processed_path = "../data/processed/housing_cleaned.csv"


def main():
    # Header for console output
    print("\n" + "★" * 45)
    print(f"{'★' * 14} Data Processing {'★' * 16}")
    print("★" * 45)

    # Ensuring the output directories exist
    os.makedirs("../data/processed", exist_ok=True)
    os.makedirs("../reports/results", exist_ok=True)

    try:
        # Checking if the raw dataset exists
        if not os.path.exists(Raw_path):
            print(f"\nError: Raw data not found at {Raw_path}")
            return

        # Displaying the preprocessing steps
        print("\nProcessing steps:")
        print("1. Loading data...")
        print("2. Handling missing values (median imputation)...")
        print("3. Removing outliers (IQR method)...")
        print("4. Creating new features...")
        print("5. Encoding categorical variables...")

        # Running preprocessing pipeline
        df = preprocess_data(Raw_path, Processed_path)

        # Printing preprocessing results
        print("\n" + "★" * 29)
        print(f"{'★' * 10} RESULTS {'★' * 12}")
        print("★" * 29)

        print(f"Original shape:  20,640 x 10")
        print(f"Processed shape: {df.shape[0]:,} x {df.shape[1]}")
        print(f"Rows removed:    {20640 - df.shape[0]:,} "
              f"({((20640 - df.shape[0]) / 20640) * 100:.1f}%)")
        print(f"Features added:  {df.shape[1] - 10}")
        print(f"Missing values:  {df.isnull().sum().sum()}")

        # Identifying newly created features
        original_cols = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'median_house_value', 'ocean_proximity'
        ]

        new_cols = [col for col in df.columns if col not in original_cols]

        print(f"\nNew features created ({len(new_cols)}):")
        for col in new_cols:
            print(f"  - {col}")

        # Saving the processed dataset
        print(f"\nData saved to: {Processed_path}")

        print(f"\n{'★' * 10} PREPROCESSING COMPLETE {'★' * 12}")
        print(f"Clean dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print("\nNext: Run 03_eda_visualization.py")

    except Exception as e:
        # Catching and reporting any errors
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
