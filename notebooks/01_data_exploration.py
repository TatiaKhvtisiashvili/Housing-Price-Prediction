import pandas as pd
import os

# Path to the raw housing dataset
DATA_PATH = "../data/raw/housing.csv"


def main():
    """
    Performs initial data exploration:
    - Loads dataset
    - Prints structure, statistics, and missing values
    - Gives an overview of the target variable
    """

    # Header for terminal output
    print("\n" + "★" * 43)
    print(f"{'★' * 14} Data Exploration {'★' * 16}")
    print("★" * 43)

    # Check if the dataset exists before loading
    if not os.path.exists(DATA_PATH):
        print(f"\nError: Data file not found at {DATA_PATH}")
        print("Download from: https://www.kaggle.com/datasets/camnugent/california-housing-prices")
        return

    # Loading the dataset into a pandas DataFrame
    df = pd.read_csv(DATA_PATH)
    print(f"\nData loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Displaying the dataset structure and data types
    print(f"\n{'★' * 10} Dataset Info {'★' * 14}")
    print(df.info())

    # Showing the first 5 rows for a quick preview
    print(f"\n{'★' * 17} First 5 Rows {'★' * 20}")
    print(df.head())

    # Displaying descriptive statistics for numeric columns
    print(f"\n{'★' * 17} Descriptive Statistics {'★' * 19}")
    print(df.describe())

    # Identifying missing values in each column
    print(f"\n{'★' * 5} Missing Values {'★' * 7}")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

    # Analyzing categorical feature distribution
    print(f"\n{'★' * 5} Categorical Features {'★' * 7}")
    print("\nOcean Proximity Distribution:")
    print(df['ocean_proximity'].value_counts())

    # Summarizing statistics for the target variable
    print(f"\n{'★' * 5} Target Variable (median_house_value) {'★' * 5}")
    print(f"Mean:   ${df['median_house_value'].mean():,.2f}")
    print(f"Median: ${df['median_house_value'].median():,.2f}")
    print(f"Min:    ${df['median_house_value'].min():,.2f}")
    print(f"Max:    ${df['median_house_value'].max():,.2f}")

    # Final summary of dataset quality
    print("\n" + "★" * 43)
    print("Summary")
    print("★" * 43)
    print(f"Total samples: {df.shape[0]:,}")
    print(f"Total features: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    print("\nNext: Run 02_data_preprocessing.py")
    print("★" * 43)


if __name__ == "__main__":
    main()
