import pandas as pd
import os

DATA_PATH = "../data/raw/housing.csv"


def main():
    print("\n" + "★" * 43)
    print(f"{"★" *14} DATA EXPLORATION {"★" *16}")
    print("★" * 43)

    if not os.path.exists(DATA_PATH):
        print(f"\nError: Data file not found at {DATA_PATH}")
        print("Download from: https://www.kaggle.com/datasets/camnugent/california-housing-prices")
        return

    df = pd.read_csv(DATA_PATH)
    print(f"\nData loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print(f"\n{"★" *10} DATASET INFO {"★" *14}")
    print(df.info())

    print(f"\n{"★" * 17} FIRST 5 ROWS {"★" * 20}")
    print(df.head())

    print(f"\n{"★" * 17} DESCRIPTIVE STATISTICS {"★" * 19}")
    print(df.describe())

    print(f"\n{"★" * 5} MISSING VALUES {"★" * 7}")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

    print(f"\n{"★" * 5} CATEGORICAL FEATURES {"★" * 7}")
    print("\nOcean Proximity Distribution:")
    print(df['ocean_proximity'].value_counts())

    print(f"\n{"★" * 5} TARGET VARIABLE (median_house_value) {"★" * 5}")
    print(f"Mean:   ${df['median_house_value'].mean():,.2f}")
    print(f"Median: ${df['median_house_value'].median():,.2f}")
    print(f"Min:    ${df['median_house_value'].min():,.2f}")
    print(f"Max:    ${df['median_house_value'].max():,.2f}")

    print("\n" + "★" * 43)
    print("SUMMARY")
    print("★" * 43)
    print(f"Total samples: {df.shape[0]:,}")
    print(f"Total features: {df.shape[1]}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    print("\nNext: Run 02_data_preprocessing.py")
    print("★" * 43)


if __name__ == "__main__":
    main()