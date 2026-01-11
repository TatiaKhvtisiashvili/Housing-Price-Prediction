"""Data processing functions for housing price prediction."""
import pandas as pd
import numpy as np

def load_data(file_path):
    """Load dataset from CSV."""
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at: {file_path}")


def handle_missing_values(df):
    """Fill missing values with median."""
    df = df.copy()
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    return df

def remove_outliers(df):
    """Remove outliers from features (not target)."""
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = [col for col in numeric_cols if col != 'median_house_value']

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    return df

def create_features(df):
    """Create new features from existing data."""
    df = df.copy()

    # Household features
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_household'] = df['total_bedrooms'] / df['households']
    df['population_per_household'] = df['population'] / df['households']

    return df

def preprocess_data(file_path, save_path=None):
    """Complete preprocessing pipeline. Load data"""
    df = load_data(file_path)

    """Handle missing values"""
    df = handle_missing_values(df)

    """Remove outliers"""
    df = remove_outliers(df)

    """Create features"""
    df = create_features(df)

    """One-hot encode categorical"""
    df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=True)

    """Save if needed"""
    if save_path:
        df.to_csv(save_path, index=False)

    return df