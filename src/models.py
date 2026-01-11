"""
Machine learning functions for housing price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def prepare_data(df, target_col='median_house_value', test_size=0.2):
    """
    Splitting features and target into train and test sets.
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_train, y_test

def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Training and evaluating a Linear Regression model.
    Returns trained model and evaluation metrics.
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = {
        'model': model,
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'predictions': y_pred
    }

    return results

def train_decision_tree(X_train, y_train, X_test, y_test):
    """Training and evaluating Decision Tree."""
    model = DecisionTreeRegressor(max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = {
        'model': model,
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'predictions': y_pred,
        'feature_importances': dict(zip(X_train.columns, model.feature_importances_))
    }

    return results

def train_random_forest(X_train, y_train, X_test, y_test):
    """Training and evaluating Random Forest (bonus)."""
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results = {
        'model': model,
        'r2': r2_score(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'predictions': y_pred,
        'feature_importances': dict(zip(X_train.columns, model.feature_importances_))
    }

    return results