"""
Data Science with Python - Housing Price Prediction Project
Source code modules for data processing, visualization, and machine learning.
"""

# Version information
__version__ = "1.0.0"
__author__ = "Your Name"
__course__ = "Data Science with Python"

# Import key functions to make them accessible from the package
from .data_processing import (
    load_data,
    handle_missing_values,
    remove_outliers,
    create_features,
    preprocess_data
)

from .visualization import (
    plot_histogram,
    plot_scatter,
    plot_heatmap,
    plot_boxplot,
    plot_pairplot,
    plot_violin,
    save_plot
)

from .models import (
    prepare_data,
    train_linear_regression,
    train_decision_tree,
    train_random_forest
)

# Define what gets imported with "from src import *"
__all__ = [
    # Data processing
    'load_data',
    'handle_missing_values',
    'remove_outliers',
    'create_features',
    'preprocess_data',

    # Visualization
    'plot_histogram',
    'plot_scatter',
    'plot_heatmap',
    'plot_boxplot',
    'plot_pairplot',
    'plot_violin',
    'save_plot',

    # Models
    'prepare_data',
    'train_linear_regression',
    'train_decision_tree',
    'train_random_forest'
]