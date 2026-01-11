"""
Visualization functions for EDA.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np  # ← FIXED: Added this import

def save_plot(fig, filename):
    """Save figure to file."""
    fig.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close(fig)

def plot_histogram(df, column, title):
    """Plot histogram for a column."""
    fig, ax = plt.subplots(figsize=(10, 6))
    df[column].hist(bins=50, ax=ax, edgecolor='black', alpha=0.7)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(alpha=0.3)
    return fig

def plot_scatter(df, x_col, y_col, title):
    """Plot scatter plot with regression line."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(x=x_col, y=y_col, data=df, ax=ax, scatter_kws={'alpha': 0.3})
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.grid(alpha=0.3)
    return fig

def plot_heatmap(df, title):
    """Plot correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax,
                center=0, square=True, linewidths=1)
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    return fig

def plot_boxplot(df, x_col, y_col, title):
    """Plot boxplot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x=x_col, y=y_col, data=df, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(alpha=0.3, axis='y')
    return fig

def plot_pairplot(df, columns, title=None):
    """Create pair plot for selected columns."""
    pairplot = sns.pairplot(df[columns], diag_kind='kde',
                            plot_kws={'alpha': 0.6, 's': 30},
                            diag_kws={'alpha': 0.7})
    if title:
        pairplot.fig.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
    return pairplot.fig

def plot_violin(df, x_col, y_col, title):
    """Plot violin plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(x=x_col, y=y_col, data=df, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(x_col, fontsize=12)
    ax.set_ylabel(y_col, fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.grid(alpha=0.3, axis='y')
    return fig