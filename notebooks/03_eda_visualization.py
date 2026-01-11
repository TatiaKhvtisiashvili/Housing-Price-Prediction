import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Adding the project root directory to Python's module search path, this allows importing custom
# modules from the src/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importing reusable visualization functions
from src.visualization import (
    plot_histogram, plot_scatter, plot_heatmap,
    plot_boxplot, save_plot
)

# Path to cleaned dataset produced during preprocessing
data_path = "../data/processed/housing_cleaned.csv"

# Directory where generated figures will be saved
figures_path = "../reports/figures"


def main():
    # Printing header for EDA stage
    print("\n" + "★" * 40)
    print("Exploratory Data Analysis")
    print("★" * 40)

    # Creating directory for figures if it does not exist
    os.makedirs(figures_path, exist_ok=True)

    try:
        # Checking if the processed dataset exists before loading it
        if not os.path.exists(data_path):
            print(f"\nError: Processed data not found at {data_path}")
            return

        # Loading cleaned housing dataset
        df = pd.read_csv(data_path)
        print(f"\nData loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

        # Displaying summary statistics for numerical features
        print(f"\n{'★' * 14} Statistical Analysis {'★' * 14}")
        print("\nDescriptive Statistics:")
        print(df.describe())

        # Computing correlation of each feature with target variable
        print("\nCorrelation with Target:")
        corr = df.corr()['median_house_value'].sort_values(ascending=False)

        # Displaying top correlations including target itself
        print("\nTop 5 correlations:")
        print(corr.head(6))

        # Beginning visualization generation
        print(f"\n{'★' * 14} Creating Visualisations {'★' * 14}")

        # Creating a counter to track number of plots created
        viz_count = 0

        # Plot 1: Distribution of median house prices
        # Helps understand skewness and price concentration
        print(f"{viz_count + 1}. Price distribution...")
        fig = plot_histogram(df, 'median_house_value', 'Distribution of House Prices')
        save_plot(fig, f"{figures_path}/01_price_distribution.png")
        viz_count += 1

        # Plot 2: Distribution of median income
        # Shows income spread across districts
        print(f"{viz_count + 1}. Income distribution...")
        fig = plot_histogram(df, 'median_income', 'Distribution of Median Income')
        save_plot(fig, f"{figures_path}/02_income_distribution.png")
        viz_count += 1

        # Plot 3: Scatter plot of income vs house price
        # Used to visually inspect linear relationship
        print(f"{viz_count + 1}. Income vs Price scatter...")
        fig = plot_scatter(
            df,
            'median_income',
            'median_house_value',
            'House Price vs Median Income'
        )
        save_plot(fig, f"{figures_path}/03_income_vs_price.png")
        viz_count += 1

        # Plot 4: Correlation heatmap
        # Highlights relationships between all numerical features
        print(f"{viz_count + 1}. Correlation heatmap...")
        fig = plot_heatmap(df, 'Feature Correlation Heatmap')
        save_plot(fig, f"{figures_path}/04_correlation_heatmap.png")
        viz_count += 1

        # Plot 5: Boxplot of house prices by room density categories
        # Requires engineered feature 'rooms_per_household'
        if 'rooms_per_household' in df.columns:
            print(f"{viz_count + 1}. Rooms vs Price boxplot...")

            # Create categorical bins for room density
            df_temp = df.copy()
            df_temp['rooms_category'] = pd.cut(
                df_temp['rooms_per_household'],
                bins=5,
                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
            )

            fig = plot_boxplot(
                df_temp,
                'rooms_category',
                'median_house_value',
                'House Price by Rooms per Household'
            )
            save_plot(fig, f"{figures_path}/05_rooms_vs_price.png")
            viz_count += 1

        # Plot 6: Geographic distribution of house prices
        # Visualizes spatial price patterns across California
        if 'longitude' in df.columns and 'latitude' in df.columns:
            print(f"{viz_count + 1}. Geographic distribution...")

            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                df['longitude'],
                df['latitude'],
                c=df['median_house_value'],
                cmap='viridis',
                s=5,
                alpha=0.6
            )

            plt.colorbar(scatter, label='Median House Value', ax=ax)
            ax.set_title('Geographic Distribution of House Prices', fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.grid(alpha=0.3)

            save_plot(fig, f"{figures_path}/06_geographic_distribution.png")
            viz_count += 1

        # Summary of visualization output
        print(f"\nTotal visualizations created: {viz_count}")
        print(f"Saved to: {figures_path}/")

        # Printing key insights derived from EDA
        print("\n" + "★" * 40)
        print(f"{'★' * 14} Key Insights {'★' * 14}")
        print("★" * 40)
        print("1. Median income has strongest correlation with price (r ~ 0.68)")
        print("2. Geographic location significantly impacts prices")
        print("3. Coastal areas show premium pricing")
        print("4. Rooms per household positively correlates with price")
        print("\nNext: Run 04_machine_learning.py")

    except Exception as e:
        # Catching and displaying any unexpected errors
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()
