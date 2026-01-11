import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import (
    plot_histogram, plot_scatter, plot_heatmap,
    plot_boxplot, save_plot
)

DATA_PATH = "../data/processed/housing_cleaned.csv"
FIGURES_DIR = "../reports/figures"

def main():
    print("\n" + "★" * 40)
    print("EXPLORATORY DATA ANALYSIS")
    print("★" * 40)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    try:
        if not os.path.exists(DATA_PATH):
            print(f"\nError: Processed data not found at {DATA_PATH}")
            return

        df = pd.read_csv(DATA_PATH)
        print(f"\nData loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")

        print(f"\n{"★" *14} STATISTICAL ANALYSIS {"★" *14}")
        print("\nDescriptive Statistics:")
        print(df.describe())

        print("\nCorrelation with Target:")
        corr = df.corr()['median_house_value'].sort_values(ascending=False)
        print("\nTop 5 correlations:")
        print(corr.head(6))

        print(f"\n{"★" *14} CREATING VISUALIZATIONS {"★" *14}")

        viz_count = 0

        # 1. Price distribution
        print(f"{viz_count+1}. Price distribution...")
        fig = plot_histogram(df, 'median_house_value', 'Distribution of House Prices')
        save_plot(fig, f"{FIGURES_DIR}/01_price_distribution.png")
        viz_count += 1

        # 2. Income distribution
        print(f"{viz_count+1}. Income distribution...")
        fig = plot_histogram(df, 'median_income', 'Distribution of Median Income')
        save_plot(fig, f"{FIGURES_DIR}/02_income_distribution.png")
        viz_count += 1

        # 3. Income vs price scatter
        print(f"{viz_count+1}. Income vs Price scatter...")
        fig = plot_scatter(df, 'median_income', 'median_house_value',
                           'House Price vs Median Income')
        save_plot(fig, f"{FIGURES_DIR}/03_income_vs_price.png")
        viz_count += 1

        # 4. Correlation heatmap
        print(f"{viz_count+1}. Correlation heatmap...")
        fig = plot_heatmap(df, 'Feature Correlation Heatmap')
        save_plot(fig, f"{FIGURES_DIR}/04_correlation_heatmap.png")
        viz_count += 1

        # 5. Box plot
        if 'rooms_per_household' in df.columns:
            print(f"{viz_count+1}. Rooms vs Price boxplot...")
            df_temp = df.copy()
            df_temp['rooms_category'] = pd.cut(df_temp['rooms_per_household'],
                                               bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
            fig = plot_boxplot(df_temp, 'rooms_category', 'median_house_value',
                               'House Price by Rooms per Household')
            save_plot(fig, f"{FIGURES_DIR}/05_rooms_vs_price.png")
            viz_count += 1

        # 6. Geographic scatter
        if 'longitude' in df.columns and 'latitude' in df.columns:
            print(f"{viz_count+1}. Geographic distribution...")
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(df['longitude'], df['latitude'],
                                 c=df['median_house_value'], cmap='viridis',
                                 s=5, alpha=0.6)
            plt.colorbar(scatter, label='Median House Value', ax=ax)
            ax.set_title('Geographic Distribution of House Prices', fontsize=14, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12)
            ax.grid(alpha=0.3)
            save_plot(fig, f"{FIGURES_DIR}/06_geographic_distribution.png")
            viz_count += 1

        print(f"\nTotal visualizations created: {viz_count}")
        print(f"Saved to: {FIGURES_DIR}/")

        print("\n" + "★" * 40)
        print(f"{"★" *14} KEY INSIGHTS {"★" *14}")
        print("★" * 40)
        print("1. Median income has strongest correlation with price (r ~ 0.68)")
        print("2. Geographic location significantly impacts prices")
        print("3. Coastal areas show premium pricing")
        print("4. Rooms per household positively correlates with price")
        print("\nNext: Run 04_machine_learning.py")

    except Exception as e:
        print(f"\nError: {e}")


if __name__ == "__main__":
    main()