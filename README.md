# Housing Price Prediction

**Course:** Data Science with Python  
**Project:** Housing Price Prediction (Option B)  
**Author:** Tatia Khvtisiashvili  
**Date:** January 2026

---

## Overview

This project implements a complete machine learning pipeline to predict **median house values in California districts** using demographic and geographic data from the 1990 census.

**Workflow:**
- Data exploration and validation
- Data cleaning and feature engineering
- Exploratory data analysis (EDA) with visualizations
- Training, evaluating, and comparing regression models

The goal is to understand the main drivers of housing prices and evaluate how different machine learning models perform on this task.

---

## Dataset

**Source:** California Housing Prices (1990 Census)  
**Kaggle:** https://www.kaggle.com/datasets/camnugent/california-housing-prices  
**Size:** 20,640 rows × 10 columns  
**Target variable:** `median_house_value`

### Features

| Feature | Description |
|--------|-------------|
| longitude, latitude | Geographic coordinates |
| housing_median_age | Median age of houses |
| total_rooms, total_bedrooms | Housing characteristics |
| population, households | Demographic information |
| median_income | Median income (in tens of thousands of dollars) |
| **median_house_value** | **Target variable (USD)** |
| ocean_proximity | Categorical location feature |

### Engineered Features
Created during preprocessing:
- `rooms_per_household` = total_rooms / households
- `bedrooms_per_household` = total_bedrooms / households
- `population_per_household` = population / households
- One-hot encoded `ocean_proximity` (4 binary features)

---

## Installation & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
````

Required libraries: pandas, numpy, scikit-learn, matplotlib, seaborn

### 2. Download Dataset

* Download `housing.csv` from Kaggle
* Place it in `data/raw/housing.csv`

### 3. Run the Pipeline

```bash
cd notebooks
python 01_data_exploration.py
python 02_data_preprocessing.py
python 03_eda_visualization.py
python 04_machine_learning.py
python 05_predict_example.py  # Optional demonstration
```

---

## Project Structure

```
house-price-prediction/
├── data/
│   ├── raw/housing.csv                    # Original dataset (20,640 rows)
│   └── processed/housing_cleaned.csv      # Cleaned dataset (17,532 rows)
├── src/
│   ├── __init__.py                        # Package initialization
│   ├── data_processing.py                 # Data cleaning & feature engineering
│   ├── visualization.py                   # Plotting utilities
│   └── models.py                          # Model training & evaluation
├── notebooks/
│   ├── 01_data_exploration.py             # Initial inspection
│   ├── 02_data_preprocessing.py           # Cleaning & preprocessing
│   ├── 03_eda_visualization.py            # EDA & plots
│   ├── 04_machine_learning.py             # Model training & comparison
│   └── 05_predict_example.py              # Prediction demo (optional)
├── reports/
│   ├── figures/                           # Generated visualizations
│   │   ├── 01_price_distribution.png
│   │   ├── 02_income_distribution.png
│   │   ├── 03_income_vs_price.png
│   │   ├── 04_correlation_heatmap.png
│   │   ├── 05_rooms_vs_price.png
│   │   ├── 06_geographic_distribution.png
│   │   └── 10_model_comparison.png
│   ├── results/
│   │   ├── model_results.json
│   │   └── model_comparison.csv
│   └── models/
│       ├── linear_regression.pkl
│       ├── decision_tree.pkl
│       └── random_forest.pkl
├── requirements.txt
├── README.md
├── CONTRIBUTIONS.md
└── .gitignore
```

---

## Methodology

### 1. Data Exploration (`01_data_exploration.py`)

* Loaded 20,640 housing districts with 10 features
* Identified **207 missing values** in `total_bedrooms`
* No duplicate records found
* Target prices range from **$14,999 to $500,001**

---

### 2. Data Preprocessing (`02_data_preprocessing.py`)

**Steps performed:**

* **Missing values:** Median imputation for `total_bedrooms`
* **Outlier removal:** IQR method (1.5×IQR), removing 3,108 rows (~15%)
* **Feature engineering:** Created three ratio-based features
* **Encoding:** One-hot encoding of `ocean_proximity`

**Output:** Cleaned dataset with **17,532 rows × 16 columns**, saved to:

```
data/processed/housing_cleaned.csv
```

---

### 3. Exploratory Data Analysis (`03_eda_visualization.py`)

Six visualizations were generated and saved to `reports/figures/`.

#### Image 1: Price Distribution

* Right-skewed distribution
* Majority of homes priced below $300k
* Visible spike at $500k reflects the dataset’s price ceiling

#### Image 2: Income Distribution

* Right-skewed with most districts in the lower-to-middle income range
* Few high-income districts above $10k
* Typical census income distribution

#### Image 3: Income vs. Price Scatter

* Clear positive linear relationship
* Correlation ≈ **0.63**
* Confirms median income as the strongest single predictor of price

#### Image 4: Correlation Heatmap

* `median_income` shows the highest correlation with price
* Room-related features are strongly correlated with each other
* Latitude shows a mild negative correlation, indicating regional effects

#### Image 5: Rooms vs. Price

* Higher room density per household corresponds to higher median prices
* Greater variance observed for high room-density groups
* Indicates housing size as a secondary but meaningful factor

#### Image 6: Geographic Distribution

* Clear coastal premium visible along California’s coastline
* Highest prices clustered around the Bay Area and Los Angeles
* Inland regions dominated by lower-priced housing

**Key EDA Insights:**

1. Median income is the strongest predictor of housing value
2. Geographic location plays a critical role
3. Housing size (room density) positively affects prices
4. Housing age has relatively limited influence

---

### 4. Machine Learning (`04_machine_learning.py`)

**Train/Test Split:**

* Training: 14,025 samples (80%)
* Testing: 3,507 samples (20%)

**Models Trained:**

1. **Linear Regression** – Baseline model
2. **Decision Tree Regressor** – Captures non-linear patterns
3. **Random Forest Regressor** – Ensemble model (bonus)

**Evaluation Metrics:** R², RMSE, MAE

---

## Results

### Model Performance

| Model             | R²         | RMSE ($)   | MAE ($)    |
| ----------------- | ---------- | ---------- | ---------- |
| Linear Regression | 0.5990     | 68,193     | 49,341     |
| Decision Tree     | 0.6786     | 61,053     | 40,121     |
| **Random Forest** | **0.7722** | **51,392** | **33,345** |

#### Image 7: Model Comparison

* Random Forest achieves the highest R² and lowest RMSE
* Demonstrates clear improvement over baseline
* Confirms ensemble methods are best suited for this dataset

**Best Model:** Random Forest

* Explains ~77% of variance
* Lowest prediction error
* Most stable overall performance

### Feature Importance (Random Forest)

* `median_income` (~42%) – dominant factor
* `population_per_household` (~15%)
* `ocean_proximity_INLAND` (~14%)
* `longitude` and `latitude` (~13% combined)

**Insight:** Income is the primary driver, with geography contributing significantly.

---

## Key Findings

* Income and location dominate housing price prediction
* Coastal and urban areas command substantial premiums
* Tree-based models outperform linear regression
* Feature engineering improves predictive accuracy

---

## Limitations

* Data is from 1990 and may not reflect current markets
* Artificial price ceiling at $500k
* Important external factors (schools, crime, amenities) not included
* Approximately 23% of variance remains unexplained

---

## Usage Example

`05_predict_example.py` demonstrates:

* Loading trained models from `reports/models/`
* Preparing input data with correct features
* Making predictions for a sample district
* Comparing predictions across models

---

## Author

**★Tatia Khvtisiashvili★**

★Course: Data Science with Python

★Project Type: Solo project
★Date: January 2026

---

**Last Updated:** January 2026
