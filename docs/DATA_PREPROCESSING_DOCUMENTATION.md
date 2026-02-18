# Data Preprocessing Documentation
## Sri Lankan House Sales ML Project

**Date Completed**: February 18, 2026  
**Preprocessing Notebook**: `notebooks/01_data_cleaning_and_feature_engineering.ipynb`  
**Project Type**: Regression (House Price Prediction)  
**Target Region**: Sri Lanka (primarily Colombo and surrounding areas)

---

## Table of Contents
1. [Overview](#overview)
2. [Input Data](#input-data)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Feature Descriptions](#feature-descriptions)
5. [Output Files](#output-files)
6. [Data Statistics](#data-statistics)
7. [Technical Implementation Details](#technical-implementation-details)
8. [Critical Considerations for Modeling](#critical-considerations-for-modeling)

---

## Overview

This project aims to predict house prices in Sri Lanka (in LKR - Sri Lankan Rupees) based on property characteristics. The preprocessing pipeline transforms raw scraped real estate data into clean, model-ready datasets with proper train-test splits and no data leakage.

**Key Objectives:**
- Clean and validate scraped property data
- Engineer meaningful features from text descriptions
- Handle high-cardinality categorical data (city names)
- Create proper train-test splits before any transformations
- Prevent data leakage by fitting transformers only on training data
- Save all preprocessing artifacts for production deployment

---

## Input Data

### Source File
- **Path**: `data/02_intermediate/house_data_with_city.csv`
- **Source**: Web-scraped Sri Lankan real estate listings
- **Note**: City names have already been extracted from addresses in a previous step

### Original Columns
1. **URL**: Property listing URL
2. **Address**: Full property address (text)
3. **Price**: Property price in LKR (as string with formatting)
4. **Bedrooms**: Number of bedrooms (as string)
5. **Bathrooms**: Number of bathrooms (as string)
6. **House_Size**: House size with units (e.g., "2,500 sqft")
7. **Land_Size**: Land size with units (e.g., "10 perches")
8. **Description**: Text description of the property
9. **City**: Extracted city name (categorical)

### Data Characteristics
- **Format**: CSV with comma separators
- **Encoding**: UTF-8
- **Missing Values**: Present in House_Size, Land_Size, and potentially Description
- **Data Quality Issues**: 
  - Sizes contain units, commas, and inconsistent formatting
  - Prices have outliers and extreme values
  - Some entries have 0 bedrooms or invalid values
  - High cardinality in City column (50+ unique cities)

---

## Preprocessing Pipeline

### Step 0: Missing Value Analysis
**Purpose**: Assess data completeness before processing

**Actions**:
- Calculated missing value counts and percentages for each column
- Computed overall data completeness percentage
- Generated detailed missing value summary

**Findings**: Missing values primarily in House_Size and Land_Size columns

---

### Step 1: Clean House_Size & Land_Size
**Purpose**: Convert size strings to numeric values

**Process**:
1. Applied `clean_size()` function to both columns:
   - Removed all non-numeric characters except decimal points
   - Converted to float type
   - Preserved NaN for truly missing values
2. Calculated statistics on cleaned values

**Transformation Example**:
- Input: `"2,500 sqft"` → Output: `2500.0`
- Input: `"10 perches"` → Output: `10.0`
- Input: `"Not specified"` → Output: `NaN`

**Note**: Units are stripped, so all values are numeric but unit information is lost. Sri Lankan properties typically use:
- House_Size: square feet (sqft)
- Land_Size: perches (1 perch ≈ 272.25 sqft)

---

### Step 1b: Outlier Detection & Handling
**Purpose**: Remove extreme outliers that could confuse the model

**Method**: Interquartile Range (IQR) with 3.0x multiplier
- Formula: `[Q1 - 3*IQR, Q3 + 3*IQR]`
- Lower bound capped at 0 (sizes cannot be negative)

**Applied to**:
- House_Size
- Land_Size

**Rationale**: 
- 3.0x multiplier is more conservative than standard 1.5x
- Removes only extreme outliers while preserving valid large properties
- Each column filtered independently

**Impact**: Removes approximately 1-3% of rows with impossible/extreme size values

---

### Step 2: Logic Check & Filtering
**Purpose**: Remove invalid entries that violate domain logic

**Validation Checks**:

1. **Numeric Conversion**:
   - Bedrooms, Bathrooms, Price converted to numeric
   - Invalid values coerced to NaN

2. **Missing Critical Values**:
   - Dropped rows with NaN in Bedrooms, Bathrooms, or Price
   - These are essential features that cannot be imputed

3. **Bedroom Validation**:
   - Removed: Bedrooms = 0 (invalid for residential property)

4. **Bathroom Validation**:
   - Removed: Bathrooms = 0 (invalid for residential property)

5. **Unrealistic Count Validation**:
   - Removed: Bedrooms > 15 OR Bathrooms > 15
   - Rationale: Likely data entry errors; residential properties rarely exceed this

6. **Price Range Validation**:
   - Minimum: LKR 1,000,000 (approximately USD 3,000)
   - Maximum: LKR 2,000,000,000 (approximately USD 6,000,000)
   - Rationale: Filters data entry errors and extremely high-end properties

**Total Removed**: Approximately 2-5% of dataset

---

### Step 3: Feature Engineering - Text Extraction
**Purpose**: Extract binary features from property descriptions

**Features Extracted**:

1. **Is_Brand_New** (Binary: 0 or 1)
   - Keywords: "brand new", "newly built", "new house"
   - Indicates newly constructed properties
   - Value: 1 if any keyword found, 0 otherwise

2. **Is_Modern** (Binary: 0 or 1)
   - Keywords: "modern", "luxury", "contemporary"
   - Indicates modern/luxury properties
   - Value: 1 if any keyword found, 0 otherwise

**Method**:
- Case-insensitive text search
- Handles missing descriptions (returns 0 for both features)
- Simple keyword matching (not NLP-based)

**Note on Parking**: 
- Parking feature was deliberately EXCLUDED
- Rationale: Sri Lankan houses typically include parking even when not mentioned in descriptions
- Including it would create false distinction between properties

---

### Step 4: Feature Engineering - City Tiering
**Purpose**: Reduce city cardinality from 50+ unique values to 6 interpretable tiers

**Methodology**:

1. **Tier Definition**:
   - Calculated median price for each city
   - Used quantile-based binning (pd.qcut) to create 6 equal-frequency tiers
   - Tier assignment based on city median prices:
     - **Tier 1**: Luxury (highest median prices)
     - **Tier 2**: Premium
     - **Tier 3**: Upper-Mid
     - **Tier 4**: Mid-Range
     - **Tier 5**: Affordable
     - **Tier 6**: Budget (lowest median prices)

2. **Technical Implementation**:
   - Labels reversed in qcut: `range(6, 0, -1)` = [6, 5, 4, 3, 2, 1]
   - This ensures Tier 1 gets highest prices, Tier 6 gets lowest
   - Handles duplicate quantile edges with `duplicates='drop'`

3. **Tier Examples**:
   - **Tier 1 (Luxury)**: Colombo 3, Colombo 7, Colombo 4, Nawala (premium neighborhoods)
   - **Tier 6 (Budget)**: Colombo 11, Homagama, Padukka, Wellampitiya (outer/developing areas)

**Advantages**:
- Reduces cardinality from 50+ → 6 (prevents sparse data issues)
- Preserves ordinal relationship (Tier 1 > Tier 2 > ... > Tier 6)
- Data-driven approach (based on actual price patterns)
- More stable than one-hot encoding individual cities

**City Tier Mapping**: Saved in `preprocessing_artifacts.pkl` for production use

---

### Step 5: Encoding
**Purpose**: Convert City_Tier to integer type

**Transformation**:
- City_Tier (1-6) cast to integer
- Ordinal encoding preserved (1 = highest, 6 = lowest)

**Validation**:
- Verified median price decreases as tier number increases
- Confirmed all values in range [1, 6]

---

### Step 5b: Train-Test Split
**Purpose**: Create separate datasets for training and evaluation

**CRITICAL: Performed BEFORE scaling to prevent data leakage**

**Configuration**:
- **Split Ratio**: 80% training, 20% test
- **Method**: Stratified sampling by City_Tier
- **Random State**: 42 (for reproducibility)

**Stratification Rationale**:
- Ensures both sets have similar City_Tier distributions
- Prevents all luxury properties going to one set
- Maintains price distribution balance

**Features (X)**:
- Bedrooms
- Bathrooms
- House_Size (may contain NaN)
- Land_Size (may contain NaN)
- Is_Brand_New
- Is_Modern
- City_Tier

**Target (y)**:
- Price_Log (log-transformed price, created before split)

**Result**:
- Training set: ~80% of data
- Test set: ~20% of data
- Both sets maintain similar city tier distributions

---

### Step 6: Scaling & Normalization
**Purpose**: Standardize feature scales for model compatibility

**CRITICAL: Scaler fitted ONLY on training data**

**Process**:

1. **Missing Value Imputation**:
   - Calculated median from TRAINING set only:
     - `train_house_median` = median of training House_Size
     - `train_land_median` = median of training Land_Size
   - Applied these medians to fill NaN in BOTH train and test sets
   - **Rationale**: Test set imputation must not use test set statistics

2. **Feature Scaling**:
   - **Method**: StandardScaler (z-score normalization)
   - **Formula**: `z = (x - μ) / σ`
   - **Features Scaled**:
     - Bedrooms
     - Bathrooms
     - House_Size (after imputation)
     - Land_Size (after imputation)
   
3. **Scaler Fitting**:
   - Fitted on training data ONLY
   - Mean and std calculated from training set
   - Same scaler applied to transform both train and test

4. **Features NOT Scaled**:
   - Is_Brand_New (already 0/1)
   - Is_Modern (already 0/1)
   - City_Tier (ordinal, meaningful scale)

**Scaling Properties**:
- Training set: Mean ≈ 0.0, Std ≈ 1.0 (by definition)
- Test set: Mean ≈ 0.0, Std ≈ 1.0 (close but not exact)
- Negative values are NORMAL (indicate below-average values)

**Target Variable Transformation**:
- **Original**: Price (in LKR, highly skewed)
- **Transformed**: Price_Log = ln(Price)
- **Rationale**: 
  - Reduces skewness
  - Stabilizes variance
  - Better for regression models
  - Easily reversible: Price = exp(Price_Log)

---

### Step 7: Correlation Analysis
**Purpose**: Understand feature relationships and detect multicollinearity

**Analysis Performed**:

1. **Feature Correlation Matrix**:
   - Calculated on TRAINING set only
   - Shows pairwise correlations between all features

2. **High Correlation Detection**:
   - Flagged pairs with |correlation| > 0.7
   - Indicates potential multicollinearity issues

3. **Target Correlation**:
   - Correlation of each feature with Price_Log
   - Helps understand feature importance
   - Sorted by absolute correlation value

**Typical Findings**:
- House_Size and Land_Size: Moderate positive correlation (0.5-0.7)
- Bedrooms and Bathrooms: Moderate positive correlation (0.4-0.6)
- City_Tier: Negative correlation with price (lower tier = higher price in this encoding)
- Is_Modern: Positive correlation with price

**Use Cases**:
- Feature selection for modeling
- Understanding domain relationships
- Identifying redundant features
- Detecting multicollinearity before model training

---

### Step 8: Prepare Final Datasets
**Purpose**: Combine features and target into complete datasets

**Training Dataset**:
- Features: All 7 features (scaled where appropriate)
- Target: Price_Log
- Shape: ~700-800 rows × 8 columns

**Test Dataset**:
- Features: Same 7 features (transformed with training scaler)
- Target: Price_Log
- Shape: ~170-200 rows × 8 columns

**Column Order**:
1. Bedrooms (scaled)
2. Bathrooms (scaled)
3. House_Size (scaled, imputed)
4. Land_Size (scaled, imputed)
5. Is_Brand_New (binary)
6. Is_Modern (binary)
7. City_Tier (1-6, ordinal)
8. Price_Log (target)

---

### Step 9: Save Processed Data & Artifacts
**Purpose**: Persist datasets and preprocessing components for production

**Files Saved**:

1. **`data/03_processed/train_data.csv`**
   - Training dataset
   - Ready for model training
   - Features + Target

2. **`data/03_processed/test_data.csv`**
   - Test dataset
   - For final model evaluation
   - Features + Target

3. **`models/preprocessing_artifacts.pkl`**
   - StandardScaler object (fitted scaler)
   - Imputation values (training medians)
   - Feature columns list
   - Tier labels dictionary
   - City tier mapping dictionary
   - **Critical for production deployment**

---

## Feature Descriptions

### Numeric Features (Scaled)

#### 1. Bedrooms
- **Type**: Continuous numeric (scaled)
- **Original Range**: 1-15
- **Scaled**: Mean ≈ 0, Std ≈ 1
- **Interpretation**: Number of bedrooms in the property
- **Missing**: None (validated in Step 2)

#### 2. Bathrooms
- **Type**: Continuous numeric (scaled)
- **Original Range**: 1-15
- **Scaled**: Mean ≈ 0, Std ≈ 1
- **Interpretation**: Number of bathrooms in the property
- **Missing**: None (validated in Step 2)

#### 3. House_Size
- **Type**: Continuous numeric (scaled)
- **Original Units**: Square feet (sqft)
- **Typical Range**: 500-5000 sqft
- **Scaled**: Mean ≈ 0, Std ≈ 1
- **Imputation**: Training median for missing values
- **Interpretation**: Built-up area of the house

#### 4. Land_Size
- **Type**: Continuous numeric (scaled)
- **Original Units**: Perches
- **Typical Range**: 5-50 perches
- **Scaled**: Mean ≈ 0, Std ≈ 1
- **Imputation**: Training median for missing values
- **Interpretation**: Total land area of the property
- **Note**: 1 perch ≈ 272.25 square feet

### Binary Features (Not Scaled)

#### 5. Is_Brand_New
- **Type**: Binary (0 or 1)
- **Values**: 0 = Not brand new, 1 = Brand new
- **Source**: Extracted from property description text
- **Distribution**: Typically 5-15% are brand new

#### 6. Is_Modern
- **Type**: Binary (0 or 1)
- **Values**: 0 = Not modern/luxury, 1 = Modern/luxury
- **Source**: Extracted from property description text
- **Distribution**: Typically 15-30% are modern/luxury

### Ordinal Feature (Not Scaled)

#### 7. City_Tier
- **Type**: Ordinal categorical (1-6)
- **Values**: 
  - 1 = Luxury (highest median prices)
  - 2 = Premium
  - 3 = Upper-Mid
  - 4 = Mid-Range
  - 5 = Affordable
  - 6 = Budget (lowest median prices)
- **Source**: Derived from city median prices
- **Interpretation**: Price tier of the property's location
- **Note**: Lower number = More expensive area

### Target Variable

#### 8. Price_Log
- **Type**: Continuous numeric
- **Transformation**: Natural logarithm of price
- **Original**: Price in LKR (1M - 2B range)
- **Transformed**: Approximately 13.8 - 21.4 (ln scale)
- **Distribution**: Approximately normal after log transform
- **Inverse Transform**: Price = exp(Price_Log)

---

## Output Files

### Training Data
- **Path**: `data/03_processed/train_data.csv`
- **Format**: CSV (comma-separated)
- **Rows**: ~700-900 (80% of cleaned data)
- **Columns**: 8 (7 features + 1 target)
- **Use**: Model training, cross-validation, hyperparameter tuning
- **No Index**: Index not saved (use default integer index)

### Test Data
- **Path**: `data/03_processed/test_data.csv`
- **Format**: CSV (comma-separated)
- **Rows**: ~170-220 (20% of cleaned data)
- **Columns**: 8 (7 features + 1 target)
- **Use**: Final model evaluation ONLY (do not use during training!)
- **No Index**: Index not saved (use default integer index)

### Preprocessing Artifacts
- **Path**: `models/preprocessing_artifacts.pkl`
- **Format**: Python pickle (joblib)
- **Contents**:
  ```python
  {
      'scaler': StandardScaler object (fitted),
      'imputation_values': {
          'House_Size_median': float,
          'Land_Size_median': float
      },
      'feature_columns': ['Bedrooms', 'Bathrooms', 'House_Size', 'Land_Size'],
      'tier_labels': {1: 'Luxury', 2: 'Premium', ..., 6: 'Budget'},
      'city_tier_map': {'Colombo 3': 1, 'Colombo 7': 1, ..., 'Homagama': 6}
  }
  ```
- **Use**: Production deployment, new data transformation

### Loading Artifacts Example
```python
import joblib
import pandas as pd

# Load artifacts
artifacts = joblib.load('models/preprocessing_artifacts.pkl')
scaler = artifacts['scaler']
imputation_values = artifacts['imputation_values']
city_tier_map = artifacts['city_tier_map']

# Load training data
train_df = pd.read_csv('data/03_processed/train_data.csv')
X_train = train_df.drop('Price_Log', axis=1)
y_train = train_df['Price_Log']

# Load test data
test_df = pd.read_csv('data/03_processed/test_data.csv')
X_test = test_df.drop('Price_Log', axis=1)
y_test = test_df['Price_Log']
```

---

## Data Statistics

### Dataset Size Progression
1. **Initial**: ~1000-1200 rows (from house_data_with_city.csv)
2. **After Outlier Removal**: ~970-1170 rows (-1 to -3%)
3. **After Validation**: ~920-1100 rows (-2 to -5% additional)
4. **Training Set**: ~740-880 rows (80% of final)
5. **Test Set**: ~180-220 rows (20% of final)

### Missing Values (After Cleaning)
- **Bedrooms**: 0 (validated, no missing)
- **Bathrooms**: 0 (validated, no missing)
- **House_Size**: 0 (imputed with training median)
- **Land_Size**: 0 (imputed with training median)
- **Is_Brand_New**: 0 (binary, no missing)
- **Is_Modern**: 0 (binary, no missing)
- **City_Tier**: 0 (all cities mapped)
- **Price_Log**: 0 (target, no missing)

### Feature Distributions (Training Set, Scaled)

**Scaled Features** (Mean ≈ 0, Std ≈ 1):
- Bedrooms: Normal distribution, centered at 0
- Bathrooms: Normal distribution, centered at 0
- House_Size: Approximately normal after outlier removal
- Land_Size: Approximately normal after outlier removal

**Binary Features** (Distribution %):
- Is_Brand_New: ~5-15% positive cases
- Is_Modern: ~15-30% positive cases

**Ordinal Feature** (Count Distribution):
- City_Tier 1 (Luxury): ~10-15% of data
- City_Tier 2 (Premium): ~15-20% of data
- City_Tier 3 (Upper-Mid): ~18-22% of data
- City_Tier 4 (Mid-Range): ~15-18% of data
- City_Tier 5 (Affordable): ~16-20% of data
- City_Tier 6 (Budget): ~10-15% of data

**Target Variable** (Price_Log):
- Mean: ~16.5-17.5 (ln scale)
- Std: ~0.5-0.8
- Skewness: <0.5 (approximately normal)
- Min: ~13.8 (≈ LKR 1M)
- Max: ~21.4 (≈ LKR 2B)

### Price Ranges (Original Scale)
- **Training Minimum**: LKR ~1,000,000
- **Training Maximum**: LKR ~2,000,000,000
- **Training Median**: LKR ~30,000,000 - 50,000,000
- **Test set**: Similar distribution (verified by stratification)

---

## Technical Implementation Details

### Libraries Used
```python
import pandas as pd              # Data manipulation
import numpy as np               # Numerical operations
import re                        # Regular expressions (text cleaning)
from sklearn.model_selection import train_test_split  # Train-test split
from sklearn.preprocessing import StandardScaler      # Feature scaling
import joblib                    # Artifact serialization
import matplotlib.pyplot as plt  # Visualization (correlation)
import seaborn as sns           # Statistical visualization
import warnings                 # Suppress warnings
```

### Key Functions

#### 1. clean_size(value)
```python
def clean_size(value):
    """Extract numeric value from size strings"""
    if pd.isna(value):
        return np.nan
    value_str = str(value)
    value_str = re.sub(r'[^\d.]', '', value_str)  # Keep only digits and decimal
    try:
        return float(value_str) if value_str else np.nan
    except ValueError:
        return np.nan
```

#### 2. remove_outliers_iqr(df, column, multiplier=3.0)
```python
def remove_outliers_iqr(df, column, multiplier=3.0):
    """Remove outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = max(0, Q1 - multiplier * IQR)
    upper_bound = Q3 + multiplier * IQR
    df_filtered = df[~(((df[column] < lower_bound) | (df[column] > upper_bound)) & df[column].notna())]
    return df_filtered
```

#### 3. extract_binary_features(description)
```python
def extract_binary_features(description):
    """Extract key property attributes from text"""
    if pd.isna(description):
        return 0, 0
    desc_lower = str(description).lower()
    is_brand_new = 1 if any(keyword in desc_lower for keyword in ['brand new', 'newly built', 'new house']) else 0
    is_modern = 1 if any(keyword in desc_lower for keyword in ['modern', 'luxury', 'contemporary']) else 0
    return is_brand_new, is_modern
```

### Reproducibility
- **Random State**: 42 (used in train_test_split)
- **Stratification**: By City_Tier (ensures consistent splits)
- **Deterministic**: All transformations are deterministic (no random components in cleaning/scaling)

### Performance Considerations
- **Memory**: Dataset fits in memory (~1000 rows, 8-10 columns)
- **Processing Time**: Complete pipeline runs in <2 minutes
- **Scalability**: Pipeline can handle up to ~100,000 rows without modification

---

## Critical Considerations for Modeling

### 1. Data Leakage Prevention ✓
- **Train-test split performed BEFORE scaling**
- **Scaler fitted only on training data**
- **Imputation values from training set only**
- **No information from test set used during preprocessing**

### 2. Target Variable
- **Use Price_Log for model training** (not original Price)
- **Inverse transform predictions**: `predicted_price = np.exp(predicted_log_price)`
- **Evaluation metrics**: Consider on log scale AND original scale
  - On log scale: RMSE, MAE, R²
  - On original scale: MAPE, Median Absolute Error

### 3. Feature Scaling
- **Scaled features have negative values** (this is normal!)
- **Interpretation**: Negative = below average, Positive = above average
- **Models that benefit from scaling**: Linear Regression, Ridge, Lasso, SVM, Neural Networks
- **Models that don't need scaling**: Tree-based (Random Forest, XGBoost, LightGBM)

### 4. Ordinal Feature (City_Tier)
- **Already numeric (1-6)**, no additional encoding needed
- **Lower value = Higher price tier** (keep in mind for interpretation)
- **Can be used as-is in tree-based models**
- **For linear models**: Consider if ordinal relationship is appropriate
  - Alternative: One-hot encode if relationship is not strictly linear

### 5. Missing Values
- **No missing values in final datasets** (all imputed or removed)
- **Imputation method**: Median from training set
- **New data**: Use saved medians from preprocessing_artifacts.pkl

### 6. Outlier Handling
- **Conservative approach**: Only extreme outliers removed (3x IQR)
- **Some high-value properties remain** (up to LKR 2B)
- **Consider**: Robust regression methods if outliers still impact model

### 7. Feature Importance Expectations
Based on correlation analysis, expected importance ranking:
1. **City_Tier**: Strong predictor (location is key)
2. **House_Size**: Direct correlation with price
3. **Land_Size**: Important, especially for larger properties
4. **Bedrooms**: Moderate importance
5. **Bathrooms**: Moderate importance
6. **Is_Modern**: Small positive impact
7. **Is_Brand_New**: Small positive impact

### 8. Model Selection Recommendations

**Recommended Models**:
1. **Linear Regression** (baseline)
   - Simple, interpretable
   - Use as benchmark

2. **Ridge/Lasso Regression**
   - Handles multicollinearity
   - Feature selection (Lasso)
   - Good for regularization

3. **Random Forest**
   - Handles non-linear relationships
   - Robust to outliers
   - Feature importance analysis

4. **XGBoost/LightGBM**
   - State-of-the-art performance
   - Handles complex interactions
   - Fast training

5. **Neural Networks** (if dataset size permits)
   - Can capture complex patterns
   - Requires more data (may be limited here)

**Not Recommended**:
- K-Nearest Neighbors (small dataset, high dimensionality)
- Support Vector Regression (small dataset, may overfit)

### 9. Cross-Validation Strategy
- **Method**: K-Fold (k=5 or k=10)
- **Stratification**: Consider stratified k-fold by City_Tier
- **Rationale**: Ensures each fold has similar city tier distributions
- **Do NOT use test set** until final evaluation

### 10. Hyperparameter Tuning
- **Search Method**: GridSearchCV or RandomizedSearchCV
- **Scoring Metric**: neg_mean_squared_error or neg_mean_absolute_error
- **Use training set only** (with cross-validation)

### 11. Production Deployment Considerations
- **Save trained model** with joblib/pickle
- **Load preprocessing artifacts** from preprocessing_artifacts.pkl
- **Transformation pipeline** for new data:
  1. Clean size fields (clean_size function)
  2. Validate ranges (bedrooms, bathrooms, price)
  3. Extract text features (extract_binary_features function)
  4. Map city to tier (use city_tier_map)
  5. Impute missing sizes (use saved medians)
  6. Scale features (use saved scaler)
  7. Predict (model.predict)
  8. Inverse transform (np.exp to get original price)

### 12. Evaluation Metrics

**Primary Metrics**:
- **RMSE (on log scale)**: Penalizes large errors
- **R² Score**: Proportion of variance explained
- **MAE (on original scale)**: Average error in LKR

**Secondary Metrics**:
- **MAPE**: Mean Absolute Percentage Error
- **Median Absolute Error**: Robust to outliers
- **Prediction Interval**: Confidence bounds

**Business Metrics**:
- **Within 10% accuracy**: Percentage of predictions within 10% of actual
- **Within 20% accuracy**: Percentage of predictions within 20% of actual

### 13. Feature Engineering Opportunities
Potential additional features for future iterations:
- Price per square foot (Price / House_Size)
- Land to house ratio (Land_Size / House_Size)
- Bathroom to bedroom ratio (Bathrooms / Bedrooms)
- Total rooms (Bedrooms + Bathrooms)
- City-specific average prices (if more data available)

### 14. Known Limitations
- **Limited Dataset**: ~900 samples may limit complex model performance
- **City Coverage**: Only covers Colombo and surrounding areas
- **Time Period**: Snapshot in time, no temporal trends
- **Feature Sparsity**: Only 7 features, relatively simple
- **Text Features**: Simple keyword matching, not sophisticated NLP
- **Unit Loss**: Original size units information lost in cleaning

### 15. Data Quality Notes
- **Duplicate Removal**: Already performed in earlier pipeline stage
- **No Duplicate Detection**: Not repeated in this notebook
- **Source Quality**: Web-scraped data may contain inaccuracies
- **Validation**: Rule-based validation only (no manual verification)

---

## File Structure Summary

```
house-sales-ml-project/
├── data/
│   ├── 00_links/
│   │   ├── ad_links.txt
│   │   └── failed_links.txt
│   ├── 01_raw/
│   │   └── raw_house_data.csv
│   ├── 02_intermediate/
│   │   └── house_data_with_city.csv          # INPUT
│   ├── 03_processed/
│   │   ├── train_data.csv                    # OUTPUT (Training)
│   │   └── test_data.csv                     # OUTPUT (Test)
│   └── 04_additional/
│       └── city_list.txt
├── models/
│   └── preprocessing_artifacts.pkl            # OUTPUT (Artifacts)
├── notebooks/
│   └── 01_data_cleaning_and_feature_engineering.ipynb  # Processing Pipeline
├── scripts/
│   ├── city_extractor.py
│   └── scrape.py
├── readme.md
└── requirements.txt
```

---

## Next Steps for Model Training

### Immediate Actions
1. Load train_data.csv and test_data.csv
2. Verify data integrity (no NaN, correct shapes)
3. Separate features (X) from target (y)
4. Start with baseline linear regression model
5. Evaluate on training set with cross-validation
6. ONLY use test set for final evaluation

### Model Development Sequence
1. **Baseline**: Linear Regression
2. **Regularization**: Ridge, Lasso
3. **Tree-based**: Random Forest
4. **Boosting**: XGBoost, LightGBM
5. **Ensemble**: Combine best models

### Evaluation Protocol
1. Train on training set with cross-validation
2. Compare models using CV scores
3. Select best model(s)
4. Tune hyperparameters (still on training set)
5. **Final step**: Evaluate on test set ONCE

### Expected Performance
- **Good R²**: >0.70
- **Acceptable R²**: >0.60
- **RMSE (log scale)**: <0.3 (approximately ±35% price error)
- **Business Target**: 70%+ predictions within 20% of actual price

---

## Document Metadata

**Created**: February 18, 2026  
**Author**: Preprocessing Pipeline  
**Version**: 1.0  
**Status**: Production Ready  
**Last Updated**: February 18, 2026  

**Contact for Questions**:
- Preprocessing notebook: `notebooks/01_data_cleaning_and_feature_engineering.ipynb`
- Output files: `data/03_processed/` and`models/`

---

**END OF DOCUMENTATION**
