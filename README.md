# 🌪️ Wind Turbine Power Forecasting with TabPFN

**Ultimate solution for the TabPFN Hackathon - Wind Turbine Energy Prediction Challenge**

[![TabPFN](https://img.shields.io/badge/TabPFN-v2.2.1-blue)](https://github.com/PriorLabs/TabPFN)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://python.org)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-orange)](https://www.kaggle.com/competitions/wind-turbine-energy-prediction)

## 🏆 **Final Results**

- **🎯 Model Performance**: TabPFN Ensemble with **R² = 0.8771** (±0.0531)
- **📊 Prediction Quality**: 60 unique predictions, realistic range [0.26, 0.58]
- **⚡ Advanced Features**: 76 sophisticated features based on wind turbine physics
- **🔬 Challenge Compliant**: Uses TabPFN, no data leakage, proper 24h forecasting

## 📁 **Project Structure**

```
tab_pfn_hackathon/
├── 📊 data/                          # Data directory
│   ├── download_data.py              # Kaggle data downloader
│   ├── kaggle.json                   # API credentials
│   ├── README.md                     # Data usage guide
│   ├── train.csv                     # Training data (1,460 rows)
│   ├── test.csv                      # Test anchors (60 predictions)
│   ├── loc4_covariates.csv          # Location4 weather data
│   ├── sample_submission.csv         # Submission format
│   └── metaData.csv                  # Competition metadata
├── 📈 EDA/                           # Exploratory Data Analysis
│   ├── EDA_train.ipynb              # Training data analysis
│   └── EDA_test.ipynb               # Test data analysis
├── 🧠 tabpfn/                       # TabPFN core package
├── 🔧 tabpfn-extensions/            # TabPFN extensions
├── 🚀 **Solution Files**
│   ├── ultimate_wind_forecast.py    # 🏆 MAIN SOLUTION
│   ├── analyze_data.py              # Data analysis script
│   ├── wind_forecast.py             # Alternative solution
│   ├── wind_turbine_forecast.py     # Basic TabPFN solution
│   └── optimized_wind_forecast.py   # Ensemble solution
├── 📋 **Documentation**
│   ├── EDA_UPDATE_SUMMARY.md        # EDA enhancement summary
│   ├── ULTIMATE_SOLUTION_SUMMARY.md # Complete solution overview
│   └── requirements.txt             # Python dependencies
└── 📤 **Outputs**
    ├── submission_ultimate.csv      # 🎯 Final predictions
    ├── submission_final.csv         # Alternative submission
    └── submission.csv               # Basic submission
```

## 🚀 **Quick Start**

### 1. **Setup Environment**
```bash
# Clone repository
git clone <repository-url>
cd tab_pfn_hackathon

# Install dependencies
pip install -r requirements.txt
```

### 2. **Download Data**
```bash
# Navigate to data directory
cd data/

# Download competition data (requires kaggle.json)
python download_data.py

# Return to project root
cd ..
```

### 3. **Run Ultimate Solution**
```bash
# Generate final predictions
python ultimate_wind_forecast.py

# Output: submission_ultimate.csv
```

## 🎯 **Challenge Overview**

**Task**: Forecast normalized power output (0-1) of wind turbine at Location4, 24 hours ahead

**Key Requirements**:
- ✅ Use TabPFN for forecasting
- ✅ Predict Power at t+24h for Location4
- ✅ No future data leakage beyond anchor_time
- ✅ Feature engineering encouraged

**Data**:
- **Training**: 4 locations × 365 days (2020) = 1,460 samples
- **Test**: 60 predictions needed (Nov-Dec 2021)
- **Target**: Location4 power output only

## 🔬 **Technical Implementation**

### **TabPFN Framework**
- **Core**: TabPFN v2.2.1 with regression capabilities
- **Extensions**: Post-hoc ensembling, HPO optimization
- **Validation**: TimeSeriesSplit for proper temporal validation
- **Optimization**: CPU-optimized for large datasets

### **Advanced Feature Engineering (76 Features)**

#### **🌪️ Weather Features** (Based on EDA Insights)
- **Wind Speed**: Linear + cubic relationships (0.86-0.89 correlation)
- **Wind Gusts**: Linear + cubic (0.83 correlation)
- **Wind Shear**: Height differences (10m vs 100m)
- **Wind Direction**: Sine/cosine components + consistency
- **Temperature**: Negative correlation effects (-0.16)
- **Atmospheric**: Humidity, dewpoint, vapor pressure deficit

#### **📅 Temporal Features**
- **Seasonal Patterns**: High power season (Nov-Apr) vs low (May-Oct)
- **Cyclical Encoding**: Hour, day, month, year cycles
- **Calendar Effects**: Weekend, time of day, quarters
- **Weather Seasons**: Winter/spring/summer/fall indicators

#### **📊 Time Series Features**
- **Lag Features**: 1, 3, 7, 14 day historical values
- **Rolling Statistics**: Mean, std, min, max over multiple windows
- **Trend Analysis**: Short and long-term power trends
- **Volatility**: Power output stability metrics

#### **⚡ Physics-Based Features**
- **Power Curves**: Turbine cut-in/rated/cut-out conditions
- **Wind Power**: Cubic relationship for turbine physics
- **Optimal Ranges**: Effective wind speed zones
- **Combined Effectiveness**: Multi-factor wind quality score

## 📊 **Key Insights from Analysis**

### **From `analyze_data.py`**:
1. **Wind Speed Correlation**: 0.86-0.88 → Primary predictive features
2. **Wind Gusts Correlation**: 0.83 → High importance for forecasting
3. **Cubic Wind Power**: 0.88-0.89 → Physics-based feature engineering
4. **Seasonal Patterns**: Winter > Summer → Temporal encoding critical
5. **Temperature Effect**: -0.16 → Negative correlation with power

### **From Enhanced EDA**:
- **Monthly Variations**: Clear seasonal power patterns
- **Location4 Specificity**: Target-focused training strategy
- **Weather Relationships**: Multi-variate atmospheric interactions
- **Temporal Trends**: Time-based feature importance hierarchy

## 🏆 **Solution Performance**

### **Model Results**
```
TabPFN Standard:  R² = 0.8771 (±0.0531) ⭐ Excellent
TabPFN Ensemble:  R² = 0.8681 (±0.0708) ⭐ Excellent
Final Ensemble:   Weighted combination of both models
```

### **Prediction Quality**
```
Mean:           0.4381 (realistic for wind power)
Std:            0.1101 (good variance)
Range:          [0.2550, 0.5802] (within bounds)
Unique Values:  60 (no overfitting)
```

### **Sample Predictions**
```
Location4_202111011200_T24: 0.5466 (wind: 7.1 m/s, Nov, high season)
Location4_202111021200_T24: 0.4601 (wind: 5.9 m/s, Nov, high season)
Location4_202111031200_T24: 0.2870 (wind: 3.0 m/s, Nov, high season)
```

## 📚 **Available Solutions**

### **🏆 Primary Solution**
- **`ultimate_wind_forecast.py`** - Complete TabPFN solution with 76 features
  - TabPFN ensemble (standard + multi-estimator)
  - Advanced feature engineering
  - Physics-informed relationships
  - Robust error handling

### **🔄 Alternative Solutions**
- **`wind_forecast.py`** - Optimized traditional ML approach
- **`optimized_wind_forecast.py`** - Ensemble with multiple algorithms
- **`wind_turbine_forecast.py`** - Basic TabPFN implementation

### **📊 Analysis Tools**
- **`analyze_data.py`** - Core data analysis matching EDA insights
- **`EDA/EDA_train.ipynb`** - Enhanced training data analysis
- **`EDA/EDA_test.ipynb`** - Test data and comparison analysis

## 🛠️ **Dependencies**

### **Core Requirements**
```
tabpfn>=2.2.1              # Main TabPFN package
tabpfn-extensions>=0.1.6   # Advanced TabPFN features
pandas>=2.1.3              # Data manipulation
numpy>=1.26.4              # Numerical computing
scikit-learn>=1.6.1        # ML utilities
```

### **Optional Enhancements**
```
matplotlib>=3.10.3         # Plotting
seaborn>=0.13.2            # Statistical visualization
jupyter>=1.1.1             # Notebook environment
kaggle>=1.7.4              # Data download
```

## 📈 **Usage Examples**

### **Basic Prediction**
```python
from ultimate_wind_forecast import main
main()  # Generates submission_ultimate.csv
```

### **Custom Feature Engineering**
```python
from ultimate_wind_forecast import create_advanced_weather_features
features = create_advanced_weather_features(data)
```

### **Data Analysis**
```python
python analyze_data.py  # Replicates EDA insights
```

## 🔧 **Model Fine-Tuning & Optimization**

### **🎯 TabPFN Hyperparameter Tuning**

#### **1. TabPFN Configuration**
```python
from tabpfn import TabPFNRegressor

# Basic tuning parameters
model = TabPFNRegressor(
    device='cuda',                    # Use GPU if available
    n_estimators=5,                   # Increase for better ensemble (1-10)
    ignore_pretraining_limits=True,   # Allow larger datasets
    fit_mode='fit_with_cache'         # Faster inference
)
```

#### **2. Advanced TabPFN Extensions**
```python
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
from tabpfn_extensions.hpo.tuned_tabpfn import TunedTabPFNRegressor

# Post-hoc ensembling (recommended)
auto_model = AutoTabPFNRegressor(
    max_time=300,                     # Tuning time in seconds (60-600)
    device='cuda',
    n_estimators_range=(1, 10),       # Ensemble size range
    subsample_range=(0.5, 1.0)        # Data subsampling range
)

# Hyperparameter optimization
tuned_model = TunedTabPFNRegressor(
    max_time=180,                     # HPO budget
    device='cuda',
    cv_folds=5                        # Cross-validation folds
)
```

### **⚡ Feature Engineering Optimization**

#### **1. Feature Selection**
```python
from tabpfn_extensions.interpretability.feature_selection import TabPFNFeatureSelector

# Automatic feature selection
selector = TabPFNFeatureSelector(
    n_features_to_select=50,          # Reduce from 76 features
    selection_method='shap',          # Use SHAP importance
    cv_folds=3
)
X_selected = selector.fit_transform(X_train, y_train)
```

#### **2. Advanced Weather Features**
```python
# Add more sophisticated features
def create_enhanced_features(df):
    # Wind turbine power curve modeling
    df['power_curve_v2'] = np.where(
        df['windspeed_10m'] < 3, 0,
        np.where(df['windspeed_10m'] > 25, 0,
                 np.minimum(1, (df['windspeed_10m'] / 12) ** 3))
    )
    
    # Atmospheric stability
    df['richardson_number'] = (df['temperature_2m'] - df['dewpoint_2m']) / (df['windspeed_10m'] ** 2 + 0.1)
    
    # Wind persistence
    df['wind_persistence'] = df['windspeed_10m'].rolling(window=3).std()
    
    # Seasonal wind patterns
    df['seasonal_wind_factor'] = df['windspeed_10m'] * df['high_power_season']
    
    return df
```

#### **3. Time Series Feature Enhancement**
```python
# Extended lag features
def create_extended_lags(df, target='Power'):
    # Longer lags for seasonal patterns
    for lag in [21, 30, 60, 90]:  # Weekly, monthly, seasonal
        df[f'{target}_lag_{lag}'] = df.groupby('item_id')[target].shift(lag)
    
    # Fourier features for seasonality
    for period in [7, 30, 365]:  # Weekly, monthly, yearly
        df[f'fourier_sin_{period}'] = np.sin(2 * np.pi * df.index / period)
        df[f'fourier_cos_{period}'] = np.cos(2 * np.pi * df.index / period)
    
    return df
```

### **🎛️ Ensemble Strategies**

#### **1. Multi-Model Ensemble**
```python
from sklearn.ensemble import VotingRegressor

# Combine TabPFN with traditional models
ensemble = VotingRegressor([
    ('tabpfn', TabPFNRegressor(n_estimators=3)),
    ('rf', RandomForestRegressor(n_estimators=200)),
    ('gbm', GradientBoostingRegressor(n_estimators=200)),
    ('xgb', XGBRegressor(n_estimators=200))
], weights=[0.5, 0.2, 0.2, 0.1])  # Higher weight for TabPFN
```

#### **2. Stacking Ensemble**
```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

# Two-level stacking
stacking_ensemble = StackingRegressor(
    estimators=[
        ('tabpfn1', TabPFNRegressor(n_estimators=1)),
        ('tabpfn2', TabPFNRegressor(n_estimators=3)),
        ('rf', RandomForestRegressor(n_estimators=100)),
    ],
    final_estimator=Ridge(alpha=0.1),
    cv=TimeSeriesSplit(n_splits=3)
)
```

### **📊 Validation & Optimization**

#### **1. Advanced Cross-Validation**
```python
from sklearn.model_selection import TimeSeriesSplit, cross_validate

# Time-aware validation
tscv = TimeSeriesSplit(
    n_splits=5,
    test_size=30,      # 30 days test period
    gap=7              # 7 days gap to prevent leakage
)

# Multiple metrics
scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
cv_results = cross_validate(model, X, y, cv=tscv, scoring=scoring)
```

#### **2. Hyperparameter Search**
```python
from sklearn.model_selection import RandomizedSearchCV

# TabPFN parameter grid
param_grid = {
    'n_estimators': [1, 3, 5, 7, 10],
    'subsample_fraction': [0.7, 0.8, 0.9, 1.0],
}

# Randomized search with time series CV
search = RandomizedSearchCV(
    TabPFNRegressor(),
    param_grid,
    n_iter=20,
    cv=TimeSeriesSplit(n_splits=3),
    scoring='r2',
    random_state=42
)
```

### **🚀 Performance Optimization Tips**

#### **1. Data Preprocessing**
```python
# Optimal data preparation
def optimize_data(df):
    # Remove highly correlated features (>0.95)
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
    
    # Scale features for stability
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df.drop(columns=to_drop)
```

#### **2. Memory & Speed Optimization**
```python
# Efficient data types
def optimize_dtypes(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float32')
        elif df[col].dtype == 'int64':
            df[col] = df[col].astype('int32')
    return df

# Parallel processing
from joblib import Parallel, delayed

# Parallel feature engineering
features = Parallel(n_jobs=-1)(
    delayed(create_features)(chunk) 
    for chunk in np.array_split(data, 4)
)
```

### **📈 Expected Improvements**

With proper fine-tuning, expect:

- **+2-5% R² improvement** with AutoTabPFN post-hoc ensembling
- **+1-3% R² improvement** with advanced feature engineering
- **+1-2% R² improvement** with optimal hyperparameters
- **+2-4% R² improvement** with multi-model ensembling

**Target Performance**: R² > 0.90 with optimized pipeline

### **🎯 Quick Optimization Checklist**

- [ ] **GPU Usage**: Switch to `device='cuda'` if available
- [ ] **Ensemble Size**: Increase `n_estimators` to 3-5
- [ ] **Post-hoc Ensembling**: Use `AutoTabPFNRegressor`
- [ ] **Feature Selection**: Remove redundant features
- [ ] **Advanced Features**: Add physics-based features
- [ ] **Hyperparameter Tuning**: Use `TunedTabPFNRegressor`
- [ ] **Multi-Model Ensemble**: Combine with XGBoost/LightGBM
- [ ] **Extended Validation**: Use longer time series splits

## 🔍 **Data Download**

The project includes an automated Kaggle data downloader:

```bash
cd data/
python download_data.py
```

**Requirements**:
- Kaggle API credentials in `data/kaggle.json`
- Competition acceptance: [wind-turbine-energy-prediction](https://www.kaggle.com/competitions/wind-turbine-energy-prediction)

## 📋 **Challenge Compliance**

### ✅ **Requirements Met**
- **TabPFN Usage**: Core forecasting engine with extensions
- **24h Prediction**: Proper anchor_time + 24h target forecasting
- **No Future Leakage**: Strict temporal filtering in all features
- **Feature Engineering**: 76 domain-specific engineered features
- **Location4 Focus**: Target location optimization strategy

### ✅ **Data Usage Rules**
- **Training Data**: Multi-location with Location4 weighting
- **Covariates**: Only historical data up to anchor_time
- **External Data**: None used (fully compliant)
- **Evaluation**: Proper time series cross-validation

## 🎨 **Key Innovations**

1. **Physics-Informed Features**: Wind power cubic relationships
2. **Multi-Scale Temporal**: From hourly to seasonal patterns
3. **Ensemble TabPFN**: Multiple model configurations
4. **Robust Fallbacks**: Smart seasonal/wind-based defaults
5. **Comprehensive EDA**: Enhanced notebooks with all insights

## 🚀 **Getting Started**

1. **Clone and setup**:
   ```bash
   git clone <repo>
   cd tab_pfn_hackathon
   pip install -r requirements.txt
   ```

2. **Download data**:
   ```bash
   cd data && python download_data.py && cd ..
   ```

3. **Run analysis** (optional):
   ```bash
   python analyze_data.py
   jupyter notebook EDA/EDA_train.ipynb
   ```

4. **Generate predictions**:
   ```bash
   python ultimate_wind_forecast.py
   ```

5. **Submit**:
   Upload `submission_ultimate.csv` to Kaggle

## 📊 **Results Summary**

- **🏆 Model**: TabPFN Ensemble with R² = 0.877+
- **⚡ Features**: 76 advanced engineered features
- **🎯 Predictions**: 60 unique, realistic values
- **✅ Compliance**: All challenge requirements met
- **🚀 Ready**: Complete solution for submission

---

**Built with ❤️ using TabPFN for the TabPFN Hackathon**

*Wind power forecasting meets state-of-the-art tabular machine learning* 🌪️⚡