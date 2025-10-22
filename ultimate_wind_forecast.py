#!/usr/bin/env python3
"""
Ultimate Wind Turbine Power Forecasting with TabPFN
===================================================

This is the ultimate solution combining:
1. TabPFN time-series capabilities
2. TabPFN extensions (post-hoc ensembling, HPO)
3. Optimal feature engineering based on EDA insights
4. Advanced time-series features
5. Multi-location training with Location4 focus

Key insights from analysis:
- Wind speed: 0.86-0.88 correlation with power
- Wind gusts: 0.83 correlation
- Wind power (cubic): 0.88-0.89 correlation
- Strong seasonal patterns (winter > summer)
- Temperature: negative correlation (-0.16)

Challenge requirements:
- Use TabPFN for forecasting
- Predict Power at t+24h for Location4
- No future data leakage beyond anchor_time
- Feature engineering encouraged
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
import joblib

warnings.filterwarnings('ignore')

# Set environment variables for TabPFN
os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'
os.environ['TABPFN_DISABLE_TELEMETRY'] = '1'

def load_data():
    """Load and prepare all datasets"""
    print("Loading data...")
    
    train_df = pd.read_csv('data/train.csv')
    train_df['Time'] = pd.to_datetime(train_df['Time'])
    
    loc4_covariates = pd.read_csv('data/loc4_covariates.csv')
    loc4_covariates['Time'] = pd.to_datetime(loc4_covariates['Time'])
    
    test_df = pd.read_csv('data/test.csv')
    test_df['anchor_time'] = pd.to_datetime(test_df['anchor_time'])
    test_df['Time'] = pd.to_datetime(test_df['Time'])
    
    sample_submission = pd.read_csv('data/sample_submission.csv')
    
    print(f"Training data: {train_df.shape}")
    print(f"Location4 covariates: {loc4_covariates.shape}")
    print(f"Test anchors: {len(test_df)}")
    
    return train_df, loc4_covariates, test_df, sample_submission

def create_advanced_weather_features(df, prefix=''):
    """Create advanced weather features based on EDA insights"""
    
    # Column names (handle anchor prefix)
    ws_10m = f'{prefix}windspeed_10m'
    ws_100m = f'{prefix}windspeed_100m'
    wg_10m = f'{prefix}windgusts_10m'
    wd_10m = f'{prefix}winddirection_10m'
    wd_100m = f'{prefix}winddirection_100m'
    temp = f'{prefix}temperature_2m'
    humidity = f'{prefix}relativehumidity_2m'
    dewpoint = f'{prefix}dewpoint_2m'
    
    features = {}
    
    # Core wind features (highest correlations)
    features[f'{prefix}windspeed_10m'] = df[ws_10m]
    features[f'{prefix}windspeed_100m'] = df[ws_100m]
    features[f'{prefix}windgusts_10m'] = df[wg_10m]
    
    # Wind power features (cubic relationship - 0.88-0.89 correlation)
    features[f'{prefix}wind_power_10m'] = np.power(df[ws_10m], 3)
    features[f'{prefix}wind_power_100m'] = np.power(df[ws_100m], 3)
    features[f'{prefix}gust_power'] = np.power(df[wg_10m], 3)
    
    # Wind relationships
    features[f'{prefix}wind_shear'] = df[ws_100m] - df[ws_10m]
    features[f'{prefix}wind_ratio'] = df[ws_100m] / (df[ws_10m] + 0.1)
    features[f'{prefix}gust_factor'] = df[wg_10m] / (df[ws_10m] + 0.1)
    features[f'{prefix}wind_intensity'] = np.sqrt(df[ws_10m].astype(float)**2 + df[ws_100m].astype(float)**2)
    
    # Wind direction features (convert to components)
    wd_10m_rad = np.radians(df[wd_10m].astype(float))
    wd_100m_rad = np.radians(df[wd_100m].astype(float))
    
    features[f'{prefix}wind_dir_10m_sin'] = np.sin(wd_10m_rad)
    features[f'{prefix}wind_dir_10m_cos'] = np.cos(wd_10m_rad)
    features[f'{prefix}wind_dir_100m_sin'] = np.sin(wd_100m_rad)
    features[f'{prefix}wind_dir_100m_cos'] = np.cos(wd_100m_rad)
    
    # Direction consistency and shear
    wind_dir_diff = np.abs(df[wd_100m].astype(float) - df[wd_10m].astype(float))
    wind_dir_diff = np.minimum(wind_dir_diff, 360 - wind_dir_diff)
    features[f'{prefix}wind_dir_shear'] = wind_dir_diff
    features[f'{prefix}wind_dir_consistency'] = np.exp(-wind_dir_diff / 45)  # Exponential decay
    
    # Temperature effects (negative correlation -0.16)
    features[f'{prefix}temperature_2m'] = df[temp]
    features[f'{prefix}temp_effect'] = -df[temp] / 50.0  # Normalized negative effect
    features[f'{prefix}temp_squared'] = df[temp] ** 2
    
    # Humidity and dewpoint
    features[f'{prefix}humidity'] = df[humidity]
    features[f'{prefix}dewpoint'] = df[dewpoint]
    features[f'{prefix}temp_dewpoint_diff'] = df[temp] - df[dewpoint]
    features[f'{prefix}vapor_pressure_deficit'] = df[temp] - df[dewpoint]  # VPD approximation
    
    # Optimal wind conditions for turbines
    features[f'{prefix}optimal_wind_10m'] = ((df[ws_10m] >= 3) & (df[ws_10m] <= 25)).astype(float)
    features[f'{prefix}optimal_wind_100m'] = ((df[ws_100m] >= 3) & (df[ws_100m] <= 25)).astype(float)
    features[f'{prefix}cut_in_wind'] = (df[ws_10m] >= 3).astype(float)
    features[f'{prefix}rated_wind'] = ((df[ws_10m] >= 12) & (df[ws_10m] <= 25)).astype(float)
    
    # Combined effectiveness score
    features[f'{prefix}wind_effectiveness'] = (
        features[f'{prefix}optimal_wind_10m'] * 
        features[f'{prefix}optimal_wind_100m'] * 
        features[f'{prefix}wind_dir_consistency']
    )
    
    # Power curve approximation
    wind_speed_norm = np.clip(df[ws_10m] / 25.0, 0, 1)  # Normalize to [0,1]
    features[f'{prefix}power_curve_approx'] = np.where(
        df[ws_10m] < 3, 0,  # Below cut-in
        np.where(df[ws_10m] > 25, 0,  # Above cut-out
                 wind_speed_norm ** 3)  # Cubic in between
    )
    
    return pd.DataFrame(features)

def create_temporal_features(df, time_col='Time'):
    """Create comprehensive temporal features"""
    features = {}
    
    # Basic time components
    features['hour'] = df[time_col].dt.hour
    features['day_of_week'] = df[time_col].dt.dayofweek
    features['day_of_year'] = df[time_col].dt.dayofyear
    features['month'] = df[time_col].dt.month
    features['quarter'] = df[time_col].dt.quarter
    features['week_of_year'] = df[time_col].dt.isocalendar().week
    
    # Seasonal patterns (based on EDA - winter has higher power)
    high_power_months = [11, 12, 1, 2, 3, 4]  # Nov-Apr
    features['high_power_season'] = df[time_col].dt.month.isin(high_power_months).astype(float)
    features['winter_season'] = df[time_col].dt.month.isin([12, 1, 2]).astype(float)
    features['spring_season'] = df[time_col].dt.month.isin([3, 4, 5]).astype(float)
    features['summer_season'] = df[time_col].dt.month.isin([6, 7, 8]).astype(float)
    features['fall_season'] = df[time_col].dt.month.isin([9, 10, 11]).astype(float)
    
    # Cyclical encoding for smooth transitions
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    features['day_of_year_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365.25)
    features['day_of_year_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365.25)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    
    # Weekend/weekday
    features['is_weekend'] = (features['day_of_week'] >= 5).astype(float)
    
    # Time of day categories
    features['is_morning'] = ((features['hour'] >= 6) & (features['hour'] < 12)).astype(float)
    features['is_afternoon'] = ((features['hour'] >= 12) & (features['hour'] < 18)).astype(float)
    features['is_evening'] = ((features['hour'] >= 18) & (features['hour'] < 24)).astype(float)
    features['is_night'] = ((features['hour'] >= 0) & (features['hour'] < 6)).astype(float)
    
    return pd.DataFrame(features)

def create_lag_and_rolling_features(df, target_col='Power', group_col='item_id'):
    """Create sophisticated lag and rolling features"""
    df = df.copy()
    df = df.sort_values([group_col, 'Time'])
    
    # Lag features (1, 3, 7, 14 days)
    for lag in [1, 3, 7, 14]:
        df[f'{target_col}_lag_{lag}'] = df.groupby(group_col)[target_col].shift(lag)
    
    # Rolling statistics (multiple windows)
    for window in [3, 7, 14, 30]:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}'] = (
            df.groupby(group_col)[target_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Rolling std
        df[f'{target_col}_rolling_std_{window}'] = (
            df.groupby(group_col)[target_col]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
        
        # Rolling min/max
        df[f'{target_col}_rolling_min_{window}'] = (
            df.groupby(group_col)[target_col]
            .rolling(window=window, min_periods=1)
            .min()
            .reset_index(0, drop=True)
        )
        
        df[f'{target_col}_rolling_max_{window}'] = (
            df.groupby(group_col)[target_col]
            .rolling(window=window, min_periods=1)
            .max()
            .reset_index(0, drop=True)
        )
    
    # Trend features
    df[f'{target_col}_trend_3'] = (
        df[f'{target_col}_lag_1'] - df[f'{target_col}_lag_3']
    )
    df[f'{target_col}_trend_7'] = (
        df[f'{target_col}_lag_1'] - df[f'{target_col}_lag_7']
    )
    
    # Volatility
    df[f'{target_col}_volatility_7'] = (
        df[f'{target_col}_rolling_std_7'] / (df[f'{target_col}_rolling_mean_7'] + 0.001)
    )
    
    return df

def prepare_training_data(train_df, loc4_covariates):
    """Prepare comprehensive training dataset"""
    print("Creating comprehensive features...")
    
    # Combine training data with Location4 covariates for complete history
    loc4_train = train_df[train_df['item_id'] == 'Location4'].copy()
    
    # Add Location4 covariates (without Power) to extend the time series
    loc4_covariates_extended = loc4_covariates.copy()
    loc4_covariates_extended['item_id'] = 'Location4'
    loc4_covariates_extended['Power'] = np.nan
    
    # Combine Location4 data
    loc4_combined = pd.concat([loc4_train, loc4_covariates_extended], ignore_index=True)
    loc4_combined = loc4_combined.sort_values('Time').drop_duplicates(subset=['Time'], keep='first')
    
    # Add other locations for pattern diversity (sample to manage size)
    other_locations = train_df[train_df['item_id'] != 'Location4'].copy()
    if len(other_locations) > 500:
        other_locations = other_locations.sample(n=500, random_state=42)
    
    # Combine all data with Location4 weighted more heavily
    combined_data = pd.concat([
        loc4_combined,
        loc4_combined,  # Double Location4 for more weight
        other_locations
    ], ignore_index=True)
    
    # Create weather features
    weather_features = create_advanced_weather_features(combined_data)
    
    # Create temporal features
    temporal_features = create_temporal_features(combined_data)
    
    # Combine features
    feature_df = pd.concat([
        combined_data[['item_id', 'Time', 'Power']],
        weather_features,
        temporal_features
    ], axis=1)
    
    # Add lag and rolling features
    feature_df = create_lag_and_rolling_features(feature_df)
    
    return feature_df

def create_prediction_features(test_row, train_df, loc4_covariates):
    """Create prediction features for a single test row"""
    
    anchor_time = test_row['anchor_time']
    target_time = test_row['Time']
    
    # Create weather features from anchor data
    anchor_weather_features = create_advanced_weather_features(
        test_row.to_frame().T, prefix='anchor_'
    )
    
    # Create temporal features for target time
    target_temporal_features = create_temporal_features(
        pd.DataFrame({'Time': [target_time]})
    )
    
    # Get historical Location4 data for lag features
    loc4_historical = train_df[
        (train_df['item_id'] == 'Location4') & 
        (train_df['Time'] <= anchor_time)
    ].copy()
    
    # Add covariates up to anchor time
    loc4_covariates_filtered = loc4_covariates[
        loc4_covariates['Time'] <= anchor_time
    ].copy()
    
    if len(loc4_covariates_filtered) > 0:
        loc4_covariates_filtered['item_id'] = 'Location4'
        loc4_covariates_filtered['Power'] = np.nan
        
        # Combine historical data
        combined_historical = pd.concat([loc4_historical, loc4_covariates_filtered], ignore_index=True)
        combined_historical = combined_historical.sort_values('Time').drop_duplicates(subset=['Time'], keep='first')
    else:
        combined_historical = loc4_historical
    
    # Calculate lag features
    lag_features = {}
    if len(combined_historical) > 0:
        # Sort by time
        combined_historical = combined_historical.sort_values('Time')
        recent_power = combined_historical['Power'].dropna().values
        
        if len(recent_power) >= 1:
            lag_features['Power_lag_1'] = recent_power[-1]
        if len(recent_power) >= 3:
            lag_features['Power_lag_3'] = recent_power[-3]
            lag_features['Power_rolling_mean_3'] = np.mean(recent_power[-3:])
            lag_features['Power_rolling_std_3'] = np.std(recent_power[-3:])
            lag_features['Power_rolling_min_3'] = np.min(recent_power[-3:])
            lag_features['Power_rolling_max_3'] = np.max(recent_power[-3:])
        if len(recent_power) >= 7:
            lag_features['Power_lag_7'] = recent_power[-7]
            lag_features['Power_rolling_mean_7'] = np.mean(recent_power[-7:])
            lag_features['Power_rolling_std_7'] = np.std(recent_power[-7:])
            lag_features['Power_rolling_min_7'] = np.min(recent_power[-7:])
            lag_features['Power_rolling_max_7'] = np.max(recent_power[-7:])
            lag_features['Power_volatility_7'] = np.std(recent_power[-7:]) / (np.mean(recent_power[-7:]) + 0.001)
        if len(recent_power) >= 14:
            lag_features['Power_lag_14'] = recent_power[-14]
            lag_features['Power_rolling_mean_14'] = np.mean(recent_power[-14:])
            lag_features['Power_rolling_std_14'] = np.std(recent_power[-14:])
            lag_features['Power_rolling_min_14'] = np.min(recent_power[-14:])
            lag_features['Power_rolling_max_14'] = np.max(recent_power[-14:])
        if len(recent_power) >= 30:
            lag_features['Power_rolling_mean_30'] = np.mean(recent_power[-30:])
            lag_features['Power_rolling_std_30'] = np.std(recent_power[-30:])
            lag_features['Power_rolling_min_30'] = np.min(recent_power[-30:])
            lag_features['Power_rolling_max_30'] = np.max(recent_power[-30:])
        
        # Trend features
        if len(recent_power) >= 3:
            lag_features['Power_trend_3'] = recent_power[-1] - recent_power[-3]
        if len(recent_power) >= 7:
            lag_features['Power_trend_7'] = recent_power[-1] - recent_power[-7]
    
    # Combine all features
    prediction_features = {}
    
    # Add weather features (remove anchor prefix)
    for col in anchor_weather_features.columns:
        new_col = col.replace('anchor_', '')
        prediction_features[new_col] = anchor_weather_features[col].iloc[0]
    
    # Add temporal features
    for col in target_temporal_features.columns:
        prediction_features[col] = target_temporal_features[col].iloc[0]
    
    # Add lag features
    for col, value in lag_features.items():
        prediction_features[col] = value
    
    return pd.DataFrame([prediction_features])

def train_tabpfn_models(X_train, y_train, feature_cols):
    """Train TabPFN models with different configurations"""
    models = {}
    
    try:
        from tabpfn import TabPFNRegressor
        
        print("Training TabPFN models...")
        
        # Standard TabPFN
        print("  - Standard TabPFN...")
        tabpfn_standard = TabPFNRegressor(
            device='cpu',
            ignore_pretraining_limits=True,
            n_estimators=1
        )
        tabpfn_standard.fit(X_train, y_train)
        models['TabPFN_Standard'] = tabpfn_standard
        
        # TabPFN with multiple estimators
        print("  - TabPFN with ensemble...")
        tabpfn_ensemble = TabPFNRegressor(
            device='cpu',
            ignore_pretraining_limits=True,
            n_estimators=3
        )
        tabpfn_ensemble.fit(X_train, y_train)
        models['TabPFN_Ensemble'] = tabpfn_ensemble
        
    except Exception as e:
        print(f"TabPFN training failed: {e}")
    
    # Try TabPFN Extensions
    try:
        from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNRegressor
        
        print("  - AutoTabPFN with post-hoc ensembling...")
        auto_tabpfn = AutoTabPFNRegressor(
            max_time=60,  # 1 minute tuning
            device='cpu'
        )
        auto_tabpfn.fit(X_train, y_train)
        models['AutoTabPFN'] = auto_tabpfn
        
    except Exception as e:
        print(f"AutoTabPFN not available: {e}")
    
    # Try time-series specific TabPFN
    try:
        from tabpfn_time_series import TabPFNTimeSeriesRegressor
        
        print("  - TabPFN Time Series...")
        tabpfn_ts = TabPFNTimeSeriesRegressor(
            device='cpu',
            ignore_pretraining_limits=True
        )
        tabpfn_ts.fit(X_train, y_train)
        models['TabPFN_TimeSeries'] = tabpfn_ts
        
    except Exception as e:
        print(f"TabPFN Time Series not available: {e}")
    
    return models

def evaluate_models(models, X_train, y_train):
    """Evaluate models and return weights"""
    model_scores = {}
    
    # Use TimeSeriesSplit for proper time series validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    for name, model in models.items():
        try:
            scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='r2')
            score = np.mean(scores)
            model_scores[name] = max(0, score)  # Ensure non-negative
            print(f"{name} CV R² score: {score:.4f} (±{np.std(scores):.4f})")
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            model_scores[name] = 0
    
    # Normalize scores to weights
    total_score = sum(model_scores.values())
    if total_score > 0:
        model_weights = {k: v/total_score for k, v in model_scores.items()}
    else:
        model_weights = {k: 1/len(model_scores) for k in model_scores.keys()}
    
    return model_weights

def main():
    """Main execution function"""
    
    print("=== Ultimate Wind Turbine Power Forecasting ===")
    print("Using TabPFN with advanced feature engineering\n")
    
    # Load data
    train_df, loc4_covariates, test_df, sample_submission = load_data()
    
    # Prepare comprehensive training data
    train_featured = prepare_training_data(train_df, loc4_covariates)
    
    # Clean training data (remove rows without Power)
    train_clean = train_featured.dropna(subset=['Power']).copy()
    
    print(f"Training data shape after feature engineering: {train_clean.shape}")
    
    # Define feature columns
    exclude_cols = ['item_id', 'Time', 'Power']
    feature_cols = [col for col in train_clean.columns if col not in exclude_cols]
    
    print(f"Number of features: {len(feature_cols)}")
    
    # Prepare training matrices
    X_train = train_clean[feature_cols].fillna(0)
    y_train = train_clean['Power']
    
    print(f"Training set size: {len(X_train)}")
    print(f"Target statistics: mean={y_train.mean():.4f}, std={y_train.std():.4f}")
    
    # Train TabPFN models
    models = train_tabpfn_models(X_train, y_train, feature_cols)
    
    if not models:
        print("No TabPFN models available, falling back to RandomForest...")
        from sklearn.ensemble import RandomForestRegressor
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        models['RandomForest'] = rf_model
    
    # Evaluate models and get weights
    model_weights = evaluate_models(models, X_train, y_train)
    print(f"\nModel weights: {model_weights}")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = []
    
    for idx, row in test_df.iterrows():
        try:
            # Create prediction features
            pred_features = create_prediction_features(row, train_df, loc4_covariates)
            
            # Ensure all features are present
            for col in feature_cols:
                if col not in pred_features.columns:
                    pred_features[col] = 0
            
            # Get features in same order as training
            X_pred = pred_features[feature_cols].fillna(0)
            
            # Ensemble prediction
            ensemble_pred = 0
            total_weight = 0
            
            for name, model in models.items():
                try:
                    pred = model.predict(X_pred)[0]
                    weight = model_weights.get(name, 0)
                    ensemble_pred += pred * weight
                    total_weight += weight
                except Exception as e:
                    print(f"Prediction error with {name}: {e}")
            
            if total_weight > 0:
                final_pred = ensemble_pred / total_weight
            else:
                # Smart fallback based on seasonal patterns and wind
                anchor_time = row['anchor_time']
                month = anchor_time.month
                wind_speed = row['anchor_windspeed_10m']
                
                # Seasonal baseline
                if month in [11, 12, 1, 2, 3, 4]:  # High power season
                    seasonal_base = 0.35
                else:  # Low power season
                    seasonal_base = 0.20
                
                # Wind adjustment
                wind_factor = min(1.0, max(0.0, (wind_speed - 2) / 20))
                final_pred = seasonal_base * wind_factor
            
            predictions.append(max(0, min(1, final_pred)))
            
        except Exception as e:
            print(f"Prediction error for row {idx}: {e}")
            predictions.append(0.257)  # Overall mean fallback
    
    # Create submission
    submission = sample_submission.copy()
    submission['Power'] = predictions
    
    # Ensure predictions are in [0, 1] range
    submission['Power'] = submission['Power'].clip(0, 1)
    
    # Save submission
    submission.to_csv('submission_ultimate.csv', index=False)
    
    print(f"\n=== Ultimate Predictions Completed! ===")
    print(f"Saved to: submission_ultimate.csv")
    print(f"\nPrediction Statistics:")
    print(f"  Mean: {np.mean(predictions):.4f}")
    print(f"  Std:  {np.std(predictions):.4f}")
    print(f"  Min:  {np.min(predictions):.4f}")
    print(f"  Max:  {np.max(predictions):.4f}")
    print(f"  Unique values: {len(set(predictions))}")
    
    # Show sample predictions with context
    print(f"\nSample Predictions:")
    for i in range(min(10, len(predictions))):
        row = test_df.iloc[i]
        print(f"  {row['row_id']}: {predictions[i]:.4f} "
              f"(wind: {row['anchor_windspeed_10m']:.1f} m/s, "
              f"month: {row['anchor_time'].month}, "
              f"season: {'high' if row['anchor_time'].month in [11,12,1,2,3,4] else 'low'})")
    
    print(f"\n=== Model Performance Summary ===")
    for name, weight in model_weights.items():
        print(f"  {name}: {weight:.3f}")

if __name__ == "__main__":
    main()