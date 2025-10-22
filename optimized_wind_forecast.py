#!/usr/bin/env python3
"""
Highly Optimized Wind Turbine Power Forecasting
===============================================

Based on data analysis insights:
- Wind speed has 0.86-0.88 correlation with power
- Wind gusts have 0.83 correlation  
- Strong seasonal patterns (winter > summer)
- Focus on the most predictive features
- Use ensemble of best models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
warnings.filterwarnings('ignore')

os.environ['TABPFN_ALLOW_CPU_LARGE_DATASET'] = '1'

def load_data():
    """Load and prepare the dataset"""
    print("Loading data...")
    
    train_df = pd.read_csv('data/train.csv')
    train_df['Time'] = pd.to_datetime(train_df['Time'])
    
    loc4_covariates = pd.read_csv('data/loc4_covariates.csv')
    loc4_covariates['Time'] = pd.to_datetime(loc4_covariates['Time'])
    
    test_df = pd.read_csv('data/test.csv')
    test_df['anchor_time'] = pd.to_datetime(test_df['anchor_time'])
    test_df['Time'] = pd.to_datetime(test_df['Time'])
    
    sample_submission = pd.read_csv('data/sample_submission.csv')
    
    return train_df, loc4_covariates, test_df, sample_submission

def create_power_features(df, prefix=''):
    """Create the most predictive features based on analysis"""
    
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
    
    # Most important: Wind speeds (linear and cubic)
    features[f'{prefix}windspeed_10m_raw'] = df[ws_10m]
    features[f'{prefix}windspeed_100m_raw'] = df[ws_100m]
    features[f'{prefix}wind_power_10m'] = np.power(df[ws_10m], 3)
    features[f'{prefix}wind_power_100m'] = np.power(df[ws_100m], 3)
    
    # Wind gusts (very important - 0.83 correlation)
    features[f'{prefix}windgusts_raw'] = df[wg_10m]
    features[f'{prefix}gust_power'] = np.power(df[wg_10m], 3)
    features[f'{prefix}gust_factor'] = df[wg_10m] / (df[ws_10m] + 0.1)
    
    # Wind shear and consistency
    features[f'{prefix}wind_shear'] = df[ws_100m] - df[ws_10m]
    features[f'{prefix}wind_ratio'] = df[ws_100m] / (df[ws_10m] + 0.1)
    
    # Wind direction (convert to components for ML)
    features[f'{prefix}wind_dir_10m_sin'] = np.sin(np.radians(df[wd_10m]))
    features[f'{prefix}wind_dir_10m_cos'] = np.cos(np.radians(df[wd_10m]))
    features[f'{prefix}wind_dir_100m_sin'] = np.sin(np.radians(df[wd_100m]))
    features[f'{prefix}wind_dir_100m_cos'] = np.cos(np.radians(df[wd_100m]))
    
    # Direction consistency
    wind_dir_diff = np.abs(df[wd_100m] - df[wd_10m])
    # Handle wrap-around (e.g., 350° to 10° is 20°, not 340°)
    wind_dir_diff = np.minimum(wind_dir_diff, 360 - wind_dir_diff)
    features[f'{prefix}wind_dir_consistency'] = 1 / (1 + wind_dir_diff)
    
    # Temperature effects (negative correlation)
    features[f'{prefix}temp_effect'] = -df[temp] / 100.0  # Normalize and invert
    features[f'{prefix}temp_dewpoint_diff'] = df[temp] - df[dewpoint]
    
    # Optimal wind ranges for turbines
    features[f'{prefix}optimal_wind_10m'] = ((df[ws_10m] >= 3) & (df[ws_10m] <= 25)).astype(float)
    features[f'{prefix}optimal_wind_100m'] = ((df[ws_100m] >= 3) & (df[ws_100m] <= 25)).astype(float)
    
    # Combined wind effectiveness
    features[f'{prefix}wind_effectiveness'] = (
        features[f'{prefix}optimal_wind_10m'] * 
        features[f'{prefix}optimal_wind_100m'] * 
        features[f'{prefix}wind_dir_consistency']
    )
    
    return pd.DataFrame(features)

def create_seasonal_features(df, time_col='Time'):
    """Create seasonal features based on strong monthly patterns"""
    features = {}
    
    # Extract time components
    features['month'] = df[time_col].dt.month
    features['day_of_year'] = df[time_col].dt.dayofyear
    features['hour'] = df[time_col].dt.hour
    features['day_of_week'] = df[time_col].dt.dayofweek
    
    # Seasonal encoding (based on power analysis)
    # High power months: 2,3,4,10,11,12 (winter/spring/fall)
    # Low power months: 6,7,8 (summer)
    high_power_months = [2, 3, 4, 10, 11, 12]
    features['high_power_season'] = df[time_col].dt.month.isin(high_power_months).astype(float)
    
    # Cyclical encoding for smooth transitions
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
    features['day_of_year_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365.25)
    features['day_of_year_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365.25)
    
    # Hour effects (wind patterns change during day)
    features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
    features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
    
    return pd.DataFrame(features)

def create_lag_features(df, target_col='Power', group_col='item_id'):
    """Create simple but effective lag features"""
    df = df.copy()
    df = df.sort_values([group_col, 'Time'])
    
    # Most recent values (1 and 7 days ago)
    df[f'{target_col}_lag_1'] = df.groupby(group_col)[target_col].shift(1)
    df[f'{target_col}_lag_7'] = df.groupby(group_col)[target_col].shift(7)
    
    # Short-term trend
    df[f'{target_col}_rolling_mean_3'] = (
        df.groupby(group_col)[target_col]
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )
    
    # Medium-term trend
    df[f'{target_col}_rolling_mean_14'] = (
        df.groupby(group_col)[target_col]
        .rolling(window=14, min_periods=1)
        .mean()
        .reset_index(0, drop=True)
    )
    
    return df

def prepare_training_data(train_df):
    """Prepare optimized training dataset"""
    print("Creating optimized features...")
    
    # Focus on Location4 + some other locations for pattern diversity
    loc4_data = train_df[train_df['item_id'] == 'Location4'].copy()
    
    # Add some other location data for robustness (but prioritize Location4)
    other_data = train_df[train_df['item_id'] != 'Location4'].sample(n=300, random_state=42)
    
    # Combine with more weight on Location4
    combined_data = pd.concat([loc4_data, loc4_data, other_data], ignore_index=True)  # Double Location4
    
    # Create power features
    power_features = create_power_features(combined_data)
    
    # Create seasonal features
    seasonal_features = create_seasonal_features(combined_data)
    
    # Combine features
    feature_df = pd.concat([
        combined_data[['item_id', 'Time', 'Power']],
        power_features,
        seasonal_features
    ], axis=1)
    
    # Add lag features
    feature_df = create_lag_features(feature_df)
    
    return feature_df

def create_prediction_features(test_row, train_df, loc4_covariates):
    """Create prediction features using anchor weather data"""
    
    anchor_time = test_row['anchor_time']
    target_time = test_row['Time']
    
    # Create power features from anchor weather
    anchor_power_features = create_power_features(test_row.to_frame().T, prefix='anchor_')
    
    # Create seasonal features for target time
    target_seasonal_features = create_seasonal_features(pd.DataFrame({'Time': [target_time]}))
    
    # Get historical Location4 data for lag features
    loc4_historical = train_df[
        (train_df['item_id'] == 'Location4') & 
        (train_df['Time'] <= anchor_time)
    ].copy()
    
    # Calculate lag features
    lag_features = {}
    if len(loc4_historical) > 0:
        recent_power = loc4_historical['Power'].values
        if len(recent_power) >= 1:
            lag_features['Power_lag_1'] = recent_power[-1]
        if len(recent_power) >= 7:
            lag_features['Power_lag_7'] = recent_power[-7]
        if len(recent_power) >= 3:
            lag_features['Power_rolling_mean_3'] = np.mean(recent_power[-3:])
        if len(recent_power) >= 14:
            lag_features['Power_rolling_mean_14'] = np.mean(recent_power[-14:])
    
    # Combine all features
    prediction_features = {}
    
    # Add anchor power features (remove prefix)
    for col in anchor_power_features.columns:
        new_col = col.replace('anchor_', '')
        prediction_features[new_col] = anchor_power_features[col].iloc[0]
    
    # Add seasonal features
    for col in target_seasonal_features.columns:
        prediction_features[col] = target_seasonal_features[col].iloc[0]
    
    # Add lag features
    for col, value in lag_features.items():
        prediction_features[col] = value
    
    return pd.DataFrame([prediction_features])

def create_ensemble_models():
    """Create ensemble of best performing models"""
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'Ridge': Ridge(alpha=1.0)
    }
    
    return models

def main():
    """Main execution function"""
    
    # Load data
    train_df, loc4_covariates, test_df, sample_submission = load_data()
    
    # Prepare training data
    train_featured = prepare_training_data(train_df)
    
    # Remove rows with NaN in Power
    train_clean = train_featured.dropna(subset=['Power']).copy()
    
    print(f"Training data shape: {train_clean.shape}")
    
    # Define feature columns
    exclude_cols = ['item_id', 'Time', 'Power']
    feature_cols = [col for col in train_clean.columns if col not in exclude_cols]
    
    print(f"Number of features: {len(feature_cols)}")
    
    # Prepare training data
    X_train = train_clean[feature_cols].fillna(0)
    y_train = train_clean['Power']
    
    print("Training ensemble models...")
    
    # Create and train models
    models = create_ensemble_models()
    trained_models = {}
    model_weights = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        try:
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            score = np.mean(cv_scores)
            
            print(f"{name} CV R² score: {score:.4f} (±{np.std(cv_scores):.4f})")
            
            trained_models[name] = model
            model_weights[name] = max(0, score)  # Use R² as weight
            
        except Exception as e:
            print(f"Error training {name}: {e}")
    
    # Normalize weights
    total_weight = sum(model_weights.values())
    if total_weight > 0:
        model_weights = {k: v/total_weight for k, v in model_weights.items()}
    
    print(f"\nModel weights: {model_weights}")
    
    # Try TabPFN as well
    try:
        from tabpfn import TabPFNRegressor
        
        # Use only Location4 data for TabPFN
        loc4_train = train_clean[train_clean['item_id'] == 'Location4'].copy()
        X_train_loc4 = loc4_train[feature_cols].fillna(0)
        y_train_loc4 = loc4_train['Power']
        
        if len(X_train_loc4) > 50:  # Ensure we have enough data
            print(f"Training TabPFN on {len(X_train_loc4)} Location4 samples...")
            
            tabpfn_model = TabPFNRegressor(device='cpu', ignore_pretraining_limits=True)
            tabpfn_model.fit(X_train_loc4, y_train_loc4)
            
            # Simple validation
            score = tabpfn_model.score(X_train_loc4, y_train_loc4)
            print(f"TabPFN R² score: {score:.4f}")
            
            if score > 0.5:  # Only use if reasonable performance
                trained_models['TabPFN'] = tabpfn_model
                model_weights['TabPFN'] = score * 0.5  # Give TabPFN moderate weight
                
                # Renormalize weights
                total_weight = sum(model_weights.values())
                model_weights = {k: v/total_weight for k, v in model_weights.items()}
                print(f"Updated model weights: {model_weights}")
            
    except Exception as e:
        print(f"TabPFN not available: {e}")
    
    # Make predictions
    print("Making ensemble predictions...")
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
            
            for name, model in trained_models.items():
                try:
                    if name == 'TabPFN':
                        # Use Location4 features for TabPFN
                        pred = model.predict(X_pred)[0]
                    else:
                        pred = model.predict(X_pred)[0]
                    
                    weight = model_weights.get(name, 0)
                    ensemble_pred += pred * weight
                    total_weight += weight
                    
                except Exception as e:
                    print(f"Prediction error with {name}: {e}")
            
            if total_weight > 0:
                final_pred = ensemble_pred / total_weight
            else:
                # Fallback to seasonal mean
                anchor_time = row['anchor_time']
                month = anchor_time.month
                loc4_monthly = train_df[
                    (train_df['item_id'] == 'Location4') & 
                    (train_df['Time'].dt.month == month)
                ]['Power']
                final_pred = loc4_monthly.mean() if len(loc4_monthly) > 0 else 0.257
            
            predictions.append(max(0, min(1, final_pred)))
            
        except Exception as e:
            print(f"Prediction error for row {idx}: {e}")
            # Fallback
            predictions.append(0.257)
    
    # Create submission
    submission = sample_submission.copy()
    submission['Power'] = predictions
    
    # Ensure predictions are in [0, 1] range
    submission['Power'] = submission['Power'].clip(0, 1)
    
    # Save submission
    submission.to_csv('submission_optimized.csv', index=False)
    
    print(f"\nEnsemble predictions completed! Saved to submission_optimized.csv")
    print(f"Prediction statistics:")
    print(f"  Mean: {np.mean(predictions):.4f}")
    print(f"  Std:  {np.std(predictions):.4f}")
    print(f"  Min:  {np.min(predictions):.4f}")
    print(f"  Max:  {np.max(predictions):.4f}")
    print(f"  Unique values: {len(set(predictions))}")
    
    # Show sample predictions with anchor weather
    print(f"\nSample predictions with anchor weather:")
    for i in range(min(5, len(predictions))):
        row = test_df.iloc[i]
        print(f"  {row['row_id']}: {predictions[i]:.4f} (wind: {row['anchor_windspeed_10m']:.1f} m/s, month: {row['anchor_time'].month})")

if __name__ == "__main__":
    main()