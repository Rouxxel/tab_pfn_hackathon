#!/usr/bin/env python3
"""
Wind Turbine Power Forecasting with TabPFN
==========================================

This script solves the wind turbine power forecasting challenge using TabPFN
for time series prediction. The task is to predict normalized power output 
24 hours ahead for Location4 using historical meteorology and turbine data.

Challenge Requirements:
- Use TabPFN for forecasting (via tabpfn-time-series package)
- Predict Power at t+24h for Location4
- Feature engineering is encouraged
- Only use data up to anchor_time (no future leakage)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load and prepare the dataset"""
    print("Loading data...")
    
    # Load training data (all locations, 2020)
    train_df = pd.read_csv('data/train.csv')
    train_df['Time'] = pd.to_datetime(train_df['Time'])
    
    # Load Location4 covariates for 2021 (evaluation year)
    loc4_covariates = pd.read_csv('data/loc4_covariates.csv')
    loc4_covariates['Time'] = pd.to_datetime(loc4_covariates['Time'])
    
    # Load test anchors
    test_df = pd.read_csv('data/test.csv')
    test_df['anchor_time'] = pd.to_datetime(test_df['anchor_time'])
    test_df['Time'] = pd.to_datetime(test_df['Time'])
    
    # Load sample submission
    sample_submission = pd.read_csv('data/sample_submission.csv')
    
    print(f"Training data shape: {train_df.shape}")
    print(f"Location4 covariates shape: {loc4_covariates.shape}")
    print(f"Test anchors: {len(test_df)}")
    
    return train_df, loc4_covariates, test_df, sample_submission

def create_calendar_features(df, time_col='Time'):
    """Create calendar-based features"""
    df = df.copy()
    
    # Extract time components
    df['hour'] = df[time_col].dt.hour
    df['day_of_week'] = df[time_col].dt.dayofweek
    df['day_of_year'] = df[time_col].dt.dayofyear
    df['month'] = df[time_col].dt.month
    df['quarter'] = df[time_col].dt.quarter
    
    # Cyclical encoding for periodic features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    
    return df

def create_lag_features(df, target_col='Power', lag_periods=[1, 2, 3, 7, 14]):
    """Create lag features for time series"""
    df = df.copy()
    df = df.sort_values(['item_id', 'Time'])
    
    for lag in lag_periods:
        df[f'{target_col}_lag_{lag}'] = df.groupby('item_id')[target_col].shift(lag)
    
    return df

def create_rolling_features(df, target_col='Power', windows=[3, 7, 14]):
    """Create rolling statistics features"""
    df = df.copy()
    df = df.sort_values(['item_id', 'Time'])
    
    for window in windows:
        # Rolling mean
        df[f'{target_col}_rolling_mean_{window}'] = (
            df.groupby('item_id')[target_col]
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        # Rolling std
        df[f'{target_col}_rolling_std_{window}'] = (
            df.groupby('item_id')[target_col]
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(0, drop=True)
        )
    
    return df

def create_weather_features(df):
    """Create weather-derived features"""
    df = df.copy()
    
    # Wind power approximation (cubic relationship)
    df['wind_power_10m'] = df['windspeed_10m'] ** 3
    df['wind_power_100m'] = df['windspeed_100m'] ** 3
    
    # Wind shear (difference between heights)
    df['wind_shear'] = df['windspeed_100m'] - df['windspeed_10m']
    
    # Temperature-humidity interaction
    df['temp_humidity_interaction'] = df['temperature_2m'] * df['relativehumidity_2m']
    
    # Wind direction differences
    df['wind_direction_diff'] = np.abs(df['winddirection_100m'] - df['winddirection_10m'])
    
    return df

def prepare_training_data(train_df):
    """Prepare training data with all features"""
    print("Creating features for training data...")
    
    # Add calendar features
    train_df = create_calendar_features(train_df)
    
    # Add weather features
    train_df = create_weather_features(train_df)
    
    # Add lag features (for power output)
    train_df = create_lag_features(train_df)
    
    # Add rolling features
    train_df = create_rolling_features(train_df)
    
    # Sort by location and time
    train_df = train_df.sort_values(['item_id', 'Time'])
    
    return train_df

def prepare_prediction_features(anchor_time, train_df, loc4_covariates):
    """
    Prepare features for a single prediction at anchor_time + 24h
    
    This function ensures no data leakage by only using information
    available up to and including the anchor_time.
    """
    
    # Get historical data up to anchor_time (no leakage)
    historical_data = train_df[
        (train_df['item_id'] == 'Location4') & 
        (train_df['Time'] <= anchor_time)
    ].copy()
    
    # Get covariates up to anchor_time from loc4_covariates
    anchor_covariates = loc4_covariates[
        loc4_covariates['Time'] <= anchor_time
    ].copy()
    
    # Combine historical training data with evaluation year covariates
    if len(anchor_covariates) > 0:
        # Add missing Power column to covariates (we don't have it for eval year)
        anchor_covariates['Power'] = np.nan
        
        # Combine datasets
        combined_data = pd.concat([historical_data, anchor_covariates], ignore_index=True)
    else:
        combined_data = historical_data
    
    # Sort by time
    combined_data = combined_data.sort_values('Time')
    
    # Create features
    combined_data = create_calendar_features(combined_data)
    combined_data = create_weather_features(combined_data)
    combined_data = create_lag_features(combined_data)
    combined_data = create_rolling_features(combined_data)
    
    # Get the most recent row (at anchor_time) for prediction
    if len(combined_data) > 0:
        latest_row = combined_data.iloc[-1:].copy()
        
        # Add target time features (24 hours ahead)
        target_time = anchor_time + timedelta(hours=24)
        latest_row['target_hour'] = target_time.hour
        latest_row['target_day_of_week'] = target_time.weekday()
        latest_row['target_day_of_year'] = target_time.timetuple().tm_yday
        latest_row['target_month'] = target_time.month
        
        # Cyclical encoding for target time
        latest_row['target_hour_sin'] = np.sin(2 * np.pi * latest_row['target_hour'] / 24)
        latest_row['target_hour_cos'] = np.cos(2 * np.pi * latest_row['target_hour'] / 24)
        latest_row['target_day_of_week_sin'] = np.sin(2 * np.pi * latest_row['target_day_of_week'] / 7)
        latest_row['target_day_of_week_cos'] = np.cos(2 * np.pi * latest_row['target_day_of_week'] / 7)
        
        return latest_row
    else:
        return None

def main():
    """Main execution function"""
    
    # Load data
    train_df, loc4_covariates, test_df, sample_submission = load_data()
    
    # Prepare training data with features
    train_featured = prepare_training_data(train_df)
    
    # Remove rows with NaN in Power (target variable) for training
    train_clean = train_featured.dropna(subset=['Power']).copy()
    
    print(f"Training data after feature engineering: {train_clean.shape}")
    
    # Define feature columns (exclude metadata and target)
    exclude_cols = ['item_id', 'Time', 'Power']
    feature_cols = [col for col in train_clean.columns if col not in exclude_cols]
    
    print(f"Number of features: {len(feature_cols)}")
    
    # Prepare training data
    X_train = train_clean[feature_cols].fillna(0)  # Fill NaN with 0
    y_train = train_clean['Power']
    
    print("Training TabPFN model...")
    
    # Try to use TabPFN for regression
    try:
        from tabpfn import TabPFNRegressor
        
        # Initialize TabPFN regressor
        model = TabPFNRegressor(device='cpu')  # Use CPU for compatibility
        
        # Fit the model
        model.fit(X_train, y_train)
        
        print("TabPFN model trained successfully!")
        
    except Exception as e:
        print(f"Error with TabPFN: {e}")
        print("Falling back to simple baseline...")
        
        # Simple baseline: use mean power for Location4
        loc4_train = train_clean[train_clean['item_id'] == 'Location4']
        baseline_prediction = loc4_train['Power'].mean()
        
        # Create predictions
        predictions = []
        for _, row in test_df.iterrows():
            predictions.append(baseline_prediction)
        
        # Create submission
        submission = sample_submission.copy()
        submission['Power'] = predictions
        
        # Clip to [0, 1] as mentioned in challenge
        submission['Power'] = submission['Power'].clip(0, 1)
        
        # Save submission
        submission.to_csv('submission.csv', index=False)
        print("Baseline submission saved to submission.csv")
        return
    
    # Make predictions for test set
    print("Making predictions...")
    predictions = []
    
    for idx, row in test_df.iterrows():
        anchor_time = row['anchor_time']
        
        # Prepare features for this anchor time
        pred_features = prepare_prediction_features(anchor_time, train_featured, loc4_covariates)
        
        if pred_features is not None:
            # Get features in same order as training
            X_pred = pred_features[feature_cols].fillna(0)
            
            # Make prediction
            try:
                pred = model.predict(X_pred)[0]
                predictions.append(pred)
            except Exception as e:
                print(f"Prediction error for {anchor_time}: {e}")
                # Use Location4 mean as fallback
                loc4_mean = train_clean[train_clean['item_id'] == 'Location4']['Power'].mean()
                predictions.append(loc4_mean)
        else:
            # Use Location4 mean as fallback
            loc4_mean = train_clean[train_clean['item_id'] == 'Location4']['Power'].mean()
            predictions.append(loc4_mean)
    
    # Create submission
    submission = sample_submission.copy()
    submission['Power'] = predictions
    
    # Clip predictions to [0, 1] range as mentioned in challenge
    submission['Power'] = submission['Power'].clip(0, 1)
    
    # Save submission
    submission.to_csv('submission.csv', index=False)
    
    print(f"Predictions completed! Saved to submission.csv")
    print(f"Prediction statistics:")
    print(f"  Mean: {np.mean(predictions):.4f}")
    print(f"  Std:  {np.std(predictions):.4f}")
    print(f"  Min:  {np.min(predictions):.4f}")
    print(f"  Max:  {np.max(predictions):.4f}")

if __name__ == "__main__":
    main()