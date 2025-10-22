import pandas as pd
import numpy as np

# Load data
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
loc4_covariates = pd.read_csv('data/loc4_covariates.csv')

# Analyze Location4 specifically
loc4 = train_df[train_df['item_id'] == 'Location4']

print("Location4 Power Statistics:")
print(f"Mean: {loc4['Power'].mean():.4f}")
print(f"Std: {loc4['Power'].std():.4f}")
print(f"Min: {loc4['Power'].min():.4f}")
print(f"Max: {loc4['Power'].max():.4f}")
print(f"Median: {loc4['Power'].median():.4f}")

print("\nWind Speed vs Power Correlation:")
print(f"10m wind: {loc4['windspeed_10m'].corr(loc4['Power']):.4f}")
print(f"100m wind: {loc4['windspeed_100m'].corr(loc4['Power']):.4f}")

print("\nWind Power (cubic) vs Power Correlation:")
loc4_copy = loc4.copy()
loc4_copy['wind_power_10m'] = loc4_copy['windspeed_10m'] ** 3
loc4_copy['wind_power_100m'] = loc4_copy['windspeed_100m'] ** 3
print(f"10m wind^3: {loc4_copy['wind_power_10m'].corr(loc4_copy['Power']):.4f}")
print(f"100m wind^3: {loc4_copy['wind_power_100m'].corr(loc4_copy['Power']):.4f}")

print("\nOther correlations:")
weather_cols = ['temperature_2m', 'relativehumidity_2m', 'dewpoint_2m', 'winddirection_10m', 'windgusts_10m']
for col in weather_cols:
    corr = loc4[col].corr(loc4['Power'])
    print(f"{col}: {corr:.4f}")

print("\nPower distribution by month:")
loc4['Time'] = pd.to_datetime(loc4['Time'])
monthly_power = loc4.groupby(loc4['Time'].dt.month)['Power'].agg(['mean', 'std', 'count'])
print(monthly_power)

print("\nTest data anchor weather vs training weather comparison:")
test_df['anchor_time'] = pd.to_datetime(test_df['anchor_time'])
print(f"Test anchor wind speeds (10m): {test_df['anchor_windspeed_10m'].describe()}")
print(f"Training wind speeds (10m): {loc4['windspeed_10m'].describe()}")