#!/usr/bin/env python3
"""
Kaggle Data Downloader for Wind Turbine Energy Prediction Competition
====================================================================

This script downloads all data files from the Kaggle competition:
https://www.kaggle.com/competitions/wind-turbine-energy-prediction/data

Requirements:
- kaggle package: pip install kaggle
- kaggle.json file with API credentials in same directory as this script

Usage:
  cd data/
  python download_data.py
"""

import os
import json
import shutil
import zipfile
from pathlib import Path
import subprocess
import sys

def setup_kaggle_credentials():
    """Setup Kaggle API credentials"""
    
    # Check if kaggle.json exists in current directory (since we're in data/)
    kaggle_json_path = Path("kaggle.json")
    if not kaggle_json_path.exists():
        print("❌ kaggle.json not found in current directory")
        print("Please download your kaggle.json from https://www.kaggle.com/settings")
        return False
    
    # Read credentials
    try:
        with open(kaggle_json_path, 'r') as f:
            credentials = json.load(f)
        
        username = credentials.get('username')
        key = credentials.get('key')
        
        if not username or not key:
            print("❌ Invalid kaggle.json format. Should contain 'username' and 'key'")
            return False
            
        print(f"✅ Found Kaggle credentials for user: {username}")
        
    except Exception as e:
        print(f"❌ Error reading kaggle.json: {e}")
        return False
    
    # Setup Kaggle directory and credentials
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    
    # Copy kaggle.json to ~/.kaggle/
    kaggle_config_path = kaggle_dir / "kaggle.json"
    shutil.copy2(kaggle_json_path, kaggle_config_path)
    
    # Set proper permissions (required by Kaggle API)
    if os.name != 'nt':  # Not Windows
        os.chmod(kaggle_config_path, 0o600)
    
    print(f"✅ Kaggle credentials setup at: {kaggle_config_path}")
    return True

def download_competition_data():
    """Download all data from the competition"""
    
    competition_name = "wind-turbine-energy-prediction"
    download_path = Path(".")  # Current directory (data/)
    
    print(f"📥 Downloading data from competition: {competition_name}")
    print(f"📁 Download location: {download_path.absolute()}")
    
    try:
        # Import kaggle after installation
        import kaggle
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        # Initialize API
        api = KaggleApi()
        api.authenticate()
        
        print("✅ Kaggle API authenticated successfully")
        
        # Create download directory
        download_path.mkdir(exist_ok=True)
        
        # Download all competition files
        print("📥 Downloading competition files...")
        api.competition_download_files(
            competition=competition_name,
            path=str(download_path),
            quiet=False
        )
        
        # Check for zip file and extract
        zip_file = download_path / f"{competition_name}.zip"
        if zip_file.exists():
            print(f"📦 Extracting {zip_file.name}...")
            
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(download_path)
            
            # Remove zip file after extraction
            zip_file.unlink()
            print("✅ Zip file extracted and removed")
        
        # List downloaded files
        print("\n📋 Downloaded files:")
        for file_path in sorted(download_path.glob("*.csv")):
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"  📄 {file_path.name} ({file_size:.1f} KB)")
        
        # Check for other files
        for file_path in sorted(download_path.glob("*")):
            if file_path.suffix not in ['.csv', '.json']:
                file_size = file_path.stat().st_size / 1024  # KB
                print(f"  📄 {file_path.name} ({file_size:.1f} KB)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return False

def verify_data_files():
    """Verify that all expected data files are present"""
    
    expected_files = [
        "train.csv",
        "test.csv", 
        "sample_submission.csv",
        "loc4_covariates.csv",
        "metaData.csv"
    ]
    
    data_path = Path(".")  # Current directory (data/)
    missing_files = []
    present_files = []
    
    print("\n🔍 Verifying data files...")
    
    for filename in expected_files:
        file_path = data_path / filename
        if file_path.exists():
            file_size = file_path.stat().st_size / 1024  # KB
            present_files.append(f"✅ {filename} ({file_size:.1f} KB)")
        else:
            missing_files.append(f"❌ {filename}")
    
    # Print results
    for file_info in present_files:
        print(f"  {file_info}")
    
    for file_info in missing_files:
        print(f"  {file_info}")
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} files are missing")
        return False
    else:
        print(f"\n✅ All {len(expected_files)} expected files are present!")
        return True

def show_data_info():
    """Show basic information about the downloaded data"""
    
    print("\n📊 Data Information:")
    
    data_path = Path(".")  # Current directory (data/)
    
    try:
        import pandas as pd
        
        # Train data info
        train_path = data_path / "train.csv"
        if train_path.exists():
            train_df = pd.read_csv(train_path)
            print(f"  📈 train.csv: {train_df.shape[0]:,} rows × {train_df.shape[1]} columns")
            print(f"      Locations: {train_df['item_id'].nunique()}")
            print(f"      Date range: {train_df['Time'].min()} to {train_df['Time'].max()}")
        
        # Test data info  
        test_path = data_path / "test.csv"
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            print(f"  📊 test.csv: {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")
            print(f"      Anchor times: {test_df['anchor_time'].min()} to {test_df['anchor_time'].max()}")
        
        # Covariates info
        cov_path = data_path / "loc4_covariates.csv"
        if cov_path.exists():
            cov_df = pd.read_csv(cov_path)
            print(f"  🌤️  loc4_covariates.csv: {cov_df.shape[0]:,} rows × {cov_df.shape[1]} columns")
            print(f"      Date range: {cov_df['Time'].min()} to {cov_df['Time'].max()}")
            
    except ImportError:
        print("  ℹ️  Install pandas to see detailed data information: pip install pandas")
    except Exception as e:
        print(f"  ⚠️  Error reading data files: {e}")

def main():
    """Main function to download and setup Kaggle competition data"""
    
    print("🌪️  Wind Turbine Energy Prediction - Data Downloader")
    print("=" * 55)
    
    # Step 1: Setup credentials
    if not setup_kaggle_credentials():
        return False
    
    # Step 2: Download data
    if not download_competition_data():
        return False
    
    # Step 3: Verify files
    if not verify_data_files():
        return False
    
    # Step 4: Show data info
    show_data_info()
    
    print("\n🎉 Data download completed successfully!")
    print("\n📝 Next steps:")
    print("  1. Run your analysis scripts")
    print("  2. Train your models")
    print("  3. Generate predictions")
    print("  4. Submit to Kaggle!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Data download failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\n✅ All done! Ready to start forecasting! 🚀")