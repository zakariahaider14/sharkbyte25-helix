#!/usr/bin/env python3
"""
COVID-19 Data Preprocessing
Processes COVID-19 dataset and splits into train/test
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime

# Paths
DATA_DIR = Path(__file__).parent / "processed_data"
DATA_DIR.mkdir(exist_ok=True)

def load_covid_data(kaggle_path: str):
    """Load COVID dataset from Kaggle download path"""
    path = Path(kaggle_path)
    csv_files = list(path.glob("**/*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {kaggle_path}")
    
    # Find the main COVID data file (usually country_wise_latest.csv or full_grouped.csv)
    covid_file = [f for f in csv_files if 'country' in f.name.lower() or 'full' in f.name.lower()]
    
    if covid_file:
        df = pd.read_csv(covid_file[0])
    else:
        df = pd.read_csv(csv_files[0])
    
    print(f"✅ Loaded COVID data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_covid_data(df: pd.DataFrame):
    """Preprocess COVID dataset"""
    print("🔧 Preprocessing COVID data...")
    
    # Make a copy
    df = df.copy()
    
    # Standardize column names
    column_mapping = {
        'Country/Region': 'country_name',
        'Country': 'country_name',
        'Confirmed': 'confirmed_cases',
        'Deaths': 'deaths',
        'Recovered': 'recovered',
        'Active': 'active_cases',
        'New cases': 'new_cases',
        'New deaths': 'new_deaths',
        'New recovered': 'new_recovered',
        'Deaths / 100 Cases': 'death_rate',
        'Recovered / 100 Cases': 'recovery_rate',
        'Deaths / 100 Recovered': 'death_recovery_ratio',
        'WHO Region': 'who_region'
    }
    
    df.rename(columns=column_mapping, inplace=True)
    
    # Ensure required columns exist
    required_cols = ['country_name', 'confirmed_cases', 'deaths']
    for col in required_cols:
        if col not in df.columns:
            print(f"⚠️  Warning: {col} not found in dataset")
    
    # Handle missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col].fillna(0, inplace=True)
    
    # Calculate additional features
    if 'confirmed_cases' in df.columns and 'deaths' in df.columns:
        df['death_rate'] = np.where(
            df['confirmed_cases'] > 0,
            (df['deaths'] / df['confirmed_cases']) * 100,
            0
        )
    
    if 'confirmed_cases' in df.columns and 'recovered' in df.columns:
        df['recovery_rate'] = np.where(
            df['confirmed_cases'] > 0,
            (df['recovered'] / df['confirmed_cases']) * 100,
            0
        )
    
    if 'confirmed_cases' in df.columns and 'active_cases' not in df.columns:
        if 'recovered' in df.columns and 'deaths' in df.columns:
            df['active_cases'] = df['confirmed_cases'] - df['recovered'] - df['deaths']
    
    # Create risk level based on death rate and active cases
    if 'death_rate' in df.columns and 'active_cases' in df.columns:
        df['risk_score'] = (
            (df['death_rate'] / 10) * 0.5 +  # Death rate contribution
            (df['active_cases'] / df['active_cases'].max()) * 0.5  # Active cases contribution
        )
        df['risk_level'] = pd.cut(
            df['risk_score'],
            bins=[0, 0.3, 0.6, 1.0],
            labels=['LOW', 'MEDIUM', 'HIGH']
        )
    
    # Add timestamp
    df['timestamp'] = datetime.now().isoformat()
    
    # Define feature columns
    feature_cols = [
        'confirmed_cases', 'deaths', 'recovered', 'active_cases',
        'death_rate', 'recovery_rate', 'risk_score'
    ]
    
    # Filter to only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    print(f"✅ Preprocessed data: {len(feature_cols)} features")
    return df, feature_cols

def split_and_save(df, feature_cols):
    """Split data into train/test and save"""
    print("✂️  Splitting data into train/test...")
    
    # Prepare features and target
    X = df[['country_name'] + feature_cols]
    y = df['risk_level'] if 'risk_level' in df.columns else None
    
    if y is None:
        # If no risk_level, create a simple target based on death_rate
        if 'death_rate' in df.columns:
            y = pd.cut(df['death_rate'], bins=[0, 2, 5, 100], labels=['LOW', 'MEDIUM', 'HIGH'])
        else:
            raise ValueError("Cannot create target variable")
    
    # Remove rows with NaN in target variable
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    print(f"📊 Valid samples after removing NaN: {len(X)} rows")
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Add target back
    train_df = X_train.copy()
    train_df['risk_level'] = y_train.values
    
    test_df = X_test.copy()
    test_df['risk_level'] = y_test.values
    
    # Add timestamp
    train_df['timestamp'] = datetime.now().isoformat()
    test_df['timestamp'] = datetime.now().isoformat()
    
    # Save to CSV
    train_path = DATA_DIR / "covid_train.csv"
    test_path = DATA_DIR / "covid_test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ Train set saved: {train_path} ({len(train_df)} rows)")
    print(f"✅ Test set saved: {test_path} ({len(test_df)} rows)")
    
    return train_df, test_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocess_covid.py <kaggle_download_path>")
        print("Example: python preprocess_covid.py ~/.cache/kagglehub/datasets/imdevskp/corona-virus-report/...")
        sys.exit(1)
    
    kaggle_path = sys.argv[1]
    
    print("=" * 60)
    print("COVID-19 DATA PREPROCESSING")
    print("=" * 60)
    
    # Load data
    df = load_covid_data(kaggle_path)
    
    # Preprocess
    df_processed, feature_cols = preprocess_covid_data(df)
    
    # Split and save
    train_df, test_df = split_and_save(df_processed, feature_cols)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"\nRisk distribution (train): {train_df['risk_level'].value_counts().to_dict()}")
