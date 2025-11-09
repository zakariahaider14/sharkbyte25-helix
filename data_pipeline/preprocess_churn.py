#!/usr/bin/env python3
"""
Churn Data Preprocessing
Processes Telco Customer Churn dataset and splits into train/test
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Paths
DATA_DIR = Path(__file__).parent / "processed_data"
DATA_DIR.mkdir(exist_ok=True)

def load_churn_data(kaggle_path: str):
    """Load churn dataset from Kaggle download path"""
    # Kaggle downloads to a versioned directory, find the CSV
    path = Path(kaggle_path)
    csv_files = list(path.glob("**/*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {kaggle_path}")
    
    # Usually the main file is WA_Fn-UseC_-Telco-Customer-Churn.csv
    churn_file = [f for f in csv_files if 'Telco' in f.name or 'churn' in f.name.lower()]
    
    if churn_file:
        df = pd.read_csv(churn_file[0])
    else:
        df = pd.read_csv(csv_files[0])
    
    print(f"✅ Loaded churn data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def preprocess_churn_data(df: pd.DataFrame):
    """Preprocess churn dataset"""
    print("🔧 Preprocessing churn data...")
    
    # Make a copy
    df = df.copy()
    
    # Handle missing values in TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Create binary encoding for Yes/No columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Handle MultipleLines, OnlineSecurity, etc.
    yes_no_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0})
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['gender', 'Contract', 'PaymentMethod', 'InternetService']
    for col in categorical_cols:
        if col in df.columns:
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    
    # Create feature columns
    feature_cols = [
        'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
        'MonthlyCharges', 'TotalCharges', 'gender_encoded', 'Contract_encoded',
        'PaymentMethod_encoded', 'InternetService_encoded'
    ]
    
    # Filter to only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    # Add customer_id if exists
    if 'customerID' in df.columns:
        df['customer_id'] = df['customerID']
    
    print(f"✅ Preprocessed data: {len(feature_cols)} features")
    return df, feature_cols

def split_and_save(df: pd.DataFrame, feature_cols: list):
    """Split data into train/test and save"""
    print("✂️  Splitting data into train/test...")
    
    # Prepare features and target
    X = df[feature_cols]
    y = df['Churn'] if 'Churn' in df.columns else None
    
    if y is None:
        raise ValueError("Churn column not found in dataset")
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Add target back
    train_df = X_train.copy()
    train_df['churn'] = y_train.values
    
    test_df = X_test.copy()
    test_df['churn'] = y_test.values
    
    # Add customer_id if available
    if 'customer_id' in df.columns:
        train_df['customer_id'] = df.loc[X_train.index, 'customer_id'].values
        test_df['customer_id'] = df.loc[X_test.index, 'customer_id'].values
    else:
        # Generate customer IDs
        train_df['customer_id'] = [f'CUST_{i:06d}' for i in range(len(train_df))]
        test_df['customer_id'] = [f'CUST_{i:06d}' for i in range(len(train_df), len(train_df) + len(test_df))]
    
    # Save to CSV
    train_path = DATA_DIR / "churn_train.csv"
    test_path = DATA_DIR / "churn_test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"✅ Train set saved: {train_path} ({len(train_df)} rows)")
    print(f"✅ Test set saved: {test_path} ({len(test_df)} rows)")
    
    return train_df, test_df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocess_churn.py <kaggle_download_path>")
        print("Example: python preprocess_churn.py ~/.cache/kagglehub/datasets/blastchar/telco-customer-churn/...")
        sys.exit(1)
    
    kaggle_path = sys.argv[1]
    
    print("=" * 60)
    print("CHURN DATA PREPROCESSING")
    print("=" * 60)
    
    # Load data
    df = load_churn_data(kaggle_path)
    
    # Preprocess
    df_processed, feature_cols = preprocess_churn_data(df)
    
    # Split and save
    train_df, test_df = split_and_save(df_processed, feature_cols)
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Features: {len(feature_cols)}")
    print(f"\nChurn distribution (train): {train_df['churn'].value_counts().to_dict()}")
