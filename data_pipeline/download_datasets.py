#!/usr/bin/env python3
"""
Dataset Download Script
Downloads COVID-19 and Telco Churn datasets from Kaggle
"""

import os
import kagglehub
from pathlib import Path

# Create data directory
DATA_DIR = Path(__file__).parent / "raw_data"
DATA_DIR.mkdir(exist_ok=True)

def download_churn_dataset():
    """Download Telco Customer Churn dataset"""
    print("📥 Downloading Telco Customer Churn dataset...")
    try:
        path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        print(f"✅ Churn dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading churn dataset: {e}")
        return None

def download_covid_dataset():
    """Download COVID-19 dataset"""
    print("📥 Downloading COVID-19 dataset...")
    try:
        path = kagglehub.dataset_download("imdevskp/corona-virus-report")
        print(f"✅ COVID dataset downloaded to: {path}")
        return path
    except Exception as e:
        print(f"❌ Error downloading COVID dataset: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("KAGGLE DATASET DOWNLOADER")
    print("=" * 60)
    
    # Download datasets
    churn_path = download_churn_dataset()
    covid_path = download_covid_dataset()
    
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Churn Dataset: {churn_path if churn_path else 'FAILED'}")
    print(f"COVID Dataset: {covid_path if covid_path else 'FAILED'}")
    print("\nNext steps:")
    print("1. Run preprocess_churn.py to process churn data")
    print("2. Run preprocess_covid.py to process COVID data")
    print("3. Run load_to_mysql.py to load data into database")


# Churn Dataset: /Users/mohammadzakariahaider/.cache/kagglehub/datasets/blastchar/telco-customer-churn/versions/1
# COVID Dataset: /Users/mohammadzakariahaider/.cache/kagglehub/datasets/imdevskp/corona-virus-report/versions/166