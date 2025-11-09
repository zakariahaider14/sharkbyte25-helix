"""
Feature Engineering Script for COVID-19 Prediction Model

This script processes raw COVID-19 data from GCS, performs feature engineering,
and uploads the processed features to GCS and BigQuery for use in model training.

Input: Raw COVID-19 dataset from Kaggle
Output: Processed features in GCS and BigQuery
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime, timedelta
import argparse

import pandas as pd
import numpy as np
from google.cloud import storage, bigquery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CovidFeatureEngineer:
    """Handles feature engineering for COVID-19 data."""
    
    def __init__(self, gcp_project_id: str, dataset_id: str):
        """Initialize the feature engineer with GCP clients."""
        self.gcp_project_id = gcp_project_id
        self.dataset_id = dataset_id
        self.storage_client = storage.Client(project=gcp_project_id)
        self.bq_client = bigquery.Client(project=gcp_project_id)
        
    def load_raw_data_from_gcs(self, bucket_name: str, prefix: str) -> pd.DataFrame:
        """Load raw COVID-19 data from GCS."""
        logger.info(f"Loading raw data from gs://{bucket_name}/{prefix}")
        
        bucket = self.storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        dataframes = []
        for blob in blobs:
            if blob.name.endswith('.csv'):
                logger.info(f"Processing file: {blob.name}")
                data = blob.download_as_bytes()
                df = pd.read_csv(pd.io.common.BytesIO(data))
                dataframes.append(df)
        
        if not dataframes:
            raise ValueError(f"No CSV files found in gs://{bucket_name}/{prefix}")
        
        # Combine all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} rows from GCS")
        return combined_df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the raw data."""
        logger.info("Cleaning data...")
        
        # Handle missing values
        df = df.fillna(0)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-z0-9_]', '', regex=True)
        
        # Convert date columns to datetime
        date_columns = [col for col in df.columns if 'date' in col or 'time' in col]
        for col in date_columns:
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception as e:
                logger.warning(f"Could not convert {col} to datetime: {str(e)}")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        logger.info(f"Data cleaned: {len(df)} rows remaining")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from raw data."""
        logger.info("Engineering features...")
        
        # Ensure required columns exist
        required_cols = ['confirmed', 'deaths', 'recovered', 'country']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Required column '{col}' not found. Using placeholder.")
                df[col] = 0
        
        # Basic epidemiological features
        df['active_cases'] = df['confirmed'] - df['deaths'] - df['recovered']
        df['active_cases'] = df['active_cases'].clip(lower=0)  # Ensure non-negative
        
        # Rates and proportions
        df['case_fatality_rate'] = (df['deaths'] / df['confirmed'].replace(0, 1)).fillna(0)
        df['recovery_rate'] = (df['recovered'] / df['confirmed'].replace(0, 1)).fillna(0)
        
        # Ensure rates are between 0 and 1
        df['case_fatality_rate'] = df['case_fatality_rate'].clip(0, 1)
        df['recovery_rate'] = df['recovery_rate'].clip(0, 1)
        
        # Time-based features (if date column exists)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['days_since_first_case'] = (df['date'] - df['date'].min()).dt.days
            
            # Rolling averages for trend analysis
            df_sorted = df.sort_values('date')
            df['new_cases_7day_avg'] = df_sorted['confirmed'].rolling(window=7, min_periods=1).mean()
            df['new_deaths_7day_avg'] = df_sorted['deaths'].rolling(window=7, min_periods=1).mean()
        else:
            df['days_since_first_case'] = 0
            df['new_cases_7day_avg'] = df['confirmed']
            df['new_deaths_7day_avg'] = df['deaths']
        
        # Per capita metrics (assuming population data exists)
        if 'population' in df.columns:
            df['cases_per_million'] = (df['confirmed'] / (df['population'] / 1_000_000)).fillna(0)
            df['deaths_per_million'] = (df['deaths'] / (df['population'] / 1_000_000)).fillna(0)
        else:
            df['cases_per_million'] = 0
            df['deaths_per_million'] = 0
        
        # Vaccination and testing features (if available)
        if 'vaccination_rate' not in df.columns:
            df['vaccination_rate'] = 0.0
        if 'testing_rate' not in df.columns:
            df['testing_rate'] = 0.0
        
        # Trend direction
        df['trend_direction'] = 'stable'
        if 'new_cases_7day_avg' in df.columns:
            df.loc[df['new_cases_7day_avg'].diff() > 0, 'trend_direction'] = 'up'
            df.loc[df['new_cases_7day_avg'].diff() < 0, 'trend_direction'] = 'down'
        
        # Trend strength (volatility)
        if 'new_cases_7day_avg' in df.columns:
            df['trend_strength'] = df['new_cases_7day_avg'].rolling(window=7, min_periods=1).std().fillna(0)
        else:
            df['trend_strength'] = 0.0
        
        # Volatility
        if 'confirmed' in df.columns:
            df['volatility'] = df['confirmed'].rolling(window=7, min_periods=1).std().fillna(0)
        else:
            df['volatility'] = 0.0
        
        # Doubling time estimation
        df['doubling_time_days'] = np.inf
        if 'new_cases_7day_avg' in df.columns:
            mask = df['new_cases_7day_avg'] > 0
            df.loc[mask, 'doubling_time_days'] = np.log(2) / (
                np.log(df.loc[mask, 'new_cases_7day_avg'] / df.loc[mask, 'new_cases_7day_avg'].shift(1))
            )
            df['doubling_time_days'] = df['doubling_time_days'].replace([np.inf, -np.inf], 999)
        
        logger.info(f"Features engineered. Total features: {len(df.columns)}")
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for the model."""
        logger.info("Selecting relevant features...")
        
        feature_columns = [
            'country',
            'confirmed',
            'deaths',
            'recovered',
            'active_cases',
            'case_fatality_rate',
            'recovery_rate',
            'population',
            'cases_per_million',
            'deaths_per_million',
            'vaccination_rate',
            'testing_rate',
            'new_cases_7day_avg',
            'new_deaths_7day_avg',
            'trend_direction',
            'trend_strength',
            'volatility',
            'doubling_time_days',
            'days_since_first_case'
        ]
        
        # Select only columns that exist
        available_columns = [col for col in feature_columns if col in df.columns]
        df_selected = df[available_columns].copy()
        
        logger.info(f"Selected {len(available_columns)} features")
        return df_selected
    
    def save_to_gcs(self, df: pd.DataFrame, bucket_name: str, file_path: str) -> None:
        """Save processed features to GCS."""
        logger.info(f"Saving features to gs://{bucket_name}/{file_path}")
        
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(file_path)
        
        # Convert to CSV
        csv_data = df.to_csv(index=False)
        blob.upload_from_string(csv_data, content_type='text/csv')
        
        logger.info(f"Features saved to GCS")
    
    def save_to_bigquery(self, df: pd.DataFrame, table_id: str) -> None:
        """Save processed features to BigQuery."""
        logger.info(f"Saving features to BigQuery: {table_id}")
        
        # Add timestamp columns
        df['event_timestamp'] = datetime.utcnow()
        df['created_timestamp'] = datetime.utcnow()
        
        # Load to BigQuery
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            autodetect=True
        )
        
        job = self.bq_client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        job.result()
        
        logger.info(f"Features saved to BigQuery: {table_id}")
    
    def validate_features(self, df: pd.DataFrame) -> bool:
        """Validate the engineered features."""
        logger.info("Validating features...")
        
        # Check for required columns
        required_cols = ['country', 'confirmed', 'deaths', 'recovered']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in features")
        
        # Check data types
        numeric_cols = ['confirmed', 'deaths', 'recovered', 'case_fatality_rate', 'recovery_rate']
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"Column {col} should be numeric")
                return False
        
        logger.info("Feature validation passed")
        return True


def main():
    """Main function to orchestrate feature engineering."""
    parser = argparse.ArgumentParser(
        description='Feature engineering for COVID-19 prediction model'
    )
    parser.add_argument(
        '--gcp-project',
        required=True,
        help='GCP project ID'
    )
    parser.add_argument(
        '--input-bucket',
        required=True,
        help='GCS bucket containing raw data'
    )
    parser.add_argument(
        '--input-prefix',
        default='covid-19',
        help='Prefix in GCS bucket for raw data'
    )
    parser.add_argument(
        '--output-bucket',
        required=True,
        help='GCS bucket for processed features'
    )
    parser.add_argument(
        '--output-prefix',
        default='covid-features',
        help='Prefix in GCS bucket for processed features'
    )
    parser.add_argument(
        '--bq-dataset',
        required=True,
        help='BigQuery dataset ID'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize feature engineer
        engineer = CovidFeatureEngineer(args.gcp_project, args.bq_dataset)
        
        # Load raw data
        raw_df = engineer.load_raw_data_from_gcs(args.input_bucket, args.input_prefix)
        
        # Clean data
        cleaned_df = engineer.clean_data(raw_df)
        
        # Engineer features
        featured_df = engineer.engineer_features(cleaned_df)
        
        # Select relevant features
        selected_df = engineer.select_features(featured_df)
        
        # Validate features
        if not engineer.validate_features(selected_df):
            logger.error("Feature validation failed")
            return
        
        # Save to GCS
        output_file = f"{args.output_prefix}/covid_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        engineer.save_to_gcs(selected_df, args.output_bucket, output_file)
        
        # Save to BigQuery
        table_id = f"{args.gcp_project}.{args.bq_dataset}.covid_features"
        engineer.save_to_bigquery(selected_df, table_id)
        
        logger.info("=" * 80)
        logger.info("Feature engineering completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
