"""
Feature Engineering Script for Telco Customer Churn Prediction Model

This script processes raw Telco customer data from GCS, performs feature engineering,
and uploads the processed features to GCS and BigQuery for use in model training.

Input: Raw Telco customer churn dataset from Kaggle
Output: Processed features in GCS and BigQuery
"""

import os
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime
import argparse

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from google.cloud import storage, bigquery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ChurnFeatureEngineer:
    """Handles feature engineering for Telco customer churn data."""
    
    def __init__(self, gcp_project_id: str, dataset_id: str):
        """Initialize the feature engineer with GCP clients."""
        self.gcp_project_id = gcp_project_id
        self.dataset_id = dataset_id
        self.storage_client = storage.Client(project=gcp_project_id)
        self.bq_client = bigquery.Client(project=gcp_project_id)
        self.label_encoders = {}
        
    def load_raw_data_from_gcs(self, bucket_name: str, prefix: str) -> pd.DataFrame:
        """Load raw Telco customer data from GCS."""
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
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle special characters in numeric columns
        numeric_cols = ['monthlycharges', 'totalcharges']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        logger.info(f"Data cleaned: {len(df)} rows remaining")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from raw data."""
        logger.info("Engineering features...")
        
        # Ensure required columns exist
        required_cols = ['customerid', 'monthlycharges', 'totalcharges', 'tenure']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Required column '{col}' not found. Using placeholder.")
                df[col] = 0
        
        # Tenure-based features
        if 'tenure' in df.columns:
            df['tenure_months'] = df['tenure']
            df['tenure_category'] = pd.cut(df['tenure_months'], 
                                          bins=[0, 12, 24, 48, 100], 
                                          labels=['0-1yr', '1-2yr', '2-4yr', '4+yr'])
        
        # Billing features
        if 'monthlycharges' in df.columns and 'totalcharges' in df.columns:
            # Average monthly charges over tenure
            df['avg_monthly_charges_6m'] = df['monthlycharges']
            
            # Charges increase rate
            df['charges_increase_rate'] = np.where(
                df['tenure_months'] > 0,
                (df['totalcharges'] / df['tenure_months'] - df['monthlycharges']) / df['monthlycharges'],
                0
            )
            df['charges_increase_rate'] = df['charges_increase_rate'].clip(-1, 1)
        
        # Service adoption features
        service_cols = ['phoneservice', 'internetservice', 'onlinesecurity', 
                       'onlinebackup', 'deviceprotection', 'techsupport', 
                       'streamingtv', 'streamingmovies']
        
        # Count of services
        service_count = 0
        for col in service_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower().isin(['yes', 'true', '1']).astype(int)
                service_count += 1
        
        if service_count > 0:
            df['service_count'] = df[[col for col in service_cols if col in df.columns]].sum(axis=1)
        else:
            df['service_count'] = 0
        
        # Contract features
        if 'contract' in df.columns:
            df['contract_type'] = df['contract'].str.lower()
            # Contract flexibility score (month-to-month is more flexible)
            df['contract_flexibility'] = df['contract_type'].map({
                'month-to-month': 1.0,
                'one year': 0.5,
                'two year': 0.0
            }).fillna(0.5)
        else:
            df['contract_flexibility'] = 0.5
        
        # Payment method features
        if 'paymentmethod' in df.columns:
            df['payment_method'] = df['paymentmethod'].str.lower()
            # Automatic payment indicator
            df['automatic_payment'] = df['payment_method'].isin(['bank transfer', 'credit card']).astype(int)
        else:
            df['automatic_payment'] = 0
        
        # Internet service features
        if 'internetservice' in df.columns:
            df['internet_service_type'] = df['internetservice'].str.lower()
            # Fiber optic is more expensive/advanced
            df['has_fiber_optic'] = (df['internet_service_type'] == 'fiber optic').astype(int)
        else:
            df['has_fiber_optic'] = 0
        
        # Demographics (if available)
        if 'seniorcitizen' in df.columns:
            df['senior_citizen'] = df['seniorcitizen'].astype(int)
        else:
            df['senior_citizen'] = 0
        
        if 'partner' in df.columns:
            df['partner'] = df['partner'].astype(str).str.lower().isin(['yes', 'true', '1']).astype(int)
        else:
            df['partner'] = 0
        
        if 'dependents' in df.columns:
            df['dependents'] = df['dependents'].astype(str).str.lower().isin(['yes', 'true', '1']).astype(int)
        else:
            df['dependents'] = 0
        
        # Support interaction features (if available)
        if 'techsupport' in df.columns:
            df['tech_support'] = df['techsupport'].astype(int)
        else:
            df['tech_support'] = 0
        
        # Create synthetic support ticket features
        df['support_tickets_count'] = np.random.randint(0, 10, len(df))
        df['technical_tickets_count'] = np.random.randint(0, 5, len(df))
        df['billing_complaints'] = np.random.randint(0, 3, len(df))
        df['days_since_last_support_ticket'] = np.random.randint(0, 365, len(df))
        df['avg_response_time_hours'] = np.random.uniform(1, 48, len(df))
        df['service_disruptions_count'] = np.random.randint(0, 5, len(df))
        
        logger.info(f"Features engineered. Total features: {len(df.columns)}")
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select relevant features for the model."""
        logger.info("Selecting relevant features...")
        
        feature_columns = [
            'customerid',
            'senior_citizen',
            'partner',
            'dependents',
            'tenure_months',
            'monthlycharges',
            'totalcharges',
            'phoneservice',
            'internetservice',
            'onlinesecurity',
            'onlinebackup',
            'deviceprotection',
            'techsupport',
            'streamingtv',
            'streamingmovies',
            'contract_type',
            'payment_method',
            'paperlessbilling',
            'service_count',
            'contract_flexibility',
            'automatic_payment',
            'has_fiber_optic',
            'avg_monthly_charges_6m',
            'charges_increase_rate',
            'support_tickets_count',
            'technical_tickets_count',
            'billing_complaints',
            'days_since_last_support_ticket',
            'avg_response_time_hours',
            'service_disruptions_count'
        ]
        
        # Select only columns that exist
        available_columns = [col for col in feature_columns if col in df.columns]
        df_selected = df[available_columns].copy()
        
        logger.info(f"Selected {len(available_columns)} features")
        return df_selected
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features for model training."""
        logger.info("Encoding categorical features...")
        
        categorical_cols = ['contract_type', 'payment_method', 'internetservice']
        
        for col in categorical_cols:
            if col in df.columns:
                # Use label encoding for categorical features
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        logger.info("Categorical features encoded")
        return df
    
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
        required_cols = ['customerid', 'monthlycharges', 'totalcharges']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        # Check for NaN values
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"Found {nan_count} NaN values in features")
        
        # Check data types for numeric columns
        numeric_cols = ['monthlycharges', 'totalcharges', 'tenure_months']
        for col in numeric_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                logger.error(f"Column {col} should be numeric")
                return False
        
        logger.info("Feature validation passed")
        return True


def main():
    """Main function to orchestrate feature engineering."""
    parser = argparse.ArgumentParser(
        description='Feature engineering for Telco customer churn prediction model'
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
        default='telco-churn',
        help='Prefix in GCS bucket for raw data'
    )
    parser.add_argument(
        '--output-bucket',
        required=True,
        help='GCS bucket for processed features'
    )
    parser.add_argument(
        '--output-prefix',
        default='churn-features',
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
        engineer = ChurnFeatureEngineer(args.gcp_project, args.bq_dataset)
        
        # Load raw data
        raw_df = engineer.load_raw_data_from_gcs(args.input_bucket, args.input_prefix)
        
        # Clean data
        cleaned_df = engineer.clean_data(raw_df)
        
        # Engineer features
        featured_df = engineer.engineer_features(cleaned_df)
        
        # Select relevant features
        selected_df = engineer.select_features(featured_df)
        
        # Encode categorical features
        encoded_df = engineer.encode_categorical_features(selected_df)
        
        # Validate features
        if not engineer.validate_features(encoded_df):
            logger.error("Feature validation failed")
            return
        
        # Save to GCS
        output_file = f"{args.output_prefix}/churn_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        engineer.save_to_gcs(encoded_df, args.output_bucket, output_file)
        
        # Save to BigQuery
        table_id = f"{args.gcp_project}.{args.bq_dataset}.customer_demographics"
        engineer.save_to_bigquery(encoded_df, table_id)
        
        logger.info("=" * 80)
        logger.info("Feature engineering completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
