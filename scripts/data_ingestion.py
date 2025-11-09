"""
Data Ingestion Script: Download Kaggle datasets and upload to Google Cloud Storage.

This script downloads two datasets from Kaggle:
1. COVID-19 Report Dataset
2. Telco Customer Churn Dataset

The datasets are then uploaded to a specified GCS bucket for further processing.

Requirements:
- kagglehub: pip install kagglehub
- google-cloud-storage: pip install google-cloud-storage
- GCP credentials configured (gcloud auth application-default login)
"""

import os
import logging
from pathlib import Path
from typing import Optional
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def download_kaggle_dataset(dataset_identifier: str, download_path: str) -> str:
    """
    Download a dataset from Kaggle using kagglehub.
    
    Args:
        dataset_identifier: Kaggle dataset identifier (e.g., 'imdevskp/corona-virus-report')
        download_path: Local path to download the dataset
        
    Returns:
        Path to the downloaded dataset
        
    Raises:
        ImportError: If kagglehub is not installed
        Exception: If dataset download fails
    """
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub is not installed. Install it with: pip install kagglehub")
        raise
    
    try:
        logger.info(f"Downloading dataset: {dataset_identifier}")
        path = kagglehub.dataset_download(dataset_identifier)
        logger.info(f"Dataset downloaded to: {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to download dataset {dataset_identifier}: {str(e)}")
        raise


def upload_to_gcs(local_path: str, bucket_name: str, gcs_prefix: str) -> None:
    """
    Upload local files to Google Cloud Storage.
    
    Args:
        local_path: Local directory or file path to upload
        bucket_name: GCS bucket name
        gcs_prefix: Prefix (folder path) in GCS bucket
        
    Raises:
        ImportError: If google-cloud-storage is not installed
        Exception: If upload fails
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error("google-cloud-storage is not installed. Install it with: pip install google-cloud-storage")
        raise
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        local_path_obj = Path(local_path)
        
        if local_path_obj.is_file():
            # Upload single file
            blob_name = f"{gcs_prefix}/{local_path_obj.name}"
            blob = bucket.blob(blob_name)
            logger.info(f"Uploading file: {local_path} → gs://{bucket_name}/{blob_name}")
            blob.upload_from_filename(local_path)
            logger.info(f"File uploaded successfully")
        elif local_path_obj.is_dir():
            # Upload directory recursively
            for file_path in local_path_obj.rglob('*'):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path_obj)
                    blob_name = f"{gcs_prefix}/{relative_path}"
                    blob = bucket.blob(blob_name)
                    logger.info(f"Uploading: {file_path} → gs://{bucket_name}/{blob_name}")
                    blob.upload_from_filename(str(file_path))
            logger.info(f"Directory uploaded successfully")
        else:
            logger.error(f"Path does not exist: {local_path}")
            raise FileNotFoundError(f"Path does not exist: {local_path}")
            
    except Exception as e:
        logger.error(f"Failed to upload to GCS: {str(e)}")
        raise


def main():
    """Main function to orchestrate data ingestion."""
    parser = argparse.ArgumentParser(
        description='Download Kaggle datasets and upload to Google Cloud Storage'
    )
    parser.add_argument(
        '--bucket',
        required=True,
        help='GCS bucket name for storing raw data'
    )
    parser.add_argument(
        '--covid-dataset',
        default='imdevskp/corona-virus-report',
        help='Kaggle COVID-19 dataset identifier'
    )
    parser.add_argument(
        '--churn-dataset',
        default='blastchar/telco-customer-churn',
        help='Kaggle Telco Churn dataset identifier'
    )
    parser.add_argument(
        '--local-cache',
        default='/tmp/kaggle_datasets',
        help='Local directory to cache downloaded datasets'
    )
    parser.add_argument(
        '--skip-covid',
        action='store_true',
        help='Skip COVID-19 dataset download'
    )
    parser.add_argument(
        '--skip-churn',
        action='store_true',
        help='Skip Telco Churn dataset download'
    )
    
    args = parser.parse_args()
    
    # Create local cache directory
    cache_dir = Path(args.local_cache)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download COVID-19 dataset
        if not args.skip_covid:
            logger.info("=" * 80)
            logger.info("STEP 1: Downloading COVID-19 Report Dataset")
            logger.info("=" * 80)
            covid_path = download_kaggle_dataset(
                args.covid_dataset,
                str(cache_dir / 'covid')
            )
            upload_to_gcs(covid_path, args.bucket, 'covid-19')
        
        # Download Telco Churn dataset
        if not args.skip_churn:
            logger.info("=" * 80)
            logger.info("STEP 2: Downloading Telco Customer Churn Dataset")
            logger.info("=" * 80)
            churn_path = download_kaggle_dataset(
                args.churn_dataset,
                str(cache_dir / 'churn')
            )
            upload_to_gcs(churn_path, args.bucket, 'telco-churn')
        
        logger.info("=" * 80)
        logger.info("Data ingestion completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
