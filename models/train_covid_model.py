"""
COVID-19 Prediction Model Training Script

This script trains a machine learning model to predict COVID-19 risk levels
using epidemiological and demographic features from the Feast feature store.

The trained model is saved to GCS for deployment.
"""

import os
import logging
from typing import Tuple, Dict, Any
from datetime import datetime
import argparse
import pickle
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from google.cloud import storage, bigquery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CovidModelTrainer:
    """Handles training and evaluation of COVID-19 prediction model."""
    
    def __init__(self, gcp_project_id: str, dataset_id: str):
        """Initialize the model trainer with GCP clients."""
        self.gcp_project_id = gcp_project_id
        self.dataset_id = dataset_id
        self.storage_client = storage.Client(project=gcp_project_id)
        self.bq_client = bigquery.Client(project=gcp_project_id)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metrics = {}
        
    def load_features_from_bigquery(self, table_id: str) -> pd.DataFrame:
        """Load engineered features from BigQuery."""
        logger.info(f"Loading features from BigQuery: {table_id}")
        
        query = f"SELECT * FROM `{table_id}`"
        df = self.bq_client.query(query).to_dataframe()
        
        logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns from BigQuery")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
        """Prepare features and target variable for training."""
        logger.info("Preparing data for training...")
        
        # Create target variable based on risk level
        # High case fatality rate or high active cases = high risk
        df['risk_level'] = 0
        df.loc[df['case_fatality_rate'] > 0.05, 'risk_level'] = 1
        df.loc[df['active_cases'] > 100000, 'risk_level'] = 1
        
        # Select feature columns
        feature_columns = [
            'confirmed', 'deaths', 'recovered', 'active_cases',
            'case_fatality_rate', 'recovery_rate', 'population',
            'cases_per_million', 'deaths_per_million',
            'vaccination_rate', 'testing_rate',
            'new_cases_7day_avg', 'new_deaths_7day_avg',
            'trend_strength', 'volatility', 'doubling_time_days'
        ]
        
        # Filter to available columns
        available_features = [col for col in feature_columns if col in df.columns]
        
        X = df[available_features].fillna(0).values
        y = df['risk_level'].values
        
        self.feature_names = available_features
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Target distribution: {np.bincount(y.astype(int))}")
        
        return X, y, available_features
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data into training and testing sets."""
        logger.info("Splitting data into train and test sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple:
        """Scale features using StandardScaler."""
        logger.info("Scaling features...")
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info("Features scaled")
        return X_train_scaled, X_test_scaled
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the COVID-19 prediction model."""
        logger.info("Training COVID-19 prediction model...")
        
        # Use Gradient Boosting for better performance
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            verbose=1
        )
        
        self.model.fit(X_train, y_train)
        
        logger.info("Model training completed")
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate the trained model on test data."""
        logger.info("Evaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
        }
        
        # Log metrics
        logger.info("Model Performance Metrics:")
        for metric_name, metric_value in self.metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Log classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        # Log confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"\nConfusion Matrix:\n{cm}")
        
        return self.metrics
    
    def save_model_to_gcs(self, bucket_name: str, model_path: str) -> str:
        """Save trained model to GCS."""
        logger.info(f"Saving model to gs://{bucket_name}/{model_path}")
        
        # Serialize model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Save to temporary file
        temp_file = '/tmp/covid_model.pkl'
        with open(temp_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Upload to GCS
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(model_path)
        blob.upload_from_filename(temp_file)
        
        logger.info(f"Model saved to GCS: gs://{bucket_name}/{model_path}")
        
        # Clean up temporary file
        os.remove(temp_file)
        
        return f"gs://{bucket_name}/{model_path}"
    
    def save_model_metadata(self, bucket_name: str, metadata_path: str) -> None:
        """Save model metadata to GCS."""
        logger.info(f"Saving model metadata to gs://{bucket_name}/{metadata_path}")
        
        metadata = {
            'model_type': 'GradientBoostingClassifier',
            'training_date': datetime.utcnow().isoformat(),
            'features': self.feature_names,
            'metrics': self.metrics,
            'feature_count': len(self.feature_names),
            'model_version': '1.0.0'
        }
        
        # Save to temporary file
        temp_file = '/tmp/covid_model_metadata.json'
        with open(temp_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Upload to GCS
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(metadata_path)
        blob.upload_from_filename(temp_file)
        
        logger.info(f"Metadata saved to GCS: gs://{bucket_name}/{metadata_path}")
        
        # Clean up temporary file
        os.remove(temp_file)


def main():
    """Main function to orchestrate model training."""
    parser = argparse.ArgumentParser(
        description='Train COVID-19 prediction model'
    )
    parser.add_argument(
        '--gcp-project',
        required=True,
        help='GCP project ID'
    )
    parser.add_argument(
        '--bq-dataset',
        required=True,
        help='BigQuery dataset ID'
    )
    parser.add_argument(
        '--bq-table',
        default='covid_features',
        help='BigQuery table with features'
    )
    parser.add_argument(
        '--model-bucket',
        required=True,
        help='GCS bucket for saving model artifacts'
    )
    parser.add_argument(
        '--model-path',
        default='covid/model.pkl',
        help='Path in GCS bucket for model artifact'
    )
    parser.add_argument(
        '--metadata-path',
        default='covid/model_metadata.json',
        help='Path in GCS bucket for model metadata'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (0-1)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = CovidModelTrainer(args.gcp_project, args.bq_dataset)
        
        # Load features from BigQuery
        table_id = f"{args.gcp_project}.{args.bq_dataset}.{args.bq_table}"
        df = trainer.load_features_from_bigquery(table_id)
        
        # Prepare data
        X, y, feature_names = trainer.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.split_data(X, y, test_size=args.test_size)
        
        # Scale features
        X_train_scaled, X_test_scaled = trainer.scale_features(X_train, X_test)
        
        # Train model
        trainer.train_model(X_train_scaled, y_train)
        
        # Evaluate model
        metrics = trainer.evaluate_model(X_test_scaled, y_test)
        
        # Save model to GCS
        model_uri = trainer.save_model_to_gcs(args.model_bucket, args.model_path)
        
        # Save metadata
        trainer.save_model_metadata(args.model_bucket, args.metadata_path)
        
        logger.info("=" * 80)
        logger.info("Model training completed successfully!")
        logger.info(f"Model saved at: {model_uri}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
