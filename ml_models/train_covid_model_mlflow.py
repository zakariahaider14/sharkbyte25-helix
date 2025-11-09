#!/usr/bin/env python3
"""
COVID-19 Model Training with MLflow Tracking
Logs experiments, parameters, metrics, and models to MLflow
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import joblib
import mlflow
import mlflow.xgboost
import mlflow.sklearn
from datetime import datetime
import os

# Set MLflow tracking URI (will be set via environment variable in production)
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Set experiment name
mlflow.set_experiment("covid-prediction")

def load_and_prepare_data():
    """Load and prepare COVID-19 dataset"""
    print("📊 Loading COVID-19 dataset...")
    
    # Load processed data
    df = pd.read_csv('../data_pipeline/processed_data/covid_features.csv')
    
    print(f"✅ Loaded {len(df)} records")
    print(f"Features: {df.columns.tolist()}")
    
    return df

def train_model_with_mlflow(df, params=None):
    """Train XGBoost model with MLflow tracking"""
    
    # Default parameters
    if params is None:
        params = {
            'max_depth': 5,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'multi:softmax',
            'num_class': 3,
            'random_state': 42
        }
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"covid_xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("model_type", "XGBoost")
        
        # Prepare features and target
        feature_cols = ['confirmed_cases', 'deaths', 'recovered', 'active_cases', 
                       'death_rate', 'recovery_rate', 'risk_score']
        
        X = df[feature_cols]
        y = df['risk_level']
        
        # Encode target
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        
        # Train model
        print("🚀 Training XGBoost model...")
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        print(f"\n📈 Model Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        
        # Log classification report as artifact
        report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
        with open("classification_report.txt", "w") as f:
            f.write(report)
        mlflow.log_artifact("classification_report.txt")
        os.remove("classification_report.txt")
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv("feature_importance.csv", index=False)
        mlflow.log_artifact("feature_importance.csv")
        os.remove("feature_importance.csv")
        
        # Log model
        mlflow.xgboost.log_model(model, "model")
        
        # Save model and metadata locally as well
        model_dir = "saved_models"
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(model, f"{model_dir}/covid_model.joblib")
        joblib.dump({
            'version': '1.0',
            'feature_columns': feature_cols,
            'label_encoder': label_encoder,
            'accuracy': accuracy,
            'f1_score': f1,
            'trained_at': datetime.now().isoformat()
        }, f"{model_dir}/covid_model_metadata.joblib")
        
        print(f"\n✅ Model saved locally to {model_dir}/")
        print(f"✅ Model logged to MLflow at {MLFLOW_TRACKING_URI}")
        
        # Register model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        model_details = mlflow.register_model(model_uri, "COVID-Predictor")
        
        print(f"✅ Model registered: {model_details.name} version {model_details.version}")
        
        return model, label_encoder, accuracy

def run_experiment_with_different_params():
    """Run multiple experiments with different hyperparameters"""
    
    df = load_and_prepare_data()
    
    # Different parameter combinations to try
    param_grid = [
        {'max_depth': 3, 'learning_rate': 0.1, 'n_estimators': 50},
        {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 100},
        {'max_depth': 7, 'learning_rate': 0.05, 'n_estimators': 150},
        {'max_depth': 5, 'learning_rate': 0.2, 'n_estimators': 100},
    ]
    
    best_accuracy = 0
    best_params = None
    
    for params in param_grid:
        print(f"\n{'='*60}")
        print(f"Training with params: {params}")
        print(f"{'='*60}")
        
        model, encoder, accuracy = train_model_with_mlflow(df, params)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
    
    print(f"\n{'='*60}")
    print(f"🏆 Best Model:")
    print(f"  Params: {best_params}")
    print(f"  Accuracy: {best_accuracy:.4f}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("🚀 COVID-19 Model Training with MLflow")
    print("="*60)
    
    # Option 1: Train single model
    df = load_and_prepare_data()
    train_model_with_mlflow(df)
    
    # Option 2: Run hyperparameter tuning (uncomment to use)
    # run_experiment_with_different_params()
    
    print("\n✅ Training complete!")
    print(f"📊 View experiments at: {MLFLOW_TRACKING_URI}")
