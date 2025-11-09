#!/usr/bin/env python3
"""
COVID-19 Model Training
Trains XGBoost model for COVID-19 risk prediction
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Paths
DATA_DIR = Path(__file__).parent.parent / "data_pipeline" / "processed_data"
MODEL_DIR = Path(__file__).parent / "saved_models"
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    """Load preprocessed train and test data"""
    print("📥 Loading COVID data...")
    
    train_path = DATA_DIR / "covid_train.csv"
    test_path = DATA_DIR / "covid_test.csv"
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Preprocessed data not found. Run preprocessing first.")
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"✅ Loaded {len(train_df)} train and {len(test_df)} test samples")
    return train_df, test_df

def prepare_features(train_df, test_df):
    """Prepare features and target"""
    print("🔧 Preparing features...")
    
    # Define feature columns (exclude country_name, risk_level, timestamp)
    exclude_cols = ['country_name', 'risk_level', 'timestamp']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df['risk_level']
    
    X_test = test_df[feature_cols]
    y_test = test_df['risk_level']
    
    # Encode target labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    print(f"✅ Features: {len(feature_cols)}")
    print(f"   {feature_cols}")
    print(f"✅ Classes: {list(le.classes_)}")
    
    return X_train, X_test, y_train_encoded, y_test_encoded, feature_cols, le

def train_model(X_train, y_train):
    """Train XGBoost classifier"""
    print("🤖 Training XGBoost model...")
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='mlogloss'
    )
    
    model.fit(X_train, y_train)
    
    print("✅ Model training complete")
    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate model performance"""
    print("📊 Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro'),
        'recall_macro': recall_score(y_test, y_pred, average='macro'),
        'f1_score_macro': f1_score(y_test, y_pred, average='macro')
    }
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    # Per-class metrics
    print("\nPER-CLASS METRICS:")
    for i, class_name in enumerate(label_encoder.classes_):
        class_mask = (y_test == i)
        if class_mask.sum() > 0:
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            print(f"  {class_name}: {class_acc:.4f}")
    
    return metrics

def save_model(model, feature_cols, label_encoder, metrics):
    """Save trained model and metadata"""
    print("\n💾 Saving model...")
    
    model_path = MODEL_DIR / "covid_model.joblib"
    metadata_path = MODEL_DIR / "covid_model_metadata.joblib"
    
    # Save model
    joblib.dump(model, model_path)
    
    # Save metadata
    metadata = {
        'feature_columns': feature_cols,
        'label_encoder': label_encoder,
        'metrics': metrics,
        'model_type': 'XGBClassifier',
        'version': '1.0.0'
    }
    joblib.dump(metadata, metadata_path)
    
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    print("=" * 60)
    print("COVID-19 MODEL TRAINING")
    print("=" * 60)
    
    try:
        # Load data
        train_df, test_df = load_data()
        
        # Prepare features
        X_train, X_test, y_train, y_test, feature_cols, le = prepare_features(train_df, test_df)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        metrics = evaluate_model(model, X_test, y_test, le)
        
        # Save model
        save_model(model, feature_cols, le, metrics)
        
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print("✅ COVID-19 risk prediction model is ready for deployment")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
