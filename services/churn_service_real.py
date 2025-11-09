#!/usr/bin/env python3
"""
Real Churn Prediction Service
Uses trained XGBoost model and Feast feature store
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Feast imports
try:
    from feast import FeatureStore
    FEAST_AVAILABLE = True
except ImportError:
    FEAST_AVAILABLE = False
    print("⚠️  Feast not available, using direct model inference")

app = FastAPI(title="Churn Prediction Service (Real)")

# Paths
MODEL_DIR = Path(__file__).parent.parent / "ml_models" / "saved_models"
FEAST_DIR = Path(__file__).parent.parent / "feast_store"

# Load model and metadata
try:
    model = joblib.load(MODEL_DIR / "churn_model.joblib")
    metadata = joblib.load(MODEL_DIR / "churn_model_metadata.joblib")
    feature_cols = metadata['feature_columns']
    print(f"✅ Loaded Churn model (v{metadata['version']})")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    metadata = None
    feature_cols = []

# Initialize Feast
feast_store = None
if FEAST_AVAILABLE and FEAST_DIR.exists():
    try:
        feast_store = FeatureStore(repo_path=str(FEAST_DIR))
        print("✅ Feast feature store initialized")
    except Exception as e:
        print(f"⚠️  Could not initialize Feast: {e}")

class ChurnPredictionRequest(BaseModel):
    customer_id: Optional[str] = None  # Make optional, will generate default if missing
    age: Optional[int] = None
    tenure_months: Optional[int] = None
    monthly_charges: Optional[float] = None
    total_charges: Optional[float] = None
    contract_type: Optional[str] = None
    internet_service_type: Optional[str] = None
    tech_support: Optional[bool] = None
    online_security: Optional[bool] = None
    support_tickets_count: Optional[int] = None
    use_feast: bool = True  # Whether to fetch features from Feast

class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_risk: str
    confidence: float
    timestamp: str
    model_version: str
    explanation: str
    feature_source: str  # "feast" or "request"

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "churn-prediction-real",
        "model_loaded": model is not None,
        "feast_available": feast_store is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/churn", response_model=ChurnPredictionResponse)
def predict_churn(request: ChurnPredictionRequest):
    """
    Real Churn prediction endpoint
    Uses trained model and optionally Feast for feature retrieval
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Generate default customer ID if not provided
        customer_id = request.customer_id or f"CUSTOMER_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        feature_source = "request"
        
        # Try to get features from Feast if available and requested
        if request.use_feast and feast_store is not None and request.customer_id:
            try:
                entity_df = pd.DataFrame({
                    "customer_id": [request.customer_id],
                    "event_timestamp": [datetime.now()]
                })
                
                features = feast_store.get_online_features(
                    entity_rows=entity_df.to_dict('records'),
                    features=[
                        "churn_features:SeniorCitizen",
                        "churn_features:Partner",
                        "churn_features:Dependents",
                        "churn_features:tenure",
                        "churn_features:PhoneService",
                        "churn_features:MultipleLines",
                        "churn_features:OnlineSecurity",
                        "churn_features:OnlineBackup",
                        "churn_features:DeviceProtection",
                        "churn_features:TechSupport",
                        "churn_features:StreamingTV",
                        "churn_features:StreamingMovies",
                        "churn_features:PaperlessBilling",
                        "churn_features:MonthlyCharges",
                        "churn_features:TotalCharges",
                        "churn_features:gender_encoded",
                        "churn_features:Contract_encoded",
                        "churn_features:PaymentMethod_encoded",
                        "churn_features:InternetService_encoded",
                    ]
                ).to_dict()
                
                # Extract features and convert to numeric types
                feature_dict = {
                    'SeniorCitizen': float(features['SeniorCitizen'][0]),
                    'Partner': float(features['Partner'][0]),
                    'Dependents': float(features['Dependents'][0]),
                    'tenure': float(features['tenure'][0]),
                    'PhoneService': float(features['PhoneService'][0]),
                    'MultipleLines': float(features['MultipleLines'][0]),
                    'OnlineSecurity': float(features['OnlineSecurity'][0]),
                    'OnlineBackup': float(features['OnlineBackup'][0]),
                    'DeviceProtection': float(features['DeviceProtection'][0]),
                    'TechSupport': float(features['TechSupport'][0]),
                    'StreamingTV': float(features['StreamingTV'][0]),
                    'StreamingMovies': float(features['StreamingMovies'][0]),
                    'PaperlessBilling': float(features['PaperlessBilling'][0]),
                    'MonthlyCharges': float(features['MonthlyCharges'][0]),
                    'TotalCharges': float(features['TotalCharges'][0]),
                    'gender_encoded': float(features['gender_encoded'][0]),
                    'Contract_encoded': float(features['Contract_encoded'][0]),
                    'PaymentMethod_encoded': float(features['PaymentMethod_encoded'][0]),
                    'InternetService_encoded': float(features['InternetService_encoded'][0]),
                }
                feature_source = "feast"
                print(f"✅ Retrieved features from Feast for {request.customer_id}")
                
            except Exception as e:
                print(f"⚠️  Feast retrieval failed: {e}, using request data")
                feature_dict = None
        else:
            feature_dict = None
        
        # Fall back to request data if Feast not available
        if feature_dict is None:
            # Use provided values or intelligent defaults
            tenure = request.tenure_months or 24  # Default: 24 months (2 years)
            monthly_charges = request.monthly_charges or 65.0  # Default: $65/month (average)
            total_charges = request.total_charges or (monthly_charges * tenure)
            
            # Map request to features with defaults
            feature_dict = {
                'SeniorCitizen': 0,  # Default: not senior
                'Partner': 1,  # Default: has partner (more stable)
                'Dependents': 0,  # Default: no dependents
                'tenure': tenure,
                'PhoneService': 1,  # Default: has phone service
                'MultipleLines': 0,  # Default: single line
                'OnlineSecurity': int(request.online_security) if request.online_security is not None else 0,
                'OnlineBackup': 0,  # Default: no backup
                'DeviceProtection': 0,  # Default: no protection
                'TechSupport': int(request.tech_support) if request.tech_support is not None else 0,
                'StreamingTV': 0,  # Default: no streaming
                'StreamingMovies': 0,  # Default: no streaming
                'PaperlessBilling': 1,  # Default: paperless
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'gender_encoded': 0,  # Default: encoded value
                'Contract_encoded': 0,  # Default: month-to-month
                'PaymentMethod_encoded': 0,  # Default: electronic check
                'InternetService_encoded': 1,  # Default: has internet
            }
        
        # Prepare features for model
        feature_df = pd.DataFrame([feature_dict])
        
        # Ensure all required features are present
        for col in feature_cols:
            if col not in feature_df.columns:
                feature_df[col] = 0.0
        
        feature_df = feature_df[feature_cols]
        
        # Ensure all columns are numeric (float64)
        feature_df = feature_df.astype(float)
        
        # Make prediction
        churn_proba = model.predict_proba(feature_df)[0]
        churn_probability = float(churn_proba[1])  # Probability of churn
        
        # Determine risk level
        if churn_probability < 0.3:
            churn_risk = "LOW"
        elif churn_probability < 0.6:
            churn_risk = "MEDIUM"
        else:
            churn_risk = "HIGH"
        
        # Generate explanation
        tenure = feature_dict.get('tenure', 0)
        monthly_charges = feature_dict.get('MonthlyCharges', 0)
        using_defaults = not request.tenure_months and feature_source != "feast"
        
        explanation = (
            f"Customer {customer_id} has a {churn_probability*100:.1f}% probability of churning. "
            f"Risk level: {churn_risk}. "
            f"Key factors: {tenure} months tenure, ${monthly_charges:.2f} monthly charges"
        )
        
        if churn_probability > 0.6:
            explanation += ". Recommend immediate retention action."
        elif churn_probability > 0.3:
            explanation += ". Monitor and consider retention offers."
        else:
            explanation += ". Customer appears stable."
        
        if using_defaults:
            explanation += " (Note: Using estimated default values for missing customer data.)"
        
        return ChurnPredictionResponse(
            customer_id=customer_id,
            churn_probability=round(churn_probability, 3),
            churn_risk=churn_risk,
            confidence=round(max(churn_proba), 2),
            timestamp=datetime.now().isoformat(),
            model_version=metadata['version'],
            explanation=explanation,
            feature_source=feature_source
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "Churn Prediction Service (Real)",
        "version": metadata['version'] if metadata else "unknown",
        "status": "running",
        "model_loaded": model is not None,
        "feast_enabled": feast_store is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict/churn"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("📞 Starting Real Churn Prediction Service on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
