#!/usr/bin/env python3
"""
Real COVID-19 Prediction Service
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

app = FastAPI(title="COVID-19 Prediction Service (Real)")

# Paths
MODEL_DIR = Path(__file__).parent.parent / "ml_models" / "saved_models"
FEAST_DIR = Path(__file__).parent.parent / "feast_store"

# Load model and metadata
try:
    model = joblib.load(MODEL_DIR / "covid_model.joblib")
    metadata = joblib.load(MODEL_DIR / "covid_model_metadata.joblib")
    feature_cols = metadata['feature_columns']
    label_encoder = metadata['label_encoder']
    print(f"✅ Loaded COVID model (v{metadata['version']})")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    metadata = None
    feature_cols = []
    label_encoder = None

# Initialize Feast
feast_store = None
if FEAST_AVAILABLE and FEAST_DIR.exists():
    try:
        feast_store = FeatureStore(repo_path=str(FEAST_DIR))
        print("✅ Feast feature store initialized")
    except Exception as e:
        print(f"⚠️  Could not initialize Feast: {e}")

class CovidPredictionRequest(BaseModel):
    country_name: str
    confirmed_cases: Optional[int] = None
    deaths: Optional[int] = None
    recovered: Optional[int] = None
    population: Optional[int] = None
    vaccination_rate: Optional[float] = None
    testing_rate: Optional[float] = None
    use_feast: bool = True  # Whether to fetch features from Feast

class CovidPredictionResponse(BaseModel):
    country_name: str
    prediction: float
    risk_level: str
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
        "service": "covid-prediction-real",
        "model_loaded": model is not None,
        "feast_available": feast_store is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/covid", response_model=CovidPredictionResponse)
def predict_covid(request: CovidPredictionRequest):
    """
    Real COVID-19 prediction endpoint
    Uses trained model and optionally Feast for feature retrieval
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        feature_source = "request"
        
        # Try to get features from Feast if available and requested
        if request.use_feast and feast_store is not None:
            try:
                entity_df = pd.DataFrame({
                    "country_name": [request.country_name],
                    "event_timestamp": [datetime.now()]
                })
                
                features = feast_store.get_online_features(
                    entity_rows=entity_df.to_dict('records'),
                    features=[
                        "covid_features:confirmed_cases",
                        "covid_features:deaths",
                        "covid_features:recovered",
                        "covid_features:active_cases",
                        "covid_features:death_rate",
                        "covid_features:recovery_rate",
                        "covid_features:risk_score",
                    ]
                ).to_dict()
                
                # Extract features and convert to numeric types
                feature_dict = {
                    'confirmed_cases': float(features['confirmed_cases'][0]),
                    'deaths': float(features['deaths'][0]),
                    'recovered': float(features['recovered'][0]),
                    'active_cases': float(features['active_cases'][0]),
                    'death_rate': float(features['death_rate'][0]),
                    'recovery_rate': float(features['recovery_rate'][0]),
                    'risk_score': float(features['risk_score'][0]),
                }
                feature_source = "feast"
                print(f"✅ Retrieved features from Feast for {request.country_name}")
                
            except Exception as e:
                print(f"⚠️  Feast retrieval failed: {e}, using request data")
                feature_dict = None
        else:
            feature_dict = None
        
        # Fall back to request data if Feast not available
        if feature_dict is None:
            # Use provided values or intelligent defaults
            confirmed_cases = request.confirmed_cases or 100000  # Default: 100k cases
            deaths = request.deaths or int(confirmed_cases * 0.02)  # Default: 2% death rate
            recovered = request.recovered or int(confirmed_cases * 0.70)  # Default: 70% recovery
            active_cases = confirmed_cases - deaths - recovered
            
            death_rate = (deaths / confirmed_cases * 100) if confirmed_cases > 0 else 2.0
            recovery_rate = (recovered / confirmed_cases * 100) if confirmed_cases > 0 else 70.0
            
            # Simple risk score calculation
            risk_score = min(
                (death_rate / 10) * 0.5 + 
                (active_cases / max(confirmed_cases, 1)) * 0.5,
                1.0
            )
            
            feature_dict = {
                'confirmed_cases': confirmed_cases,
                'deaths': deaths,
                'recovered': recovered,
                'active_cases': active_cases,
                'death_rate': death_rate,
                'recovery_rate': recovery_rate,
                'risk_score': risk_score,
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
        prediction_encoded = model.predict(feature_df)[0]
        prediction_proba = model.predict_proba(feature_df)[0]
        
        # Decode prediction
        risk_level = label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = float(prediction_proba[prediction_encoded])
        
        # Generate explanation
        using_defaults = not request.confirmed_cases and feature_source != "feast"
        explanation = (
            f"Based on {feature_dict['confirmed_cases']:,} confirmed cases, "
            f"{feature_dict['deaths']:,} deaths, and {feature_dict['recovered']:,} recovered "
            f"in {request.country_name}, the predicted risk level is {risk_level}. "
            f"Death rate: {feature_dict['death_rate']:.2f}%, "
            f"Recovery rate: {feature_dict['recovery_rate']:.2f}%. "
        )
        
        if using_defaults:
            explanation += "(Note: Using estimated default values for missing data.)"
        
        return CovidPredictionResponse(
            country_name=request.country_name,
            prediction=float(feature_dict['risk_score']),
            risk_level=risk_level,
            confidence=round(confidence, 2),
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
        "service": "COVID-19 Prediction Service (Real)",
        "version": metadata['version'] if metadata else "unknown",
        "status": "running",
        "model_loaded": model is not None,
        "feast_enabled": feast_store is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict/covid"
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🦠 Starting Real COVID-19 Prediction Service on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
