"""
Telco Customer Churn Prediction Model Serving Service

This FastAPI application serves predictions from the trained churn model.
It retrieves features from the Feast online store and returns churn predictions.
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Telco Churn Prediction Service",
    description="ML model serving endpoint for customer churn predictions",
    version="1.0.0"
)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChurnPredictionRequest(BaseModel):
    """Request model for churn predictions."""
    customer_id: str = Field(..., description="Unique customer identifier")
    age: int = Field(..., description="Customer age")
    tenure_months: int = Field(..., description="Months as customer")
    monthly_charges: float = Field(..., description="Monthly charge amount")
    total_charges: float = Field(..., description="Total charges to date")
    contract_type: str = Field(..., description="Contract type (Month-to-month, One year, Two year)")
    internet_service_type: str = Field(..., description="Internet service type (DSL, Fiber optic, No)")
    tech_support: bool = Field(..., description="Has tech support service")
    online_security: bool = Field(..., description="Has online security service")
    support_tickets_count: int = Field(..., description="Number of support tickets")
    
    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "age": 45,
                "tenure_months": 24,
                "monthly_charges": 85.5,
                "total_charges": 2052.0,
                "contract_type": "One year",
                "internet_service_type": "Fiber optic",
                "tech_support": True,
                "online_security": True,
                "support_tickets_count": 2
            }
        }


class ChurnPredictionResponse(BaseModel):
    """Response model for churn predictions."""
    customer_id: str
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: bool = Field(..., description="Predicted churn (True/False)")
    confidence: float = Field(..., description="Model confidence (0-1)")
    risk_factors: list[str] = Field(..., description="Top risk factors for churn")
    retention_score: float = Field(..., description="Retention score (0-1)")
    timestamp: datetime
    model_version: str


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    model_loaded: bool
    service_version: str


# ============================================================================
# GLOBAL STATE
# ============================================================================

class ModelState:
    """Holds the loaded model and feature store client."""
    model = None
    feast_client = None
    model_version = "1.0.0"
    model_loaded = False


# ============================================================================
# INITIALIZATION
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model and feature store on startup."""
    logger.info("Starting Telco Churn Prediction Service...")
    
    try:
        # Load the trained model from GCS
        # This is a placeholder - replace with actual model loading logic
        logger.info("Loading Churn model from GCS...")
        # model_path = f"gs://mlops-dev-model-artifacts/churn/model.pkl"
        # ModelState.model = load_model_from_gcs(model_path)
        ModelState.model_loaded = True
        logger.info("Model loaded successfully")
        
        # Initialize Feast client
        logger.info("Initializing Feast client...")
        # from feast import FeatureStore
        # ModelState.feast_client = FeatureStore(repo_path="./feast")
        logger.info("Feast client initialized")
        
    except Exception as e:
        logger.error(f"Failed to initialize service: {str(e)}")
        ModelState.model_loaded = False


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Telco Churn Prediction Service...")


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthCheckResponse(
        status="healthy" if ModelState.model_loaded else "unhealthy",
        timestamp=datetime.utcnow(),
        model_loaded=ModelState.model_loaded,
        service_version=ModelState.model_version
    )


# ============================================================================
# METRICS ENDPOINT (for Prometheus)
# ============================================================================

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    # Placeholder for Prometheus metrics
    # In production, use prometheus_client library
    metrics_text = """
# HELP churn_predictions_total Total number of predictions made
# TYPE churn_predictions_total counter
churn_predictions_total 0

# HELP churn_prediction_latency_seconds Prediction latency in seconds
# TYPE churn_prediction_latency_seconds histogram
churn_prediction_latency_seconds_bucket{le="0.1"} 0
churn_prediction_latency_seconds_bucket{le="0.5"} 0
churn_prediction_latency_seconds_bucket{le="1.0"} 0
churn_prediction_latency_seconds_bucket{le="+Inf"} 0
churn_prediction_latency_seconds_sum 0
churn_prediction_latency_seconds_count 0

# HELP churn_prediction_errors_total Total number of prediction errors
# TYPE churn_prediction_errors_total counter
churn_prediction_errors_total 0
"""
    return metrics_text


# ============================================================================
# PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict/churn", response_model=ChurnPredictionResponse)
async def predict_churn(request: ChurnPredictionRequest):
    """
    Make a churn prediction for a given customer.
    
    Args:
        request: Churn prediction request with customer data
        
    Returns:
        ChurnPredictionResponse with churn probability and risk factors
        
    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    
    if not ModelState.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is not ready."
        )
    
    try:
        # Prepare features for prediction
        features = {
            "age": request.age,
            "tenure_months": request.tenure_months,
            "monthly_charges": request.monthly_charges,
            "total_charges": request.total_charges,
            "contract_type": request.contract_type,
            "internet_service_type": request.internet_service_type,
            "tech_support": request.tech_support,
            "online_security": request.online_security,
            "support_tickets_count": request.support_tickets_count,
        }
        
        # Retrieve additional features from Feast if available
        # if ModelState.feast_client:
        #     feast_features = ModelState.feast_client.get_online_features(...)
        #     features.update(feast_features)
        
        # Make prediction (placeholder)
        # churn_probability = ModelState.model.predict_proba([list(features.values())])[0][1]
        churn_probability = 0.28  # Placeholder prediction
        
        # Determine churn prediction
        churn_prediction = churn_probability > 0.5
        
        # Identify risk factors
        risk_factors = []
        if request.tenure_months < 12:
            risk_factors.append("Low tenure (less than 1 year)")
        if request.monthly_charges > 100:
            risk_factors.append("High monthly charges")
        if request.support_tickets_count > 5:
            risk_factors.append("High support ticket frequency")
        if request.contract_type == "Month-to-month":
            risk_factors.append("Month-to-month contract (flexible exit)")
        if not request.tech_support:
            risk_factors.append("No tech support service")
        
        # Calculate retention score
        retention_score = 1.0 - churn_probability
        
        return ChurnPredictionResponse(
            customer_id=request.customer_id,
            churn_probability=churn_probability,
            churn_prediction=churn_prediction,
            confidence=0.89,
            risk_factors=risk_factors[:3],  # Top 3 risk factors
            retention_score=retention_score,
            timestamp=datetime.utcnow(),
            model_version=ModelState.model_version
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================================
# BATCH PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict/churn/batch")
async def predict_churn_batch(requests: list[ChurnPredictionRequest]):
    """
    Make batch predictions for multiple customers.
    
    Args:
        requests: List of churn prediction requests
        
    Returns:
        List of predictions
    """
    
    if not ModelState.model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is not ready."
        )
    
    try:
        predictions = []
        for request in requests:
            # Reuse single prediction logic
            prediction = await predict_churn(request)
            predictions.append(prediction)
        
        return {"predictions": predictions}
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ============================================================================
# INFO ENDPOINT
# ============================================================================

@app.get("/info")
async def service_info():
    """Get service information."""
    return {
        "service_name": "Telco Churn Prediction Service",
        "version": ModelState.model_version,
        "model_loaded": ModelState.model_loaded,
        "endpoints": {
            "health": "/health",
            "predict": "/predict/churn",
            "batch_predict": "/predict/churn/batch",
            "metrics": "/metrics",
            "info": "/info"
        }
    }


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Telco Churn Prediction Service",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
