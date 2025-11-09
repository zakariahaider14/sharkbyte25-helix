#!/usr/bin/env python3
"""
Mock Telco Customer Churn Prediction Service
Simulates the FastAPI service for local development
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import random
import uvicorn

app = FastAPI(title="Churn Prediction Service (Mock)")

class ChurnPredictionRequest(BaseModel):
    customer_id: str
    age: Optional[int] = None
    tenure_months: Optional[int] = None
    monthly_charges: Optional[float] = None
    total_charges: Optional[float] = None
    contract_type: Optional[str] = None
    internet_service_type: Optional[str] = None
    tech_support: Optional[bool] = None
    online_security: Optional[bool] = None
    support_tickets_count: Optional[int] = None

class ChurnPredictionResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    confidence: float
    risk_factors: List[str]
    retention_score: float
    timestamp: str
    model_version: str

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "churn-prediction",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/churn", response_model=ChurnPredictionResponse)
def predict_churn(request: ChurnPredictionRequest):
    """
    Mock churn prediction endpoint
    Returns simulated predictions based on input parameters
    """
    try:
        # Simulate prediction logic
        tenure_months = request.tenure_months or 12
        monthly_charges = request.monthly_charges or 50.0
        tech_support = request.tech_support or False
        online_security = request.online_security or False
        support_tickets = request.support_tickets_count or 0
        contract_type = request.contract_type or "Month-to-month"
        
        # Calculate churn probability (mock logic)
        churn_prob = 0.5
        risk_factors = []
        
        # Tenure factor
        if tenure_months < 12:
            churn_prob += 0.2
            risk_factors.append("Low tenure (less than 1 year)")
        elif tenure_months > 36:
            churn_prob -= 0.15
        
        # Monthly charges factor
        if monthly_charges > 80:
            churn_prob += 0.15
            risk_factors.append("High monthly charges")
        
        # Contract type factor
        if contract_type.lower() == "month-to-month":
            churn_prob += 0.1
            risk_factors.append("Month-to-month contract")
        elif "two year" in contract_type.lower():
            churn_prob -= 0.2
        
        # Support services factor
        if not tech_support:
            churn_prob += 0.05
            risk_factors.append("No tech support")
        
        if not online_security:
            churn_prob += 0.05
            risk_factors.append("No online security")
        
        # Support tickets factor
        if support_tickets > 3:
            churn_prob += 0.1
            risk_factors.append(f"High support ticket count ({support_tickets})")
        
        # Add some randomness
        churn_prob += random.uniform(-0.05, 0.05)
        churn_prob = max(0.0, min(1.0, churn_prob))
        
        # Determine churn prediction
        will_churn = churn_prob > 0.5
        
        # Calculate retention score (inverse of churn probability)
        retention_score = 1.0 - churn_prob
        
        # If no risk factors, add positive note
        if not risk_factors:
            risk_factors.append("No significant risk factors identified")
        
        return ChurnPredictionResponse(
            customer_id=request.customer_id,
            churn_probability=round(churn_prob, 2),
            churn_prediction=will_churn,
            confidence=round(random.uniform(0.82, 0.93), 2),
            risk_factors=risk_factors,
            retention_score=round(retention_score, 2),
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0-mock"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "Telco Churn Prediction Service",
        "version": "1.0.0-mock",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict/churn"
        }
    }

if __name__ == "__main__":
    print("📞 Starting Mock Churn Prediction Service on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
