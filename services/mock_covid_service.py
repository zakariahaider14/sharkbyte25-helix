#!/usr/bin/env python3
"""
Mock COVID-19 Prediction Service
Simulates the FastAPI service for local development
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import random
import uvicorn

app = FastAPI(title="COVID-19 Prediction Service (Mock)")

class CovidPredictionRequest(BaseModel):
    country_name: str
    confirmed_cases: Optional[int] = None
    deaths: Optional[int] = None
    recovered: Optional[int] = None
    population: Optional[int] = None
    vaccination_rate: Optional[float] = None
    testing_rate: Optional[float] = None

class CovidPredictionResponse(BaseModel):
    country_name: str
    prediction: float
    risk_level: str
    confidence: float
    timestamp: str
    model_version: str
    explanation: str

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "covid-prediction",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/covid", response_model=CovidPredictionResponse)
def predict_covid(request: CovidPredictionRequest):
    """
    Mock COVID-19 prediction endpoint
    Returns simulated predictions based on input parameters
    """
    try:
        # Simulate prediction logic
        confirmed_cases = request.confirmed_cases or 0
        deaths = request.deaths or 0
        vaccination_rate = request.vaccination_rate or 0.0
        
        # Simple risk calculation (mock)
        if confirmed_cases > 0:
            death_rate = deaths / confirmed_cases if confirmed_cases > 0 else 0
        else:
            death_rate = 0
            
        # Calculate risk score (0-1)
        risk_score = min(
            (death_rate * 10 + (1 - vaccination_rate) * 0.5 + random.uniform(-0.1, 0.1)),
            1.0
        )
        risk_score = max(0.0, risk_score)
        
        # Determine risk level
        if risk_score < 0.3:
            risk_level = "LOW"
        elif risk_score < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        # Generate explanation
        explanation = (
            f"Based on {confirmed_cases:,} confirmed cases, {deaths:,} deaths, "
            f"and {vaccination_rate*100:.1f}% vaccination rate in {request.country_name}, "
            f"the predicted risk level is {risk_level}. "
        )
        
        if vaccination_rate > 0.7:
            explanation += "High vaccination rate is helping to reduce risk. "
        elif vaccination_rate < 0.3:
            explanation += "Low vaccination rate increases risk. "
            
        return CovidPredictionResponse(
            country_name=request.country_name,
            prediction=round(risk_score, 2),
            risk_level=risk_level,
            confidence=round(random.uniform(0.85, 0.95), 2),
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0-mock",
            explanation=explanation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "service": "COVID-19 Prediction Service",
        "version": "1.0.0-mock",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict/covid"
        }
    }

if __name__ == "__main__":
    print("🦠 Starting Mock COVID-19 Prediction Service on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
