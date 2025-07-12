from fastapi import FastAPI, HTTPException
import mlflow
import pandas as pd
import numpy as np
from .pydantic_models import PredictionRequest, PredictionResponse, HealthResponse

app = FastAPI(title="Credit Risk API", version="1.0.0")

# Load model from MLflow registry
try:
    model = mlflow.sklearn.load_model("models:/credit-risk-proxy-best/Production")
except:
    # Fallback to local model if registry is not available
    model = mlflow.sklearn.load_model("mlruns/0/latest/artifacts/best_model")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_version="1.0.0"
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_risk(request: PredictionRequest):
    try:
        # Convert request to DataFrame
        features = pd.DataFrame([request.dict()])
        
        # Make prediction
        risk_probability = model.predict_proba(features)[0][1]
        
        # Determine risk category
        if risk_probability < 0.3:
            risk_category = "low"
        elif risk_probability < 0.7:
            risk_category = "medium"
        else:
            risk_category = "high"
        
        # Calculate confidence based on probability distance from 0.5
        prediction_confidence = abs(risk_probability - 0.5) * 2
        
        return PredictionResponse(
            risk_probability=float(risk_probability),
            risk_category=risk_category,
            prediction_confidence=float(prediction_confidence)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Credit Risk API - Use /predict for predictions"}
