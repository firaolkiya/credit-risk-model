from pydantic import BaseModel
from typing import List, Optional

class PredictionRequest(BaseModel):
    total_amount: float
    avg_amount: float
    std_amount: float
    transaction_count: int
    total_value: float
    avg_value: float
    fraud_count: int
    fraud_rate: float
    avg_category_fraud_rate: float
    avg_provider_fraud_rate: float
    high_value_count: int
    low_value_count: int
    weekend_ratio: float
    avg_hour: float
    std_hour: float
    avg_day_of_week: float
    std_day_of_week: float
    amount_volatility: float
    value_volatility: float
    high_value_ratio: float
    low_value_ratio: float

class PredictionResponse(BaseModel):
    risk_probability: float
    risk_category: str
    prediction_confidence: float

class HealthResponse(BaseModel):
    status: str
    model_version: str
