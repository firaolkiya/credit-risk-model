# Credit Risk API

A FastAPI-based REST API for credit risk prediction using machine learning models.

## Features

- **Risk Prediction**: Predict credit risk probability for new customers
- **Health Monitoring**: API health check endpoint
- **Model Versioning**: Integrated with MLflow model registry
- **Containerized**: Docker support for easy deployment

## Quick Start

### Using Docker Compose

```bash
# Build and run the service
docker-compose up --build

# The API will be available at http://localhost:8000
```

### Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Risk Prediction
```bash
POST /predict
```

**Request Body:**
```json
{
  "total_amount": 10000.0,
  "avg_amount": 1000.0,
  "std_amount": 500.0,
  "transaction_count": 10,
  "total_value": 12000.0,
  "avg_value": 1200.0,
  "fraud_count": 0,
  "fraud_rate": 0.0,
  "avg_category_fraud_rate": 0.01,
  "avg_provider_fraud_rate": 0.005,
  "high_value_count": 1,
  "low_value_count": 2,
  "weekend_ratio": 0.3,
  "avg_hour": 14.5,
  "std_hour": 3.2,
  "avg_day_of_week": 3.1,
  "std_day_of_week": 1.8,
  "amount_volatility": 0.5,
  "value_volatility": 0.4,
  "high_value_ratio": 0.1,
  "low_value_ratio": 0.2
}
```

**Response:**
```json
{
  "risk_probability": 0.25,
  "risk_category": "low",
  "prediction_confidence": 0.5
}
```

## Development

### Running Tests
```bash
pytest tests/ -v
```

### Code Linting
```bash
flake8 src/ tests/
```

### CI/CD
The project includes GitHub Actions for automated testing and linting on every push to main branch. 