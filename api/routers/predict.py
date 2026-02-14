"""
Predict Router
==============
POST /predict — accepts a customer profile and returns churn prediction.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import CustomerInput, PredictionResponse
from app.churn_service import predict_churn
from log_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerInput) -> PredictionResponse:
    """Run the ensemble model on a single customer and return risk assessment."""
    try:
        result = predict_churn(customer.model_dump())
        return PredictionResponse(**result)
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")
