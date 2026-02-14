"""
Advice Router
=============
POST /advice — predicts churn then asks the LLM for retention strategies.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import AdviceRequest, AdviceResponse, PredictionResponse
from app.churn_service import predict_churn
from app.llm_client import get_retention_advice
from log_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Advice"])


@router.post("/advice", response_model=AdviceResponse)
def advice(payload: AdviceRequest) -> AdviceResponse:
    """Predict churn and generate LLM-powered retention recommendations."""
    # 1. Run the prediction
    try:
        churn_result = predict_churn(payload.customer.model_dump())
    except Exception as exc:
        logger.exception("Prediction step failed in /advice")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # 2. Call the LLM for retention advice
    try:
        llm_advice = get_retention_advice(
            churn_result=churn_result,
            user_message=payload.question,
        )
    except Exception as exc:
        logger.exception("LLM call failed in /advice")
        raise HTTPException(status_code=502, detail=f"LLM error: {exc}")

    return AdviceResponse(
        prediction=PredictionResponse(**churn_result),
        advice=llm_advice,
    )
