"""
Health & Metadata Router
========================
Lightweight endpoints for liveness checks and model info.
"""

from fastapi import APIRouter, HTTPException

from config import settings
from api.schemas import HealthResponse
from app.churn_service import get_model_metadata
from log_config import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Liveness probe — confirms the API is running."""
    return HealthResponse(status="ok", version=settings.API_VERSION)


@router.get("/model/metadata")
def model_metadata() -> dict:
    """Return training metrics, date, and model details."""
    try:
        return get_model_metadata()
    except RuntimeError as exc:
        logger.error("Failed to load model metadata: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
