"""
ChurnGuard AI — FastAPI Application
====================================
Entry point for the REST API.  Run with:
    uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from log_config import get_logger
from api.routers import health, predict, advice

logger = get_logger(__name__)


# ── Startup / Shutdown lifecycle ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Pre-load model artifacts so the first request isn't slow."""
    logger.info("Loading model artifacts at startup …")
    from app.churn_service import _load_artifacts
    _load_artifacts()
    logger.info("Model artifacts ready — API is live")
    yield
    logger.info("API shutting down")


# ── App factory ─────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=(
        "REST API for the ChurnGuard AI telco churn prediction model. "
        "Predict customer churn risk and get AI-powered retention advice."
    ),
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(advice.router)
