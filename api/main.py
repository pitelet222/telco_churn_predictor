"""
ChurnGuard AI — FastAPI Application
====================================
Entry point for the REST API.  Run with:
    uvicorn api.main:app --reload
"""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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
# In production, lock down origins instead of allowing "*"
cors_origins = settings.API_CORS_ORIGINS
if settings.ENVIRONMENT == "production" and cors_origins == ["*"]:
    logger.warning(
        "CORS is set to ['*'] in production — consider restricting API_CORS_ORIGINS"
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API Key Authentication Middleware ───────────────────────────────────────
# Only active when API_KEY is set in the environment.
# Protects all endpoints except /health and /docs from unauthorized access.

if settings.API_KEY:
    @app.middleware("http")
    async def verify_api_key(request: Request, call_next):
        # Allow health checks, OpenAPI docs, and favicon without auth
        exempt_paths = {"/health", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}
        if request.url.path not in exempt_paths:
            api_key = request.headers.get("X-API-Key", "")
            if api_key != settings.API_KEY:
                logger.warning(
                    "Rejected request to %s — invalid or missing API key",
                    request.url.path,
                )
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
        return await call_next(request)
    logger.info("API key authentication enabled")

# ── Routers ─────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(advice.router)
