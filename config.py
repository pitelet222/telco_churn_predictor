"""
Centralized Configuration Management
=====================================
Uses Pydantic BaseSettings to load config from environment variables
and .env files, with sensible defaults for every setting.

Usage
-----
    from config import settings

    # Access any setting:
    settings.OPENAI_MODEL          # "gpt-4o-mini"
    settings.TEST_SIZE             # 0.3
    settings.MODELS_DIR            # Path("...telco-churn-prediction/models")

How it works
------------
1. Pydantic reads the .env file at project root automatically.
2. Environment variables ALWAYS override .env values (12-factor app pattern).
3. If neither is set, the default in this class is used.
4. Types are validated at startup — wrong types crash early with a clear message.

To change a setting for one run without editing files:
    $env:OPENAI_MODEL = "gpt-4o"
    streamlit run app/app.py
"""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Resolve project root (one level up from this file) ──────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    # ── Pydantic-settings configuration ─────────────────────────────────────
    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",       # auto-load .env from project root
        env_file_encoding="utf-8",
        extra="ignore",                        # ignore unknown env vars
        case_sensitive=False,                   # OPENAI_MODEL == openai_model
    )

    # ── OpenAI / LLM ───────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""                   # required for chatbot features
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 600

    # ── Environment ────────────────────────────────────────────────────────
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # ── Paths ──────────────────────────────────────────────────────────────
    DATA_PATH: Path = _PROJECT_ROOT / "data" / "processed" / "telco_churn_cleaned.csv"
    MODELS_DIR: Path = _PROJECT_ROOT / "models"

    # ── Training ───────────────────────────────────────────────────────────
    TEST_SIZE: float = 0.3
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5

    # ── Prediction / SHAP ─────────────────────────────────────────────────
    SHAP_TOP_N: int = 5                        # number of top SHAP factors shown
    CHURN_THRESHOLD: float = 0.5               # classification cutoff (lower → more recall)

    # ── Streamlit UI ───────────────────────────────────────────────────────
    APP_TITLE: str = "ChurnGuard AI"
    APP_ICON: str = "🛡️"

    # ── FastAPI ────────────────────────────────────────────────────────────
    API_TITLE: str = "ChurnGuard AI API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_CORS_ORIGINS: list[str] = ["*"]

    # ── API Security ───────────────────────────────────────────────────────
    API_KEY: str = ""                          # if set, require X-API-Key header

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"                    # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_DIR: Path = _PROJECT_ROOT / "logs"     # where log files are written


# ── Singleton instance — import this everywhere ────────────────────────────
settings = Settings()
