"""
API Request & Response Schemas
==============================
Pydantic models that define the contract for every endpoint.
Field constraints mirror CUSTOMER_FIELDS in churn_service.py.
"""

from typing import Literal
from pydantic import BaseModel, Field


# ── Request Models ──────────────────────────────────────────────────────────

class CustomerInput(BaseModel):
    """Payload accepted by POST /predict and POST /advice."""

    gender: Literal["Male", "Female"] = Field(
        ..., examples=["Female"],
    )
    SeniorCitizen: Literal["No", "Yes"] = Field(
        ..., examples=["No"],
    )
    Partner: Literal["No", "Yes"] = Field(
        ..., examples=["Yes"],
    )
    Dependents: Literal["No", "Yes"] = Field(
        ..., examples=["No"],
    )
    tenure: int = Field(
        ..., ge=0, le=72, examples=[12],
        description="Months the customer has been with the company",
    )
    PhoneService: Literal["No", "Yes"] = Field(
        ..., examples=["Yes"],
    )
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(
        ..., examples=["No"],
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., examples=["Fiber optic"],
    )
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(
        ..., examples=["No"],
    )
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(
        ..., examples=["No"],
    )
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(
        ..., examples=["No"],
    )
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(
        ..., examples=["No"],
    )
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(
        ..., examples=["Yes"],
    )
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(
        ..., examples=["Yes"],
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., examples=["Month-to-month"],
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(
        ..., examples=["Electronic check"],
    )
    PaperlessBilling: Literal["No", "Yes"] = Field(
        ..., examples=["Yes"],
    )
    MonthlyCharges: float = Field(
        ..., ge=0, le=200, examples=[70.35],
    )
    TotalCharges: float = Field(
        ..., ge=0, le=10000, examples=[840.50],
    )


# ── Response Models ─────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Returned by POST /predict."""

    churn_probability: float = Field(
        ..., ge=0, le=1, description="Probability of churn (0-1)",
    )
    risk_level: Literal["Low", "Medium", "High", "Very High"]
    risk_factors: list[str]
    risk_source: Literal["shap", "manual"]
    customer_summary: str


class AdviceRequest(BaseModel):
    """Payload for POST /advice — customer data plus an optional question."""

    customer: CustomerInput
    question: str = Field(
        default="",
        max_length=500,
        description="Optional follow-up question for the retention advisor",
    )


class AdviceResponse(BaseModel):
    """Returned by POST /advice."""

    prediction: PredictionResponse
    advice: str = Field(
        ..., description="Markdown-formatted retention advice from the LLM",
    )


class HealthResponse(BaseModel):
    """Returned by GET /health."""

    status: str = "ok"
    version: str
