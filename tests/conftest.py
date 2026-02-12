"""
Shared test fixtures for the Telco Churn test suite.

Fixtures defined here are automatically available in every test file
under the tests/ directory — no imports needed.
"""

import sys
from pathlib import Path

import pytest

# ── Make project modules importable ─────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))


# ── Customer profile fixtures ──────────────────────────────────────────────


@pytest.fixture
def high_risk_customer() -> dict:
    """A customer profile that should produce a HIGH churn probability.

    Key risk signals: month-to-month, fiber optic, electronic check,
    short tenure, no support add-ons.
    """
    return {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "No",
        "Dependents": "No",
        "tenure": 3,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check",
        "PaperlessBilling": "Yes",
        "MonthlyCharges": 75.0,
        "TotalCharges": 225.0,
    }


@pytest.fixture
def low_risk_customer() -> dict:
    """A customer profile that should produce a LOW churn probability.

    Key safety signals: two-year contract, long tenure, automatic payment,
    many services.
    """
    return {
        "gender": "Male",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 60,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaymentMethod": "Bank transfer (automatic)",
        "PaperlessBilling": "No",
        "MonthlyCharges": 85.0,
        "TotalCharges": 5100.0,
    }


@pytest.fixture
def minimal_customer() -> dict:
    """A bare-minimum customer dict (only required fields).

    Tests that defaults and edge cases are handled without crashing.
    """
    return {
        "tenure": 0,
        "MonthlyCharges": 0.0,
        "TotalCharges": 0.0,
    }
