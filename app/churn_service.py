"""
Churn Prediction Service
Loads the trained ensemble model + scaler and exposes a predict function
that returns churn probability and risk level for a single customer.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path
from typing import Any

import sys
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from config import settings

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = settings.MODELS_DIR

# ── Feature schema (must match training order) ──────────────────────────────
FEATURE_COLUMNS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "PaperlessBilling", "MonthlyCharges", "TotalCharges",
    "MultipleLines_no phone service", "MultipleLines_yes",
    "OnlineSecurity_no internet service", "OnlineSecurity_yes",
    "OnlineBackup_no internet service", "OnlineBackup_yes",
    "DeviceProtection_no internet service", "DeviceProtection_yes",
    "TechSupport_no internet service", "TechSupport_yes",
    "StreamingTV_no internet service", "StreamingTV_yes",
    "StreamingMovies_no internet service", "StreamingMovies_yes",
    "InternetService_fiber optic", "InternetService_no",
    "Contract_one year", "Contract_two year",
    "PaymentMethod_credit card (automatic)",
    "PaymentMethod_electronic check", "PaymentMethod_mailed check",
    "AvgMonthlyCharges", "TenureGroup_1-2yr", "TenureGroup_2-4yr",
    "TenureGroup_4-6yr", "TotalServices",
]

# ── Human-readable input fields (what the chatbot collects) ─────────────────
CUSTOMER_FIELDS = {
    "gender":            {"type": "select",  "options": ["Male", "Female"], "label": "Gender"},
    "SeniorCitizen":     {"type": "select",  "options": ["No", "Yes"],     "label": "Senior Citizen"},
    "Partner":           {"type": "select",  "options": ["No", "Yes"],     "label": "Has Partner"},
    "Dependents":        {"type": "select",  "options": ["No", "Yes"],     "label": "Has Dependents"},
    "tenure":            {"type": "number",  "min": 0, "max": 72,         "label": "Tenure (months)"},
    "PhoneService":      {"type": "select",  "options": ["No", "Yes"],     "label": "Phone Service"},
    "MultipleLines":     {"type": "select",  "options": ["No", "Yes", "No phone service"], "label": "Multiple Lines"},
    "InternetService":   {"type": "select",  "options": ["DSL", "Fiber optic", "No"], "label": "Internet Service"},
    "OnlineSecurity":    {"type": "select",  "options": ["No", "Yes", "No internet service"], "label": "Online Security"},
    "OnlineBackup":      {"type": "select",  "options": ["No", "Yes", "No internet service"], "label": "Online Backup"},
    "DeviceProtection":  {"type": "select",  "options": ["No", "Yes", "No internet service"], "label": "Device Protection"},
    "TechSupport":       {"type": "select",  "options": ["No", "Yes", "No internet service"], "label": "Tech Support"},
    "StreamingTV":       {"type": "select",  "options": ["No", "Yes", "No internet service"], "label": "Streaming TV"},
    "StreamingMovies":   {"type": "select",  "options": ["No", "Yes", "No internet service"], "label": "Streaming Movies"},
    "Contract":          {"type": "select",  "options": ["Month-to-month", "One year", "Two year"], "label": "Contract Type"},
    "PaymentMethod":     {"type": "select",  "options": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], "label": "Payment Method"},
    "PaperlessBilling":  {"type": "select",  "options": ["No", "Yes"],     "label": "Paperless Billing"},
    "MonthlyCharges":    {"type": "number",  "min": 0, "max": 200,        "label": "Monthly Charges ($)"},
    "TotalCharges":      {"type": "number",  "min": 0, "max": 10000,      "label": "Total Charges ($)"},
}


# ── Singleton model loader ──────────────────────────────────────────────────
_models = None
_scaler = None
_metadata = None
_explainer = None


def _load_artifacts():
    """Load individual models, scaler, and metadata once (cached).
    We load the 3 component models separately instead of the VotingClassifier
    to avoid scikit-learn version mismatch issues.
    """
    global _models, _scaler, _metadata
    if _models is None:
        _models = {
            "lr": joblib.load(MODELS_DIR / "logistic_regression.pkl"),
            "gb": joblib.load(MODELS_DIR / "gradient_boosting.pkl"),
            "cat": joblib.load(MODELS_DIR / "catboost_model.pkl"),
        }
        _scaler = joblib.load(MODELS_DIR / "scaler.pkl")
        with open(MODELS_DIR / "model_metadata.json") as f:
            _metadata = json.load(f)
    return _models, _scaler, _metadata


def _get_explainer() -> shap.TreeExplainer:
    """Load SHAP TreeExplainer for model explanations (cached).

    Uses the Gradient Boosting model because tree-based explainers
    are fast and produce accurate SHAP values.
    """
    global _explainer
    if _explainer is None:
        models, _, _ = _load_artifacts()
        _explainer = shap.TreeExplainer(models["gb"])
    return _explainer


# ── Feature engineering (mirrors notebook preprocessing) ────────────────────
def _encode_customer(raw: dict) -> pd.DataFrame:
    """
    Convert human-readable customer dict → model-ready feature vector.
    Applies the same one-hot encoding and derived features used during training.
    """
    row = {}

    # Binary mappings
    yes_no = lambda v: 1 if str(v).lower() == "yes" else 0
    row["gender"]           = 1 if raw.get("gender", "Male") == "Male" else 0
    row["SeniorCitizen"]    = yes_no(raw.get("SeniorCitizen", "No"))
    row["Partner"]          = yes_no(raw.get("Partner", "No"))
    row["Dependents"]       = yes_no(raw.get("Dependents", "No"))
    row["tenure"]           = float(raw.get("tenure", 0))
    row["PhoneService"]     = yes_no(raw.get("PhoneService", "Yes"))
    row["PaperlessBilling"] = yes_no(raw.get("PaperlessBilling", "No"))
    row["MonthlyCharges"]   = float(raw.get("MonthlyCharges", 0))
    row["TotalCharges"]     = float(raw.get("TotalCharges", 0))

    # One-hot: MultipleLines
    ml = raw.get("MultipleLines", "No")
    row["MultipleLines_no phone service"] = 1 if ml == "No phone service" else 0
    row["MultipleLines_yes"]              = 1 if ml == "Yes" else 0

    # One-hot: Internet-dependent services
    inet = raw.get("InternetService", "DSL")
    no_inet = (inet == "No")

    for svc, col_prefix in [
        ("OnlineSecurity",  "OnlineSecurity"),
        ("OnlineBackup",    "OnlineBackup"),
        ("DeviceProtection","DeviceProtection"),
        ("TechSupport",     "TechSupport"),
        ("StreamingTV",     "StreamingTV"),
        ("StreamingMovies", "StreamingMovies"),
    ]:
        val = raw.get(svc, "No")
        row[f"{col_prefix}_no internet service"] = 1 if no_inet or val == "No internet service" else 0
        row[f"{col_prefix}_yes"]                 = 1 if (not no_inet and val == "Yes") else 0

    # One-hot: InternetService
    row["InternetService_fiber optic"] = 1 if inet == "Fiber optic" else 0
    row["InternetService_no"]          = 1 if inet == "No" else 0

    # One-hot: Contract
    contract = raw.get("Contract", "Month-to-month")
    row["Contract_one year"] = 1 if contract == "One year" else 0
    row["Contract_two year"] = 1 if contract == "Two year" else 0

    # One-hot: PaymentMethod
    pm = raw.get("PaymentMethod", "Electronic check")
    row["PaymentMethod_credit card (automatic)"] = 1 if pm == "Credit card (automatic)" else 0
    row["PaymentMethod_electronic check"]        = 1 if pm == "Electronic check" else 0
    row["PaymentMethod_mailed check"]            = 1 if pm == "Mailed check" else 0

    # Derived features (same logic as 02_data_cleaning notebook)
    tenure = max(row["tenure"], 1)
    row["AvgMonthlyCharges"] = row["TotalCharges"] / tenure

    # TenureGroup buckets
    row["TenureGroup_1-2yr"] = 1 if 12 <= tenure < 24 else 0
    row["TenureGroup_2-4yr"] = 1 if 24 <= tenure < 48 else 0
    row["TenureGroup_4-6yr"] = 1 if 48 <= tenure      else 0

    # TotalServices = count of active services
    service_flags = [
        row["PhoneService"],
        row["MultipleLines_yes"],
        row["OnlineSecurity_yes"],
        row["OnlineBackup_yes"],
        row["DeviceProtection_yes"],
        row["TechSupport_yes"],
        row["StreamingTV_yes"],
        row["StreamingMovies_yes"],
    ]
    row["TotalServices"] = sum(service_flags)

    # Build DataFrame in exact training column order
    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return df


# ── Public API ──────────────────────────────────────────────────────────────
def predict_churn(customer_data: dict) -> dict:
    """
    Predict churn probability for a single customer.

    Parameters
    ----------
    customer_data : dict
        Human-readable customer attributes (see CUSTOMER_FIELDS).

    Returns
    -------
    dict with keys:
        churn_probability  – float 0-1
        risk_level         – "Low" / "Medium" / "High" / "Very High"
        risk_factors       – list[str] key drivers for this customer
        recommendation     – str short retention suggestion
    """
    models, scaler, metadata = _load_artifacts()
    if scaler is None:
        raise RuntimeError("Scaler failed to load from artifacts")
    features = _encode_customer(customer_data)
    features_scaled = pd.DataFrame(
        scaler.transform(features), columns=features.columns
    )

    # Soft voting: average predicted probabilities from 3 models
    probas = np.array([
        m.predict_proba(features_scaled)[0][1]
        for m in models.values()
    ])
    proba = float(probas.mean())

    # Risk level
    if proba < 0.25:
        risk = "Low"
    elif proba < 0.50:
        risk = "Medium"
    elif proba < 0.75:
        risk = "High"
    else:
        risk = "Very High"

    # Detect key risk factors using SHAP (model-based),
    # with fallback to manual rules if SHAP fails.
    try:
        risk_factors = _detect_risk_factors_shap(features_scaled.values, features)
        risk_source = "shap"
    except Exception:
        risk_factors = _detect_risk_factors(customer_data, proba)
        risk_source = "manual"

    return {
        "churn_probability": round(float(proba), 4),
        "risk_level": risk,
        "risk_factors": risk_factors,
        "risk_source": risk_source,
        "customer_summary": _build_customer_summary(customer_data),
    }


# ── SHAP-based risk factor detection ────────────────────────────────────────

# Human-readable names for each encoded feature column.
_FEATURE_LABELS: dict[str, str] = {
    "gender":                                "Gender (Male)",
    "SeniorCitizen":                         "Senior citizen",
    "Partner":                               "Has partner",
    "Dependents":                            "Has dependents",
    "tenure":                                "Customer tenure",
    "PhoneService":                          "Phone service",
    "PaperlessBilling":                      "Paperless billing",
    "MonthlyCharges":                        "Monthly charges",
    "TotalCharges":                          "Total charges",
    "MultipleLines_no phone service":        "No phone service (multiple lines)",
    "MultipleLines_yes":                     "Multiple lines",
    "OnlineSecurity_no internet service":    "No internet service (online security)",
    "OnlineSecurity_yes":                    "Online security",
    "OnlineBackup_no internet service":      "No internet service (online backup)",
    "OnlineBackup_yes":                      "Online backup",
    "DeviceProtection_no internet service":  "No internet service (device protection)",
    "DeviceProtection_yes":                  "Device protection",
    "TechSupport_no internet service":       "No internet service (tech support)",
    "TechSupport_yes":                       "Tech support",
    "StreamingTV_no internet service":       "No internet service (streaming TV)",
    "StreamingTV_yes":                       "Streaming TV",
    "StreamingMovies_no internet service":   "No internet service (streaming movies)",
    "StreamingMovies_yes":                   "Streaming movies",
    "InternetService_fiber optic":           "Fiber optic internet",
    "InternetService_no":                    "No internet service",
    "Contract_one year":                     "One-year contract",
    "Contract_two year":                     "Two-year contract",
    "PaymentMethod_credit card (automatic)": "Credit card payment (automatic)",
    "PaymentMethod_electronic check":        "Electronic check payment",
    "PaymentMethod_mailed check":            "Mailed check payment",
    "AvgMonthlyCharges":                     "Average monthly charges",
    "TenureGroup_1-2yr":                     "Tenure 1-2 years",
    "TenureGroup_2-4yr":                     "Tenure 2-4 years",
    "TenureGroup_4-6yr":                     "Tenure 4-6 years",
    "TotalServices":                         "Total active services",
}


def _format_shap_factor(feature: str, shap_value: float, feat_value: float) -> str:
    """Convert a SHAP result into a human-readable risk factor string.

    Parameters
    ----------
    feature    : encoded column name (e.g. "Contract_two year")
    shap_value : SHAP contribution (positive = pushes toward churn)
    feat_value : raw feature value for this customer
    """
    label = _FEATURE_LABELS.get(feature, feature)
    pct = abs(shap_value) * 100
    direction = "increases" if shap_value > 0 else "decreases"

    # For binary features interpret the 0/1 value
    if feat_value in (0.0, 1.0) and feature not in (
        "tenure", "MonthlyCharges", "TotalCharges",
        "AvgMonthlyCharges", "TotalServices",
    ):
        status = "Yes" if feat_value == 1 else "No"
        return f"{label}: {status} — {direction} risk by {pct:.1f}%"

    # For numeric features include the value
    if feature == "tenure":
        return f"{label}: {int(feat_value)} months — {direction} risk by {pct:.1f}%"
    if feature in ("MonthlyCharges", "TotalCharges", "AvgMonthlyCharges"):
        return f"{label}: ${feat_value:,.2f} — {direction} risk by {pct:.1f}%"
    if feature == "TotalServices":
        return f"{label}: {int(feat_value)} — {direction} risk by {pct:.1f}%"

    return f"{label} — {direction} risk by {pct:.1f}%"


def _detect_risk_factors_shap(
    features_scaled: np.ndarray,
    features_df: pd.DataFrame,
    top_n: int = settings.SHAP_TOP_N,
) -> list[str]:
    """Use SHAP to find the features that most influence THIS prediction.

    Unlike the manual heuristic rules, this reflects what the model
    actually learned and is personalised to each individual customer.

    Parameters
    ----------
    features_scaled : scaled feature array (output of scaler.transform)
    features_df     : unscaled DataFrame (to display original values)
    top_n           : how many top risk factors to return

    Returns
    -------
    list[str]  human-readable risk factor descriptions
    """
    explainer = _get_explainer()
    shap_values = explainer.shap_values(features_scaled)

    # Handle models that return [class_0_shap, class_1_shap]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Take Churn (class 1) contributions

    # Build a DataFrame of feature impacts
    impacts = pd.DataFrame({
        "feature": features_df.columns,
        "shap_value": shap_values[0],                   # SHAP for this row
        "feat_value": features_df.iloc[0].values,        # Original values
    })

    # Sort by absolute impact (most influential first)
    impacts["abs_impact"] = impacts["shap_value"].abs()
    impacts = impacts.sort_values("abs_impact", ascending=False)

    # Take the top_n most impactful features
    top = impacts.head(top_n)

    factors = [
        _format_shap_factor(row["feature"], row["shap_value"], row["feat_value"])
        for _, row in top.iterrows()
    ]
    return factors


# ── Fallback: manual heuristic rules ────────────────────────────────────────

def _detect_risk_factors(data: dict, proba: float) -> list[str]:
    """Heuristic risk-factor detection based on known EDA insights.

    Kept as a fallback in case SHAP computation fails.
    """
    factors = []
    if data.get("Contract", "") == "Month-to-month":
        factors.append("Month-to-month contract (highest churn segment)")
    if data.get("InternetService", "") == "Fiber optic":
        factors.append("Fiber optic internet (higher churn vs DSL)")
    if data.get("PaymentMethod", "") == "Electronic check":
        factors.append("Electronic check payment (correlated with churn)")
    if float(data.get("tenure", 0)) < 12:
        factors.append(f"Short tenure ({data.get('tenure', 0)} months)")
    if float(data.get("MonthlyCharges", 0)) > 70:
        factors.append(f"High monthly charges (${data.get('MonthlyCharges', 0)})")
    if data.get("TechSupport", "No") == "No" and data.get("InternetService", "") != "No":
        factors.append("No tech support")
    if data.get("OnlineSecurity", "No") == "No" and data.get("InternetService", "") != "No":
        factors.append("No online security")
    if data.get("PaperlessBilling", "No") == "Yes":
        factors.append("Paperless billing enabled")
    if data.get("Dependents", "No") == "No" and data.get("Partner", "No") == "No":
        factors.append("No partner or dependents (less sticky)")
    return factors


def _build_customer_summary(data: dict) -> str:
    """One-paragraph summary of customer profile for LLM context."""
    tenure = data.get("tenure", 0)
    monthly = data.get("MonthlyCharges", 0)
    total = data.get("TotalCharges", 0)
    contract = data.get("Contract", "Month-to-month")
    inet = data.get("InternetService", "No")
    return (
        f"Customer profile: tenure={tenure} months, contract={contract}, "
        f"internet={inet}, monthly=${monthly}, total=${total}, "
        f"payment={data.get('PaymentMethod','N/A')}, "
        f"services: phone={data.get('PhoneService','No')}, "
        f"security={data.get('OnlineSecurity','No')}, "
        f"backup={data.get('OnlineBackup','No')}, "
        f"techsupport={data.get('TechSupport','No')}, "
        f"streaming={data.get('StreamingTV','No')}/{data.get('StreamingMovies','No')}."
    )


def get_model_metadata() -> dict[str, Any]:
    """Return model metadata (performance, training date, etc.)."""
    _, _, metadata = _load_artifacts()
    if metadata is None:
        raise RuntimeError("Model metadata not loaded")
    return metadata
