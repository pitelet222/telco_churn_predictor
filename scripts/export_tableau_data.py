"""
Export a Tableau-ready CSV with human-readable labels + engineered features + model predictions.

Run from project root:
    venv/Scripts/python.exe scripts/export_tableau_data.py
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import joblib
import numpy as np
import pandas as pd

# ── 1. Load raw data (human-readable labels) ────────────────────────────────
raw = pd.read_csv("data/raw/teclo_custommer_churn.csv")

# Fix TotalCharges (some are whitespace → NaN) and fill with 0
raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0)

# Convert Churn to numeric for calculations
raw["Churn_Flag"] = (raw["Churn"] == "Yes").astype(int)

# ── 2. Add engineered features ──────────────────────────────────────────────
# AvgMonthlyCharges
raw["AvgMonthlyCharges"] = raw["TotalCharges"] / raw["tenure"].replace(0, 1)

# TenureGroup (human-readable)
bins = [0, 12, 24, 48, 72]
labels = ["0-1 Year", "1-2 Years", "2-4 Years", "4-6 Years"]
raw["TenureGroup"] = pd.cut(raw["tenure"], bins=bins, labels=labels)

# TotalServices count
service_cols = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
raw["TotalServices"] = raw[service_cols].apply(
    lambda row: sum(1 for v in row if v == "Yes"), axis=1
)

# ── 3. Add model predictions ────────────────────────────────────────────────
# Load the processed (encoded) data and models to generate predictions
processed = pd.read_csv("data/processed/telco_churn_cleaned.csv")
X = processed.drop("Churn", axis=1)
scaler = joblib.load("models/scaler.pkl")
X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

# Load ensemble component models
lr = joblib.load("models/logistic_regression.pkl")
gb = joblib.load("models/gradient_boosting.pkl")
cb = joblib.load("models/catboost_model.pkl")

# Soft-voting probability
proba = np.mean([
    lr.predict_proba(X_scaled)[:, 1],
    gb.predict_proba(X_scaled)[:, 1],
    cb.predict_proba(X_scaled)[:, 1],
], axis=0)

raw["Churn_Probability"] = proba
raw["Predicted_Risk"] = pd.cut(
    proba,
    bins=[0, 0.3, 0.6, 1.0],
    labels=["Low Risk", "Medium Risk", "High Risk"],
)

# ── 4. Add customer value segment ───────────────────────────────────────────
raw["MonthlyCharges_Segment"] = pd.cut(
    raw["MonthlyCharges"],
    bins=[0, 35, 70, 120],
    labels=["Low Spender", "Mid Spender", "High Spender"],
)

# ── 5. Export ────────────────────────────────────────────────────────────────
output_dir = os.path.join(ROOT, "data", "tableau")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "telco_churn_tableau.csv")

raw.to_csv(output_path, index=False)
print(f"Exported {raw.shape[0]} rows × {raw.shape[1]} columns to:")
print(f"  {output_path}")
print(f"\nColumns: {list(raw.columns)}")
