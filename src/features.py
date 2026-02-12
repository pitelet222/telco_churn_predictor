"""
Feature engineering module for Telco Churn.

Creates derived features from the cleaned dataset:
- AvgMonthlyCharges: average spending per month
- TenureGroup: binned tenure into 4 buckets
- TotalServices: count of active service subscriptions

These mirror the exact transformations in notebooks/02_data_cleaning.ipynb.
"""

from __future__ import annotations

import pandas as pd


# ── Tenure bucket boundaries ────────────────────────────────────────────────
TENURE_BINS = [0, 12, 24, 48, 72]
TENURE_LABELS = ["0-1yr", "1-2yr", "2-4yr", "4-6yr"]

# Service columns whose "_yes" dummies count towards TotalServices
SERVICE_YES_SUFFIXES = [
    "PhoneService",
    "MultipleLines_yes",
    "OnlineSecurity_yes",
    "OnlineBackup_yes",
    "DeviceProtection_yes",
    "TechSupport_yes",
    "StreamingTV_yes",
    "StreamingMovies_yes",
]


def add_avg_monthly_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Create AvgMonthlyCharges = TotalCharges / tenure (safe for tenure=0)."""
    df = df.copy()
    df["AvgMonthlyCharges"] = df["TotalCharges"] / df["tenure"].replace(0, 1)
    return df


def add_tenure_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Bin tenure into categorical groups and one-hot encode them (drop_first)."""
    df = df.copy()
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS
    )
    df = pd.get_dummies(df, columns=["TenureGroup"], drop_first=True, dtype=int)
    return df


def add_total_services(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of active services (columns ending in '_yes' + PhoneService)."""
    df = df.copy()
    yes_cols = [c for c in SERVICE_YES_SUFFIXES if c in df.columns]
    if yes_cols:
        df["TotalServices"] = df[yes_cols].sum(axis=1)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and encoded DataFrame (output of data_preprocessing.preprocess_dataframe).

    Returns
    -------
    pd.DataFrame with 3 new derived features added.
    """
    df = add_avg_monthly_charges(df)
    df = add_tenure_groups(df)
    df = add_total_services(df)
    return df
