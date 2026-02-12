"""
Model evaluation utilities for Telco Churn.

Provides functions to compute classification metrics, generate comparison
tables, and print formatted reports. Used by both the training pipeline
(src/train.py) and the notebooks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ── Single-model evaluation ─────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute standard classification metrics for a fitted model.

    Parameters
    ----------
    model : fitted sklearn-compatible estimator
    X_test : test features (already scaled)
    y_test : true labels
    threshold : decision threshold for converting probabilities to classes

    Returns
    -------
    dict with keys: Accuracy, Precision, Recall, F1-Score, ROC-AUC
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_proba)),
    }


def get_confusion_matrix(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return the confusion matrix for a fitted model."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return confusion_matrix(y_test, y_pred)


def get_classification_report(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> str:
    """Return sklearn's classification report as a string."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    return str(classification_report(y_test, y_pred, target_names=["Not Churned", "Churned"], output_dict=False))


def get_roc_curve(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fpr, tpr, thresholds) for plotting the ROC curve."""
    y_proba = model.predict_proba(X_test)[:, 1]
    return roc_curve(y_test, y_proba)


# ── Multi-model comparison ──────────────────────────────────────────────────

def compare_models(
    models: dict[str, Any],
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Evaluate multiple models and return a sorted comparison DataFrame.

    Parameters
    ----------
    models : dict  {model_name: fitted_model}
    X_test : test features (scaled)
    y_test : true labels

    Returns
    -------
    pd.DataFrame sorted by ROC-AUC descending.
    """
    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        row = {"Model": name, **metrics}
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[["Model", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]]
    return df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


# ── Formatted printing ──────────────────────────────────────────────────────

def print_evaluation_report(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    model_name: str = "Model",
) -> None:
    """Print a formatted evaluation report for a single model."""
    metrics = evaluate_model(model, X_test, y_test)
    cm = get_confusion_matrix(model, X_test, y_test)

    print(f"\n{'=' * 60}")
    print(f"  {model_name} – Evaluation Report")
    print(f"{'=' * 60}")
    print(f"  Accuracy:   {metrics['Accuracy']:.4f}")
    print(f"  Precision:  {metrics['Precision']:.4f}")
    print(f"  Recall:     {metrics['Recall']:.4f}")
    print(f"  F1-Score:   {metrics['F1-Score']:.4f}")
    print(f"  ROC-AUC:    {metrics['ROC-AUC']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
    print(f"  FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")
    print(f"{'=' * 60}\n")


def print_comparison_table(comparison_df: pd.DataFrame) -> None:
    """Pretty-print the model comparison DataFrame with ranking medals."""
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON (sorted by ROC-AUC)")
    print("=" * 80 + "\n")

    medals = ["🥇", "🥈", "🥉"] + ["  "] * 10
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        print(
            f"  {medals[i]} {row['Model']:25s} | "
            f"AUC: {row['ROC-AUC']:.4f} | "
            f"Acc: {row['Accuracy']:.4f} | "
            f"F1: {row['F1-Score']:.4f}"
        )
    print("\n" + "=" * 80)
