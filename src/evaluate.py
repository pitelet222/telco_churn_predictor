"""
Model evaluation utilities for Telco Churn.

Provides functions to compute classification metrics, generate comparison
tables, and print formatted reports. Used by both the training pipeline
(src/train.py) and the notebooks.

With a churn rate of ~26.5%, accuracy is a misleading headline metric (a
trivial "nobody churns" classifier already scores ~73.5%). PR-AUC (average
precision) is reported alongside ROC-AUC because it is more informative for
an imbalanced positive class, and calibration (Brier score) matters because
predicted probabilities are consumed directly by the API and the LLM advisor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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
    dict with keys: Accuracy, Precision, Recall, F1-Score, ROC-AUC, PR-AUC, Brier
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_test, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_test, y_proba)),
        "PR-AUC": float(average_precision_score(y_test, y_proba)),
        "Brier": float(brier_score_loss(y_test, y_proba)),
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


def get_pr_curve(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, thresholds) for plotting the PR curve.

    More informative than the ROC curve here because the positive class
    (Churn=1) is the minority class (~26.5%).
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    return precision_recall_curve(y_test, y_proba)


def get_calibration_curve(
    model: Any,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (prob_true, prob_pred) reliability points for a calibration plot.

    Compare against the diagonal y=x: points below it mean the model is
    over-confident (predicted probability higher than the observed
    frequency), which is common for tree-based models like Gradient
    Boosting / CatBoost.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    return calibration_curve(y_test, y_proba, n_bins=n_bins)


def find_optimal_threshold(
    y_test: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    cost_fn: float = 5.0,
    cost_fp: float = 1.0,
) -> dict[str, float]:
    """Find the decision threshold that minimizes business cost.

    A false negative (missed churner) is assumed to be more expensive than
    a false positive (an unnecessary retention offer): industry estimates
    place the cost of acquiring a new customer at 5-25x the cost of
    retaining an existing one, so ``cost_fn`` defaults to the conservative
    low end of that range (5x ``cost_fp``).

    Parameters
    ----------
    y_test : true labels
    y_proba : predicted probabilities of the positive class
    cost_fn : relative cost of a false negative
    cost_fp : relative cost of a false positive

    Returns
    -------
    dict with keys: threshold, total_cost, precision, recall, f1
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best: dict[str, float] | None = None

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        total_cost = cost_fn * fn + cost_fp * fp

        if best is None or total_cost < best["total_cost"]:
            best = {
                "threshold": float(t),
                "total_cost": float(total_cost),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            }

    assert best is not None
    return best


# ── Bias-variance / regularization diagnostics ──────────────────────────────

def compute_overfitting_gap(
    model: Any,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Compare train vs. test ROC-AUC / PR-AUC to flag overfitting.

    A large positive gap (train >> test) indicates the model has memorized
    the training data rather than learning generalizable patterns. With
    5-fold stratified CV already in place, this gap is a cheap additional
    signal to track run-over-run.

    Returns
    -------
    dict with keys: train_roc_auc, test_roc_auc, roc_auc_gap,
    train_pr_auc, test_pr_auc, pr_auc_gap
    """
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test = model.predict_proba(X_test)[:, 1]

    train_roc_auc = roc_auc_score(y_train, proba_train)
    test_roc_auc = roc_auc_score(y_test, proba_test)
    train_pr_auc = average_precision_score(y_train, proba_train)
    test_pr_auc = average_precision_score(y_test, proba_test)

    return {
        "train_roc_auc": float(train_roc_auc),
        "test_roc_auc": float(test_roc_auc),
        "roc_auc_gap": float(train_roc_auc - test_roc_auc),
        "train_pr_auc": float(train_pr_auc),
        "test_pr_auc": float(test_pr_auc),
        "pr_auc_gap": float(train_pr_auc - test_pr_auc),
    }


def l1_feature_selection_report(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    C: float = 1.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit an L1-penalized Logistic Regression to report embedded feature selection.

    L1 regularization drives uninformative coefficients to exactly zero,
    giving a quick read on which of the 35 engineered features are
    redundant. This is a diagnostic only — it does not change the deployed
    Logistic Regression, which uses L2 (see ``MODEL_CONFIGS`` in
    ``src/train.py``).

    Returns
    -------
    dict with keys: C, n_features_total, n_features_selected,
    selected_features, zeroed_features
    """
    model = LogisticRegression(
        penalty="l1", solver="liblinear", C=C, max_iter=1000, random_state=random_state,
    )
    model.fit(X_train, y_train)

    coefs = model.coef_[0]
    zeroed = [col for col, coef in zip(X_train.columns, coefs) if coef == 0.0]
    selected = [col for col in X_train.columns if col not in zeroed]

    return {
        "C": C,
        "n_features_total": X_train.shape[1],
        "n_features_selected": len(selected),
        "selected_features": selected,
        "zeroed_features": zeroed,
    }


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
    pd.DataFrame sorted by PR-AUC descending (more informative than ROC-AUC
    for the minority churn class).
    """
    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, X_test, y_test)
        row = {"Model": name, **metrics}
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df[["Model", "PR-AUC", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score", "Brier"]]
    return df.sort_values("PR-AUC", ascending=False).reset_index(drop=True)


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
    print(f"  PR-AUC:     {metrics['PR-AUC']:.4f}")
    print(f"  Brier:      {metrics['Brier']:.4f}  (lower = better calibrated)")
    print(f"\n  Confusion Matrix:")
    print(f"  TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
    print(f"  FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")
    print(f"{'=' * 60}\n")


def print_comparison_table(comparison_df: pd.DataFrame) -> None:
    """Pretty-print the model comparison DataFrame with ranking medals."""
    print("\n" + "=" * 80)
    print("  MODEL COMPARISON (sorted by PR-AUC)")
    print("=" * 80 + "\n")

    medals = ["🥇", "🥈", "🥉"] + ["  "] * 10
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        print(
            f"  {medals[i]} {row['Model']:25s} | "
            f"PR-AUC: {row['PR-AUC']:.4f} | "
            f"ROC-AUC: {row['ROC-AUC']:.4f} | "
            f"Acc: {row['Accuracy']:.4f} | "
            f"F1: {row['F1-Score']:.4f}"
        )
    print("\n" + "=" * 80)
