"""
Training pipeline for Telco Churn models.

Loads the processed data, splits, scales, trains 6 models + an ensemble,
and persists the best artifacts to the models/ directory.

Can be run as a script:
    python -m src.train
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.utils import ClassifierTags, Tags, TargetTags

from src.evaluate import (
    compute_overfitting_gap,
    evaluate_model,
    find_optimal_threshold,
    l1_feature_selection_report,
)

# catboost 1.2.8 predates scikit-learn's __sklearn_tags__ API (sklearn >= 1.6),
# so is_classifier(CatBoostClassifier()) fails with AttributeError. This is
# needed for CalibratedClassifierCV (see calibrate_models) to accept CatBoost.
if not hasattr(CatBoostClassifier, "__sklearn_tags__"):
    CatBoostClassifier.__sklearn_tags__ = lambda self: Tags(
        estimator_type="classifier",
        target_tags=TargetTags(required=True),
        transformer_tags=None,
        regressor_tags=None,
        classifier_tags=ClassifierTags(),
    )
from config import settings
from log_config import get_logger

logger = get_logger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = settings.DATA_PATH
MODELS_DIR = settings.MODELS_DIR

# ── Default hyper-parameters (same as notebook 04) ───────────────────────────
MODEL_CONFIGS: dict[str, tuple] = {
    "Logistic Regression": (
        LogisticRegression,
        # L2 (ridge) regularization: shrinks coefficients without zeroing them
        # out. C is the inverse regularization strength (lower C → stronger
        # regularization). See l1_feature_selection_report() in
        # src/evaluate.py for an L1-based feature-selection diagnostic.
        {"random_state": 42, "max_iter": 1000, "n_jobs": -1, "C": 1.0, "penalty": "l2"},
    ),
    "Random Forest": (
        RandomForestClassifier,
        {"n_estimators": 100, "max_depth": 15, "random_state": 42, "n_jobs": -1},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier,
        # Tuned via src/tune.py (Optuna, 30 trials, 5-fold CV PR-AUC):
        # 0.6576 -> 0.6684. More regularized than the prior defaults
        # (fewer/shallower trees, subsampling, larger leaves).
        {"n_estimators": 50, "max_depth": 4, "learning_rate": 0.0888,
         "subsample": 0.8656, "min_samples_leaf": 28,
         "random_state": 42, "n_iter_no_change": 10},
    ),
    "XGBoost": (
        xgb.XGBClassifier,
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1,
         "random_state": 42, "eval_metric": "logloss", "verbosity": 0},
    ),
    "LightGBM": (
        lgb.LGBMClassifier,
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1,
         "random_state": 42, "verbose": -1, "num_leaves": 31},
    ),
    "CatBoost": (
        CatBoostClassifier,
        # Tuned via src/tune.py (Optuna, 30 trials, 5-fold CV PR-AUC):
        # 0.6651 -> 0.6707. More iterations but shallower trees and
        # stronger L2 than the prior defaults.
        {"iterations": 220, "depth": 3, "learning_rate": 0.0366,
         "l2_leaf_reg": 8.0883, "random_state": 42, "verbose": False,
         "allow_writing_files": False},
    ),
}

# Models selected for the soft-voting ensemble (top 3 by ROC-AUC)
ENSEMBLE_COMPONENTS = ["Logistic Regression", "Gradient Boosting", "CatBoost"]

# Tree ensembles tend to be overconfident; calibrate their probabilities with
# isotonic regression before they feed the soft-voting ensemble, the API, and
# the LLM advisor. Logistic Regression is already well-calibrated by design.
CALIBRATED_COMPONENTS = ["Gradient Boosting", "CatBoost"]


# ── Core functions ───────────────────────────────────────────────────────────

def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load the processed CSV and split into features (X) and target (y)."""
    df = pd.read_csv(path)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y


def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = settings.TEST_SIZE,
    random_state: int = settings.RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Stratified train/test split + StandardScaler (fitted on train only).

    Returns
    -------
    X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_single_model(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple:
    """Instantiate and fit a single model from MODEL_CONFIGS.

    Returns
    -------
    (fitted_model, training_time_seconds)
    """
    cls, params = MODEL_CONFIGS[name]
    model = cls(**params)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, tuple]:
    """Train every model in MODEL_CONFIGS.

    Returns
    -------
    dict  {model_name: (fitted_model, train_time)}
    """
    trained = {}
    for name in MODEL_CONFIGS:
        logger.info("Training %s...", name)
        model, elapsed = train_single_model(name, X_train, y_train)
        logger.info("Trained %s in %.2fs", name, elapsed)
        trained[name] = (model, elapsed)
    return trained


def calibrate_models(
    trained: dict[str, tuple],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    names: list[str] = CALIBRATED_COMPONENTS,
) -> tuple[dict[str, tuple], dict[str, dict[str, float]]]:
    """Calibrate probability outputs of tree ensembles with isotonic regression.

    Returns a copy of ``trained`` where each model in ``names`` is replaced
    by a ``CalibratedClassifierCV`` (5-fold isotonic), plus a report of the
    Brier score before/after calibration for each calibrated model.
    """
    calibrated = dict(trained)
    report: dict[str, dict[str, float]] = {}

    for name in names:
        raw_model, train_time = trained[name]
        brier_before = brier_score_loss(y_test, raw_model.predict_proba(X_test)[:, 1])

        cls, params = MODEL_CONFIGS[name]
        calibrated_model = CalibratedClassifierCV(cls(**params), method="isotonic", cv=5)
        calibrated_model.fit(X_train, y_train)
        brier_after = brier_score_loss(y_test, calibrated_model.predict_proba(X_test)[:, 1])

        calibrated[name] = (calibrated_model, train_time)
        report[name] = {"brier_before": float(brier_before), "brier_after": float(brier_after)}
        logger.info(
            "Calibrated %s — Brier %.4f -> %.4f", name, brier_before, brier_after
        )

    return calibrated, report


def ensemble_predict_proba(
    models: dict,
    X: pd.DataFrame,
    component_names: list[str] = ENSEMBLE_COMPONENTS,
) -> np.ndarray:
    """Soft-voting: average P(Churn=1) from selected component models."""
    probas = np.array([
        models[name][0].predict_proba(X)[:, 1] for name in component_names
    ])
    return probas.mean(axis=0)


def cross_validate_ensemble(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = settings.CV_FOLDS,
) -> np.ndarray:
    """5-fold stratified CV on the soft-voting ensemble."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=settings.RANDOM_STATE)
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_cv_train, X_cv_val = X.iloc[train_idx], X.iloc[val_idx]
        y_cv_train, y_cv_val = y.iloc[train_idx], y.iloc[val_idx]

        # Re-fit ensemble components on CV fold
        fold_models = {}
        for name in ENSEMBLE_COMPONENTS:
            cls, params = MODEL_CONFIGS[name]
            m = cls(**params)
            m.fit(X_cv_train, y_cv_train)
            fold_models[name] = (m, 0)

        proba = ensemble_predict_proba(fold_models, X_cv_val)
        scores.append(roc_auc_score(y_cv_val, proba))

    return np.array(scores)


# ── Persistence ──────────────────────────────────────────────────────────────

def save_artifacts(
    models: dict,
    scaler: StandardScaler,
    ensemble_metrics: dict,
    cv_scores: np.ndarray,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    calibration_report: dict | None = None,
    threshold_analysis: dict | None = None,
    regularization: dict | None = None,
    l1_feature_selection: dict | None = None,
    per_model_gaps: dict | None = None,
    output_dir: Path = MODELS_DIR,
) -> None:
    """Save ensemble component models, scaler, and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save ensemble components
    artifact_map = {
        "Logistic Regression": "logistic_regression.pkl",
        "Gradient Boosting": "gradient_boosting.pkl",
        "CatBoost": "catboost_model.pkl",
    }
    for name, filename in artifact_map.items():
        joblib.dump(models[name][0], output_dir / filename)
        logger.info("Saved %s", filename)

    # Scaler
    joblib.dump(scaler, output_dir / "scaler.pkl")
    logger.info("Saved scaler.pkl")

    # Metadata
    metadata = {
        "model_name": "Telco Customer Churn - Ensemble Predictor",
        "model_type": "Soft Voting Ensemble",
        "ensemble_components": ENSEMBLE_COMPONENTS,
        "weights": [round(1 / len(ENSEMBLE_COMPONENTS), 6)] * len(ENSEMBLE_COMPONENTS),
        "voting_method": "soft (probability averaging)",
        "calibration": calibration_report or {},
        **ensemble_metrics,
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "threshold_analysis": threshold_analysis or {},
        "regularization": regularization or {},
        "l1_feature_selection": l1_feature_selection or {},
        "per_model_train_test_gap": per_model_gaps or {},
        "train_set_size": len(X_train),
        "test_set_size": len(X_test),
        "n_features": X_train.shape[1],
        "feature_scaling": "StandardScaler (fitted on training data)",
        "target_variable": "Churn",
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notes": "Ensemble selected over individual models due to marginally better generalization",
    }
    with open(output_dir / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info("Saved model_metadata.json")


# ── Main entry point ─────────────────────────────────────────────────────────

def run_training_pipeline() -> None:
    """Execute the full training pipeline end-to-end."""
    logger.info("=" * 70)
    logger.info("TELCO CHURN – TRAINING PIPELINE")
    logger.info("=" * 70)

    # 1. Load data
    logger.info("Loading data...")
    X, y = load_data()
    logger.info("%d records, %d features", X.shape[0], X.shape[1])

    # 2. Split & scale
    logger.info("Splitting & scaling...")
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    # 3. Train all models
    logger.info("Training models...")
    trained = train_all_models(X_train, y_train)

    # 4. Evaluate each model (test metrics + train-test gap for overfitting)
    logger.info("Evaluating models...")
    results = []
    gap_reports = {}
    for name, (model, train_time) in trained.items():
        metrics = evaluate_model(model, X_test, y_test)
        gap = compute_overfitting_gap(model, X_train, y_train, X_test, y_test)
        gap_reports[name] = gap
        result = {
            **metrics,
            "Model": name,
            "Training Time": train_time,
            "ROC-AUC Gap": gap["roc_auc_gap"],
            "PR-AUC Gap": gap["pr_auc_gap"],
        }
        results.append(result)

    results_df = pd.DataFrame(results).sort_values("PR-AUC", ascending=False)
    logger.info("Model comparison:\n%s", results_df.to_string(index=False))

    # 5. Calibration analysis (isotonic regression) — reported, not yet deployed.
    # The saved gradient_boosting.pkl / catboost_model.pkl stay uncalibrated
    # because app/churn_service.py builds a shap.TreeExplainer directly from
    # these artifacts, which requires the raw tree estimator.
    logger.info("Checking calibration of %s...", ", ".join(CALIBRATED_COMPONENTS))
    _, calibration_report = calibrate_models(trained, X_train, y_train, X_test, y_test)

    # 6. Ensemble evaluation
    logger.info("Evaluating ensemble...")
    proba_ensemble = ensemble_predict_proba(trained, X_test)
    proba_ensemble_train = ensemble_predict_proba(trained, X_train)
    y_pred_ensemble = (proba_ensemble >= settings.CHURN_THRESHOLD).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ensemble_train_roc_auc = float(roc_auc_score(y_train, proba_ensemble_train))
    ensemble_train_pr_auc = float(average_precision_score(y_train, proba_ensemble_train))
    ensemble_test_roc_auc = float(roc_auc_score(y_test, proba_ensemble))
    ensemble_test_pr_auc = float(average_precision_score(y_test, proba_ensemble))
    ensemble_metrics = {
        "test_roc_auc": ensemble_test_roc_auc,
        "test_pr_auc": ensemble_test_pr_auc,
        "test_brier": float(brier_score_loss(y_test, proba_ensemble)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_ensemble)),
        "test_precision": float(precision_score(y_test, y_pred_ensemble, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred_ensemble, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred_ensemble, zero_division=0)),
        "train_roc_auc": ensemble_train_roc_auc,
        "train_pr_auc": ensemble_train_pr_auc,
        "roc_auc_gap": ensemble_train_roc_auc - ensemble_test_roc_auc,
        "pr_auc_gap": ensemble_train_pr_auc - ensemble_test_pr_auc,
    }
    logger.info(
        "Ensemble ROC-AUC: %.4f | PR-AUC: %.4f | Brier: %.4f",
        ensemble_metrics["test_roc_auc"],
        ensemble_metrics["test_pr_auc"],
        ensemble_metrics["test_brier"],
    )
    logger.info(
        "Ensemble train-test gap — ROC-AUC: %.4f | PR-AUC: %.4f",
        ensemble_metrics["roc_auc_gap"],
        ensemble_metrics["pr_auc_gap"],
    )

    # 7. Cost-based threshold optimization
    threshold_analysis = find_optimal_threshold(
        y_test,
        proba_ensemble,
        cost_fn=settings.COST_FALSE_NEGATIVE,
        cost_fp=settings.COST_FALSE_POSITIVE,
    )
    logger.info(
        "Optimal threshold (cost FN=%.1f, FP=%.1f): %.2f "
        "(precision=%.4f, recall=%.4f, f1=%.4f) vs configured CHURN_THRESHOLD=%.2f",
        settings.COST_FALSE_NEGATIVE,
        settings.COST_FALSE_POSITIVE,
        threshold_analysis["threshold"],
        threshold_analysis["precision"],
        threshold_analysis["recall"],
        threshold_analysis["f1"],
        settings.CHURN_THRESHOLD,
    )

    # 8. Cross-validation
    logger.info("Cross-validating ensemble (%d-fold)...", settings.CV_FOLDS)
    cv_scores = cross_validate_ensemble(trained, X_train, y_train)
    logger.info("CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # 9. L1 feature-selection diagnostic (separate from the deployed L2 model)
    lr_params = MODEL_CONFIGS["Logistic Regression"][1]
    l1_report = l1_feature_selection_report(
        X_train, y_train, C=lr_params.get("C", 1.0), random_state=settings.RANDOM_STATE
    )
    logger.info(
        "L1 feature selection: %d/%d features kept (C=%.2f)",
        l1_report["n_features_selected"], l1_report["n_features_total"], l1_report["C"],
    )
    if l1_report["zeroed_features"]:
        logger.info("L1 zeroed out: %s", ", ".join(l1_report["zeroed_features"]))

    regularization = {
        "Logistic Regression": {
            "C": lr_params.get("C", 1.0),
            "penalty": lr_params.get("penalty", "l2"),
        }
    }

    # 10. Save
    logger.info("Saving artifacts...")
    save_artifacts(
        trained, scaler, ensemble_metrics, cv_scores, X_train, X_test,
        calibration_report=calibration_report, threshold_analysis=threshold_analysis,
        regularization=regularization, l1_feature_selection=l1_report,
        per_model_gaps=gap_reports,
    )

    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
