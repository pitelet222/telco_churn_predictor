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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from src.evaluate import evaluate_model

import sys
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

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
        {"random_state": 42, "max_iter": 1000, "n_jobs": -1},
    ),
    "Random Forest": (
        RandomForestClassifier,
        {"n_estimators": 100, "max_depth": 15, "random_state": 42, "n_jobs": -1},
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier,
        {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1,
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
        {"iterations": 100, "depth": 5, "learning_rate": 0.1,
         "random_state": 42, "verbose": False, "allow_writing_files": False},
    ),
}

# Models selected for the soft-voting ensemble (top 3 by ROC-AUC)
ENSEMBLE_COMPONENTS = ["Logistic Regression", "Gradient Boosting", "CatBoost"]


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
        **ensemble_metrics,
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
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

    # 4. Evaluate each model
    logger.info("Evaluating models...")
    results = []
    for name, (model, train_time) in trained.items():
        metrics = evaluate_model(model, X_test, y_test)
        result = {**metrics, "Model": name, "Training Time": train_time}
        results.append(result)

    results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
    logger.info("Model comparison:\n%s", results_df.to_string(index=False))

    # 5. Ensemble evaluation
    logger.info("Evaluating ensemble...")
    proba_ensemble = ensemble_predict_proba(trained, X_test)
    y_pred_ensemble = (proba_ensemble >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    ensemble_metrics = {
        "test_roc_auc": float(roc_auc_score(y_test, proba_ensemble)),
        "test_accuracy": float(accuracy_score(y_test, y_pred_ensemble)),
        "test_precision": float(precision_score(y_test, y_pred_ensemble)),
        "test_recall": float(recall_score(y_test, y_pred_ensemble)),
        "test_f1": float(f1_score(y_test, y_pred_ensemble)),
    }
    logger.info("Ensemble ROC-AUC: %.4f", ensemble_metrics["test_roc_auc"])

    # 6. Cross-validation
    logger.info("Cross-validating ensemble (%d-fold)...", settings.CV_FOLDS)
    cv_scores = cross_validate_ensemble(trained, X_train, y_train)
    logger.info("CV ROC-AUC: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())

    # 7. Save
    logger.info("Saving artifacts...")
    save_artifacts(trained, scaler, ensemble_metrics, cv_scores, X_train, X_test)

    logger.info("=" * 70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    run_training_pipeline()
