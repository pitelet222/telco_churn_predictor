"""
Hyperparameter tuning for the ensemble's component models.

Runs an Optuna (TPE) search over each component's hyperparameter space,
scoring candidates with stratified k-fold CV on PR-AUC (average precision) —
the metric used throughout src/evaluate.py because the churn class is a
~26.5% minority.

Each model's tuned score is reported next to the current MODEL_CONFIGS
baseline (same CV scoring) so the gain, if any, can be judged before
adopting the new parameters.

This is a diagnostic tool: it does NOT modify MODEL_CONFIGS or the deployed
artifacts in models/. If a result looks worth adopting, copy the
``best_params`` from models/tuned_hyperparameters.json into MODEL_CONFIGS
(src/train.py) and re-run `python -m src.train`.

Can be run as a script:
    python -m src.tune
    python -m src.tune --n-trials 50 --cv-folds 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from catboost import CatBoostClassifier

from config import settings
from log_config import get_logger
from src.train import ENSEMBLE_COMPONENTS, MODEL_CONFIGS, load_data, split_and_scale

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODELS_DIR = settings.MODELS_DIR

# PR-AUC, not ROC-AUC: matches the rationale in src/evaluate.py for the
# ~26.5% churn minority class.
SCORING = "average_precision"

DEFAULT_N_TRIALS = 30


# ── Search spaces ────────────────────────────────────────────────────────────
# Each entry: (estimator class, fixed kwargs, fn(trial) -> tunable kwargs)

def _lr_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
        "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
    }


def _gb_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=10),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 50),
    }


def _catboost_space(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 50, 300, step=10),
        "depth": trial.suggest_int("depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }


# liblinear (unlike the lbfgs default used in MODEL_CONFIGS) supports both
# l1 and l2 penalties, which is what makes "penalty" tunable here.
SEARCH_SPACES: dict[str, tuple[type, dict[str, Any], Callable[[optuna.Trial], dict[str, Any]]]] = {
    "Logistic Regression": (
        LogisticRegression,
        {"solver": "liblinear", "max_iter": 1000, "random_state": settings.RANDOM_STATE},
        _lr_space,
    ),
    "Gradient Boosting": (
        GradientBoostingClassifier,
        {"n_iter_no_change": 10, "random_state": settings.RANDOM_STATE},
        _gb_space,
    ),
    "CatBoost": (
        CatBoostClassifier,
        {"verbose": False, "allow_writing_files": False, "random_state": settings.RANDOM_STATE},
        _catboost_space,
    ),
}


# ── Core functions ───────────────────────────────────────────────────────────

def evaluate_baseline(
    name: str,
    X_train,
    y_train,
    cv_folds: int = settings.CV_FOLDS,
    random_state: int = settings.RANDOM_STATE,
) -> float:
    """Return mean CV PR-AUC for the current MODEL_CONFIGS hyperparameters."""
    cls, params = MODEL_CONFIGS[name]
    params = {k: v for k, v in params.items() if k != "n_jobs"}
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    # n_jobs left at default (sequential): CatBoost's __sklearn_tags__ monkeypatch
    # (src/train.py) doesn't survive joblib's subprocess workers, and CatBoost
    # already parallelizes internally across cores.
    scores = cross_val_score(cls(**params), X_train, y_train, cv=cv, scoring=SCORING)
    return float(scores.mean())


def tune_model(
    name: str,
    X_train,
    y_train,
    n_trials: int = DEFAULT_N_TRIALS,
    cv_folds: int = settings.CV_FOLDS,
    random_state: int = settings.RANDOM_STATE,
) -> dict[str, Any]:
    """Run an Optuna TPE search for one model and return a results dict.

    Returns
    -------
    dict with keys: model, n_trials, cv_folds, best_params,
    best_cv_pr_auc, baseline_cv_pr_auc, improvement, elapsed_seconds
    """
    cls, fixed_params, space_fn = SEARCH_SPACES[name]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = {**fixed_params, **space_fn(trial)}
        scores = cross_val_score(cls(**params), X_train, y_train, cv=cv, scoring=SCORING)
        return float(scores.mean())

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=random_state))

    start = time.time()
    study.optimize(objective, n_trials=n_trials)
    elapsed = time.time() - start

    baseline = evaluate_baseline(name, X_train, y_train, cv_folds, random_state)
    best_params = {**fixed_params, **study.best_params}

    return {
        "model": name,
        "n_trials": n_trials,
        "cv_folds": cv_folds,
        "best_params": best_params,
        "best_cv_pr_auc": float(study.best_value),
        "baseline_cv_pr_auc": baseline,
        "improvement": float(study.best_value - baseline),
        "elapsed_seconds": round(elapsed, 1),
    }


def tune_all(
    X_train,
    y_train,
    n_trials: int = DEFAULT_N_TRIALS,
    cv_folds: int = settings.CV_FOLDS,
    names: list[str] = ENSEMBLE_COMPONENTS,
) -> dict[str, dict[str, Any]]:
    """Tune every model in ``names`` and return {model_name: result_dict}."""
    results = {}
    for name in names:
        logger.info("Tuning %s (%d trials, %d-fold CV)...", name, n_trials, cv_folds)
        result = tune_model(name, X_train, y_train, n_trials=n_trials, cv_folds=cv_folds)
        results[name] = result
        logger.info(
            "%s: baseline PR-AUC %.4f -> tuned %.4f (%+.4f) in %.1fs",
            name,
            result["baseline_cv_pr_auc"],
            result["best_cv_pr_auc"],
            result["improvement"],
            result["elapsed_seconds"],
        )
    return results


# ── Persistence ──────────────────────────────────────────────────────────────

def save_results(results: dict[str, dict[str, Any]], output_dir: Path = MODELS_DIR) -> Path:
    """Write tuning results to models/tuned_hyperparameters.json."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "tuned_hyperparameters.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    logger.info("Saved %s", path)
    return path


# ── Main entry point ─────────────────────────────────────────────────────────

def run_tuning_pipeline(n_trials: int = DEFAULT_N_TRIALS, cv_folds: int = settings.CV_FOLDS) -> dict[str, dict[str, Any]]:
    """Tune every ensemble component and save a comparison report.

    Does not modify MODEL_CONFIGS or retrain the deployed ensemble. Review
    models/tuned_hyperparameters.json and update src/train.py manually if a
    result looks worth adopting, then re-run `python -m src.train`.
    """
    logger.info("=" * 70)
    logger.info("TELCO CHURN - HYPERPARAMETER TUNING")
    logger.info("=" * 70)

    X, y = load_data()
    X_train, X_test, y_train, y_test, _ = split_and_scale(X, y)
    logger.info("Train: %d | Test: %d", len(X_train), len(X_test))

    results = tune_all(X_train, y_train, n_trials=n_trials, cv_folds=cv_folds)
    save_results(results)

    logger.info("=" * 70)
    logger.info("TUNING COMPLETE")
    logger.info("=" * 70)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for ensemble components")
    parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS, help="Optuna trials per model")
    parser.add_argument("--cv-folds", type=int, default=settings.CV_FOLDS, help="CV folds for scoring")
    args = parser.parse_args()

    run_tuning_pipeline(n_trials=args.n_trials, cv_folds=args.cv_folds)
