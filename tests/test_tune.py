"""
Tests for src/tune.py — Optuna-based hyperparameter search.

Covers:
    - SEARCH_SPACES: defined for every ensemble component
    - evaluate_baseline: returns a valid PR-AUC score
    - tune_model: returns the expected result schema and a valid search
    - tune_all / save_results: end-to-end on a small synthetic dataset

Uses tiny n_trials/cv_folds and a small synthetic dataset throughout, since
this is a search procedure rather than a fixed computation — the goal is to
verify wiring and output shape, not search quality.
"""

import json

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.train import ENSEMBLE_COMPONENTS
from src.tune import (
    SEARCH_SPACES,
    evaluate_baseline,
    save_results,
    tune_all,
    tune_model,
)


@pytest.fixture(scope="module")
def small_train_data():
    """Small synthetic dataset, just enough for a 2-fold CV search."""
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=4,
        weights=[0.7, 0.3],
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(8)])
    y = pd.Series(y)
    return X, y


# ═══════════════════════════════════════════════════════════════════════════
#  SEARCH SPACE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchSpaces:
    """SEARCH_SPACES must cover every model used in the ensemble."""

    def test_covers_all_ensemble_components(self):
        assert set(ENSEMBLE_COMPONENTS).issubset(SEARCH_SPACES.keys())

    def test_each_entry_has_class_fixed_params_and_space_fn(self):
        for name, (cls, fixed_params, space_fn) in SEARCH_SPACES.items():
            assert isinstance(fixed_params, dict)
            assert callable(space_fn)
            assert callable(cls)


# ═══════════════════════════════════════════════════════════════════════════
#  evaluate_baseline TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluateBaseline:
    """Tests for evaluate_baseline()."""

    @pytest.mark.parametrize("name", ENSEMBLE_COMPONENTS)
    def test_returns_valid_pr_auc(self, small_train_data, name):
        X, y = small_train_data
        score = evaluate_baseline(name, X, y, cv_folds=2)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  tune_model TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestTuneModel:
    """Tests for tune_model()."""

    @pytest.fixture(scope="class")
    def lr_result(self, small_train_data):
        X, y = small_train_data
        return tune_model("Logistic Regression", X, y, n_trials=2, cv_folds=2)

    def test_returns_expected_keys(self, lr_result):
        expected = {
            "model", "n_trials", "cv_folds", "best_params",
            "best_cv_pr_auc", "baseline_cv_pr_auc", "improvement", "elapsed_seconds",
        }
        assert set(lr_result.keys()) == expected

    def test_model_name_matches(self, lr_result):
        assert lr_result["model"] == "Logistic Regression"

    def test_scores_in_valid_range(self, lr_result):
        assert 0.0 <= lr_result["best_cv_pr_auc"] <= 1.0
        assert 0.0 <= lr_result["baseline_cv_pr_auc"] <= 1.0

    def test_best_params_includes_fixed_and_tuned(self, lr_result):
        params = lr_result["best_params"]
        # fixed params for Logistic Regression search space
        assert params["solver"] == "liblinear"
        # tuned params
        assert "C" in params
        assert params["penalty"] in {"l1", "l2"}

    def test_improvement_is_difference(self, lr_result):
        assert lr_result["improvement"] == pytest.approx(
            lr_result["best_cv_pr_auc"] - lr_result["baseline_cv_pr_auc"]
        )


# ═══════════════════════════════════════════════════════════════════════════
#  tune_all / save_results TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestTuneAllAndSaveResults:
    """End-to-end: tune every ensemble component and persist the report."""

    @pytest.fixture(scope="class")
    def all_results(self, small_train_data):
        X, y = small_train_data
        return tune_all(X, y, n_trials=2, cv_folds=2, names=ENSEMBLE_COMPONENTS)

    def test_one_result_per_component(self, all_results):
        assert set(all_results.keys()) == set(ENSEMBLE_COMPONENTS)

    def test_save_results_writes_valid_json(self, all_results, tmp_path):
        path = save_results(all_results, output_dir=tmp_path)
        assert path.exists()
        with open(path) as f:
            loaded = json.load(f)
        assert set(loaded.keys()) == set(ENSEMBLE_COMPONENTS)
