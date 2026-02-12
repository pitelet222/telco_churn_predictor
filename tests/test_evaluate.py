"""
Tests for src/evaluate.py — metric computation utilities.

Covers:
    - evaluate_model: returns correct metric keys and value ranges
    - get_confusion_matrix: shape and values
    - compare_models: multi-model comparison table
    - Threshold sensitivity
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.evaluate import (
    evaluate_model,
    get_confusion_matrix,
    get_classification_report,
    get_roc_curve,
    compare_models,
)


# ── Helper fixture ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def trained_model_and_data():
    """Create a small synthetic dataset and fit a model for testing.

    scope='module' means this runs once for all tests in this file,
    avoiding repeated training (faster tests).
    """
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(y)

    # Simple train/test split
    X_train, X_test = X.iloc[:150], X.iloc[150:]
    y_train, y_test = y.iloc[:150], y.iloc[150:]

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════
#  evaluate_model TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluateModel:
    """Tests for evaluate_model()."""

    def test_returns_all_metric_keys(self, trained_model_and_data):
        """Output must contain all five standard metrics."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        expected = {"Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"}
        assert set(metrics.keys()) == expected

    def test_all_metrics_are_floats(self, trained_model_and_data):
        """Every metric value must be a Python float."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        for key, value in metrics.items():
            assert isinstance(value, float), f"{key} is {type(value)}, not float"

    def test_metrics_in_valid_range(self, trained_model_and_data):
        """All metrics must be between 0 and 1."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} is out of [0, 1] range"

    def test_accuracy_better_than_random(self, trained_model_and_data):
        """A trained model should do better than 50% accuracy."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        assert metrics["Accuracy"] > 0.5

    def test_custom_threshold(self, trained_model_and_data):
        """Changing threshold should affect precision/recall trade-off."""
        model, X_test, y_test = trained_model_and_data
        low_thresh = evaluate_model(model, X_test, y_test, threshold=0.3)
        high_thresh = evaluate_model(model, X_test, y_test, threshold=0.7)

        # Lower threshold → more positives → higher recall, lower precision
        assert low_thresh["Recall"] >= high_thresh["Recall"]


# ═══════════════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestConfusionMatrix:
    """Tests for get_confusion_matrix()."""

    def test_shape_is_2x2(self, trained_model_and_data):
        """Binary classification confusion matrix must be 2×2."""
        model, X_test, y_test = trained_model_and_data
        cm = get_confusion_matrix(model, X_test, y_test)
        assert cm.shape == (2, 2)

    def test_values_sum_to_test_size(self, trained_model_and_data):
        """All cells must sum to total number of test samples."""
        model, X_test, y_test = trained_model_and_data
        cm = get_confusion_matrix(model, X_test, y_test)
        assert cm.sum() == len(y_test)

    def test_no_negative_values(self, trained_model_and_data):
        """Confusion matrix cannot contain negative values."""
        model, X_test, y_test = trained_model_and_data
        cm = get_confusion_matrix(model, X_test, y_test)
        assert (cm >= 0).all()


# ═══════════════════════════════════════════════════════════════════════════
#  CLASSIFICATION REPORT TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestClassificationReport:
    """Tests for get_classification_report()."""

    def test_returns_string(self, trained_model_and_data):
        """Report must be a string."""
        model, X_test, y_test = trained_model_and_data
        report = get_classification_report(model, X_test, y_test)
        assert isinstance(report, str)

    def test_contains_class_names(self, trained_model_and_data):
        """Report should contain the target class names."""
        model, X_test, y_test = trained_model_and_data
        report = get_classification_report(model, X_test, y_test)
        assert "Not Churned" in report
        assert "Churned" in report


# ═══════════════════════════════════════════════════════════════════════════
#  ROC CURVE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestRocCurve:
    """Tests for get_roc_curve()."""

    def test_returns_three_arrays(self, trained_model_and_data):
        """Must return (fpr, tpr, thresholds) as numpy arrays."""
        model, X_test, y_test = trained_model_and_data
        fpr, tpr, thresholds = get_roc_curve(model, X_test, y_test)
        assert isinstance(fpr, np.ndarray)
        assert isinstance(tpr, np.ndarray)
        assert isinstance(thresholds, np.ndarray)

    def test_fpr_tpr_between_0_and_1(self, trained_model_and_data):
        """FPR and TPR values must be between 0 and 1."""
        model, X_test, y_test = trained_model_and_data
        fpr, tpr, _ = get_roc_curve(model, X_test, y_test)
        assert fpr.min() >= 0.0 and fpr.max() <= 1.0
        assert tpr.min() >= 0.0 and tpr.max() <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  COMPARE MODELS TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCompareModels:
    """Tests for compare_models(): multi-model comparison table."""

    def test_returns_dataframe(self, trained_model_and_data):
        """Output must be a pandas DataFrame."""
        model, X_test, y_test = trained_model_and_data
        # Use the same model twice with different names
        models = {"Model_A": model, "Model_B": model}
        df = compare_models(models, X_test, y_test)
        assert isinstance(df, pd.DataFrame)

    def test_has_correct_columns(self, trained_model_and_data):
        """DataFrame must have Model + all metric columns."""
        model, X_test, y_test = trained_model_and_data
        models = {"Model_A": model}
        df = compare_models(models, X_test, y_test)
        expected_cols = ["Model", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]
        assert list(df.columns) == expected_cols

    def test_sorted_by_roc_auc_descending(self, trained_model_and_data):
        """Results should be sorted by ROC-AUC (best first)."""
        model, X_test, y_test = trained_model_and_data
        models = {"Model_A": model, "Model_B": model}
        df = compare_models(models, X_test, y_test)
        auc_values = df["ROC-AUC"].tolist()
        assert auc_values == sorted(auc_values, reverse=True)

    def test_row_count_matches_models(self, trained_model_and_data):
        """Number of rows should equal number of models."""
        model, X_test, y_test = trained_model_and_data
        models = {"A": model, "B": model, "C": model}
        df = compare_models(models, X_test, y_test)
        assert len(df) == 3
