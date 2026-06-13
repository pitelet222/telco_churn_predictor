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
    get_pr_curve,
    get_calibration_curve,
    find_optimal_threshold,
    compare_models,
    compute_overfitting_gap,
    l1_feature_selection_report,
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


@pytest.fixture(scope="module")
def trained_model_and_full_data():
    """Like trained_model_and_data, but also exposes the training split.

    Needed for tests that compare train vs. test performance
    (compute_overfitting_gap) or fit on the training split directly
    (l1_feature_selection_report).
    """
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])
    y = pd.Series(y)

    X_train, X_test = X.iloc[:150], X.iloc[150:]
    y_train, y_test = y.iloc[:150], y.iloc[150:]

    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    return model, X_train, y_train, X_test, y_test


# ═══════════════════════════════════════════════════════════════════════════
#  evaluate_model TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluateModel:
    """Tests for evaluate_model()."""

    def test_returns_all_metric_keys(self, trained_model_and_data):
        """Output must contain all standard metrics, including PR-AUC and Brier."""
        model, X_test, y_test = trained_model_and_data
        metrics = evaluate_model(model, X_test, y_test)
        expected = {"Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "PR-AUC", "Brier"}
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
        expected_cols = ["Model", "PR-AUC", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1-Score", "Brier"]
        assert list(df.columns) == expected_cols

    def test_sorted_by_pr_auc_descending(self, trained_model_and_data):
        """Results should be sorted by PR-AUC (best first)."""
        model, X_test, y_test = trained_model_and_data
        models = {"Model_A": model, "Model_B": model}
        df = compare_models(models, X_test, y_test)
        pr_auc_values = df["PR-AUC"].tolist()
        assert pr_auc_values == sorted(pr_auc_values, reverse=True)

    def test_row_count_matches_models(self, trained_model_and_data):
        """Number of rows should equal number of models."""
        model, X_test, y_test = trained_model_and_data
        models = {"A": model, "B": model, "C": model}
        df = compare_models(models, X_test, y_test)
        assert len(df) == 3


# ═══════════════════════════════════════════════════════════════════════════
#  PR CURVE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPrCurve:
    """Tests for get_pr_curve()."""

    def test_returns_three_arrays(self, trained_model_and_data):
        """Must return (precision, recall, thresholds) as numpy arrays."""
        model, X_test, y_test = trained_model_and_data
        precision, recall, thresholds = get_pr_curve(model, X_test, y_test)
        assert isinstance(precision, np.ndarray)
        assert isinstance(recall, np.ndarray)
        assert isinstance(thresholds, np.ndarray)

    def test_precision_recall_between_0_and_1(self, trained_model_and_data):
        """Precision and recall values must be between 0 and 1."""
        model, X_test, y_test = trained_model_and_data
        precision, recall, _ = get_pr_curve(model, X_test, y_test)
        assert precision.min() >= 0.0 and precision.max() <= 1.0
        assert recall.min() >= 0.0 and recall.max() <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  CALIBRATION CURVE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestCalibrationCurve:
    """Tests for get_calibration_curve()."""

    def test_returns_two_arrays_of_equal_length(self, trained_model_and_data):
        """Must return (prob_true, prob_pred) of equal length."""
        model, X_test, y_test = trained_model_and_data
        prob_true, prob_pred = get_calibration_curve(model, X_test, y_test, n_bins=5)
        assert isinstance(prob_true, np.ndarray)
        assert isinstance(prob_pred, np.ndarray)
        assert len(prob_true) == len(prob_pred)

    def test_values_between_0_and_1(self, trained_model_and_data):
        """Calibration points must be valid probabilities."""
        model, X_test, y_test = trained_model_and_data
        prob_true, prob_pred = get_calibration_curve(model, X_test, y_test, n_bins=5)
        assert prob_true.min() >= 0.0 and prob_true.max() <= 1.0
        assert prob_pred.min() >= 0.0 and prob_pred.max() <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  OPTIMAL THRESHOLD TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestFindOptimalThreshold:
    """Tests for find_optimal_threshold()."""

    def test_returns_expected_keys(self, trained_model_and_data):
        """Output must contain threshold, total_cost, precision, recall, f1."""
        model, X_test, y_test = trained_model_and_data
        y_proba = model.predict_proba(X_test)[:, 1]
        result = find_optimal_threshold(y_test, y_proba)
        expected = {"threshold", "total_cost", "precision", "recall", "f1"}
        assert set(result.keys()) == expected

    def test_threshold_in_valid_range(self, trained_model_and_data):
        """Optimal threshold must be between 0 and 1."""
        model, X_test, y_test = trained_model_and_data
        y_proba = model.predict_proba(X_test)[:, 1]
        result = find_optimal_threshold(y_test, y_proba)
        assert 0.0 < result["threshold"] < 1.0

    def test_higher_fn_cost_lowers_threshold(self, trained_model_and_data):
        """Penalizing false negatives more heavily should not raise the threshold."""
        model, X_test, y_test = trained_model_and_data
        y_proba = model.predict_proba(X_test)[:, 1]
        low_cost_fn = find_optimal_threshold(y_test, y_proba, cost_fn=1.0, cost_fp=1.0)
        high_cost_fn = find_optimal_threshold(y_test, y_proba, cost_fn=20.0, cost_fp=1.0)
        assert high_cost_fn["threshold"] <= low_cost_fn["threshold"]


# ═══════════════════════════════════════════════════════════════════════════
#  OVERFITTING GAP TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeOverfittingGap:
    """Tests for compute_overfitting_gap()."""

    def test_returns_expected_keys(self, trained_model_and_full_data):
        """Output must contain train/test ROC-AUC and PR-AUC plus their gaps."""
        model, X_train, y_train, X_test, y_test = trained_model_and_full_data
        gap = compute_overfitting_gap(model, X_train, y_train, X_test, y_test)
        expected = {
            "train_roc_auc", "test_roc_auc", "roc_auc_gap",
            "train_pr_auc", "test_pr_auc", "pr_auc_gap",
        }
        assert set(gap.keys()) == expected

    def test_all_values_are_floats(self, trained_model_and_full_data):
        """Every value must be a Python float."""
        model, X_train, y_train, X_test, y_test = trained_model_and_full_data
        gap = compute_overfitting_gap(model, X_train, y_train, X_test, y_test)
        for key, value in gap.items():
            assert isinstance(value, float), f"{key} is {type(value)}, not float"

    def test_gaps_equal_train_minus_test(self, trained_model_and_full_data):
        """Gap values must equal train metric minus test metric."""
        model, X_train, y_train, X_test, y_test = trained_model_and_full_data
        gap = compute_overfitting_gap(model, X_train, y_train, X_test, y_test)
        assert gap["roc_auc_gap"] == pytest.approx(gap["train_roc_auc"] - gap["test_roc_auc"])
        assert gap["pr_auc_gap"] == pytest.approx(gap["train_pr_auc"] - gap["test_pr_auc"])

    def test_train_metrics_in_valid_range(self, trained_model_and_full_data):
        """Train ROC-AUC and PR-AUC must be between 0 and 1."""
        model, X_train, y_train, X_test, y_test = trained_model_and_full_data
        gap = compute_overfitting_gap(model, X_train, y_train, X_test, y_test)
        assert 0.0 <= gap["train_roc_auc"] <= 1.0
        assert 0.0 <= gap["train_pr_auc"] <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
#  L1 FEATURE SELECTION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestL1FeatureSelectionReport:
    """Tests for l1_feature_selection_report()."""

    def test_returns_expected_keys(self, trained_model_and_full_data):
        """Output must contain C, feature counts, and selected/zeroed lists."""
        _, X_train, y_train, _, _ = trained_model_and_full_data
        report = l1_feature_selection_report(X_train, y_train)
        expected = {
            "C", "n_features_total", "n_features_selected",
            "selected_features", "zeroed_features",
        }
        assert set(report.keys()) == expected

    def test_n_features_total_matches_columns(self, trained_model_and_full_data):
        """n_features_total must equal the number of input columns."""
        _, X_train, y_train, _, _ = trained_model_and_full_data
        report = l1_feature_selection_report(X_train, y_train)
        assert report["n_features_total"] == X_train.shape[1]

    def test_selected_and_zeroed_partition_all_features(self, trained_model_and_full_data):
        """Selected + zeroed features must together account for all columns."""
        _, X_train, y_train, _, _ = trained_model_and_full_data
        report = l1_feature_selection_report(X_train, y_train)
        combined = set(report["selected_features"]) | set(report["zeroed_features"])
        assert combined == set(X_train.columns)
        assert report["n_features_selected"] == len(report["selected_features"])

    def test_strong_regularization_zeroes_more_features(self, trained_model_and_full_data):
        """A much smaller C (stronger regularization) should not increase selected features."""
        _, X_train, y_train, _, _ = trained_model_and_full_data
        weak_reg = l1_feature_selection_report(X_train, y_train, C=10.0)
        strong_reg = l1_feature_selection_report(X_train, y_train, C=0.01)
        assert strong_reg["n_features_selected"] <= weak_reg["n_features_selected"]
