"""
Tests for src/train.py — data loading and splitting utilities.

Covers:
    - load_data: correct shapes and types
    - split_and_scale: stratification, scaling, no data leakage
"""

import numpy as np
import pandas as pd
import pytest

from src.train import load_data, split_and_scale


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD DATA TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestLoadData:
    """Tests for load_data(): reads the processed CSV."""

    def test_returns_dataframe_and_series(self):
        """Must return (X: DataFrame, y: Series)."""
        X, y = load_data()
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_not_in_features(self):
        """The 'Churn' column must NOT be in X (features)."""
        X, y = load_data()
        assert "Churn" not in X.columns

    def test_target_is_binary(self):
        """Target variable should only contain 0 and 1."""
        _, y = load_data()
        assert set(y.unique()).issubset({0, 1})

    def test_no_missing_values(self):
        """Processed data should have no NaN values."""
        X, y = load_data()
        assert not X.isnull().any().any(), "Features contain NaN values"
        assert not y.isnull().any(), "Target contains NaN values"

    def test_has_expected_feature_count(self):
        """Processed data should have 35 features."""
        X, _ = load_data()
        assert X.shape[1] == 35, f"Expected 35 features, got {X.shape[1]}"

    def test_has_enough_samples(self):
        """Dataset should have thousands of records (not empty or truncated)."""
        X, _ = load_data()
        assert X.shape[0] > 5000, f"Expected > 5000 rows, got {X.shape[0]}"


# ═══════════════════════════════════════════════════════════════════════════
#  SPLIT AND SCALE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestSplitAndScale:
    """Tests for split_and_scale(): stratified split + StandardScaler."""

    @pytest.fixture(scope="class")
    def split_result(self):
        """Load data and split once for all tests in this class."""
        X, y = load_data()
        return split_and_scale(X, y, test_size=0.3, random_state=42)

    def test_returns_five_items(self, split_result):
        """Must return (X_train, X_test, y_train, y_test, scaler)."""
        assert len(split_result) == 5

    def test_train_test_sizes(self, split_result):
        """70/30 split: train should be ~70% of total."""
        X_train, X_test, y_train, y_test, _ = split_result
        total = len(X_train) + len(X_test)
        train_ratio = len(X_train) / total
        assert 0.68 <= train_ratio <= 0.72, (
            f"Expected ~0.70 train ratio, got {train_ratio:.3f}"
        )

    def test_features_preserved(self, split_result):
        """Train and test must have the same number of columns."""
        X_train, X_test, _, _, _ = split_result
        assert X_train.shape[1] == X_test.shape[1]

    def test_stratification(self, split_result):
        """Churn ratio should be similar in train and test sets."""
        _, _, y_train, y_test, _ = split_result
        train_churn_rate = y_train.mean()
        test_churn_rate = y_test.mean()
        # Allow 2% tolerance
        assert abs(train_churn_rate - test_churn_rate) < 0.02, (
            f"Stratification broken: train={train_churn_rate:.3f}, "
            f"test={test_churn_rate:.3f}"
        )

    def test_scaling_zero_mean(self, split_result):
        """After StandardScaler, train features should have ~0 mean."""
        X_train, _, _, _, _ = split_result
        means = X_train.mean()
        # Each feature's mean should be very close to 0
        assert (means.abs() < 0.1).all(), (
            f"Some features have mean far from 0: {means[means.abs() >= 0.1]}"
        )

    def test_scaling_unit_variance(self, split_result):
        """After StandardScaler, train features should have ~1 std."""
        X_train, _, _, _, _ = split_result
        stds = X_train.std()
        # Each feature's std should be close to 1 (±0.2 tolerance)
        assert ((stds - 1.0).abs() < 0.2).all(), (
            f"Some features have std far from 1: {stds[(stds - 1.0).abs() >= 0.2]}"
        )

    def test_no_data_leakage(self, split_result):
        """Test set should NOT have zero mean (scaler fitted on train only)."""
        _, X_test, _, _, _ = split_result
        test_means = X_test.mean()
        # Test means should NOT all be exactly 0 (that would mean leakage)
        assert not (test_means.abs() < 0.001).all(), (
            "Test set has near-zero means — possible data leakage!"
        )

    def test_scaler_is_fitted(self, split_result):
        """The returned scaler should already be fitted (has mean_ attribute)."""
        _, _, _, _, scaler = split_result
        assert hasattr(scaler, "mean_"), "Scaler is not fitted"
        assert hasattr(scaler, "scale_"), "Scaler is not fitted"
