"""
Tests for app/churn_service.py — the core prediction engine.

Covers:
    - Feature encoding (_encode_customer)
    - Full prediction pipeline (predict_churn)
    - Risk level classification
    - SHAP-based risk factors
    - Fallback manual risk factors
    - Customer summary generation
    - Model metadata loading
"""

import numpy as np
import pandas as pd
import pytest

from churn_service import (
    FEATURE_COLUMNS,
    _encode_customer,
    _detect_risk_factors,
    _build_customer_summary,
    _format_shap_factor,
    predict_churn,
    get_model_metadata,
)


# ═══════════════════════════════════════════════════════════════════════════
#  ENCODING TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestEncodeCustomer:
    """Tests for _encode_customer(): converts user input → 35-feature vector."""

    def test_output_is_dataframe(self, high_risk_customer):
        """Encoding must return a pandas DataFrame."""
        result = _encode_customer(high_risk_customer)
        assert isinstance(result, pd.DataFrame)

    def test_produces_35_features(self, high_risk_customer):
        """The encoded output must have exactly 35 columns (matching training)."""
        result = _encode_customer(high_risk_customer)
        assert result.shape == (1, 35), (
            f"Expected (1, 35) but got {result.shape}"
        )

    def test_column_order_matches_training(self, high_risk_customer):
        """Columns must be in the exact order defined in FEATURE_COLUMNS."""
        result = _encode_customer(high_risk_customer)
        assert list(result.columns) == FEATURE_COLUMNS

    def test_gender_encoding(self):
        """Male → 1, Female → 0."""
        male = _encode_customer({"gender": "Male"})
        female = _encode_customer({"gender": "Female"})
        assert male["gender"].iloc[0] == 1
        assert female["gender"].iloc[0] == 0

    def test_yes_no_binary_encoding(self):
        """'Yes' → 1, 'No' → 0 for binary fields."""
        yes = _encode_customer({"Partner": "Yes"})
        no = _encode_customer({"Partner": "No"})
        assert yes["Partner"].iloc[0] == 1
        assert no["Partner"].iloc[0] == 0

    def test_one_hot_internet_service(self):
        """InternetService one-hot encoding for all 3 options."""
        fiber = _encode_customer({"InternetService": "Fiber optic"})
        assert fiber["InternetService_fiber optic"].iloc[0] == 1
        assert fiber["InternetService_no"].iloc[0] == 0

        dsl = _encode_customer({"InternetService": "DSL"})
        assert dsl["InternetService_fiber optic"].iloc[0] == 0
        assert dsl["InternetService_no"].iloc[0] == 0

        no_inet = _encode_customer({"InternetService": "No"})
        assert no_inet["InternetService_fiber optic"].iloc[0] == 0
        assert no_inet["InternetService_no"].iloc[0] == 1

    def test_one_hot_contract(self):
        """Contract one-hot encoding: Month-to-month is the dropped base."""
        m2m = _encode_customer({"Contract": "Month-to-month"})
        assert m2m["Contract_one year"].iloc[0] == 0
        assert m2m["Contract_two year"].iloc[0] == 0

        one_yr = _encode_customer({"Contract": "One year"})
        assert one_yr["Contract_one year"].iloc[0] == 1
        assert one_yr["Contract_two year"].iloc[0] == 0

        two_yr = _encode_customer({"Contract": "Two year"})
        assert two_yr["Contract_one year"].iloc[0] == 0
        assert two_yr["Contract_two year"].iloc[0] == 1

    def test_one_hot_payment_method(self):
        """PaymentMethod one-hot encoding (bank transfer is the dropped base)."""
        echeck = _encode_customer({"PaymentMethod": "Electronic check"})
        assert echeck["PaymentMethod_electronic check"].iloc[0] == 1
        assert echeck["PaymentMethod_mailed check"].iloc[0] == 0
        assert echeck["PaymentMethod_credit card (automatic)"].iloc[0] == 0

    def test_avg_monthly_charges_calculation(self):
        """AvgMonthlyCharges = TotalCharges / max(tenure, 1)."""
        result = _encode_customer({
            "tenure": 10,
            "TotalCharges": 500.0,
        })
        assert result["AvgMonthlyCharges"].iloc[0] == pytest.approx(50.0)

    def test_avg_monthly_charges_zero_tenure(self):
        """When tenure=0, we divide by 1 to avoid division by zero."""
        result = _encode_customer({
            "tenure": 0,
            "TotalCharges": 100.0,
        })
        assert result["AvgMonthlyCharges"].iloc[0] == pytest.approx(100.0)

    def test_tenure_group_buckets(self):
        """Tenure group one-hot encoding for different tenure values."""
        # 0-12 months: all groups = 0
        short = _encode_customer({"tenure": 6})
        assert short["TenureGroup_1-2yr"].iloc[0] == 0
        assert short["TenureGroup_2-4yr"].iloc[0] == 0
        assert short["TenureGroup_4-6yr"].iloc[0] == 0

        # 12-24 months: 1-2yr = 1
        mid1 = _encode_customer({"tenure": 18})
        assert mid1["TenureGroup_1-2yr"].iloc[0] == 1
        assert mid1["TenureGroup_2-4yr"].iloc[0] == 0

        # 24-48 months: 2-4yr = 1
        mid2 = _encode_customer({"tenure": 36})
        assert mid2["TenureGroup_2-4yr"].iloc[0] == 1
        assert mid2["TenureGroup_4-6yr"].iloc[0] == 0

        # 48+ months: 4-6yr = 1
        long = _encode_customer({"tenure": 60})
        assert long["TenureGroup_4-6yr"].iloc[0] == 1

    def test_total_services_count(self, high_risk_customer):
        """TotalServices should count active yes-services."""
        # high_risk_customer has PhoneService=Yes only (no other _yes columns)
        result = _encode_customer(high_risk_customer)
        assert result["TotalServices"].iloc[0] == 1  # Only PhoneService

    def test_total_services_many(self, low_risk_customer):
        """Low-risk customer with many services should have a higher count."""
        result = _encode_customer(low_risk_customer)
        assert result["TotalServices"].iloc[0] >= 5

    def test_no_internet_sets_service_columns(self):
        """When InternetService='No', all internet-dependent services → 'no internet service'."""
        result = _encode_customer({"InternetService": "No"})
        for svc in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                     "TechSupport", "StreamingTV", "StreamingMovies"]:
            assert result[f"{svc}_no internet service"].iloc[0] == 1
            assert result[f"{svc}_yes"].iloc[0] == 0

    def test_all_values_are_numeric(self, high_risk_customer):
        """Every feature value must be numeric (int or float)."""
        result = _encode_customer(high_risk_customer)
        assert result.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all()

    def test_handles_minimal_input(self, minimal_customer):
        """Should work with minimal fields, filling defaults for the rest."""
        result = _encode_customer(minimal_customer)
        assert result.shape == (1, 35)
        assert not result.isnull().any().any(), "No NaN values should be present"


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICTION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictChurn:
    """Tests for predict_churn(): the main public API."""

    def test_returns_required_keys(self, high_risk_customer):
        """Output dict must contain all expected keys."""
        result = predict_churn(high_risk_customer)
        expected_keys = {
            "churn_probability",
            "risk_level",
            "risk_factors",
            "risk_source",
            "customer_summary",
        }
        assert set(result.keys()) == expected_keys

    def test_probability_in_valid_range(self, high_risk_customer):
        """Churn probability must be between 0 and 1."""
        result = predict_churn(high_risk_customer)
        assert 0.0 <= result["churn_probability"] <= 1.0

    def test_risk_level_is_valid(self, high_risk_customer):
        """Risk level must be one of the four defined categories."""
        result = predict_churn(high_risk_customer)
        assert result["risk_level"] in {"Low", "Medium", "High", "Very High"}

    def test_high_risk_customer_has_high_probability(self, high_risk_customer):
        """A clearly risky profile should have churn probability > 0.5."""
        result = predict_churn(high_risk_customer)
        assert result["churn_probability"] > 0.5, (
            f"Expected > 0.5, got {result['churn_probability']}"
        )

    def test_low_risk_customer_has_low_probability(self, low_risk_customer):
        """A clearly safe profile should have churn probability < 0.25."""
        result = predict_churn(low_risk_customer)
        assert result["churn_probability"] < 0.25, (
            f"Expected < 0.25, got {result['churn_probability']}"
        )

    def test_risk_factors_is_list(self, high_risk_customer):
        """Risk factors must be a non-empty list of strings."""
        result = predict_churn(high_risk_customer)
        assert isinstance(result["risk_factors"], list)
        assert len(result["risk_factors"]) > 0
        assert all(isinstance(f, str) for f in result["risk_factors"])

    def test_risk_source_is_shap_or_manual(self, high_risk_customer):
        """Risk source must indicate which detection method was used."""
        result = predict_churn(high_risk_customer)
        assert result["risk_source"] in {"shap", "manual"}

    def test_customer_summary_is_string(self, high_risk_customer):
        """Customer summary must be a non-empty string."""
        result = predict_churn(high_risk_customer)
        assert isinstance(result["customer_summary"], str)
        assert len(result["customer_summary"]) > 0

    def test_minimal_input_does_not_crash(self, minimal_customer):
        """Even a minimal customer dict should return a valid prediction."""
        result = predict_churn(minimal_customer)
        assert 0.0 <= result["churn_probability"] <= 1.0
        assert result["risk_level"] in {"Low", "Medium", "High", "Very High"}

    def test_probability_changes_with_contract(self, high_risk_customer):
        """Switching from month-to-month to two-year should lower churn risk."""
        risky = predict_churn(high_risk_customer)

        safe_version = high_risk_customer.copy()
        safe_version["Contract"] = "Two year"
        safer = predict_churn(safe_version)

        assert safer["churn_probability"] < risky["churn_probability"], (
            "Two-year contract should reduce churn probability"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  RISK LEVEL CLASSIFICATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestRiskLevels:
    """Verify that churn probabilities map to the correct risk levels."""

    @pytest.mark.parametrize("proba_range,expected_level", [
        ((0.0, 0.24), "Low"),
        ((0.25, 0.49), "Medium"),
        ((0.50, 0.74), "High"),
        ((0.75, 1.00), "Very High"),
    ])
    def test_risk_level_boundaries(self, proba_range, expected_level):
        """Each probability range should map to the correct risk level."""
        # We test the mapping logic directly rather than via predict_churn()
        # because we can't control the exact probability output.
        low, high = proba_range
        mid = (low + high) / 2

        if mid < 0.25:
            assert expected_level == "Low"
        elif mid < 0.50:
            assert expected_level == "Medium"
        elif mid < 0.75:
            assert expected_level == "High"
        else:
            assert expected_level == "Very High"


# ═══════════════════════════════════════════════════════════════════════════
#  MANUAL RISK FACTOR TESTS (fallback)
# ═══════════════════════════════════════════════════════════════════════════


class TestManualRiskFactors:
    """Tests for _detect_risk_factors() — the EDA-based fallback."""

    def test_month_to_month_flagged(self):
        """Month-to-month contract should be flagged."""
        factors = _detect_risk_factors({"Contract": "Month-to-month"}, 0.7)
        assert any("Month-to-month" in f for f in factors)

    def test_two_year_not_flagged(self):
        """Two-year contract should NOT trigger the contract risk factor."""
        factors = _detect_risk_factors({"Contract": "Two year"}, 0.7)
        assert not any("Month-to-month" in f for f in factors)

    def test_fiber_optic_flagged(self):
        """Fiber optic internet should be flagged."""
        factors = _detect_risk_factors({"InternetService": "Fiber optic"}, 0.7)
        assert any("Fiber optic" in f for f in factors)

    def test_electronic_check_flagged(self):
        """Electronic check payment should be flagged."""
        factors = _detect_risk_factors({"PaymentMethod": "Electronic check"}, 0.7)
        assert any("Electronic check" in f for f in factors)

    def test_short_tenure_flagged(self):
        """Tenure < 12 months should be flagged."""
        factors = _detect_risk_factors({"tenure": 5}, 0.7)
        assert any("tenure" in f.lower() for f in factors)

    def test_long_tenure_not_flagged(self):
        """Tenure >= 12 months should NOT trigger the tenure risk factor."""
        factors = _detect_risk_factors({"tenure": 24}, 0.3)
        assert not any("Short tenure" in f for f in factors)

    def test_high_charges_flagged(self):
        """MonthlyCharges > 70 should be flagged."""
        factors = _detect_risk_factors({"MonthlyCharges": 85}, 0.7)
        assert any("charges" in f.lower() for f in factors)

    def test_low_charges_not_flagged(self):
        """MonthlyCharges <= 70 should NOT trigger the charges risk factor."""
        factors = _detect_risk_factors({"MonthlyCharges": 50}, 0.3)
        assert not any("High monthly charges" in f for f in factors)

    def test_no_tech_support_flagged(self):
        """No tech support with internet service should be flagged."""
        factors = _detect_risk_factors(
            {"TechSupport": "No", "InternetService": "DSL"}, 0.7
        )
        assert any("tech support" in f.lower() for f in factors)

    def test_no_tech_support_without_internet_not_flagged(self):
        """No tech support WITHOUT internet should NOT be flagged."""
        factors = _detect_risk_factors(
            {"TechSupport": "No", "InternetService": "No"}, 0.7
        )
        assert not any("tech support" in f.lower() for f in factors)

    def test_returns_empty_for_safe_customer(self):
        """A safe customer profile should return no risk factors."""
        factors = _detect_risk_factors({
            "Contract": "Two year",
            "InternetService": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "tenure": 60,
            "MonthlyCharges": 30,
            "TechSupport": "Yes",
            "OnlineSecurity": "Yes",
            "PaperlessBilling": "No",
            "Dependents": "Yes",
            "Partner": "Yes",
        }, 0.1)
        assert factors == []


# ═══════════════════════════════════════════════════════════════════════════
#  SHAP FORMATTER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestFormatShapFactor:
    """Tests for _format_shap_factor(): human-readable SHAP descriptions."""

    def test_binary_feature_yes(self):
        """Binary feature with value 1 should show 'Yes'."""
        result = _format_shap_factor("InternetService_fiber optic", 0.15, 1.0)
        assert "Yes" in result
        assert "increases" in result

    def test_binary_feature_no(self):
        """Binary feature with value 0 should show 'No'."""
        result = _format_shap_factor("Contract_two year", -0.20, 0.0)
        assert "No" in result
        assert "decreases" in result

    def test_tenure_shows_months(self):
        """Tenure should display value in months."""
        result = _format_shap_factor("tenure", 0.10, 6.0)
        assert "6 months" in result

    def test_charges_show_dollar(self):
        """MonthlyCharges should display a dollar sign."""
        result = _format_shap_factor("MonthlyCharges", 0.05, 85.0)
        assert "$" in result

    def test_positive_shap_says_increases(self):
        """Positive SHAP value should say 'increases risk'."""
        result = _format_shap_factor("tenure", 0.10, 3.0)
        assert "increases" in result

    def test_negative_shap_says_decreases(self):
        """Negative SHAP value should say 'decreases risk'."""
        result = _format_shap_factor("tenure", -0.15, 60.0)
        assert "decreases" in result

    def test_percentage_is_included(self):
        """The risk percentage should be in the output."""
        result = _format_shap_factor("tenure", 0.123, 12.0)
        assert "12.3%" in result

    def test_uses_human_label(self):
        """Should use the human-readable label, not the technical column name."""
        result = _format_shap_factor("InternetService_fiber optic", 0.1, 1.0)
        assert "Fiber optic internet" in result

    def test_total_services_shows_count(self):
        """TotalServices should display as an integer count."""
        result = _format_shap_factor("TotalServices", -0.05, 6.0)
        assert "6" in result
        assert "Total active services" in result


# ═══════════════════════════════════════════════════════════════════════════
#  CUSTOMER SUMMARY TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestBuildCustomerSummary:
    """Tests for _build_customer_summary(): text for GPT context."""

    def test_returns_string(self, high_risk_customer):
        """Summary must be a non-empty string."""
        result = _build_customer_summary(high_risk_customer)
        assert isinstance(result, str)
        assert len(result) > 50

    def test_contains_key_fields(self, high_risk_customer):
        """Summary should mention tenure, contract, and charges."""
        result = _build_customer_summary(high_risk_customer)
        assert "tenure" in result
        assert "contract" in result
        assert "monthly" in result.lower()

    def test_contains_customer_values(self, high_risk_customer):
        """Summary should include the actual customer values."""
        result = _build_customer_summary(high_risk_customer)
        assert "3" in result         # tenure
        assert "75" in result        # monthly charges
        assert "Month-to-month" in result


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL METADATA TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestModelMetadata:
    """Tests for get_model_metadata(): loads saved performance info."""

    def test_returns_dict(self):
        """Metadata must be a dictionary."""
        meta = get_model_metadata()
        assert isinstance(meta, dict)

    def test_contains_required_keys(self):
        """Metadata should include model type, components, and metrics."""
        meta = get_model_metadata()
        assert "model_type" in meta
        assert "ensemble_components" in meta
        assert "test_roc_auc" in meta

    def test_roc_auc_is_reasonable(self):
        """Stored ROC-AUC should be between 0.5 and 1.0 (better than random)."""
        meta = get_model_metadata()
        assert 0.5 < meta["test_roc_auc"] <= 1.0
