"""
Tests for the FastAPI REST API (api/).

Covers:
    - GET  /health           → liveness check
    - GET  /model/metadata   → model info
    - POST /predict          → churn prediction
    - POST /advice           → prediction + LLM retention advice
    - Input validation       → Pydantic rejects bad payloads (422)
"""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import app


# ── Test client fixture ─────────────────────────────────────────────────────

@pytest.fixture
def client():
    """FastAPI test client — no real HTTP, runs in-process."""
    return TestClient(app)


# ═══════════════════════════════════════════════════════════════════════════
#  HEALTH ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """GET /health — liveness probe."""

    def test_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_version_is_present(self, client):
        data = client.get("/health").json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
#  MODEL METADATA ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════


class TestModelMetadataEndpoint:
    """GET /model/metadata — model training info."""

    def test_returns_200(self, client):
        response = client.get("/model/metadata")
        assert response.status_code == 200

    def test_contains_key_fields(self, client):
        data = client.get("/model/metadata").json()
        expected_keys = [
            "model_name", "model_type", "test_roc_auc",
            "test_accuracy", "training_date",
        ]
        for key in expected_keys:
            assert key in data, f"Missing key: {key}"

    def test_roc_auc_is_reasonable(self, client):
        data = client.get("/model/metadata").json()
        auc = data["test_roc_auc"]
        assert 0.5 < auc < 1.0, f"ROC AUC {auc} is out of expected range"


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICT ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictEndpoint:
    """POST /predict — churn prediction."""

    def test_high_risk_returns_200(self, client, high_risk_customer):
        response = client.post("/predict", json=high_risk_customer)
        assert response.status_code == 200

    def test_low_risk_returns_200(self, client, low_risk_customer):
        response = client.post("/predict", json=low_risk_customer)
        assert response.status_code == 200

    def test_response_has_required_fields(self, client, high_risk_customer):
        data = client.post("/predict", json=high_risk_customer).json()
        required = [
            "churn_probability", "risk_level",
            "risk_factors", "risk_source", "customer_summary",
        ]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_churn_probability_range(self, client, high_risk_customer):
        data = client.post("/predict", json=high_risk_customer).json()
        prob = data["churn_probability"]
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of [0, 1]"

    def test_risk_level_is_valid(self, client, high_risk_customer):
        data = client.post("/predict", json=high_risk_customer).json()
        valid_levels = {"Low", "Medium", "High", "Very High"}
        assert data["risk_level"] in valid_levels

    def test_risk_source_is_valid(self, client, high_risk_customer):
        data = client.post("/predict", json=high_risk_customer).json()
        assert data["risk_source"] in {"shap", "manual"}

    def test_risk_factors_is_list(self, client, high_risk_customer):
        data = client.post("/predict", json=high_risk_customer).json()
        assert isinstance(data["risk_factors"], list)

    def test_high_risk_customer_has_elevated_probability(self, client, high_risk_customer):
        """A known high-risk profile should produce probability > 0.5."""
        data = client.post("/predict", json=high_risk_customer).json()
        assert data["churn_probability"] > 0.5

    def test_low_risk_customer_has_low_probability(self, client, low_risk_customer):
        """A known low-risk profile should produce probability < 0.5."""
        data = client.post("/predict", json=low_risk_customer).json()
        assert data["churn_probability"] < 0.5

    def test_high_risk_level_label(self, client, high_risk_customer):
        data = client.post("/predict", json=high_risk_customer).json()
        assert data["risk_level"] in {"High", "Very High"}

    def test_low_risk_level_label(self, client, low_risk_customer):
        data = client.post("/predict", json=low_risk_customer).json()
        assert data["risk_level"] in {"Low", "Medium"}

    def test_customer_summary_contains_key_info(self, client, high_risk_customer):
        data = client.post("/predict", json=high_risk_customer).json()
        summary = data["customer_summary"]
        assert "tenure" in summary
        assert "contract" in summary.lower() or "Contract" in summary


# ═══════════════════════════════════════════════════════════════════════════
#  PREDICT — INPUT VALIDATION (422 errors)
# ═══════════════════════════════════════════════════════════════════════════


class TestPredictValidation:
    """POST /predict with invalid payloads should return 422."""

    def test_empty_body_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_missing_field_returns_422(self, client, high_risk_customer):
        """Remove a required field → 422."""
        incomplete = {k: v for k, v in high_risk_customer.items() if k != "tenure"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_invalid_gender_returns_422(self, client, high_risk_customer):
        bad = {**high_risk_customer, "gender": "Unknown"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_invalid_contract_returns_422(self, client, high_risk_customer):
        bad = {**high_risk_customer, "Contract": "Weekly"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_tenure_negative_returns_422(self, client, high_risk_customer):
        bad = {**high_risk_customer, "tenure": -5}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_tenure_too_high_returns_422(self, client, high_risk_customer):
        bad = {**high_risk_customer, "tenure": 100}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_monthly_charges_negative_returns_422(self, client, high_risk_customer):
        bad = {**high_risk_customer, "MonthlyCharges": -10}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_total_charges_too_high_returns_422(self, client, high_risk_customer):
        bad = {**high_risk_customer, "TotalCharges": 99999}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_invalid_internet_service_returns_422(self, client, high_risk_customer):
        bad = {**high_risk_customer, "InternetService": "5G"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_invalid_payment_method_returns_422(self, client, high_risk_customer):
        bad = {**high_risk_customer, "PaymentMethod": "Bitcoin"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
#  ADVICE ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════


class TestAdviceEndpoint:
    """POST /advice — prediction + LLM retention advice."""

    @pytest.fixture
    def _mock_llm(self):
        """Patch the LLM call so tests don't hit OpenAI."""
        with patch(
            "api.routers.advice.get_retention_advice",
            return_value="## Mocked advice\n- Offer a discount\n- Upgrade contract",
        ) as mock:
            yield mock

    def test_returns_200(self, client, high_risk_customer, _mock_llm):
        payload = {"customer": high_risk_customer, "question": ""}
        response = client.post("/advice", json=payload)
        assert response.status_code == 200

    def test_response_has_prediction_and_advice(self, client, high_risk_customer, _mock_llm):
        payload = {"customer": high_risk_customer}
        data = client.post("/advice", json=payload).json()
        assert "prediction" in data
        assert "advice" in data

    def test_prediction_fields_present(self, client, high_risk_customer, _mock_llm):
        payload = {"customer": high_risk_customer}
        prediction = client.post("/advice", json=payload).json()["prediction"]
        for field in ["churn_probability", "risk_level", "risk_factors"]:
            assert field in prediction

    def test_advice_is_nonempty_string(self, client, high_risk_customer, _mock_llm):
        payload = {"customer": high_risk_customer}
        data = client.post("/advice", json=payload).json()
        assert isinstance(data["advice"], str)
        assert len(data["advice"]) > 0

    def test_with_question(self, client, low_risk_customer, _mock_llm):
        payload = {
            "customer": low_risk_customer,
            "question": "What discount should we offer?",
        }
        response = client.post("/advice", json=payload)
        assert response.status_code == 200

    def test_llm_called_with_churn_result(self, client, high_risk_customer, _mock_llm):
        """Verify the LLM receives the prediction output."""
        payload = {"customer": high_risk_customer, "question": "Help"}
        client.post("/advice", json=payload)
        _mock_llm.assert_called_once()
        call_kwargs = _mock_llm.call_args
        assert "churn_result" in call_kwargs.kwargs or len(call_kwargs.args) > 0

    def test_llm_failure_returns_502(self, client, high_risk_customer):
        """If the LLM call fails, the API should return 502."""
        with patch(
            "api.routers.advice.get_retention_advice",
            side_effect=Exception("OpenAI is down"),
        ):
            payload = {"customer": high_risk_customer}
            response = client.post("/advice", json=payload)
            assert response.status_code == 502

    def test_invalid_customer_in_advice_returns_422(self, client):
        """Bad customer data in /advice should also be rejected."""
        payload = {"customer": {"gender": "Alien"}, "question": ""}
        response = client.post("/advice", json=payload)
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════
#  EDGE CASES & CROSS-CUTTING
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Miscellaneous API-level edge cases."""

    def test_unknown_route_returns_404(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_wrong_method_returns_405(self, client):
        """GET on a POST-only endpoint should fail."""
        response = client.get("/predict")
        assert response.status_code == 405

    def test_predict_with_string_tenure_returns_422(self, client, high_risk_customer):
        bad = {**high_risk_customer, "tenure": "three"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422

    def test_predict_returns_json_content_type(self, client, high_risk_customer):
        response = client.post("/predict", json=high_risk_customer)
        assert "application/json" in response.headers["content-type"]
