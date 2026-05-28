"""
Tests for src/fraud_detection/api/app.py.

Uses a session-scoped TestClient (from tests/conftest.py) that loads the
real trained models once and reuses them across all tests in this module.

Skipped automatically if model artefacts are absent.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_response_is_json(self, api_client):
        resp = api_client.get("/health")
        assert resp.headers["content-type"].startswith("application/json")

    def test_status_field_is_healthy(self, api_client):
        body = api_client.get("/health").json()
        assert body["status"] == "healthy"

    def test_model_loaded_is_true(self, api_client):
        body = api_client.get("/health").json()
        assert body["model_loaded"] is True

    def test_threshold_is_positive_float(self, api_client):
        body = api_client.get("/health").json()
        assert isinstance(body["threshold"], float)
        assert 0.0 < body["threshold"] < 1.0

    def test_feature_count_is_230(self, api_client):
        """Pipeline produces 229 engineered features + anomaly_score = 230."""
        body = api_client.get("/health").json()
        assert body["feature_count"] == 230

    def test_uptime_seconds_is_non_negative(self, api_client):
        body = api_client.get("/health").json()
        assert body["uptime_seconds"] >= 0.0

    def test_all_expected_fields_present(self, api_client):
        body = api_client.get("/health").json()
        for field in ("status", "model_loaded", "threshold", "feature_count", "uptime_seconds"):
            assert field in body, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# POST /predict — valid requests
# ---------------------------------------------------------------------------


_MINIMAL_PAYLOAD = {"TransactionDT": 86400, "TransactionAmt": 99.0}

_RICH_PAYLOAD = {
    "TransactionDT": 5_000_000,
    "TransactionAmt": 2_500.0,
    "ProductCD": "W",
    "card1": 12_000.0,
    "card4": "visa",
    "card6": "credit",
    "addr1": 300.0,
    "addr2": 87.0,
    "P_emaildomain": "gmail.com",
    "C1": 3.0,
    "C2": 1.0,
    "D1": 150.0,
    "M1": "T",
    "M4": "M0",
    "V1": 0.5,
    "V2": 3.0,
}


class TestPredictValidRequests:
    def test_minimal_payload_returns_200(self, api_client):
        resp = api_client.post("/predict", json=_MINIMAL_PAYLOAD)
        assert resp.status_code == 200

    def test_rich_payload_returns_200(self, api_client):
        resp = api_client.post("/predict", json=_RICH_PAYLOAD)
        assert resp.status_code == 200

    def test_response_has_fraud_probability(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        assert "fraud_probability" in body

    def test_fraud_probability_in_unit_interval(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        prob = body["fraud_probability"]
        assert isinstance(prob, float)
        assert 0.0 <= prob <= 1.0

    def test_response_has_is_fraud_bool(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        assert "is_fraud" in body
        assert isinstance(body["is_fraud"], bool)

    def test_is_fraud_consistent_with_probability_and_threshold(self, api_client):
        """is_fraud must equal (fraud_probability >= threshold)."""
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        expected = body["fraud_probability"] >= body["threshold"]
        assert body["is_fraud"] == expected

    def test_response_has_threshold(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        assert "threshold" in body
        assert 0.0 < body["threshold"] < 1.0

    def test_threshold_matches_health_endpoint(self, api_client):
        predict_threshold = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()["threshold"]
        health_threshold  = api_client.get("/health").json()["threshold"]
        assert predict_threshold == health_threshold

    def test_response_has_top_risk_factors(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        assert "top_risk_factors" in body

    def test_exactly_three_risk_factors(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        assert len(body["top_risk_factors"]) == 3

    def test_risk_factor_has_feature_name(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        for rf in body["top_risk_factors"]:
            assert isinstance(rf["feature"], str)
            assert len(rf["feature"]) > 0

    def test_risk_factor_has_shap_value_float(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        for rf in body["top_risk_factors"]:
            assert isinstance(rf["shap_value"], float)

    def test_risk_factor_direction_valid(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        for rf in body["top_risk_factors"]:
            assert rf["direction"] in ("increases_risk", "decreases_risk")

    def test_risk_factor_direction_consistent_with_shap_value(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        for rf in body["top_risk_factors"]:
            if rf["shap_value"] > 0:
                assert rf["direction"] == "increases_risk"
            else:
                assert rf["direction"] == "decreases_risk"

    def test_risk_factor_feature_value_is_float_or_null(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        for rf in body["top_risk_factors"]:
            assert rf["feature_value"] is None or isinstance(rf["feature_value"], float)

    def test_model_version_present(self, api_client):
        body = api_client.post("/predict", json=_MINIMAL_PAYLOAD).json()
        assert "model_version" in body

    def test_large_amount_increases_fraud_probability(self, api_client):
        """
        A $5000 transaction should be scored higher risk than a $1 transaction
        when all other features are identical.
        """
        low  = api_client.post("/predict", json={**_MINIMAL_PAYLOAD, "TransactionAmt": 1.0}).json()
        high = api_client.post("/predict", json={**_MINIMAL_PAYLOAD, "TransactionAmt": 5_000.0}).json()
        assert high["fraud_probability"] > low["fraud_probability"]

    def test_extra_v_features_accepted(self, api_client):
        """V1–V339 and id_* fields should be passed through as extra fields."""
        payload = {**_MINIMAL_PAYLOAD, "V1": 0.1, "V2": 3.5, "V10": 1.2, "id_01": -5.0}
        resp = api_client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_all_m_flags_accepted(self, api_client):
        payload = {
            **_MINIMAL_PAYLOAD,
            "M1": "T", "M2": "F", "M3": "T",
            "M4": "M2",
            "M5": "F", "M6": "T", "M7": "F", "M8": "T", "M9": "F",
        }
        resp = api_client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_response_reproducible(self, api_client):
        """Same input must produce the same output (model is deterministic)."""
        r1 = api_client.post("/predict", json=_RICH_PAYLOAD).json()
        r2 = api_client.post("/predict", json=_RICH_PAYLOAD).json()
        assert r1["fraud_probability"] == r2["fraud_probability"]
        assert r1["is_fraud"] == r2["is_fraud"]


# ---------------------------------------------------------------------------
# POST /predict — validation errors (422)
# ---------------------------------------------------------------------------


class TestPredictValidationErrors:
    def test_missing_transaction_dt_returns_422(self, api_client):
        resp = api_client.post("/predict", json={"TransactionAmt": 50.0})
        assert resp.status_code == 422

    def test_missing_transaction_amt_returns_422(self, api_client):
        resp = api_client.post("/predict", json={"TransactionDT": 1000})
        assert resp.status_code == 422

    def test_negative_amount_returns_422(self, api_client):
        resp = api_client.post("/predict", json={"TransactionDT": 1000, "TransactionAmt": -1.0})
        assert resp.status_code == 422

    def test_zero_amount_returns_422(self, api_client):
        resp = api_client.post("/predict", json={"TransactionDT": 1000, "TransactionAmt": 0.0})
        assert resp.status_code == 422

    def test_amount_over_cap_returns_422(self, api_client):
        resp = api_client.post("/predict", json={"TransactionDT": 1000, "TransactionAmt": 2_000_000.0})
        assert resp.status_code == 422

    def test_bad_m4_value_returns_422(self, api_client):
        resp = api_client.post("/predict", json={**_MINIMAL_PAYLOAD, "M4": "X99"})
        assert resp.status_code == 422

    def test_bad_m1_value_returns_422(self, api_client):
        resp = api_client.post("/predict", json={**_MINIMAL_PAYLOAD, "M1": "yes"})
        assert resp.status_code == 422

    def test_bad_m5_value_returns_422(self, api_client):
        resp = api_client.post("/predict", json={**_MINIMAL_PAYLOAD, "M5": "1"})
        assert resp.status_code == 422

    def test_422_body_contains_detail(self, api_client):
        resp = api_client.post("/predict", json={"TransactionAmt": 50.0})
        body = resp.json()
        assert "detail" in body

    def test_422_detail_identifies_field(self, api_client):
        resp = api_client.post("/predict", json={"TransactionAmt": 50.0})
        detail = resp.json()["detail"]
        field_names = [str(err.get("loc", [])) for err in detail]
        assert any("TransactionDT" in f for f in field_names)

    def test_empty_body_returns_422(self, api_client):
        resp = api_client.post("/predict", json={})
        assert resp.status_code == 422

    def test_non_json_body_returns_422(self, api_client):
        resp = api_client.post(
            "/predict",
            content="not json at all",
            headers={"content-type": "application/json"},
        )
        assert resp.status_code == 422
