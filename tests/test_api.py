"""API-level tests: routing, validation, and the shape of what comes back."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from app.model_service import FEATURE_ORDER, model_service

APP_JS = Path(__file__).resolve().parents[1] / "app" / "static" / "app.js"
EXAMPLES_MARKER = "const EXAMPLES = "


def load_demo_examples() -> dict:
    """Pull the EXAMPLES literal out of the demo's JavaScript."""
    source = APP_JS.read_text(encoding="utf-8")
    start = source.find(EXAMPLES_MARKER)
    assert start != -1, "EXAMPLES literal not found in app/static/app.js"
    obj, _ = json.JSONDecoder().raw_decode(source, start + len(EXAMPLES_MARKER))
    return obj


def test_root_serves_the_demo_page(client):
    """/ is the browser demo; the JSON description moved to /api."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Credit Card Fraud Detection" in response.text
    assert "/static/app.js" in response.text


def test_api_endpoint_returns_service_information(client):
    response = client.get("/api")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "Credit Card Fraud Detection API"
    assert body["docs_url"] == "/docs"


def test_static_assets_are_served(client):
    for asset in ("/static/style.css", "/static/app.js"):
        response = client.get(asset)
        assert response.status_code == 200, asset
        assert response.content


def test_demo_examples_are_accepted_by_the_real_model(client):
    """The buttons in the demo must load payloads the API actually accepts.

    The examples are read out of app.js itself, so this fails if someone edits
    them into something the model cannot score.
    """
    examples = load_demo_examples()
    assert set(examples) == {"legitimate", "fraud"}

    for name, payload in examples.items():
        assert sorted(payload) == sorted(FEATURE_ORDER), f"{name} has the wrong fields"
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, f"{name} was rejected: {response.text}"
        body = response.json()
        assert body["label"] == name
        assert 0.0 <= body["fraud_probability"] <= 1.0


def test_health_is_healthy_and_model_is_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["detail"] is None


def test_swagger_docs_and_openapi_schema_are_served(client):
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json")
    assert schema.status_code == 200
    assert "/predict" in schema.json()["paths"]


def test_model_info_describes_the_real_pipeline(client):
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "RandomForestClassifier"
    assert body["pipeline_type"] == "Pipeline"
    assert body["expected_feature_count"] == 30
    assert body["feature_names"] == FEATURE_ORDER
    assert body["classes"] == [0, 1]


def test_predict_returns_a_valid_prediction(client, sample_transaction):
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200
    body = response.json()

    assert set(body) == {
        "predicted_class",
        "label",
        "fraud_probability",
        "decision_threshold",
        "is_fraud",
        "model_version",
    }
    assert body["predicted_class"] in (0, 1)
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["label"] == ("fraud" if body["predicted_class"] == 1 else "legitimate")
    assert body["is_fraud"] is (body["predicted_class"] == 1)


def test_prediction_matches_the_underlying_pipeline(client, sample_transaction):
    """The API must not post-process the model output in any way."""
    import pandas as pd

    frame = pd.DataFrame([sample_transaction])[model_service.feature_order]
    expected = float(model_service._model.predict_proba(frame)[0][1])

    body = client.post("/predict", json=sample_transaction).json()
    assert body["fraud_probability"] == round(expected, 6)


def test_missing_feature_is_rejected(client, sample_transaction):
    payload = copy.deepcopy(sample_transaction)
    del payload["V14"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert any(err["loc"][-1] == "V14" for err in response.json()["detail"])


def test_wrong_type_is_rejected(client, sample_transaction):
    payload = copy.deepcopy(sample_transaction)
    payload["Amount"] = "not-a-number"
    assert client.post("/predict", json=payload).status_code == 422


def test_unexpected_field_is_rejected(client, sample_transaction):
    payload = copy.deepcopy(sample_transaction)
    payload["Class"] = 1  # the target must never be sent to the API
    assert client.post("/predict", json=payload).status_code == 422


def test_negative_amount_is_rejected(client, sample_transaction):
    payload = copy.deepcopy(sample_transaction)
    payload["Amount"] = -10.0
    assert client.post("/predict", json=payload).status_code == 422


def test_empty_and_malformed_bodies_are_rejected(client):
    assert client.post("/predict", json={}).status_code == 422
    assert client.post("/predict", json={"foo": "bar"}).status_code == 422
    assert (
        client.post(
            "/predict", content=b"{not json", headers={"Content-Type": "application/json"}
        ).status_code
        == 422
    )


def test_health_reports_503_when_the_model_is_unavailable(client):
    """A service that cannot score must not advertise itself as healthy."""
    saved_model, saved_error = model_service._model, model_service.load_error
    model_service._model = None
    model_service.load_error = "simulated load failure"
    try:
        response = client.get("/health")
        assert response.status_code == 503
        body = response.json()
        assert body["status"] == "unhealthy"
        assert body["model_loaded"] is False
        assert body["detail"] == "simulated load failure"

        assert client.get("/model-info").status_code == 503
    finally:
        model_service._model, model_service.load_error = saved_model, saved_error
