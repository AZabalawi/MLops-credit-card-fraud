"""Tests for the model loading/inference layer."""

from __future__ import annotations

import pytest

from app.model_service import (
    FEATURE_ORDER,
    ModelNotLoadedError,
    ModelService,
    model_service,
)
from tests.conftest import MODEL_PATH


def test_the_real_artifact_is_present():
    """CI must fail loudly rather than silently test nothing."""
    assert MODEL_PATH.exists(), (
        f"{MODEL_PATH} is missing. The API is meant to serve the Phase 1 model; "
        "run `dvc repro` to rebuild it."
    )


def test_feature_order_is_the_thirty_training_columns():
    assert len(FEATURE_ORDER) == 30
    assert FEATURE_ORDER[0] == "Time"
    assert FEATURE_ORDER[-1] == "Amount"
    assert FEATURE_ORDER[1:29] == [f"V{i}" for i in range(1, 29)]
    assert "Class" not in FEATURE_ORDER


def test_service_loads_the_model_and_records_a_checksum():
    service = ModelService(model_path=MODEL_PATH)
    assert service.load() is True
    assert service.is_loaded is True
    assert service.load_error is None
    assert service.model_sha256 and len(service.model_sha256) == 64
    assert service.feature_order == FEATURE_ORDER


def test_missing_artifact_is_reported_not_raised(tmp_path):
    service = ModelService(model_path=tmp_path / "nope.pkl")
    assert service.load() is False
    assert service.is_loaded is False
    assert "FileNotFoundError" in service.load_error


def test_predicting_before_loading_raises(tmp_path, sample_transaction):
    service = ModelService(model_path=tmp_path / "nope.pkl")
    service.load()
    with pytest.raises(ModelNotLoadedError):
        service.predict_one(sample_transaction)
    with pytest.raises(ModelNotLoadedError):
        service.info()


def test_predict_one_returns_a_probability_in_range(sample_transaction):
    result = model_service.predict_one(sample_transaction)
    assert result["predicted_class"] in (0, 1)
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert result["label"] in ("legitimate", "fraud")
    assert result["decision_threshold"] == 0.5


def test_predict_one_rejects_a_missing_feature(sample_transaction):
    incomplete = {k: v for k, v in sample_transaction.items() if k != "V3"}
    with pytest.raises(ValueError, match="V3"):
        model_service.predict_one(incomplete)


def test_threshold_controls_the_decision(sample_transaction):
    """Lowering the cut-off to zero must flag everything as fraud."""
    service = ModelService(model_path=MODEL_PATH, threshold=0.0)
    assert service.load() is True
    assert service.predict_one(sample_transaction)["predicted_class"] == 1


def test_class_matches_sklearn_predict(sample_transaction):
    """At threshold 0.5 the API decision equals the pipeline's own predict()."""
    import pandas as pd

    frame = pd.DataFrame([sample_transaction])[model_service.feature_order]
    sklearn_class = int(model_service._model.predict(frame)[0])
    assert model_service.predict_one(sample_transaction)["predicted_class"] == sklearn_class


def test_info_describes_the_random_forest():
    details = model_service.info()
    assert details["model_type"] == "RandomForestClassifier"
    assert details["expected_feature_count"] == 30
    assert details["problem_type"].startswith("binary classification")
    assert details["n_estimators"] > 0
    assert "Time" in details["preprocessing"]
