"""
schemas.py
----------
Pydantic request/response models for the fraud detection API.

The request mirrors one row of the Kaggle dataset with the target removed:
``Time``, the 28 anonymised PCA components ``V1``-``V28``, and ``Amount``.
``extra="forbid"`` means a request carrying ``Class`` - or any typo such as
``V29`` - is rejected with 422 instead of being silently ignored, which is the
behaviour we want for a schema this rigid.

The Swagger example is loaded from ``app/example_transaction.json``, a real
transaction exported from the project's own test split, so the example shown
in ``/docs`` is guaranteed to be a valid input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_EXAMPLE_PATH = Path(__file__).with_name("example_transaction.json")


def _load_example() -> dict[str, Any] | None:
    try:
        with open(_EXAMPLE_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


_EXAMPLE = _load_example()

_PCA_DESCRIPTION = (
    "Anonymised principal component supplied with the public dataset. "
    "The original features were transformed with PCA for confidentiality."
)


class TransactionRequest(BaseModel):
    """One credit-card transaction to score (30 features, no target)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": _EXAMPLE} if _EXAMPLE else {},
    )

    Time: float = Field(
        ...,
        description="Seconds elapsed between this transaction and the first "
        "transaction in the dataset.",
    )
    V1: float = Field(..., description=_PCA_DESCRIPTION)
    V2: float = Field(..., description=_PCA_DESCRIPTION)
    V3: float = Field(..., description=_PCA_DESCRIPTION)
    V4: float = Field(..., description=_PCA_DESCRIPTION)
    V5: float = Field(..., description=_PCA_DESCRIPTION)
    V6: float = Field(..., description=_PCA_DESCRIPTION)
    V7: float = Field(..., description=_PCA_DESCRIPTION)
    V8: float = Field(..., description=_PCA_DESCRIPTION)
    V9: float = Field(..., description=_PCA_DESCRIPTION)
    V10: float = Field(..., description=_PCA_DESCRIPTION)
    V11: float = Field(..., description=_PCA_DESCRIPTION)
    V12: float = Field(..., description=_PCA_DESCRIPTION)
    V13: float = Field(..., description=_PCA_DESCRIPTION)
    V14: float = Field(..., description=_PCA_DESCRIPTION)
    V15: float = Field(..., description=_PCA_DESCRIPTION)
    V16: float = Field(..., description=_PCA_DESCRIPTION)
    V17: float = Field(..., description=_PCA_DESCRIPTION)
    V18: float = Field(..., description=_PCA_DESCRIPTION)
    V19: float = Field(..., description=_PCA_DESCRIPTION)
    V20: float = Field(..., description=_PCA_DESCRIPTION)
    V21: float = Field(..., description=_PCA_DESCRIPTION)
    V22: float = Field(..., description=_PCA_DESCRIPTION)
    V23: float = Field(..., description=_PCA_DESCRIPTION)
    V24: float = Field(..., description=_PCA_DESCRIPTION)
    V25: float = Field(..., description=_PCA_DESCRIPTION)
    V26: float = Field(..., description=_PCA_DESCRIPTION)
    V27: float = Field(..., description=_PCA_DESCRIPTION)
    V28: float = Field(..., description=_PCA_DESCRIPTION)
    Amount: float = Field(..., ge=0, description="Transaction amount.")


class PredictionResponse(BaseModel):
    """Model output for a single transaction."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "example": {
                "predicted_class": 0,
                "label": "legitimate",
                "fraud_probability": 0.02,
                "decision_threshold": 0.5,
                "is_fraud": False,
                "model_version": "2.0.0",
            }
        },
    )

    predicted_class: int = Field(..., description="0 = legitimate, 1 = fraud.")
    label: str = Field(..., description="Human-readable form of predicted_class.")
    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of the fraud class from the Random Forest.",
    )
    decision_threshold: float = Field(
        ..., description="Probability cut-off applied to obtain predicted_class."
    )
    is_fraud: bool = Field(..., description="True when predicted_class is 1.")
    model_version: str = Field(..., description="Version of the serving application.")


class HealthResponse(BaseModel):
    """Liveness/readiness answer. ``status`` is only 'healthy' when the model
    actually loaded - a service that cannot score anything is not healthy."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="'healthy' or 'unhealthy'.")
    model_loaded: bool = Field(..., description="Whether the pipeline is in memory.")
    model_path: str = Field(..., description="Artifact path the service tried to load.")
    version: str = Field(..., description="Serving application version.")
    detail: str | None = Field(
        None, description="Loading error message when the model is unavailable."
    )


class ModelInfoResponse(BaseModel):
    """Metadata about the served pipeline."""

    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    model_type: str
    pipeline_type: str
    problem_type: str
    expected_feature_count: int
    feature_names: list[str]
    target: str
    decision_threshold: float
    preprocessing: str | None = None
    artifact_sha256: str | None = None
    model_version: str


class RootResponse(BaseModel):
    """Short description of the service returned by ``GET /``."""

    service: str
    version: str
    description: str
    docs_url: str
    health_url: str
    predict_url: str
