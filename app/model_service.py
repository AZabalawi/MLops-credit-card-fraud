"""
model_service.py
----------------
Loads and serves the trained Phase 1 pipeline.

The artifact is the full scikit-learn ``Pipeline`` written by
``src/train.py`` (``models/model.pkl``): a ``ColumnTransformer`` that scales
only ``Time`` and ``Amount`` followed by the ``RandomForestClassifier``.
Serving the whole pipeline - rather than the bare classifier - is what keeps
preprocessing at inference time byte-for-byte identical to training.

The model is read from disk exactly once, when the API starts up. Every
request then reuses the same in-memory object; reloading a ~30 MB forest per
request would dominate the response time for no benefit.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

# Column order the pipeline was fitted on (the raw Kaggle CSV order, minus the
# target). ColumnTransformer resolves "Time"/"Amount" by name but forwards the
# remaining columns positionally, so the order matters and is asserted below.
FEATURE_ORDER: list[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

DEFAULT_MODEL_PATH = "models/model.pkl"
DEFAULT_THRESHOLD = 0.5

LABELS = {0: "legitimate", 1: "fraud"}


class ModelNotLoadedError(RuntimeError):
    """Raised when a prediction is attempted before the model is available."""


class ModelService:
    """Thin wrapper around the trained pipeline.

    Parameters
    ----------
    model_path:
        Location of the joblib artifact. Defaults to the ``MODEL_PATH``
        environment variable, then to ``models/model.pkl``.
    threshold:
        Probability cut-off used to turn ``predict_proba`` into a class.
        Defaults to 0.5, which reproduces scikit-learn's own ``predict`` for a
        binary forest and therefore matches the Phase 1 evaluation numbers.
    """

    def __init__(
        self,
        model_path: str | os.PathLike[str] | None = None,
        threshold: float | None = None,
    ) -> None:
        self.model_path = Path(
            model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
        )
        self.threshold = float(
            threshold
            if threshold is not None
            else os.getenv("DECISION_THRESHOLD", DEFAULT_THRESHOLD)
        )
        self._model: Any = None
        self._lock = threading.Lock()
        self.load_error: str | None = None
        self.model_sha256: str | None = None
        self.feature_order: list[str] = list(FEATURE_ORDER)

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------
    def load(self) -> bool:
        """Load the artifact from disk. Returns True on success.

        Failures are recorded in ``load_error`` instead of raising, so the API
        can start, report itself unhealthy, and be diagnosed over HTTP rather
        than crash-looping on the cloud platform.
        """
        with self._lock:
            try:
                if not self.model_path.exists():
                    raise FileNotFoundError(
                        f"Model artifact not found at '{self.model_path}'. "
                        "Run `dvc repro` (or `python src/train.py`) to build it."
                    )

                model = joblib.load(self.model_path)

                if not hasattr(model, "predict_proba"):
                    raise TypeError(
                        "Loaded object does not expose predict_proba(); "
                        "expected the scikit-learn Pipeline from src/train.py."
                    )

                # Prefer the column order recorded at fit time over our constant.
                fitted_names = getattr(model, "feature_names_in_", None)
                if fitted_names is not None:
                    self.feature_order = [str(c) for c in fitted_names]

                if len(self.feature_order) != len(FEATURE_ORDER):
                    raise ValueError(
                        f"Model expects {len(self.feature_order)} features but the "
                        f"API schema defines {len(FEATURE_ORDER)}."
                    )

                self.model_sha256 = self._hash_file(self.model_path)
                self._model = model
                self.load_error = None
                return True
            except Exception as exc:  # noqa: BLE001 - reported over /health
                self._model = None
                self.load_error = f"{type(exc).__name__}: {exc}"
                return False

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        """Non-sensitive description of the served model."""
        if not self.is_loaded:
            raise ModelNotLoadedError(self.load_error or "Model is not loaded.")

        classifier = self._model
        preprocessing = None
        if hasattr(self._model, "named_steps"):
            classifier = self._model.named_steps.get("model", self._model)
            preprocessing = (
                "StandardScaler on ['Time', 'Amount']; V1-V28 passed through "
                "unchanged (they are already PCA components)"
            )

        details: dict[str, Any] = {
            "model_type": type(classifier).__name__,
            "pipeline_type": type(self._model).__name__,
            "problem_type": "binary classification (fraud detection)",
            "expected_feature_count": len(self.feature_order),
            "feature_names": self.feature_order,
            "target": "Class (0 = legitimate, 1 = fraud)",
            "decision_threshold": self.threshold,
            "preprocessing": preprocessing,
            "artifact_sha256": self.model_sha256,
        }

        for attribute in ("n_estimators", "max_depth", "min_samples_split", "class_weight"):
            if hasattr(classifier, attribute):
                details[attribute] = getattr(classifier, attribute)

        classes = getattr(classifier, "classes_", None)
        if classes is not None:
            details["classes"] = [int(c) for c in classes]

        return details

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------
    def predict_one(self, features: dict[str, float]) -> dict[str, Any]:
        """Score a single transaction.

        ``features`` must contain every column in ``feature_order``; the
        Pydantic layer guarantees that before we get here.
        """
        if not self.is_loaded:
            raise ModelNotLoadedError(self.load_error or "Model is not loaded.")

        missing = [c for c in self.feature_order if c not in features]
        if missing:
            raise ValueError(f"Missing required feature(s): {', '.join(missing)}")

        frame = pd.DataFrame([{c: features[c] for c in self.feature_order}])
        frame = frame[self.feature_order]

        probability = float(self._model.predict_proba(frame)[0][1])
        predicted = int(probability >= self.threshold)

        return {
            "predicted_class": predicted,
            "label": LABELS[predicted],
            "fraud_probability": round(probability, 6),
            "decision_threshold": self.threshold,
            "is_fraud": bool(predicted),
        }


# Module-level singleton used by the FastAPI application.
model_service = ModelService()
