"""Shared pytest fixtures.

The API tests run against the REAL Phase 1 artifact (``models/model.pkl``).
Mocking only appears where we deliberately need a broken model to check the
failure path of ``/health``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.main import app  # noqa: E402
from app.model_service import FEATURE_ORDER  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
EXAMPLE_PATH = PROJECT_ROOT / "app" / "example_transaction.json"


@pytest.fixture(scope="session")
def client() -> TestClient:
    """TestClient used as a context manager so the lifespan hook runs."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def sample_transaction() -> dict[str, float]:
    """A real transaction exported from the project's own test split."""
    if not EXAMPLE_PATH.exists():
        pytest.fail(
            f"Missing {EXAMPLE_PATH}. Regenerate it with "
            "`python src/export_serving_assets.py`."
        )
    with open(EXAMPLE_PATH, encoding="utf-8") as handle:
        payload = json.load(handle)
    missing = [c for c in FEATURE_ORDER if c not in payload]
    assert not missing, f"example_transaction.json is missing {missing}"
    return payload
