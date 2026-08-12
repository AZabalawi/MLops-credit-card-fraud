"""
export_serving_assets.py
------------------------
Small helper that pulls REAL example transactions out of the held-out test
split and writes them next to the API.

Why this exists: the Swagger page needs an example body, and the live demo
needs request payloads that are guaranteed to be valid. Rather than typing
plausible-looking numbers by hand, we export actual rows from the project's
own data and record what the deployed model predicts for them, so the demo
script cannot claim an output the model does not produce.

Outputs
    app/example_transaction.json                     shown in /docs
    presentation/sample_requests/legitimate.json     demo payload (Class 0)
    presentation/sample_requests/fraud.json          demo payload (Class 1)
    presentation/sample_requests/expected_output.json  what the model returns

Usage
    python src/export_serving_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
APP_EXAMPLE = PROJECT_ROOT / "app" / "example_transaction.json"
SAMPLES_DIR = PROJECT_ROOT / "presentation" / "sample_requests"

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def _row_to_payload(row: pd.Series) -> dict[str, float]:
    return {column: float(row[column]) for column in FEATURE_ORDER}


def main() -> None:
    for path in (TEST_PATH, MODEL_PATH):
        if not path.exists():
            raise SystemExit(f"Missing '{path}'. Run `dvc repro` first.")

    test_df = pd.read_csv(TEST_PATH)
    model = joblib.load(MODEL_PATH)

    # Deterministic picks: the first correctly-predicted row of each class, so
    # the demo shows the model behaving as advertised.
    features = test_df[FEATURE_ORDER]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    payloads: dict[str, dict[str, float]] = {}
    outputs: dict[str, dict[str, object]] = {}

    for label, class_value in (("legitimate", 0), ("fraud", 1)):
        mask = (test_df[TARGET_COLUMN] == class_value) & (predictions == class_value)
        if not mask.any():
            raise SystemExit(
                f"No correctly predicted '{label}' row found in the test split."
            )
        index = int(mask.idxmax())
        payloads[label] = _row_to_payload(test_df.loc[index])
        probability = float(probabilities[index])
        outputs[label] = {
            "source_row_index_in_test_csv": index,
            "true_class": class_value,
            "predicted_class": int(predictions[index]),
            "fraud_probability": round(probability, 6),
            "label": "fraud" if predictions[index] == 1 else "legitimate",
        }

    APP_EXAMPLE.parent.mkdir(parents=True, exist_ok=True)
    with open(APP_EXAMPLE, "w", encoding="utf-8") as handle:
        json.dump(payloads["legitimate"], handle, indent=2)

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for label, payload in payloads.items():
        with open(SAMPLES_DIR / f"{label}.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
    with open(SAMPLES_DIR / "expected_output.json", "w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2)

    print(f"Wrote {APP_EXAMPLE.relative_to(PROJECT_ROOT)}")
    for label in payloads:
        print(f"Wrote {(SAMPLES_DIR / f'{label}.json').relative_to(PROJECT_ROOT)}")
        print(f"  model output -> {outputs[label]}")


if __name__ == "__main__":
    main()
