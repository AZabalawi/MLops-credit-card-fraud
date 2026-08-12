"""
generate_drift_data.py
----------------------
Builds the datasets the monitoring stage compares.

We are not running a real payment system, so there is no live production
traffic to monitor. What we do instead is take real rows from our own
processed splits and build three batches:

  reference_sample.csv  a seeded sample of the TRAINING split. This is the
                        distribution the deployed model learned, and it is
                        what every current batch is compared against. It is
                        written once and never modified afterwards.

  current_normal.csv    a seeded sample of the TEST split, untouched. This
                        stands in for a healthy production day: same data
                        generating process, so Evidently should report no
                        dataset drift.

  current_drifted.csv   the same rows with a deterministic transformation
                        applied to 17 of the 30 columns. THIS IS SIMULATED
                        DRIFT FOR DEMONSTRATION - it is not real customer
                        behaviour, and it exists only to prove the monitoring
                        pipeline reacts when the input distribution moves.

  labeled_batch.csv     a seeded sample of TRAINING rows with the same shift
                        applied, keeping the Class label. This plays the role
                        of "new labelled data that arrived after the drift"
                        and is the input to the retraining script. It is drawn
                        from the training split on purpose: the held-out test
                        split must stay untouched so it remains an honest
                        yardstick for comparing a candidate against the model
                        currently in production.

Everything is seeded, so re-running this script reproduces byte-identical
files and the drift numbers in the report do not move around.

Usage
    python monitoring/generate_drift_data.py
    python monitoring/generate_drift_data.py --seed 7 --shift-std 2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "monitoring" / "data"

TARGET_COLUMN = "Class"
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Columns the simulated shift touches: 17 of 30, i.e. a share above the 0.5
# dataset-drift threshold Evidently uses by default. The V columns chosen
# include V14, V17 and V12, which are the components most associated with
# fraud in this dataset, so the shift is not just statistically visible but
# also plausible as something that would change model behaviour.
DRIFTED_COLUMNS = (
    [f"V{i}" for i in range(1, 11)] + ["V12", "V14", "V16", "V17", "V18"] + ["Time", "Amount"]
)


def _sample(frame: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """Seeded sample without replacement, capped at the frame size."""
    n_rows = min(n_rows, len(frame))
    return frame.sample(n=n_rows, random_state=seed).reset_index(drop=True)


def apply_simulated_drift(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    shift_std: float,
    amount_factor: float,
    seed: int,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with a reproducible distribution shift.

    The scenario: the service starts receiving traffic from a different
    segment - later in the day, noticeably larger amounts, and PCA components
    centred somewhere else. Each affected V column is moved by
    ``shift_std`` times ITS OWN standard deviation in the reference sample,
    so the shift is scaled to the feature rather than being an arbitrary
    constant.
    """
    rng = np.random.default_rng(seed)
    drifted = frame.copy()

    for column in DRIFTED_COLUMNS:
        if column in ("Time", "Amount"):
            continue
        column_std = float(reference[column].std())
        # alternate the sign so the drift is not one uniform translation
        direction = 1.0 if DRIFTED_COLUMNS.index(column) % 2 == 0 else -1.0
        noise = rng.normal(0.0, 0.25 * column_std, size=len(drifted))
        drifted[column] = drifted[column] + direction * shift_std * column_std + noise

    # Amount: heavier transactions, floored at 0 because a negative amount is
    # not a thing the API would ever accept.
    drifted["Amount"] = (drifted["Amount"] * amount_factor + 75.0).clip(lower=0.0)

    # Time: pretend the batch arrives a full dataset-length later.
    drifted["Time"] = drifted["Time"] + float(reference["Time"].max())

    return drifted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build reference and simulated production batches for monitoring."
    )
    parser.add_argument("--train", default=str(PROJECT_ROOT / "data/processed/train.csv"))
    parser.add_argument("--test", default=str(PROJECT_ROOT / "data/processed/test.csv"))
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    parser.add_argument("--reference-rows", type=int, default=10000)
    parser.add_argument("--current-rows", type=int, default=5000)
    parser.add_argument("--labeled-rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shift-std", type=float, default=1.5)
    parser.add_argument("--amount-factor", type=float, default=2.5)
    args = parser.parse_args()

    train_path, test_path = Path(args.train), Path(args.test)
    for path in (train_path, test_path):
        if not path.exists():
            raise SystemExit(
                f"Missing '{path}'. Run the DVC pipeline first (`dvc repro`) so the "
                "processed splits exist."
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 1. Reference: what the model was trained on.
    reference = _sample(train_df, args.reference_rows, args.seed)[FEATURE_ORDER]
    reference.to_csv(output_dir / "reference_sample.csv", index=False)

    # 2. Healthy production batch: untouched test rows.
    current_normal = _sample(test_df, args.current_rows, args.seed + 1)[FEATURE_ORDER]
    current_normal.to_csv(output_dir / "current_normal.csv", index=False)

    # 3. Drifted production batch: same rows, shifted.
    current_drifted = apply_simulated_drift(
        current_normal, reference, args.shift_std, args.amount_factor, args.seed + 2
    )
    current_drifted.to_csv(output_dir / "current_drifted.csv", index=False)

    # 4. Newly labelled batch for retraining, taken from the training split so
    #    the test split is never used as training data.
    labeled_source = _sample(train_df, args.labeled_rows, args.seed + 3)
    labeled_features = apply_simulated_drift(
        labeled_source[FEATURE_ORDER],
        reference,
        args.shift_std,
        args.amount_factor,
        args.seed + 4,
    )
    labeled_batch = labeled_features.copy()
    labeled_batch[TARGET_COLUMN] = labeled_source[TARGET_COLUMN].to_numpy()
    labeled_batch.to_csv(output_dir / "labeled_batch.csv", index=False)

    manifest = {
        "note": "SIMULATED DRIFT FOR DEMONSTRATION - not real production traffic.",
        "seed": args.seed,
        "shift_std": args.shift_std,
        "amount_factor": args.amount_factor,
        "drifted_columns": DRIFTED_COLUMNS,
        "n_drifted_columns": len(DRIFTED_COLUMNS),
        "n_monitored_columns": len(FEATURE_ORDER),
        "files": {
            "reference_sample.csv": {"rows": len(reference), "source": "data/processed/train.csv"},
            "current_normal.csv": {
                "rows": len(current_normal),
                "source": "data/processed/test.csv",
            },
            "current_drifted.csv": {
                "rows": len(current_drifted),
                "source": "current_normal.csv + simulated shift",
            },
            "labeled_batch.csv": {
                "rows": len(labeled_batch),
                "source": "data/processed/train.csv + simulated shift, labels kept",
                "fraud_rows": int(labeled_batch[TARGET_COLUMN].sum()),
            },
        },
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Wrote monitoring datasets to {output_dir}")
    for name, meta in manifest["files"].items():
        print(f"  {name}: {meta['rows']} rows")


if __name__ == "__main__":
    main()
