"""
drift.py
--------
Data drift detection with EvidentlyAI.

Compares a current batch against the fixed reference sample using
``DataDriftPreset`` over the 30 input features. The target column ``Class`` is
deliberately excluded: the API never receives a label at prediction time, so
monitoring the label would tell us nothing about what production actually
sends us.

Evidently picks the statistical test per column from the sample size; for
numerical columns at these volumes that is the Kolmogorov-Smirnov test with a
p-value threshold of 0.05. A column counts as drifted when its test fails, and
the dataset as a whole counts as drifted when the share of drifted columns
exceeds ``--drift-share`` (0.5 by default).

Every number written to the JSON summary comes from the Evidently snapshot -
nothing here is hand-written.

Usage
    # healthy batch, expect no dataset drift
    python monitoring/drift.py --current monitoring/data/current_normal.csv \
        --name reference

    # simulated drifted batch, expect dataset drift
    python monitoring/drift.py --current monitoring/data/current_drifted.csv \
        --name drift

Exit codes
    0  report generated, no dataset drift detected
    1  bad input (missing file, missing columns)
    2  report generated, DATASET DRIFT DETECTED  (with --exit-on-drift)
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import evidently
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = PROJECT_ROOT / "monitoring" / "data" / "reference_sample.csv"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"

FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"

DRIFTED_COUNT_METRIC = "DriftedColumnsCount"
VALUE_DRIFT_METRIC = "ValueDrift"


def _display_path(path: Path) -> str:
    """Show paths relative to the project root when they live inside it.

    Reports written somewhere else (a temporary directory in the tests, for
    example) are recorded with their absolute path instead.
    """
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def load_feature_frame(path: Path) -> pd.DataFrame:
    """Read a CSV and return only the 30 model input columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing '{path}'. Run `python monitoring/generate_drift_data.py` first."
        )
    frame = pd.read_csv(path)
    missing = [c for c in FEATURE_ORDER if c not in frame.columns]
    if missing:
        raise ValueError(f"'{path}' is missing column(s): {', '.join(missing)}")
    return frame[FEATURE_ORDER].copy()


def _column_drift_from_snapshot(snapshot_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Pull the per-column drift results out of the Evidently snapshot."""
    results: dict[str, dict[str, Any]] = {}
    for metric in snapshot_dict.get("metrics", []):
        config = metric.get("config", {})
        if VALUE_DRIFT_METRIC not in str(config.get("type", "")):
            continue
        column = config.get("column")
        if column is None:
            continue
        method = str(config.get("method", ""))
        threshold = config.get("threshold")
        score = metric.get("value")
        # p-value style tests fail when the score falls BELOW the threshold;
        # distance style tests fail when it rises above it.
        if "p_value" in method and threshold is not None and score is not None:
            drifted = bool(score < threshold)
        elif threshold is not None and score is not None:
            drifted = bool(score > threshold)
        else:
            drifted = False
        results[str(column)] = {
            "method": method,
            "score": None if score is None else float(score),
            "threshold": None if threshold is None else float(threshold),
            "drifted": drifted,
        }
    return results


def _dataset_drift_from_snapshot(snapshot_dict: dict[str, Any]) -> dict[str, Any]:
    """Pull Evidently's own drifted-column count and share."""
    for metric in snapshot_dict.get("metrics", []):
        config = metric.get("config", {})
        if DRIFTED_COUNT_METRIC not in str(config.get("type", "")):
            continue
        value = metric.get("value", {}) or {}
        return {
            "drifted_columns_count": int(value.get("count", 0)),
            "drifted_columns_share": float(value.get("share", 0.0)),
            "drift_share_threshold": float(config.get("drift_share", 0.5)),
        }
    raise RuntimeError(
        "Evidently snapshot did not contain a DriftedColumnsCount metric; "
        "the installed Evidently version may be incompatible."
    )


def run_drift_report(
    reference_path: Path,
    current_path: Path,
    name: str,
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    drift_share: float = 0.5,
) -> dict[str, Any]:
    """Run Evidently, write ``<name>_report.html`` + ``<name>_summary.json``.

    Returns the summary dictionary that was written to disk.
    """
    reference = load_feature_frame(reference_path)
    current = load_feature_frame(current_path)

    reports_dir.mkdir(parents=True, exist_ok=True)

    definition = DataDefinition(numerical_columns=FEATURE_ORDER)
    reference_ds = Dataset.from_pandas(reference, data_definition=definition)
    current_ds = Dataset.from_pandas(current, data_definition=definition)

    report = Report([DataDriftPreset(drift_share=drift_share)], include_tests=True)
    snapshot = report.run(current_ds, reference_ds)

    html_path = reports_dir / f"{name}_report.html"
    snapshot.save_html(str(html_path))

    snapshot_dict = snapshot.dict()
    dataset_level = _dataset_drift_from_snapshot(snapshot_dict)
    column_level = _column_drift_from_snapshot(snapshot_dict)

    dataset_drift = (
        dataset_level["drifted_columns_share"] > dataset_level["drift_share_threshold"]
    )

    summary: dict[str, Any] = {
        "scenario": name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evidently_version": evidently.__version__,
        "reference_path": _display_path(reference_path),
        "current_path": _display_path(current_path),
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "n_monitored_columns": len(FEATURE_ORDER),
        "monitored_columns": FEATURE_ORDER,
        "excluded_columns": [TARGET_COLUMN],
        "dataset_drift_detected": bool(dataset_drift),
        **dataset_level,
        "drifted_columns": sorted(
            col for col, res in column_level.items() if res["drifted"]
        ),
        "column_drift": column_level,
        "html_report": _display_path(html_path),
    }

    json_path = reports_dir / f"{name}_summary.json"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def load_summary(path: Path) -> dict[str, Any]:
    """Read a previously generated drift summary."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run EvidentlyAI data drift detection.")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument(
        "--current",
        required=True,
        help="CSV holding the current/production-like batch.",
    )
    parser.add_argument(
        "--name",
        default="drift",
        help="Prefix for the generated report files (e.g. 'drift' or 'reference').",
    )
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--drift-share", type=float, default=0.5)
    parser.add_argument(
        "--exit-on-drift",
        action="store_true",
        help="Exit with code 2 when dataset drift is detected (useful in automation).",
    )
    args = parser.parse_args()

    try:
        summary = run_drift_report(
            reference_path=Path(args.reference),
            current_path=Path(args.current),
            name=args.name,
            reports_dir=Path(args.reports_dir),
            drift_share=args.drift_share,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Scenario           : {summary['scenario']}")
    print(f"Reference rows     : {summary['reference_rows']}")
    print(f"Current rows       : {summary['current_rows']}")
    print(
        f"Drifted columns    : {summary['drifted_columns_count']}"
        f"/{summary['n_monitored_columns']} "
        f"(share {summary['drifted_columns_share']:.3f}, "
        f"threshold {summary['drift_share_threshold']})"
    )
    print(f"Dataset drift      : {summary['dataset_drift_detected']}")
    print(f"HTML report        : {summary['html_report']}")

    if summary["dataset_drift_detected"] and args.exit_on_drift:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
