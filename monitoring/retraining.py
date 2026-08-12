"""
retraining.py
-------------
Drift-triggered retraining for the fraud detection model.

The important design decision here is that we do NOT retrain on production
traffic. The API only ever sees the 30 input features; whether a transaction
turned out to be fraud is something a human or a chargeback confirms days
later. Retraining on model predictions would just teach the model its own
mistakes, so this script requires a batch that already carries the ``Class``
label and refuses to run without one.

The sequence is:

  1. check the drift signal produced by monitoring/drift.py
  2. validate the newly labelled batch (schema, labels, no empty batch)
  3. rebuild the training frame as  original training split + new batch
  4. train a candidate with the SAME pipeline and parameters as Phase 1
     (StandardScaler on Time/Amount + RandomForestClassifier from params.yaml)
  5. score the candidate AND the model currently in production on the
     untouched held-out test split
  6. log everything to MLflow
  7. promote the candidate only if it clears the acceptance criteria
  8. if it does not, keep the existing model and say so

The test split is never added to training data, and the model family never
changes - a Random Forest is retrained as a Random Forest.

Usage
    # monitoring only, no training
    python monitoring/retraining.py monitor
    python monitoring/retraining.py monitor \
        --current monitoring/data/current_normal.csv --name reference

    # retrain from a labelled batch (drift must have been detected first)
    python monitoring/retraining.py retrain --labeled-data monitoring/data/labeled_batch.csv

    # retrain regardless of the drift signal
    python monitoring/retraining.py retrain --labeled-data <csv> --skip-drift-check
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402

from monitoring.drift import (  # noqa: E402
    DEFAULT_REFERENCE,
    DEFAULT_REPORTS_DIR,
    load_summary,
    run_drift_report,
)
from train import build_pipeline  # noqa: E402  - Phase 1 pipeline factory
from utils import load_params  # noqa: E402

TARGET_COLUMN = "Class"
FEATURE_ORDER = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
PREVIOUS_MODEL_PATH = PROJECT_ROOT / "models" / "model_previous.pkl"
CANDIDATE_MODEL_PATH = PROJECT_ROOT / "models" / "model_candidate.pkl"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"
RETRAIN_REPORT_PATH = PROJECT_ROOT / "monitoring" / "reports" / "retraining_summary.json"


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def evaluate(model: Any, test_df: pd.DataFrame) -> dict[str, float]:
    """Score a model on the held-out test split with imbalance-aware metrics."""
    x_test = test_df[FEATURE_ORDER]
    y_test = test_df[TARGET_COLUMN]
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
    }


def validate_labeled_batch(path: Path) -> pd.DataFrame:
    """Load and sanity-check a newly labelled batch."""
    if not path.exists():
        raise FileNotFoundError(f"Labelled batch not found: {path}")

    batch = pd.read_csv(path)

    if TARGET_COLUMN not in batch.columns:
        raise ValueError(
            f"'{path}' has no '{TARGET_COLUMN}' column. Retraining needs labelled "
            "data - unlabelled production requests cannot be used as training data."
        )

    missing = [c for c in FEATURE_ORDER if c not in batch.columns]
    if missing:
        raise ValueError(f"'{path}' is missing feature column(s): {', '.join(missing)}")

    if batch.empty:
        raise ValueError(f"'{path}' contains no rows.")

    labels = set(batch[TARGET_COLUMN].unique().tolist())
    if not labels.issubset({0, 1}):
        raise ValueError(
            f"'{path}' has unexpected label values {sorted(labels)}; expected 0/1."
        )
    if len(labels) < 2:
        raise ValueError(
            f"'{path}' contains only class {labels.pop()}. A retraining batch needs "
            "both legitimate and fraudulent examples."
        )

    return batch[FEATURE_ORDER + [TARGET_COLUMN]]


def decide_promotion(
    candidate: dict[str, float],
    current: dict[str, float],
    min_pr_auc_gain: float,
    max_recall_drop: float,
) -> tuple[bool, str]:
    """Acceptance criteria for replacing the deployed model.

    PR-AUC is the headline metric because the positive class is 0.17% of the
    data, and recall is guarded separately: a candidate that wins on PR-AUC by
    catching less fraud is not an improvement for this use case.
    """
    pr_gain = candidate["pr_auc"] - current["pr_auc"]
    recall_drop = current["recall"] - candidate["recall"]

    if pr_gain < min_pr_auc_gain:
        return False, (
            f"candidate PR-AUC {candidate['pr_auc']:.4f} vs current "
            f"{current['pr_auc']:.4f} (gain {pr_gain:+.4f}) does not meet the "
            f"required gain of {min_pr_auc_gain:+.4f}"
        )
    if recall_drop > max_recall_drop:
        return False, (
            f"candidate recall {candidate['recall']:.4f} is {recall_drop:.4f} below "
            f"current {current['recall']:.4f}, more than the allowed "
            f"{max_recall_drop:.4f}"
        )
    return True, (
        f"PR-AUC {pr_gain:+.4f} and recall within tolerance "
        f"(candidate {candidate['recall']:.4f} vs current {current['recall']:.4f})"
    )


# ----------------------------------------------------------------------
# commands
# ----------------------------------------------------------------------
def command_monitor(args: argparse.Namespace) -> int:
    """Run the drift check only. No training happens here."""
    summary = run_drift_report(
        reference_path=Path(args.reference),
        current_path=Path(args.current),
        name=args.name,
        reports_dir=Path(args.reports_dir),
    )
    print(
        f"[monitor] {summary['scenario']}: dataset_drift="
        f"{summary['dataset_drift_detected']} "
        f"({summary['drifted_columns_count']}/{summary['n_monitored_columns']} columns)"
    )
    print(f"[monitor] report: {summary['html_report']}")
    if summary["dataset_drift_detected"]:
        print(
            "[monitor] Drift detected. Retraining is NOT automatic - it needs a "
            "labelled batch:\n"
            "          python monitoring/retraining.py retrain --labeled-data <csv>"
        )
    return 0


def command_retrain(args: argparse.Namespace) -> int:
    reason = args.reason
    drift_summary_path = Path(args.drift_summary)

    # ---- 1. drift condition ------------------------------------------------
    if args.skip_drift_check:
        reason = reason or "manual run with --skip-drift-check"
        print("[retrain] Drift check skipped by flag.")
    else:
        if not drift_summary_path.exists():
            print(
                f"ERROR: no drift summary at '{drift_summary_path}'. Run "
                "`python monitoring/retraining.py monitor --current <csv> --name drift` "
                "first, or pass --skip-drift-check.",
                file=sys.stderr,
            )
            return 1
        drift_summary = load_summary(drift_summary_path)
        if not drift_summary.get("dataset_drift_detected"):
            print(
                f"[retrain] No dataset drift in '{drift_summary_path.name}' "
                f"({drift_summary.get('drifted_columns_count')}/"
                f"{drift_summary.get('n_monitored_columns')} columns drifted). "
                "Nothing to do - the deployed model stays as it is."
            )
            return 0
        reason = reason or (
            f"data drift detected: {drift_summary['drifted_columns_count']}/"
            f"{drift_summary['n_monitored_columns']} columns "
            f"(share {drift_summary['drifted_columns_share']:.3f})"
        )
        print(f"[retrain] Drift condition met - {reason}")

    # ---- 2. labelled data --------------------------------------------------
    batch = validate_labeled_batch(Path(args.labeled_data))
    print(
        f"[retrain] Labelled batch accepted: {len(batch)} rows, "
        f"{int(batch[TARGET_COLUMN].sum())} fraud."
    )

    for path in (TRAIN_PATH, TEST_PATH, MODEL_PATH):
        if not path.exists():
            print(f"ERROR: missing '{path}'. Run `dvc repro` first.", file=sys.stderr)
            return 1

    # ---- 3. build the new training frame -----------------------------------
    train_df = pd.read_csv(TRAIN_PATH)[FEATURE_ORDER + [TARGET_COLUMN]]
    test_df = pd.read_csv(TEST_PATH)[FEATURE_ORDER + [TARGET_COLUMN]]
    combined = pd.concat([train_df, batch], ignore_index=True)
    print(
        f"[retrain] Training frame: {len(train_df)} original + {len(batch)} new "
        f"= {len(combined)} rows. Test split ({len(test_df)} rows) untouched."
    )

    # ---- 4. train the candidate with the Phase 1 recipe ---------------------
    params = load_params(str(PROJECT_ROOT / "params.yaml"))
    model_params = params["model"]
    mlflow_cfg = params["mlflow"]

    candidate = build_pipeline(model_params, FEATURE_ORDER)
    print(f"[retrain] Training candidate RandomForest with params: {model_params}")
    candidate.fit(combined[FEATURE_ORDER], combined[TARGET_COLUMN])

    # ---- 5. compare on the held-out test split ------------------------------
    current_model = joblib.load(MODEL_PATH)
    current_metrics = evaluate(current_model, test_df)
    candidate_metrics = evaluate(candidate, test_df)
    print(f"[retrain] current  : {current_metrics}")
    print(f"[retrain] candidate: {candidate_metrics}")

    promote, decision_reason = decide_promotion(
        candidate_metrics, current_metrics, args.min_pr_auc_gain, args.max_recall_drop
    )
    status = "promoted" if promote else "rejected"
    print(f"[retrain] Decision: {status.upper()} - {decision_reason}")

    CANDIDATE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(candidate, CANDIDATE_MODEL_PATH)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
        "candidate_status": status,
        "decision_reason": decision_reason,
        "acceptance_criteria": {
            "min_pr_auc_gain": args.min_pr_auc_gain,
            "max_recall_drop": args.max_recall_drop,
        },
        "labeled_batch": str(Path(args.labeled_data).name),
        "labeled_batch_rows": int(len(batch)),
        "labeled_batch_fraud_rows": int(batch[TARGET_COLUMN].sum()),
        "training_rows_original": int(len(train_df)),
        "training_rows_total": int(len(combined)),
        "test_rows": int(len(test_df)),
        "model_params": model_params,
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
    }

    # ---- 6. MLflow ----------------------------------------------------------
    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])
    run_name = f"retrain_{datetime.now().strftime('%Y%m%d-%H%M%S')}_{status}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(model_params)
        mlflow.log_params(
            {
                "labeled_batch_rows": len(batch),
                "training_rows_total": len(combined),
                "min_pr_auc_gain": args.min_pr_auc_gain,
                "max_recall_drop": args.max_recall_drop,
            }
        )
        mlflow.log_metrics({f"candidate_{k}": v for k, v in candidate_metrics.items()})
        mlflow.log_metrics({f"current_{k}": v for k, v in current_metrics.items()})
        mlflow.set_tag("stage", "retrain")
        mlflow.set_tag("candidate_status", status)
        mlflow.set_tag("retraining_reason", reason)
        mlflow.set_tag("dataset", "creditcard_fraud")

        RETRAIN_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(RETRAIN_REPORT_PATH, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        mlflow.log_artifact(str(RETRAIN_REPORT_PATH))
        if drift_summary_path.exists():
            mlflow.log_artifact(str(drift_summary_path))
        if promote:
            mlflow.sklearn.log_model(candidate, name="model")

        run_id = mlflow.active_run().info.run_id
    print(f"[retrain] MLflow run logged: {run_name} (run_id={run_id})")
    summary["mlflow_run_id"] = run_id

    # ---- 7/8. promote or keep the existing model ----------------------------
    if promote:
        shutil.copy2(MODEL_PATH, PREVIOUS_MODEL_PATH)
        shutil.copy2(CANDIDATE_MODEL_PATH, MODEL_PATH)
        print(
            f"[retrain] Promoted. Previous model kept at "
            f"{PREVIOUS_MODEL_PATH.relative_to(PROJECT_ROOT)}."
        )
    else:
        print(
            "[retrain] Candidate rejected; models/model.pkl is unchanged. "
            f"The candidate is kept at {CANDIDATE_MODEL_PATH.relative_to(PROJECT_ROOT)} "
            "for inspection."
        )

    summary["promoted"] = promote
    with open(RETRAIN_REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[retrain] Summary written to {RETRAIN_REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drift monitoring and drift-triggered retraining."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    monitor = sub.add_parser("monitor", help="Run the Evidently drift check only.")
    monitor.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    monitor.add_argument(
        "--current",
        default=str(PROJECT_ROOT / "monitoring/data/current_drifted.csv"),
    )
    monitor.add_argument("--name", default="drift")
    monitor.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    monitor.set_defaults(func=command_monitor)

    retrain = sub.add_parser("retrain", help="Retrain from a newly labelled batch.")
    retrain.add_argument(
        "--labeled-data",
        required=True,
        help="CSV with the 30 features AND the Class label.",
    )
    retrain.add_argument(
        "--drift-summary",
        default=str(DEFAULT_REPORTS_DIR / "drift_summary.json"),
    )
    retrain.add_argument("--skip-drift-check", action="store_true")
    retrain.add_argument("--reason", default=None)
    retrain.add_argument("--min-pr-auc-gain", type=float, default=0.0)
    retrain.add_argument("--max-recall-drop", type=float, default=0.02)
    retrain.set_defaults(func=command_retrain)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
