"""Tests for the drift detection and retraining logic.

These run EvidentlyAI for real on small synthetic frames. Synthetic data is
fine here because we are testing OUR code paths - that a shifted batch is
flagged and an identical one is not. The reports that go in the submission are
generated from the project's own splits by ``monitoring/generate_drift_data.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from monitoring.drift import FEATURE_ORDER, load_feature_frame, run_drift_report
from monitoring.generate_drift_data import DRIFTED_COLUMNS, apply_simulated_drift
from monitoring.retraining import decide_promotion, validate_labeled_batch

ROWS = 800


def _frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {column: rng.normal(size=ROWS) for column in FEATURE_ORDER}
    data["Amount"] = np.abs(data["Amount"]) * 50
    data["Time"] = np.arange(ROWS, dtype=float)
    return pd.DataFrame(data)[FEATURE_ORDER]


def _write(frame: pd.DataFrame, path) -> None:
    frame.to_csv(path, index=False)


# ----------------------------------------------------------------------
# drift detection
# ----------------------------------------------------------------------
def test_identical_batches_show_no_dataset_drift(tmp_path):
    reference = _frame(1)
    ref_path, cur_path = tmp_path / "ref.csv", tmp_path / "cur.csv"
    _write(reference, ref_path)
    _write(reference.copy(), cur_path)

    summary = run_drift_report(ref_path, cur_path, "identical", tmp_path / "reports")

    assert summary["dataset_drift_detected"] is False
    assert summary["drifted_columns_count"] == 0
    assert summary["n_monitored_columns"] == 30
    assert summary["excluded_columns"] == ["Class"]
    assert (tmp_path / "reports" / "identical_report.html").exists()
    assert (tmp_path / "reports" / "identical_summary.json").exists()


def test_shifted_batch_is_flagged_as_drifted(tmp_path):
    reference = _frame(1)
    current = apply_simulated_drift(_frame(2), reference, 1.5, 2.5, seed=3)
    ref_path, cur_path = tmp_path / "ref.csv", tmp_path / "cur.csv"
    _write(reference, ref_path)
    _write(current, cur_path)

    summary = run_drift_report(ref_path, cur_path, "shifted", tmp_path / "reports")

    assert summary["dataset_drift_detected"] is True
    assert summary["drifted_columns_count"] >= len(DRIFTED_COLUMNS) - 2
    assert summary["drifted_columns_share"] > summary["drift_share_threshold"]
    assert "Amount" in summary["drifted_columns"]
    for column, result in summary["column_drift"].items():
        assert 0.0 <= result["score"] <= 1.0, column


def test_simulated_drift_leaves_untouched_columns_alone():
    reference = _frame(1)
    original = _frame(2)
    drifted = apply_simulated_drift(original, reference, 1.5, 2.5, seed=3)

    untouched = [c for c in FEATURE_ORDER if c not in DRIFTED_COLUMNS]
    assert untouched, "the scenario should leave some columns alone"
    for column in untouched:
        pd.testing.assert_series_equal(drifted[column], original[column])
    for column in DRIFTED_COLUMNS:
        assert not drifted[column].equals(original[column])
    assert (drifted["Amount"] >= 0).all()


def test_reference_data_is_never_modified():
    reference = _frame(1)
    before = reference.copy()
    apply_simulated_drift(_frame(2), reference, 1.5, 2.5, seed=3)
    pd.testing.assert_frame_equal(reference, before)


def test_missing_columns_are_rejected(tmp_path):
    bad = _frame(1).drop(columns=["V7"])
    path = tmp_path / "bad.csv"
    _write(bad, path)
    with pytest.raises(ValueError, match="V7"):
        load_feature_frame(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_feature_frame(tmp_path / "does_not_exist.csv")


# ----------------------------------------------------------------------
# retraining guards
# ----------------------------------------------------------------------
def test_unlabelled_batch_is_refused(tmp_path):
    path = tmp_path / "unlabelled.csv"
    _write(_frame(1), path)
    with pytest.raises(ValueError, match="Class"):
        validate_labeled_batch(path)


def test_single_class_batch_is_refused(tmp_path):
    frame = _frame(1)
    frame["Class"] = 0
    path = tmp_path / "one_class.csv"
    _write(frame, path)
    with pytest.raises(ValueError, match="both legitimate and fraudulent"):
        validate_labeled_batch(path)


def test_valid_labelled_batch_is_accepted(tmp_path):
    frame = _frame(1)
    frame["Class"] = 0
    frame.loc[:9, "Class"] = 1
    path = tmp_path / "good.csv"
    _write(frame, path)

    batch = validate_labeled_batch(path)
    assert list(batch.columns) == FEATURE_ORDER + ["Class"]
    assert len(batch) == ROWS


def test_better_candidate_is_promoted():
    current = {"pr_auc": 0.78, "recall": 0.72}
    candidate = {"pr_auc": 0.81, "recall": 0.74}
    promote, reason = decide_promotion(candidate, current, 0.0, 0.02)
    assert promote is True
    assert "PR-AUC" in reason


def test_worse_candidate_is_rejected():
    current = {"pr_auc": 0.78, "recall": 0.72}
    candidate = {"pr_auc": 0.70, "recall": 0.72}
    promote, reason = decide_promotion(candidate, current, 0.0, 0.02)
    assert promote is False
    assert "PR-AUC" in reason


def test_candidate_that_loses_too_much_recall_is_rejected():
    current = {"pr_auc": 0.78, "recall": 0.72}
    candidate = {"pr_auc": 0.85, "recall": 0.60}
    promote, reason = decide_promotion(candidate, current, 0.0, 0.02)
    assert promote is False
    assert "recall" in reason
