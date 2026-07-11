# MLOps Phase 1 — Submission Report

**Group Members:** Abdul Raouf Zabalawi · Mohamed Roble · Somayeh Balashi

## 1. Dataset

- **Source:** Credit Card Fraud Detection (Kaggle / ULB), raw
  `creditcard.csv` version-tracked with DVC (`dvc add`).
- **Size:** 284,807 transactions × 30 features, binary target.
- **Imbalance:** only 492 frauds (**0.17%**) — the defining challenge.
- **Cleaning:** 0 missing values; 1,081 duplicate rows removed.
- **Split:** stratified 80/20 → 226,980 train (378 frauds) /
  56,746 test (95 frauds).
- Full report auto-generated at `metrics/data_quality_report.json`; see
  `docs/dataset_report.md`.

## 2. Architecture

See `docs/architecture.png`. Four layers: DVC-versioned data source, a
3-stage DVC pipeline, MLflow experiment tracking, and Git/DVC version
control — all parameterized from a single `params.yaml`.

## 3. DVC Pipeline

3 stages in `dvc.yaml`: `prepare` → `train` → `evaluate`, each with
explicit `deps`/`outs`/`params`. The raw dataset is tracked with `dvc add`
(pointer committed to Git, 150 MB CSV kept out). Verified end-to-end with
`dvc repro` and from a fresh clone. Evidence:
`docs/evidence/01_dvc_dag.png`, `02_dvc_repro.png`,
`03_dvc_metrics_show.png`.

## 4. MLflow Experiments

Experiment `creditcard-fraud-detection`: 3 configurations × 2 stages
(train, evaluate) = 6 logged runs. Held-out test-set results:

| Experiment | n_estimators | max_depth | class_weight | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|---|---|---|
| baseline | 100 | 12 | balanced | 0.885 | 0.726 | 0.798 | 0.957 | 0.785 |
| exp-more-trees-deeper | 150 | 16 | balanced | 0.932 | 0.726 | 0.817 | 0.942 | 0.806 |
| exp-no-class-weight | 100 | 12 | None | 0.986 | 0.726 | 0.836 | 0.965 | 0.788 |

**Key finding:** we deliberately evaluate with precision/recall/PR-AUC, not
accuracy — every model scores ~99.9% accuracy simply because 99.8% of
transactions are legitimate, so accuracy is useless here. The three
experiments catch the same share of fraud (recall ≈ 0.73) but differ in
precision: more trees/depth improves precision and PR-AUC; dropping class
weighting maximizes precision (very conservative flagging). This
precision/recall trade-off — invisible if you only look at accuracy — is
exactly what MLflow tracking surfaces. Evidence:
`docs/evidence/04_dvc_exp_show.png`, `05_mlflow_runs_summary.png`, plus the
live MLflow UI screenshots described in `docs/screenshots_needed.md`.

## 5. Repository

Clean Git history; `.gitignore` excludes generated/large artifacts
(processed data, model, MLflow store) while the DVC pointer pins the exact
raw-data version. Fully reproducible from a fresh clone via `dvc repro` +
`./run_experiments.sh`.

## 6. Requirement Checklist

- [x] Dataset selected and documented (source, size, features, quality)
- [x] System architecture diagram
- [x] DVC pipeline with 3 stages (prepare, train, evaluate)
- [x] MLflow tracking: 1 baseline + 2 additional experiments
- [x] Clean GitHub-ready repository
- [x] Screenshots / evidence of DVC and MLflow working
- [x] Complete documentation (README, dataset report, this report)
