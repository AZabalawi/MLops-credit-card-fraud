# MLOps Phase 1 — Credit Card Fraud Detection Pipeline

A reproducible MLOps pipeline that trains a Random Forest classifier to
detect fraudulent credit-card transactions, orchestrated with **DVC** and
tracked with **MLflow**.

The dataset is severely imbalanced (only **0.17%** of transactions are
fraud), so the pipeline is built specifically to handle class imbalance:
stratified splitting, `class_weight="balanced"`, and imbalance-aware
evaluation metrics (precision, recall, F1, ROC-AUC, PR-AUC) rather than
plain accuracy.

## Group Members

| Name | Role |
|---|---|
| Abdul Raouf Zabalawi | DVC pipeline development, model training, and MLflow experiment tracking |
| Mohamed Roble | Dataset selection, data preparation, and data quality documentation |
| Somayeh Balashi | System architecture diagram, evaluation, results analysis, and documentation |

---

## 1. Requirement → Deliverable Map

| Assignment requirement | Where it lives |
|---|---|
| Dataset selection + documentation (source, size, features, quality) | [`docs/dataset_report.md`](docs/dataset_report.md); auto-generated `metrics/data_quality_report.json` |
| System architecture diagram | [`docs/architecture.png`](docs/architecture.png) (source: `docs/generate_architecture_diagram.py`) |
| DVC pipeline with 3+ stages | [`dvc.yaml`](dvc.yaml) → `prepare`, `train`, `evaluate` |
| MLflow tracking: baseline + 2 experiments | [`run_experiments.sh`](run_experiments.sh), logged to `mlflow.db` |
| Clean GitHub repo | this repository |
| DVC + MLflow screenshots | [`docs/evidence/`](docs/evidence/) + [`docs/screenshots_needed.md`](docs/screenshots_needed.md) |
| Submission report | [`docs/submission_report.md`](docs/submission_report.md) |

---

## 2. Project Structure

```
mlops-creditcard/
├── README.md                        <- you are here
├── requirements.txt                  <- pinned Python dependencies
├── params.yaml                       <- single source of truth for all settings
├── dvc.yaml                          <- DVC pipeline definition (3 stages)
├── dvc.lock                          <- generated; pins exact deps/outs per run
├── run_experiments.sh                <- runs baseline + 2 experiments
├── .gitignore
├── .dvc/                             <- DVC internal config
│
├── src/
│   ├── utils.py                      <- shared helpers (params loading, logging)
│   ├── prepare.py                    <- Stage 1: load, QA, de-dup, stratified split
│   ├── train.py                      <- Stage 2: scale + RandomForest, log to MLflow
│   └── evaluate.py                   <- Stage 3: imbalance-aware metrics, log to MLflow
│
├── data/
│   ├── raw/
│   │   ├── creditcard.csv            <- raw dataset (DVC-tracked, not in Git)
│   │   └── creditcard.csv.dvc        <- DVC pointer (committed to Git)
│   └── processed/{train,test}.csv    <- splits (generated, DVC-tracked)
│
├── models/model.pkl                  <- trained sklearn Pipeline (generated)
│
├── metrics/
│   ├── data_quality_report.json      <- data quality assessment (generated)
│   ├── train_metrics.json            <- training metrics (generated, DVC metric)
│   └── eval_metrics.json             <- test metrics (generated, DVC metric)
│
├── reports/
│   ├── confusion_matrix.png          <- confusion matrix (generated)
│   └── classification_report.txt     <- sklearn classification report (generated)
│
└── docs/
    ├── dataset_report.md             <- full dataset documentation
    ├── architecture.png              <- system architecture diagram
    ├── generate_architecture_diagram.py
    ├── submission_report.md          <- one-page submission summary
    ├── screenshots_needed.md         <- exact MLflow UI screenshots to capture
    └── evidence/                     <- captured CLI output proving DVC/MLflow work
```

> `data/` (except the `.dvc` pointer), `models/`, and `mlflow.db` are in
> `.gitignore` because they are **generated / large artifacts**. DVC and
> MLflow recreate them deterministically from the tracked source and
> config. This keeps the Git history small and clean.

---

## 3. The Dataset

**Credit Card Fraud Detection** (Kaggle / ULB) — 284,807 transactions, 30
numeric features (`Time`, `V1..V28` PCA components, `Amount`), binary
target `Class` (0 = legitimate, 1 = fraud). Fraud rate: **0.17%**. Full
documentation, including the automated data-quality assessment, is in
[`docs/dataset_report.md`](docs/dataset_report.md).

---

## 4. System Architecture

![Architecture Diagram](docs/architecture.png)

Flow: **DVC-versioned data → DVC Pipeline (prepare → train → evaluate) →
MLflow Tracking**, with `params.yaml` feeding settings into every stage,
and Git + DVC providing version control for code, config, data, and model.

---

## 5. The DVC Pipeline

Defined in [`dvc.yaml`](dvc.yaml), driven by [`params.yaml`](params.yaml):

1. **`prepare`** (`src/prepare.py`) — loads the raw CSV, runs a
   data-quality assessment, removes 1,081 duplicate transactions, and
   performs a **stratified** train/test split (so the 0.17% fraud class is
   preserved in both). Outputs the processed splits and the QA report.

2. **`train`** (`src/train.py`) — builds a scikit-learn `Pipeline` that
   scales only `Time` and `Amount` (inside the pipeline, fit on train only
   → no leakage) and trains a `RandomForestClassifier` with
   `class_weight="balanced"`. Logs params, training metrics, and the model
   to MLflow. Outputs `models/model.pkl`.

3. **`evaluate`** (`src/evaluate.py`) — evaluates on the held-out test set
   using precision, recall, F1, ROC-AUC, and PR-AUC (accuracy is reported
   for reference only), plus a confusion matrix. Logs everything to MLflow.

Each stage declares explicit `deps`/`outs`/`params`, so `dvc repro` only
re-runs stages whose inputs actually changed.

> **Note on evaluation strategy:** with ~285k transactions, a single
> stratified split leaves a large held-out test set (~57k transactions,
> ~95 frauds), which gives a statistically robust generalization estimate.
> We therefore report honest out-of-sample metrics on that held-out set
> instead of running slow k-fold cross-validation.

---

## 6. MLflow Experiment Tracking

Backend: local SQLite (`sqlite:///mlflow.db`) — self-contained, no server
required. Experiment name: `creditcard-fraud-detection`.

Three configurations are run (each produces a `train` + an `evaluate`
MLflow run = 6 runs total). Results on the held-out test set:

| Experiment | n_estimators | max_depth | class_weight | precision | recall | f1 | roc_auc | pr_auc |
|---|---|---|---|---|---|---|---|---|
| **baseline** | 100 | 12 | balanced | 0.885 | 0.726 | 0.798 | 0.957 | 0.785 |
| **exp-more-trees-deeper** | 150 | 16 | balanced | 0.932 | 0.726 | 0.817 | 0.942 | 0.806 |
| **exp-no-class-weight** | 100 | 12 | None | 0.986 | 0.726 | 0.836 | 0.965 | 0.788 |

*(Exact values are reproduced on every run and stored in
`metrics/eval_metrics.json` per experiment; a captured summary is in
`docs/evidence/`.)*

**Interpretation:** all three catch the same fraction of fraud (recall
≈ 0.73), but they trade off precision differently. Adding trees/depth
(`exp-more-trees-deeper`) raises precision and PR-AUC. Removing class
weighting (`exp-no-class-weight`) yields the highest *precision* — the
model becomes very conservative about flagging fraud — which is exactly
the kind of trade-off MLflow tracking is meant to expose. In a real fraud
system the choice depends on the business cost of a missed fraud (false
negative) vs. a false alarm (false positive); the balanced baseline is
usually preferred because it keeps recall up without collapsing precision.

---

## 7. Setup & Running From a Fresh Clone

> **Prerequisite:** you need the `creditcard.csv` file. Download it from
> Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and
> place it at `data/raw/creditcard.csv`. (If a DVC remote were configured,
> `dvc pull` would fetch it automatically — see note at the end.)

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd mlops-creditcard

# 2. Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Initialize DVC (skips automatically if already initialized)
dvc init

# 4. Make sure the dataset is in place
#    data/raw/creditcard.csv   (downloaded from Kaggle)

# 5. Run the full pipeline (prepare -> train -> evaluate) for the baseline
dvc repro

# 6. Inspect results
dvc metrics show                  # train + eval metrics
dvc dag                           # visualize the pipeline graph
cat metrics/eval_metrics.json     # final test-set metrics

# 7. Run the baseline + 2 additional experiments (MLflow tracking)
chmod +x run_experiments.sh
./run_experiments.sh

# 8. Compare experiments
dvc exp show

# 9. Launch the MLflow UI to visually compare all runs
mlflow ui --backend-store-uri sqlite:///mlflow.db
# then open http://127.0.0.1:5000 in your browser
```

> Training takes ~1–3 minutes per run depending on your machine (the
> dataset has ~227k training rows). This is normal.

### Changing settings manually

Edit `params.yaml` (e.g. `model.max_depth`) and re-run `dvc repro`. DVC
detects the change and re-runs only the affected stages.

---

## 8. Screenshots for Submission

See [`docs/screenshots_needed.md`](docs/screenshots_needed.md) for the
exact MLflow UI screenshots to capture, plus the CLI evidence already
captured in [`docs/evidence/`](docs/evidence/).

---

## 9. Best Practices Followed

- **Single source of config** (`params.yaml`) — no hardcoded
  hyperparameters, so every run is reproducible from Git history alone.
- **DVC data versioning** (`dvc add`) for the 150 MB raw CSV — Git tracks
  only a small pointer, not the binary.
- **No data leakage** — scaling is fit inside the model pipeline on
  training data only.
- **Imbalance handled correctly** — stratified split, balanced class
  weights, and PR-AUC / precision / recall instead of accuracy.
- **Explicit deps/outs in `dvc.yaml`** — reproducible, incremental re-runs.
- **`dvc exp run` for experiments** — keeps the baseline `params.yaml`
  intact while producing comparable experiment branches.
- **Generated artifacts excluded from Git** — reproducible via `dvc repro`.

### Optional: configuring a DVC remote (for full `dvc pull` reproducibility)

To let collaborators fetch the data/model with `dvc pull` instead of
re-downloading from Kaggle, add a remote (e.g. Google Drive, S3):

```bash
dvc remote add -d myremote gdrive://<folder-id>   # example
dvc push                                           # upload data + model
```

Then a fresh clone only needs `dvc pull` to retrieve the exact data
version. This repo works fine without a remote — it just requires the
`creditcard.csv` file to be present locally.
