# Model Card - Credit Card Fraud Detection

This card describes the model that is actually deployed by this repository:
the scikit-learn pipeline produced by the Phase 1 DVC pipeline and served by
the Phase 2 FastAPI application. Every number below was taken from the
project's own generated files (`metrics/eval_metrics.json`,
`metrics/data_quality_report.json`, `reports/classification_report.txt`) after
re-running `dvc repro` on the original dataset.

---

## 1. Model details

| Item | Value |
|---|---|
| Course / project | MAI201 MLOps - Credit Card Fraud Detection (Phase 1 + Phase 2) |
| Team | Abdulraouf Zabalawi (Project & ML Lead), Mohamed Roble (Engineering Lead), Someyah Balashi (Documentation Lead) |
| Model type | `RandomForestClassifier` inside a scikit-learn `Pipeline` |
| Problem type | Binary classification (supervised) |
| Input | 30 numeric features: `Time`, `V1`-`V28`, `Amount` |
| Output | Predicted class (0 / 1) and the probability of the fraud class |
| Artifact | `models/model.pkl`, 1,891,938 bytes |
| Artifact md5 | `eedef6853d1d43f588f4482b1b410ce3` (recorded in `dvc.lock`) |
| Artifact sha256 | `a825c533d34bcbff6e785851e32c9c692fcadb833726f7c4a41f9099efc802c2` (also reported by `/model-info`) |
| Serving version | 2.0.0 |
| Library versions | scikit-learn 1.7.2, numpy 2.2.6, pandas 2.3.3, joblib 1.5.3 |

Hyperparameters, all read from `params.yaml`:

```
n_estimators      100
max_depth         12
min_samples_split 2
class_weight      balanced
random_state      42
n_jobs            -1
```

---

## 2. Training data

The dataset is the Kaggle / ULB **Credit Card Fraud Detection** set: real card
transactions made by European cardholders over two days in September 2013.
Features `V1`-`V28` are principal components published in place of the original
fields, which is why they carry no readable meaning. `Time` and `Amount` were
left in their original units.

From the data quality report the pipeline generates automatically:

| Property | Value |
|---|---|
| Rows | 284,807 |
| Columns | 31 (30 features + `Class`) |
| Missing values | 0 |
| Duplicate rows | 1,081 |
| Fraud transactions | 492 |
| Legitimate transactions | 284,315 |
| Fraud rate | 0.1727% |
| `Amount` | mean 88.35, std 250.12, min 0.00, max 25,691.16 |
| `Time` | 0 to 172,792 seconds |

After the duplicates were removed, 283,726 rows remained. A stratified 80/20
split produced **226,980 training rows (378 fraud)** and **56,746 test rows
(95 fraud)**. Stratifying matters here: with 0.17% positives, an ordinary
random split can easily leave the two sets with noticeably different fraud
rates.

### Preprocessing

`V1`-`V28` are already PCA outputs and are on a comparable scale, so they are
passed through untouched. Only `Time` and `Amount` go through a
`StandardScaler`. The scaler sits inside the `Pipeline` rather than in the
preparation stage, which means it is fitted on training rows only and the test
split never influences the scaling statistics.

### Class imbalance

We used `class_weight="balanced"` rather than resampling. It costs nothing at
prediction time and keeps the training set as it really is, whereas SMOTE would
have introduced synthetic transactions that do not correspond to anything a
bank actually saw. The imbalance is also why accuracy is reported but never
used to make decisions: a model that answers "legitimate" to everything scores
99.83%.

---

## 3. Evaluation

Measured on the held-out test split (56,746 transactions, 95 of them fraud),
which was never used for training or for any tuning decision.

| Metric | Value |
|---|---|
| Accuracy | 0.999383 |
| Precision (fraud) | 0.8846 |
| Recall (fraud) | 0.7263 |
| F1 (fraud) | 0.7977 |
| ROC-AUC | 0.9567 |
| PR-AUC (average precision) | 0.7854 |

Confusion matrix on the test split:

|  | Predicted legitimate | Predicted fraud |
|---|---|---|
| **Actually legitimate** | 56,642 | 9 |
| **Actually fraud** | 26 | 69 |

In plain terms: of the 95 frauds in the test set the model caught 69 and missed
26, and it raised 9 false alarms out of 56,651 legitimate transactions. That is
the trade-off the `balanced` class weight buys - a model tuned purely for
precision would flag less and miss more.

Training-set scores are `train_roc_auc = 1.0` and `train_pr_auc = 0.979`. The
gap between those and the test scores is the expected overfitting signal for a
forest of depth 12; it is logged deliberately so the difference is visible
rather than hidden.

> **One small discrepancy, recorded honestly.** Phase 1 committed
> `pr_auc = 0.7853`; re-running the pipeline for Phase 2 produced `0.7854`.
> The other five metrics are identical to the last decimal. The difference is a
> 1e-4 rounding effect from pinning scikit-learn to 1.7.2 instead of 1.8.0 (see
> section 8). This card reports 0.7854 because that is what the deployed
> artifact actually scores.

---

## 4. Intended use

This model is coursework. It exists to demonstrate an end-to-end MLOps
workflow: versioned data, a reproducible training pipeline, tracked
experiments, a deployed API, automated tests and deployment, and drift
monitoring with a retraining path.

Reasonable uses:

- scoring transactions that have exactly the 30 features of this dataset
- as a teaching example of serving a scikit-learn pipeline behind FastAPI
- as a baseline to compare other fraud models against on the same split

---

## 5. Out-of-scope use

- **Any real financial decision.** Nothing here has been through the
  validation, calibration, security review or governance a payment system
  needs. It must not decline transactions or freeze accounts.
- **Other datasets.** The `V` features are principal components of one specific
  bank's data from 2013. They mean nothing outside this dataset, so the model
  cannot be pointed at another payment stream.
- **Explaining a decision to a customer.** The inputs are anonymised
  components, so "why was this flagged" has no human-readable answer.
- **Anything that needs calibrated probabilities.** The output is a forest vote
  share. It orders transactions by risk well; it is not a calibrated
  probability of fraud.

---

## 6. Limitations and risks

- **26 out of 95 frauds are missed.** Recall of 0.73 is normal for this
  benchmark, but a real deployment would have to decide what happens to the
  fraud the model does not catch.
- **The data is two days old, from 2013.** Card fraud patterns move; a model
  fitted to that window would degrade quickly against current traffic. This is
  precisely why the drift monitoring in section 7 exists.
- **Only 492 fraud examples in total, 378 of them in training.** Every fraud
  metric is computed from small counts, so a handful of transactions moves
  recall by a full percentage point. Metric differences smaller than a few
  percent should not be treated as meaningful.
- **The threshold is fixed at 0.5** and was never tuned against a cost model.
  A real system would set it from the relative cost of a missed fraud versus a
  false alarm. The API exposes the probability so a caller can apply its own
  cut-off.
- **`Time` is seconds since the first transaction in the dataset**, not a
  timestamp. It leaks information about position within those two days, and it
  will not generalise to a live feed without being redefined.

### Fairness and ethical considerations

The dataset carries no demographic attributes - no age, gender, location or
merchant category - so we cannot test whether error rates differ across groups,
and we do not claim the model is fair. That absence is itself a limitation: a
model can distribute its 9 false positives and 26 misses very unevenly across
customers without any of it being visible in these features. A production
version would need labelled subgroup data and per-group error reporting before
anyone could make a fairness claim.

There is also an asymmetry worth stating. A false positive inconveniences a
customer whose legitimate purchase is blocked; a false negative lets fraud
through and the loss usually falls on the bank or the merchant. Those costs are
not equal and are not interchangeable, and a single F1 score hides that.

---

## 7. Monitoring and retraining

**Drift detection** uses EvidentlyAI 0.7.21 (`DataDriftPreset`) over the 30
input features. `Class` is excluded, because the API never receives a label at
prediction time. The reference is a seeded 10,000-row sample of the training
split, held fixed. Two scenarios are generated and committed:

| Scenario | Report | Drifted columns | Dataset drift |
|---|---|---|---|
| Healthy batch (untouched test rows) | `monitoring/reports/reference_report.html` | 0 / 30 | No |
| Simulated drift (17 columns shifted) | `monitoring/reports/drift_report.html` | 17 / 30 (share 0.567) | Yes |

The drifted batch is **simulated for demonstration** - a reproducible,
seeded transformation of real rows, not real customer behaviour. It is labelled
as such in `monitoring/data/manifest.json`.

**Retraining policy.** Retraining never runs on production requests, because
those arrive unlabelled and training on the model's own predictions would only
reinforce its mistakes. `monitoring/retraining.py` requires a batch that
already carries `Class`, refuses a batch containing only one class, trains the
candidate on the original training split plus that batch, and scores both the
candidate and the deployed model on the untouched test split. The candidate
replaces the current model only if:

- PR-AUC is at least as good as the current model's, and
- recall has not dropped by more than 0.02.

Otherwise the current model stays and the candidate is kept aside for
inspection. The demonstration run promoted its candidate (PR-AUC 0.7854 →
0.7865, recall 0.7263 → 0.7368); the full record is in
`monitoring/reports/retraining_summary.json` and in MLflow.

---

## 8. Deployment

The model is served by a FastAPI application (`app/`), containerised from
`Dockerfile` on `python:3.11-slim` with Uvicorn, and deployed to Render as a
Docker web service. The container installs the pinned `requirements-api.txt`,
so the scikit-learn that loads `model.pkl` is the same version that fitted it.

| Endpoint | Purpose |
|---|---|
| `GET /` | service description |
| `GET /health` | 200 only when the model is loaded, 503 otherwise |
| `GET /model-info` | model type, feature count, hyperparameters, artifact hash |
| `POST /predict` | score one transaction |
| `GET /docs` | Swagger UI |

The public URL is in the README. No credentials are baked into the image; the
Render deploy hook lives in GitHub Actions secrets.

**On the scikit-learn pin.** Phase 1 pinned 1.8.0, which needs Python 3.11 or
newer. Phase 2 pins 1.7.2 so the training environment, the test environment and
the container all agree with the version that produced the pickle - loading a
scikit-learn artifact under a different minor version is not something to leave
to chance in a deployed service. Re-running the pipeline under 1.7.2 reproduces
five of the six Phase 1 metrics exactly and the sixth to within 1e-4.

---

## 9. Reproducing this model

```bash
pip install -r requirements.txt
# place the Kaggle dataset at data/raw/creditcard.csv
dvc repro
```

The dataset is pinned by `data/raw/creditcard.csv.dvc`
(md5 `e90efcb83d69faf99fcab8b0255024de`, 150,828,752 bytes), the splits and the
model by `dvc.lock`, and every setting by `params.yaml`. With the same input
file the run is deterministic: `random_state=42` is used for both the split and
the forest.

---

*Last updated when the Phase 2 deliverables were built. Maintained by the
MAI201 project team named in section 1.*
