# Dataset Documentation

## 1. Dataset Source

- **Name:** Credit Card Fraud Detection dataset
- **Origin:** Transactions made by European cardholders in September 2013,
  released by the Machine Learning Group of Université Libre de Bruxelles
  (ULB). Publicly distributed on Kaggle:
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Access method in this project:** the raw `creditcard.csv` file is
  placed at `data/raw/creditcard.csv` and **version-tracked by DVC**
  (`dvc add`), so the large file itself stays out of Git while its exact
  version is pinned by the committed `data/raw/creditcard.csv.dvc` pointer.
- **License:** Released for research/educational use (Database Contents
  License). The original features are anonymized.

## 2. Dataset Size

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Features | 30 (all numeric) |
| Target classes | 2 (binary: legitimate / fraud) |
| Fraud transactions | 492 |
| Legitimate transactions | 284,315 |
| **Fraud rate** | **0.1727%** (severely imbalanced) |
| Missing values | 0 |
| Duplicate rows | 1,081 (removed during preparation) |
| Train split (after de-dup) | 226,980 transactions (378 frauds) |
| Test split (after de-dup) | 56,746 transactions (95 frauds) |

Figures are recomputed automatically on every pipeline run and saved to
`metrics/data_quality_report.json` (Stage 1: `prepare.py`).

## 3. Features

The dataset contains 30 input features plus the target:

- **`Time`** — seconds elapsed between each transaction and the first
  transaction in the dataset.
- **`V1` … `V28`** — 28 anonymized numerical features. These are the
  result of a **PCA (Principal Component Analysis) transformation** applied
  by the data publishers to protect confidential cardholder information.
  Because they are PCA components, they are already centered/scaled.
- **`Amount`** — the transaction amount.
- **`Class`** (target) — `0` = legitimate, `1` = fraud.

Because `V1..V28` are already scaled but `Time` and `Amount` are not, our
pipeline scales **only `Time` and `Amount`** (with `StandardScaler`), and
does so *inside the model pipeline* (fit on training data only) to avoid
data leakage into the test set.

## 4. Data Quality Assessment

`prepare.py` (DVC Stage 1) runs an automatic data quality assessment on
every run, saved to `metrics/data_quality_report.json`:

- **Completeness:** 0 missing values across all 30 features — the dataset
  is complete, no imputation required.
- **Duplicates:** 1,081 fully duplicated rows were found. These are
  removed during preparation (`drop_duplicates: true` in `params.yaml`)
  because duplicate transactions can leak between train and test splits
  and inflate performance.
- **Class balance:** the dataset is **severely imbalanced** — only 0.17%
  of transactions are fraud. This is the single most important property of
  this dataset and it drives every downstream modeling decision:
  - We use a **stratified** train/test split so the tiny fraud class is
    proportionally represented in both sets.
  - We train with **`class_weight="balanced"`** so the model does not
    simply predict "always legitimate".
  - We evaluate with **precision, recall, F1, ROC-AUC and PR-AUC** rather
    than accuracy. Plain accuracy is meaningless here: a model that
    predicts "never fraud" would still score ~99.8% accuracy while
    catching zero fraud.
- **Feature ranges:** `Amount` ranges from 0 to 25,691.16 with a heavily
  skewed distribution; `Time` spans ~48 hours (0 to 172,792 seconds).
  Both are scaled inside the model pipeline.

**Conclusion:** the dataset is clean and complete but severely imbalanced.
The imbalance — not missing data or noise — is the central challenge, and
the pipeline is designed specifically to handle it correctly.
