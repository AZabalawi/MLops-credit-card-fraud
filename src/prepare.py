"""
prepare.py
----------
DVC Pipeline Stage 1: PREPARE

Responsibilities:
  1. Load the raw Credit Card Fraud Detection dataset from disk
     (data/raw/creditcard.csv).
  2. Run a data-quality assessment (shape, missing values, duplicate
     rows, class balance / fraud rate, feature summary) and save it as
     a JSON report.
  3. Clean the data by removing duplicate transactions (optional, driven
     by params.yaml).
  4. Split the data into train/test sets using a STRATIFIED split so the
     tiny fraud class (~0.17%) is proportionally represented in both
     sets -- critical for a severely imbalanced dataset.
  5. Persist the processed splits to disk so the next stage (train.py)
     can consume them as DVC-tracked dependencies.

NOTE on scaling: the V1..V28 features are already PCA components (scaled).
Only 'Time' and 'Amount' are on a different scale. To avoid data leakage,
scaling is applied INSIDE the model pipeline (fit on train only) in
train.py -- NOT here. This stage keeps the raw feature values.

Outputs:
  data/processed/train.csv         -> training split
  data/processed/test.csv          -> test split
  metrics/data_quality_report.json -> data quality assessment
"""

import json
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_params, get_logger, ensure_dir

logger = get_logger(__name__)

TARGET_COLUMN = "Class"


def load_raw_data(raw_path: str) -> pd.DataFrame:
    """
    Load the Credit Card Fraud Detection dataset from a CSV file.

    Source: Kaggle "Credit Card Fraud Detection" (transactions made by
    European cardholders in September 2013). 284,807 transactions,
    30 features, binary target ('Class': 0 = legitimate, 1 = fraud).

    Parameters
    ----------
    raw_path : str
        Path to the raw creditcard.csv file.

    Returns
    -------
    pd.DataFrame
    """
    logger.info(f"Loading raw dataset from {raw_path} ...")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded dataset with shape {df.shape}")
    return df


def assess_data_quality(df: pd.DataFrame) -> dict:
    """
    Compute a transparent data-quality report tailored to this dataset.

    Returns
    -------
    dict
        Report containing shape, missing values, duplicates, class
        balance / fraud rate, and summary stats for key raw features.
    """
    logger.info("Running data quality assessment ...")

    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    fraud_rate = float(df[TARGET_COLUMN].mean())

    report = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "n_features": int(df.shape[1] - 1),  # excluding target
        "missing_values_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_distribution": {str(k): int(v) for k, v in class_counts.items()},
        "fraud_count": int(class_counts.get(1, 0)),
        "legitimate_count": int(class_counts.get(0, 0)),
        "fraud_rate_percent": round(fraud_rate * 100, 4),
        "amount_summary": {
            "mean": round(float(df["Amount"].mean()), 4),
            "std": round(float(df["Amount"].std()), 4),
            "min": round(float(df["Amount"].min()), 4),
            "max": round(float(df["Amount"].max()), 4),
        },
        "time_summary": {
            "min": round(float(df["Time"].min()), 4),
            "max": round(float(df["Time"].max()), 4),
        },
    }

    logger.info(
        f"Data quality summary: {report['n_rows']} rows, "
        f"{report['n_features']} features, "
        f"{report['missing_values_total']} missing values, "
        f"{report['duplicate_rows']} duplicate rows, "
        f"fraud rate = {report['fraud_rate_percent']}%."
    )
    return report


def main():
    params = load_params("params.yaml")
    raw_path = params["data"]["raw_path"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]
    drop_duplicates = params["data"]["drop_duplicates"]

    ensure_dir("data/processed")
    ensure_dir("metrics")

    # 1. Load raw data
    df = load_raw_data(raw_path)

    # 2. Data quality assessment (BEFORE cleaning, to document the raw state)
    quality_report = assess_data_quality(df)
    with open("metrics/data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)
    logger.info("Saved data quality report to metrics/data_quality_report.json")

    # 3. Clean: drop duplicate transactions
    if drop_duplicates:
        before = df.shape[0]
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Dropped {before - df.shape[0]} duplicate rows "
                    f"({before} -> {df.shape[0]}).")

    # 4. Stratified train/test split (preserves the rare fraud proportion)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )
    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    logger.info(
        f"Split complete -> train: {train_df.shape[0]} rows "
        f"({int(train_df[TARGET_COLUMN].sum())} frauds), "
        f"test: {test_df.shape[0]} rows "
        f"({int(test_df[TARGET_COLUMN].sum())} frauds) "
        f"(test_size={test_size}, random_state={random_state})"
    )
    logger.info("PREPARE stage completed successfully.")


if __name__ == "__main__":
    main()
