"""
evaluate.py
-----------
DVC Pipeline Stage 3: EVALUATE

Responsibilities:
  1. Load the trained pipeline (models/model.pkl) and the held-out test
     split (data/processed/test.csv).
  2. Compute classification metrics that are meaningful for a severely
     imbalanced problem: precision, recall, F1 (on the fraud class),
     ROC-AUC, and Average Precision (PR-AUC). Plain accuracy is included
     for reference only -- it is near-100% for any model here and is NOT
     a useful signal.
  3. Generate and save a confusion matrix plot.
  4. Log all metrics + artifacts to MLflow, and write eval_metrics.json
     so DVC tracks them independently of MLflow.

Outputs:
  metrics/eval_metrics.json     -> final metrics tracked by DVC
  reports/confusion_matrix.png  -> confusion matrix visualization
  reports/classification_report.txt -> full sklearn classification report
"""

import json

import joblib
import matplotlib
import mlflow
import pandas as pd

matplotlib.use("Agg")  # headless backend, no display needed
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from utils import ensure_dir, get_logger, load_params

logger = get_logger(__name__)

TARGET_COLUMN = "Class"


def evaluate_model(model, test_df: pd.DataFrame):
    """
    Compute imbalance-aware metrics and a confusion matrix on the test set.

    Returns
    -------
    tuple(dict, np.ndarray, str)
        (metrics dict, confusion matrix array, text classification report)
    """
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        # accuracy is reported for reference only; it is misleadingly high
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 6),
        # the metrics that actually matter for fraud detection:
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 4),
    }

    cm = confusion_matrix(y_test, y_pred)
    report_text = classification_report(
        y_test, y_pred, target_names=["Legitimate", "Fraud"], zero_division=0
    )

    logger.info(f"Evaluation metrics: {metrics}")
    return metrics, cm, report_text


def plot_confusion_matrix(cm, output_path: str):
    """Save a labeled confusion matrix heatmap to disk."""
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Reds",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Test Set (Fraud Detection)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {output_path}")


def main():
    params = load_params("params.yaml")
    model_params = params["model"]
    mlflow_cfg = params["mlflow"]

    ensure_dir("metrics")
    ensure_dir("reports")

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    cw = str(model_params["class_weight"])
    run_name = (
        f"rf_n{model_params['n_estimators']}"
        f"_d{model_params['max_depth']}"
        f"_cw-{cw}_eval"
    )

    model = joblib.load("models/model.pkl")
    test_df = pd.read_csv("data/processed/test.csv")

    with mlflow.start_run(run_name=run_name):
        metrics, cm, report_text = evaluate_model(model, test_df)

        cm_path = "reports/confusion_matrix.png"
        plot_confusion_matrix(cm, cm_path)

        report_path = "reports/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report_text)

        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("stage", "evaluate")
        mlflow.set_tag("dataset", "creditcard_fraud")
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(report_path)

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow eval run logged with run_id={run_id}")

    with open("metrics/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved evaluation metrics to metrics/eval_metrics.json")
    logger.info("EVALUATE stage completed successfully.")


if __name__ == "__main__":
    main()
