"""
train.py
--------
DVC Pipeline Stage 2: TRAIN

Responsibilities:
  1. Load the processed training data produced by prepare.py.
  2. Build a scikit-learn Pipeline that:
       - Scales ONLY 'Time' and 'Amount' with StandardScaler (V1..V28 are
         already PCA-scaled), passing the rest through unchanged. Scaling
         lives inside the pipeline so it is fit on training data only ->
         no data leakage into the test set.
       - Trains a RandomForestClassifier with hyperparameters + class
         weighting from params.yaml. class_weight="balanced" is the key
         setting for this severely imbalanced dataset.
  3. Log parameters, cross-validated training metrics, and the trained
     pipeline to MLflow.
  4. Save the trained pipeline to disk (models/model.pkl) for evaluate.py.
  5. Write train_metrics.json so DVC can also show metrics via
     `dvc metrics show` without needing MLflow.

Because accuracy is meaningless on a 0.17%-positive dataset (a model that
predicts "never fraud" scores 99.8% accuracy), we evaluate with
cross-validated ROC-AUC and Average Precision (PR-AUC) instead.

Outputs:
  models/model.pkl            -> trained sklearn Pipeline (joblib)
  metrics/train_metrics.json  -> training metrics tracked by DVC
"""

import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import load_params, get_logger, ensure_dir

logger = get_logger(__name__)

TARGET_COLUMN = "Class"
SCALE_COLUMNS = ["Time", "Amount"]  # only these need scaling


def build_pipeline(model_params: dict, feature_columns: list) -> Pipeline:
    """
    Build a preprocessing + model pipeline.

    The ColumnTransformer scales 'Time' and 'Amount' and passes the
    already-scaled PCA features (V1..V28) through untouched.
    """
    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), SCALE_COLUMNS)],
        remainder="passthrough",  # keep V1..V28 as-is
    )

    # DVC param overrides (`-S model.class_weight=None`) arrive as the
    # string "None"; normalize that (and YAML null) to a real Python None.
    class_weight = model_params["class_weight"]
    if isinstance(class_weight, str) and class_weight.lower() in ("none", "null", ""):
        class_weight = None

    classifier = RandomForestClassifier(
        n_estimators=model_params["n_estimators"],
        max_depth=model_params["max_depth"],
        min_samples_split=model_params["min_samples_split"],
        class_weight=class_weight,
        random_state=model_params["random_state"],
        n_jobs=model_params["n_jobs"],
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor),
                              ("model", classifier)])
    return pipeline


def train_model(train_df: pd.DataFrame, model_params: dict):
    """
    Train the pipeline and compute cross-validated training metrics.

    Returns
    -------
    tuple(Pipeline, dict)
    """
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    pipeline = build_pipeline(model_params, list(X_train.columns))

    logger.info(f"Training RandomForest pipeline with params: {model_params}")

    # Fit the pipeline on the full training set.
    #
    # NOTE on evaluation strategy: with ~285k transactions, a single
    # stratified train/test split leaves a LARGE held-out test set
    # (~57k transactions, ~98 frauds), which gives a statistically robust
    # generalization estimate. We therefore report the honest
    # out-of-sample metrics in the EVALUATE stage (on that held-out set)
    # rather than running expensive k-fold cross-validation here. The
    # training-set scores below are logged mainly to expose the
    # train-vs-test gap (an overfitting signal).
    pipeline.fit(X_train, y_train)

    # For imbalanced data, ROC-AUC and Average Precision (PR-AUC) are the
    # meaningful scores -- not accuracy.
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]
    metrics = {
        "train_roc_auc": round(float(roc_auc_score(y_train, y_train_proba)), 4),
        "train_pr_auc": round(float(average_precision_score(y_train, y_train_proba)), 4),
    }
    logger.info(f"Training-set metrics: {metrics}")
    return pipeline, metrics


def main():
    params = load_params("params.yaml")
    model_params = params["model"]
    mlflow_cfg = params["mlflow"]

    ensure_dir("models")
    ensure_dir("metrics")

    mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])
    mlflow.set_experiment(mlflow_cfg["experiment_name"])

    # Descriptive, unique run name so runs are distinguishable in the UI
    cw = str(model_params["class_weight"])
    run_name = (
        f"rf_n{model_params['n_estimators']}"
        f"_d{model_params['max_depth']}"
        f"_cw-{cw}"
    )

    train_df = pd.read_csv("data/processed/train.csv")

    with mlflow.start_run(run_name=run_name):
        pipeline, metrics = train_model(train_df, model_params)

        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)
        mlflow.set_tag("stage", "train")
        mlflow.set_tag("dataset", "creditcard_fraud")

        mlflow.sklearn.log_model(pipeline, name="model")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run logged with run_id={run_id}, run_name={run_name}")

    joblib.dump(pipeline, "models/model.pkl")
    with open("metrics/train_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Saved trained pipeline to models/model.pkl")
    logger.info("TRAIN stage completed successfully.")


if __name__ == "__main__":
    main()
