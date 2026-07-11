# Screenshots Needed for Submission

This project already includes **captured CLI evidence** (real command
output executed against this repo) in [`docs/evidence/`](evidence/), which
proves the DVC pipeline and MLflow tracking both work end-to-end:

| File | What it proves |
|---|---|
| `evidence/01_dvc_dag.png` | The DVC pipeline graph: `prepare → train → evaluate` |
| `evidence/02_dvc_repro.png` | A full `dvc repro` run completing all 3 stages |
| `evidence/03_dvc_metrics_show.png` | `dvc metrics show` displaying tracked metrics |
| `evidence/04_dvc_exp_show.png` | `dvc exp show` comparing baseline + 2 experiments |
| `evidence/05_mlflow_runs_summary.png` | All 6 MLflow runs confirmed `FINISHED` with key metrics |

**The MLflow UI is a live local web server**, so it can't be captured from
the build environment — capture these two screenshots yourself after
cloning and running the project. It takes about two minutes.

## Steps to capture the MLflow UI screenshots

1. Follow the setup steps in the main [`README.md`](../README.md) through
   running `./run_experiments.sh` (or at least `dvc repro` once).

2. Launch the MLflow UI:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```

3. Open **http://127.0.0.1:5000** in your browser.

4. **Screenshot A — Experiment comparison:**
   Click the `creditcard-fraud-detection` experiment. You'll see 6 runs
   (3 `train` + 3 `evaluate`, e.g. `rf_n100_d12_cw-balanced` and its
   `..._eval`). Tick the checkbox next to the three `_eval` runs and click
   **Compare** to see hyperparameters vs. metrics (precision, recall, f1,
   roc_auc, pr_auc) side by side. Screenshot this view.

5. **Screenshot B — Single run detail:**
   Click into `rf_n100_d12_cw-balanced_eval` (the baseline evaluation).
   Screenshot the run page showing its logged parameters, metrics, and the
   `confusion_matrix.png` / `classification_report.txt` artifacts.

Save both into `docs/evidence/` (e.g. `06_mlflow_ui_comparison.png` and
`07_mlflow_ui_run_detail.png`) before submitting, so the grader sees the
live UI alongside the CLI evidence already provided.
