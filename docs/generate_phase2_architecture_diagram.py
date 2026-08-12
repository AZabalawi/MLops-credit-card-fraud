"""
generate_phase2_architecture_diagram.py
---------------------------------------
Draws the end-to-end Phase 1 + Phase 2 architecture
(docs/architecture_phase2.png).

The Phase 1 diagram (docs/architecture.png) still shows the training pipeline
on its own; this one adds everything Phase 2 introduced - serving, the
container, the cloud deployment, CI/CD and the monitoring/retraining loop -
so the whole system fits on one slide.

    python docs/generate_phase2_architecture_diagram.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

FIG_W, FIG_H = 15.0, 10.0

COLOR_DATA = "#2563eb"
COLOR_DVC = "#059669"
COLOR_MLFLOW = "#d97706"
COLOR_API = "#0891b2"
COLOR_CLOUD = "#7c3aed"
COLOR_CI = "#be123c"
COLOR_MON = "#ca8a04"
COLOR_TEXT = "#1f2937"
COLOR_MUTED = "#94a3b8"

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")


def box(x, y, w, h, text, color, fontsize=8.4, text_color="white", alpha=0.93):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            linewidth=1.5,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
            zorder=2,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
        zorder=3,
    )


def arrow(x1, y1, x2, y2, color=COLOR_TEXT, lw=1.6, rad=0.0, style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=14,
            linewidth=lw,
            color=color,
            zorder=1,
            linestyle=ls,
            connectionstyle=f"arc3,rad={rad}",
        )
    )


def elbow(points, color, lw=1.6, ls="-"):
    """Draw a right-angled route, arrowhead on the final segment."""
    for i in range(len(points) - 1):
        (x1, y1), (x2, y2) = points[i], points[i + 1]
        last = i == len(points) - 2
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>" if last else "-",
                mutation_scale=14,
                linewidth=lw,
                color=color,
                zorder=1,
                linestyle=ls,
            )
        )


def lane(x, y, text, color):
    ax.text(x, y, text, fontsize=10, color=color, fontweight="bold", ha="left", va="center")


# ----------------------------------------------------------------------
# title
# ----------------------------------------------------------------------
ax.text(
    FIG_W / 2,
    9.62,
    "Credit Card Fraud Detection - Full MLOps Architecture (Phase 1 + Phase 2)",
    fontsize=16,
    fontweight="bold",
    ha="center",
    color=COLOR_TEXT,
)
ax.text(
    FIG_W / 2,
    9.25,
    "MAI201 MLOps  |  DVC + MLflow  ->  FastAPI + Docker + Render"
    "  ->  GitHub Actions  ->  EvidentlyAI",
    fontsize=10,
    ha="center",
    color="#4b5563",
    style="italic",
)

COL_A, COL_B, COL_C = 0.4, 5.4, 9.6
W_A, W_B, W_C = 4.2, 3.6, 4.6
H = 0.7

# ----------------------------------------------------------------------
# lane A - Phase 1 training pipeline
# ----------------------------------------------------------------------
lane(COL_A, 8.85, "PHASE 1  -  TRAINING PIPELINE", COLOR_DVC)
a_boxes = [
    (7.9, "creditcard.csv  (Kaggle / ULB)\n284,807 transactions - tracked by DVC", COLOR_DATA),
    (7.0, "prepare.py\ndata quality, de-duplicate, stratified split", COLOR_DVC),
    (6.1, "train.py\nscale Time + Amount, then RandomForest", COLOR_DVC),
    (5.2, "evaluate.py\nprecision / recall / F1 / ROC-AUC / PR-AUC", COLOR_DVC),
    (4.3, "MLflow Tracking  (mlflow.db)\nbaseline + 2 experiments + retraining", COLOR_MLFLOW),
]
for y, text, color in a_boxes:
    box(COL_A, y, W_A, H, text, color)
for i in range(len(a_boxes) - 1):
    y_from = a_boxes[i][0]
    y_to = a_boxes[i + 1][0]
    arrow(COL_A + W_A / 2, y_from, COL_A + W_A / 2, y_to + H)

box(
    COL_A,
    3.3,
    W_A,
    0.55,
    "params.yaml  -  one config for every stage",
    "#475569",
    fontsize=8.0,
)

# ----------------------------------------------------------------------
# lane B - Phase 2 deployment path
# ----------------------------------------------------------------------
lane(COL_B, 8.85, "PHASE 2  -  DEPLOYMENT", COLOR_API)
b_boxes = [
    (7.9, "models/model.pkl\ntrained sklearn Pipeline", COLOR_MLFLOW),
    (7.0, "FastAPI  (app/)\n/health   /model-info   /predict   /docs", COLOR_API),
    (6.1, "Docker image\npython:3.11-slim + Uvicorn", COLOR_API),
    (5.2, "Render  (cloud)\nDocker web service + health check", COLOR_CLOUD),
    (4.3, "Public HTTPS API\nSwagger UI at /docs", COLOR_CLOUD),
]
for y, text, color in b_boxes:
    box(COL_B, y, W_B, H, text, color)
for i in range(len(b_boxes) - 1):
    arrow(COL_B + W_B / 2, b_boxes[i][0], COL_B + W_B / 2, b_boxes[i + 1][0] + H)

# train.py writes the artifact that Phase 2 serves
arrow(COL_A + W_A, 6.45, COL_B, 8.25, color=COLOR_MLFLOW, rad=-0.25)

# ----------------------------------------------------------------------
# lane C - CI/CD
# ----------------------------------------------------------------------
lane(COL_C, 8.85, "PHASE 2  -  CI/CD", COLOR_CI)
box(COL_C, 7.9, W_C, H, "GitHub repository\ncode + Dockerfile + model.pkl", "#334155")
box(
    COL_C,
    7.0,
    W_C,
    H,
    "GitHub Actions  (.github/workflows/ci-cd.yml)\nruns on every push and pull request",
    COLOR_CI,
)
arrow(COL_C + W_C / 2, 7.9, COL_C + W_C / 2, 7.7)

ci_steps = [
    (6.25, "1. Ruff lint"),
    (5.72, "2. pytest suite"),
    (5.19, "3. docker build + container smoke test"),
    (4.66, "4. deploy to Render  (main branch only)"),
]
for y, text in ci_steps:
    box(COL_C + 0.35, y, W_C - 0.7, 0.44, text, COLOR_CI, fontsize=8.0, alpha=0.78)
arrow(COL_C + W_C / 2, 7.0, COL_C + W_C / 2, 6.69)
for i in range(len(ci_steps) - 1):
    arrow(COL_C + W_C / 2, ci_steps[i][0], COL_C + W_C / 2, ci_steps[i + 1][0] + 0.44, lw=1.2)

# deploy step triggers the Render service through a deploy hook
arrow(COL_C + 0.35, 4.88, COL_B + W_B, 5.55, color=COLOR_CI, rad=0.18)
ax.text(
    9.3,
    4.32,
    "deploy hook\n(GitHub secret)",
    fontsize=7.4,
    color=COLOR_CI,
    ha="center",
    va="center",
    style="italic",
)

# ----------------------------------------------------------------------
# monitoring and retraining band
# ----------------------------------------------------------------------
lane(COL_A, 3.02, "PHASE 2  -  MONITORING AND RETRAINING", COLOR_MON)

box(0.4, 2.32, 3.0, 0.5, "Reference sample\n(from the training split)", COLOR_MUTED, fontsize=7.8)
box(0.4, 1.72, 3.0, 0.5, "Current batch\n(production-like, seeded)", COLOR_MUTED, fontsize=7.8)
box(
    3.9,
    1.85,
    3.0,
    0.75,
    "EvidentlyAI\nDataDriftPreset over the 30 features",
    COLOR_MON,
    fontsize=8.2,
)
box(7.4, 1.85, 3.0, 0.75, "Drift report\nHTML + drift_summary.json", COLOR_MON, fontsize=8.2)
box(
    10.9,
    1.85,
    3.0,
    0.75,
    "Retraining decision\ndrift detected AND labelled data?",
    COLOR_MON,
    fontsize=8.2,
)
arrow(3.4, 2.57, 3.9, 2.35, color=COLOR_MON)
arrow(3.4, 1.97, 3.9, 2.10, color=COLOR_MON)
arrow(6.9, 2.22, 7.4, 2.22, color=COLOR_MON)
arrow(10.4, 2.22, 10.9, 2.22, color=COLOR_MON)

box(
    3.9,
    0.72,
    3.0,
    0.75,
    "retraining.py\ntrain split + new labelled batch",
    COLOR_DVC,
    fontsize=8.2,
)
box(
    7.4,
    0.72,
    3.0,
    0.75,
    "MLflow run\nparams, metrics, candidate status",
    COLOR_MLFLOW,
    fontsize=8.2,
)
box(
    10.9,
    0.72,
    3.0,
    0.75,
    "Promote only if PR-AUC and recall hold,\notherwise keep the current model",
    COLOR_DVC,
    fontsize=7.4,
)
elbow([(12.4, 1.85), (12.4, 1.62), (5.4, 1.62), (5.4, 1.47)], COLOR_MON)
arrow(6.9, 1.09, 7.4, 1.09, color=COLOR_DVC)
arrow(10.4, 1.09, 10.9, 1.09, color=COLOR_DVC)

# a promoted candidate becomes the served artifact and ships through CI/CD again
elbow(
    [(13.9, 1.09), (14.6, 1.09), (14.6, 3.5), (5.0, 3.5), (5.0, 8.25), (5.4, 8.25)],
    COLOR_DVC,
    ls="--",
)
ax.text(
    9.6,
    3.72,
    "a promoted candidate replaces models/model.pkl and is released through the same CI/CD path",
    fontsize=7.6,
    color=COLOR_DVC,
    ha="center",
    va="center",
    style="italic",
)

plt.savefig("docs/architecture_phase2.png", dpi=170, bbox_inches="tight")
print("Saved diagram to docs/architecture_phase2.png")
