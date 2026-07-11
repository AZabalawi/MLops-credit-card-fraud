"""
generate_architecture_diagram.py
---------------------------------
Generates the system architecture diagram (docs/architecture.png) for
the MLOps credit-card fraud detection project. Regenerate anytime with:

    python docs/generate_architecture_diagram.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(13, 8.2))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8.2)
ax.axis("off")

COLOR_DATA = "#2563eb"
COLOR_DVC = "#059669"
COLOR_MLFLOW = "#d97706"
COLOR_GIT = "#7c3aed"
COLOR_TEXT = "#1f2937"
COLOR_BOX_BG = "#f8fafc"


def box(x, y, w, h, text, color, fontsize=10.5, text_color="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.08",
        linewidth=1.6, edgecolor=color, facecolor=color, alpha=0.92, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)


def arrow(x1, y1, x2, y2, color=COLOR_TEXT, lw=1.8, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
        linewidth=lw, color=color, zorder=1,
        connectionstyle=f"arc3,rad={rad}"))


def lane(x, y, text, color):
    ax.text(x, y, text, fontsize=11, color=color, fontweight="bold",
            ha="left", va="center")


ax.text(6.5, 7.8, "MLOps Phase 1 — System Architecture",
        fontsize=17, fontweight="bold", ha="center", color=COLOR_TEXT)
ax.text(6.5, 7.4, "Credit Card Fraud Detection Pipeline (DVC + MLflow)",
        fontsize=11, ha="center", color="#4b5563", style="italic")

# Data source lane
lane(0.3, 6.7, "DATA SOURCE", COLOR_DATA)
box(0.3, 5.85, 2.9, 0.7, "creditcard.csv (Kaggle)\n284,807 transactions\nversioned by DVC (.dvc)", COLOR_DATA, fontsize=8.3)

# DVC pipeline lane
lane(0.3, 5.05, "DVC PIPELINE  (dvc.yaml)", COLOR_DVC)
box(0.3, 4.05, 2.5, 0.78, "Stage 1: PREPARE\nprepare.py\nload, QA, de-dup, split", COLOR_DVC, fontsize=8.2)
box(3.2, 4.05, 2.5, 0.78, "Stage 2: TRAIN\ntrain.py\nscale + RandomForest", COLOR_DVC, fontsize=8.2)
box(6.1, 4.05, 2.5, 0.78, "Stage 3: EVALUATE\nevaluate.py\nP / R / F1 / AUC / PR-AUC", COLOR_DVC, fontsize=8.0)
arrow(2.8, 4.44, 3.2, 4.44)
arrow(5.7, 4.44, 6.1, 4.44)
arrow(1.7, 5.85, 1.55, 4.83)

# params.yaml
box(9.1, 4.05, 2.3, 0.78, "params.yaml\nsplit ratio, tree depth,\nclass_weight", "#475569", fontsize=8.2)
arrow(9.1, 4.6, 7.0, 4.55, rad=-0.25)
arrow(9.1, 4.35, 4.1, 4.5, rad=-0.35)
arrow(9.1, 4.15, 1.15, 4.5, rad=-0.45)

# Artifacts row
box(0.3, 3.0, 2.5, 0.62, "data/processed/\ntrain.csv, test.csv", COLOR_BOX_BG, fontsize=7.8, text_color=COLOR_TEXT)
box(3.2, 3.0, 2.5, 0.62, "models/model.pkl\n(sklearn Pipeline)", COLOR_BOX_BG, fontsize=7.8, text_color=COLOR_TEXT)
box(6.1, 3.0, 2.5, 0.62, "metrics/*.json\nreports/confusion_matrix.png", COLOR_BOX_BG, fontsize=7.2, text_color=COLOR_TEXT)
arrow(1.55, 4.05, 1.55, 3.62, color="#94a3b8")
arrow(4.45, 4.05, 4.45, 3.62, color="#94a3b8")
arrow(7.35, 4.05, 7.35, 3.62, color="#94a3b8")

# MLflow lane
lane(0.3, 2.25, "EXPERIMENT TRACKING", COLOR_MLFLOW)
box(0.3, 1.2, 4.1, 0.78, "MLflow Tracking (local SQLite: mlflow.db)\nParams + Metrics + Model artifacts\nper run: baseline, exp-2, exp-3", COLOR_MLFLOW, fontsize=8.2)
box(4.7, 1.2, 2.3, 0.78, "MLflow UI\n(mlflow ui)\nCompare runs", COLOR_MLFLOW, fontsize=8.5)
arrow(3.3, 4.05, 2.3, 1.98, color=COLOR_MLFLOW, rad=0.15)
arrow(7.2, 4.05, 3.4, 1.98, color=COLOR_MLFLOW, rad=0.25)
arrow(4.4, 1.59, 4.7, 1.59, color=COLOR_MLFLOW)

# Version control lane
lane(7.7, 2.25, "VERSION CONTROL", COLOR_GIT)
box(7.7, 1.2, 2.2, 0.78, "Git\ncode, dvc.yaml,\nparams.yaml, .dvc", COLOR_GIT, fontsize=8.2)
box(10.1, 1.2, 2.2, 0.78, "DVC cache/remote\nlarge data + model\nversioning", COLOR_GIT, fontsize=8.2)
box(7.7, 0.2, 4.6, 0.7, "GitHub Repository (single source of truth)\nreproducible from a fresh clone via `dvc repro`", COLOR_GIT, fontsize=8.4)
arrow(8.8, 1.2, 8.8, 0.9, color=COLOR_GIT)
arrow(11.2, 1.2, 11.2, 0.9, color=COLOR_GIT)
arrow(3.25, 5.85, 11.0, 1.98, color=COLOR_GIT, rad=-0.3)

plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=180, bbox_inches="tight")
print("Saved diagram to docs/architecture.png")
