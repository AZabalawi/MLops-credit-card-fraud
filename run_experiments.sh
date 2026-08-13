#!/usr/bin/env bash
# ==============================================================
# run_experiments.sh
#
# Runs the full DVC pipeline three times with different
# hyperparameter sets, satisfying the requirement of
# "1 baseline experiment + at least 2 additional experiments"
# tracked in MLflow. Each run uses `dvc exp run -f`, which:
#   - Re-executes the pipeline with the overridden params
#   - Keeps params.yaml on disk unchanged (safe, non-destructive)
#   - Creates separate MLflow runs (train + evaluate) per config
#
# The experiments are chosen to demonstrate what matters on a
# severely imbalanced fraud dataset:
#   1. BASELINE: balanced class weights (handles the 0.17% fraud rate)
#   2. MORE TREES / DEEPER: does extra capacity help catch more fraud?
#   3. NO CLASS WEIGHT: shows how recall collapses if the imbalance is
#      NOT handled -- a key teaching point.
#
# Usage:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh
# ==============================================================

set -e  # exit immediately if any command fails

echo "=============================================="
echo " Experiment 1/3: BASELINE (balanced class weight)"
echo "=============================================="
dvc exp run -f --name baseline \
  -S model.n_estimators=100 \
  -S model.max_depth=12 \
  -S model.class_weight=balanced

echo ""
echo "=============================================="
echo " Experiment 2/3: MORE TREES, DEEPER (balanced)"
echo "=============================================="
dvc exp run -f --name exp-more-trees-deeper \
  -S model.n_estimators=150 \
  -S model.max_depth=16 \
  -S model.class_weight=balanced

echo ""
echo "=============================================="
echo " Experiment 3/3: NO CLASS WEIGHT (imbalance NOT handled)"
echo "=============================================="
dvc exp run -f --name exp-no-class-weight \
  -S model.n_estimators=100 \
  -S model.max_depth=12 \
  -S model.class_weight=None

echo ""
echo "=============================================="
echo " All experiments complete. Compare with:"
echo "   dvc exp show"
echo "   mlflow ui --backend-store-uri sqlite:///mlflow.db"
echo "   (then open http://127.0.0.1:5000)"
echo "=============================================="
