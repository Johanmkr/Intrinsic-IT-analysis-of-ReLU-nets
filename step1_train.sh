#!/usr/bin/env bash
# Step 1 — Generate configs and train all models.
#
# Trains 30 (composite) + 30 (WBC) + 150 (MNIST) + 20 (label permutation)
# = 230 models, in parallel (one single-threaded process per network).
# Each model is saved as outputs/<sweep>/<experiment_name>/seed_<seed>.h5;
# per-network logs go to logs/train/. Already-trained models are skipped.
#
# Usage:
#   ./step1_train.sh                # all sweeps
#   ./step1_train.sh --force        # retrain even if HDF5 exists
#   ./step1_train.sh --workers 8    # limit parallelism (default: #CPUs − 2)

set -euo pipefail
cd "$(dirname "$0")"

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) ARGS+=(--overwrite); shift ;;
    --workers) ARGS+=(--workers "$2"); shift 2 ;;
    *) shift ;;
  esac
done

PYTHON="uv run python"
TS=$(date +"%Y%m%d_%H%M%S")
LOG="logs/step1_train_${TS}.log"
mkdir -p logs

banner() { echo ""; echo "=== $1 ==="; echo ""; }

# ── Generate configs ──────────────────────────────────────────────────────────
banner "Generating training configs"
$PYTHON configs/generate_composite.py         | tail -1 | tee -a "$LOG"
$PYTHON configs/generate_wbc.py               | tail -1 | tee -a "$LOG"
$PYTHON configs/generate_mnist.py             | tail -1 | tee -a "$LOG"
$PYTHON configs/generate_label_permutation.py | tail -1 | tee -a "$LOG"

# ── Train every sweep in parallel (per-network logs in logs/train/) ───────────
banner "Training"
$PYTHON run_training.py "${ARGS[@]}" --sweeps \
    composite_label_noise wbc_label_noise mnist_capacity label_permutation 2>&1 | tee -a "$LOG"

banner "Step 1 complete — log: $LOG"
