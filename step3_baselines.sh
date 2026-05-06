#!/usr/bin/env bash
# Step 3 — Compute MI baselines (binning K=8, k-means K=|Y|, KSG k=3).
#
# Reads HDF5 checkpoints from step1_train.sh and the per-experiment
# new_estimator_seed_*.csv from step2_estimate.sh.
#
# Output CSVs:
#   results/mi_baselines.csv           — composite + WBC, all layers/epochs
#   results/mnist_fc_baselines.csv     — MNIST narrow nets, last epoch
#
# Usage:
#   ./step3_baselines.sh           # run all baselines, skip already-done
#   ./step3_baselines.sh --force   # recompute

set -euo pipefail
cd "$(dirname "$0")"

FORCE=""
for arg in "$@"; do
  [[ "$arg" == "--force" ]] && FORCE="--force"
done

PYTHON="uv run python"
TS=$(date +"%Y%m%d_%H%M%S")
LOG="logs/step3_baselines_${TS}.log"
mkdir -p logs results

banner() { echo ""; echo "=== $1 ==="; echo ""; }

# ── Composite + WBC baselines ─────────────────────────────────────────────────
# Restrict to last epoch (150) and noise=0 to limit compute.
# The paper only uses these conditions in all figures.
banner "MI baselines — composite + WBC (epoch 150, noise 0, all layers)" | tee -a "$LOG"
$PYTHON scripts/run_mi_baselines.py \
    --datasets composite wbc \
    --noise 0.0 \
    --epoch-filter 150 \
    --skip-mine --skip-infonce \
    $FORCE 2>&1 | tee -a "$LOG"

banner "Aggregating mi_baselines.csv" | tee -a "$LOG"
$PYTHON scripts/run_mi_baselines.py \
    --aggregate --output results/mi_baselines.csv \
    2>&1 | tee -a "$LOG"

# ── MNIST FC baselines ────────────────────────────────────────────────────────
banner "MI baselines — MNIST FC narrow nets (epoch 150, all layers)" | tee -a "$LOG"
$PYTHON scripts/run_mnist_fc_baselines.py 2>&1 | tee -a "$LOG"

banner "Step 3 complete — log: $LOG"
echo "Results written to:"
ls -lh results/*.csv 2>/dev/null | tee -a "$LOG"
