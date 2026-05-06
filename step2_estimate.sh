#!/usr/bin/env bash
# Step 2 — Run the routing-information estimator (Recipe 1 + 2 + 3).
#
# Reads the HDF5 checkpoints from step1_train.sh and writes per-experiment
# CSVs next to each .h5 file, then aggregates into results/*.csv.
#
# Output CSVs:
#   results/composite_label_noise_new_estimator.csv
#   results/wbc_label_noise_new_estimator.csv
#   results/mnist_capacity_new_estimator.csv
#
# Already-computed per-HDF5 CSVs are skipped (resumable).
#
# Usage:
#   ./step2_estimate.sh           # run all, skip already-done
#   ./step2_estimate.sh --force   # recompute everything

set -euo pipefail
cd "$(dirname "$0")"

FORCE=""
for arg in "$@"; do
  [[ "$arg" == "--force" ]] && FORCE="--force"
done

PYTHON="uv run python"
TS=$(date +"%Y%m%d_%H%M%S")
LOG="logs/step2_estimate_${TS}.log"
mkdir -p logs results

banner() { echo ""; echo "=== $1 ==="; echo ""; }

# ε grid used in the paper figures: covers ρ_func at 0, 0.1, 0.3, 0.5, 1.0, 2.0
# plus finer values needed for capacity-bars / rho-vs-eps plots.
EPSILONS="0.0 0.1 0.2 0.3 0.4 0.5 1.0 2.0"

# ── Composite + WBC routing MI ────────────────────────────────────────────────
banner "Routing MI — composite" | tee -a "$LOG"
$PYTHON run_label_noise_estimator.py \
    --datasets composite \
    --composite-probe-size 20000 --composite-holdout-size 10000 \
    --epsilons $EPSILONS \
    $FORCE 2>&1 | tee -a "$LOG"

banner "Routing MI — WBC" | tee -a "$LOG"
$PYTHON run_label_noise_estimator.py \
    --datasets wbc \
    --epsilons $EPSILONS \
    $FORCE 2>&1 | tee -a "$LOG"

# ── Aggregate label-noise CSVs ────────────────────────────────────────────────
banner "Aggregating composite CSV" | tee -a "$LOG"
$PYTHON run_label_noise_estimator.py \
    --aggregate --output results/composite_label_noise_new_estimator.csv \
    --datasets composite 2>&1 | tee -a "$LOG"
# Filter to composite rows only
uv run python -c "
import pandas as pd
df = pd.read_csv('results/composite_label_noise_new_estimator.csv')
df[df['dataset'] == 'composite'].to_csv(
    'results/composite_label_noise_new_estimator.csv', index=False)
print(f'composite: {len(df[df[\"dataset\"]==\"composite\"])} rows')
" 2>&1 | tee -a "$LOG"

banner "Aggregating WBC CSV" | tee -a "$LOG"
$PYTHON run_label_noise_estimator.py \
    --aggregate --output results/wbc_label_noise_new_estimator.csv \
    --datasets wbc 2>&1 | tee -a "$LOG"
uv run python -c "
import pandas as pd
df = pd.read_csv('results/wbc_label_noise_new_estimator.csv')
df[df['dataset'] == 'wbc'].to_csv(
    'results/wbc_label_noise_new_estimator.csv', index=False)
print(f'wbc: {len(df[df[\"dataset\"]==\"wbc\"])} rows')
" 2>&1 | tee -a "$LOG"

# ── MNIST capacity routing MI ─────────────────────────────────────────────────
banner "Routing MI — MNIST capacity" | tee -a "$LOG"
$PYTHON run_mnist_capacity_estimator.py \
    --epsilons $EPSILONS \
    $FORCE 2>&1 | tee -a "$LOG"

banner "Aggregating MNIST capacity CSV" | tee -a "$LOG"
$PYTHON run_mnist_capacity_estimator.py \
    --aggregate --output results/mnist_capacity_new_estimator.csv \
    2>&1 | tee -a "$LOG"

banner "Step 2 complete — log: $LOG"
echo "Results written to:"
ls -lh results/*.csv 2>/dev/null | tee -a "$LOG"
