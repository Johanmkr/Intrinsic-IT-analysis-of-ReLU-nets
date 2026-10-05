#!/usr/bin/env bash
# Step 3 — MI baselines on the hidden-layer activations (run_baselines.py).
#
# For every clean network from step1_train.sh, every step-2 protocol (so the
# baselines see the same points as the routing estimate) and every hidden
# layer at the last epoch: binning (K ∈ {2,4,8,16,30}), k-means
# (K ∈ {|Y|,2|Y|,4|Y|,16,64,256}) — each scored with all six discrete
# estimators — and KSG (k ∈ {3,5,10}).
#
# Output:
#   outputs/<sweep>/<experiment>/baselines_seed_<s>_<protocol>.csv     (per job)
#   results/baselines_{composite_label_noise,wbc_label_noise,mnist_capacity}.csv
#   results/provenance.json
#
# Usage:
#   ./step3_baselines.sh                # skip already-done jobs
#   ./step3_baselines.sh --force        # recompute
#   ./step3_baselines.sh --workers 8    # limit parallelism (default: #CPUs − 2)

set -euo pipefail
cd "$(dirname "$0")"

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) ARGS+=(--force); shift ;;
    --workers) ARGS+=(--workers "$2"); shift 2 ;;
    *) shift ;;
  esac
done

PYTHON="uv run python"
TS=$(date +"%Y%m%d_%H%M%S")
LOG="logs/step3_baselines_${TS}.log"
mkdir -p logs results

# One BLAS/OpenMP thread per worker process (k-means and KD-trees included).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

banner() { echo ""; echo "=== $1 ==="; echo ""; }

banner "MI baselines — clean networks, all protocols, last epoch" | tee -a "$LOG"
$PYTHON run_baselines.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"

banner "Aggregating" | tee -a "$LOG"
$PYTHON run_baselines.py --aggregate 2>&1 | tee -a "$LOG"

banner "Step 3 complete — log: $LOG"
