#!/usr/bin/env bash
# Step 4 — Diagnostics at the last epoch (run_diagnostics.py).
#
# Region-size histograms for every network, protocol and hidden layer, and the
# ordering sensitivity of the functional quotient (first-encounter order vs 15
# random visiting orders, ε ∈ {0.1, 0.3, 0.5, 1.0}) for every clean network on
# the held-out points.
#
# Output:
#   results/region_sizes_<sweep>.csv.gz
#   results/ordering_{composite_label_noise,wbc_label_noise,mnist_capacity}.csv.gz
#   results/provenance.json
#
# Usage:
#   ./step4_diagnostics.sh                # skip already-done jobs
#   ./step4_diagnostics.sh --force        # recompute
#   ./step4_diagnostics.sh --workers 8    # limit parallelism (default: #CPUs − 2)

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
LOG="logs/step4_diagnostics_${TS}.log"
mkdir -p logs results
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

banner() { echo ""; echo "=== $1 ==="; echo ""; }

banner "Region sizes and quotient ordering sensitivity" | tee -a "$LOG"
$PYTHON run_diagnostics.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"

banner "Aggregating" | tee -a "$LOG"
$PYTHON run_diagnostics.py --aggregate 2>&1 | tee -a "$LOG"

banner "Step 4 complete — log: $LOG"
