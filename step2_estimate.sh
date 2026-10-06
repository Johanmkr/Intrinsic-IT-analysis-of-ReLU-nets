#!/usr/bin/env bash
# Step 2 — Routing information of every trained network (run_estimate.py).
#
# For every HDF5 from step1_train.sh and every protocol (held-out test split;
# plus in-sample for Composite and WBC), computes raw and ε-quotient routing MI
# with all estimators (plug-in, Miller–Madow, Grassberger, Chao–Shen,
# Chao–Wang–Jost, ANSB) at every saved epoch and hidden layer, for the 15 ε
# values in src_experiment/functional_quotient.py.
#
# Output:
#   outputs/<sweep>/<experiment>/routing_seed_<s>_<protocol>.csv   (per job)
#   results/routing_{composite_label_noise,wbc_label_noise,mnist_capacity,label_permutation}.csv.gz
#   results/provenance.json                                         (git commit, settings)
#
# Already-computed jobs are skipped (resumable).
#
# Usage:
#   ./step2_estimate.sh                # run all, skip already-done
#   ./step2_estimate.sh --force        # recompute everything
#   ./step2_estimate.sh --workers 8    # limit parallelism (default: #CPUs − 2)

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
LOG="logs/step2_estimate_${TS}.log"
mkdir -p logs results

# One BLAS thread per worker process; parallelism comes from the workers.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

banner() { echo ""; echo "=== $1 ==="; echo ""; }

banner "Routing MI — all sweeps and protocols" | tee -a "$LOG"
$PYTHON run_estimate.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"

banner "Aggregating" | tee -a "$LOG"
$PYTHON run_estimate.py --aggregate 2>&1 | tee -a "$LOG"

banner "Step 2 complete — log: $LOG"
