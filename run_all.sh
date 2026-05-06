#!/usr/bin/env bash
# Run the full paper reproducibility pipeline end-to-end.
#
# This script executes steps 1–4 in order. Steps are individually idempotent:
# already-computed HDF5s / per-experiment CSVs are automatically skipped.
#
# Usage:
#   ./run_all.sh           # full pipeline, skip already-done work
#   ./run_all.sh --force   # recompute everything from scratch
#
# See README.md for step-by-step documentation and approximate runtimes.

set -euo pipefail
cd "$(dirname "$0")"

ARGS=("$@")
chmod +x step1_train.sh step2_estimate.sh step3_baselines.sh step4_plot.sh

echo "=== run_all.sh — $(date) ==="
echo "args: ${ARGS[*]:-<none>}"
echo ""

./step1_train.sh     "${ARGS[@]}"
./step2_estimate.sh  "${ARGS[@]}"
./step3_baselines.sh "${ARGS[@]}"
./step4_plot.sh

echo ""
echo "=== All done — $(date) ==="
