#!/usr/bin/env bash
# Step 5 — Generate the paper figures from the stored results.
#
# Reads:  results/*.csv.gz only (steps 1–4), so it runs without outputs/
# Writes: figures/*.pdf and figures/*.png
#
# Figures produced:
#   pedagogical_figure1.{pdf,png}       — intro: ReLU partition + activation pattern
#   calibration_scatter_raw.{pdf,png}   — routing MI vs 3 baselines, 3 datasets
#   layer_profile_last_epoch.{pdf,png}  — layerwise bits at last epoch
#   mnist_capacity_bars_per_arch.{pdf,png} — I_raw vs I_func across PCA dims
#   mnist_rho_vs_eps.{pdf,png}          — ρ_func vs ε per architecture
#   rho_func_layerwise.{pdf,png}        — ρ_func by depth for multiple ε
#   composite_dataset.{pdf,png}         — App. B: the Composite dataset
#   training_curves_{composite,wbc,mnist}.{pdf,png} — App. F: test accuracy / loss
#
# and results/summary/ (scripts/summarize_results.py): the appendix tables as
# CSV and every number quoted in the text (numbers.json, SUMMARY.md).
#
# Usage:
#   ./step5_plot.sh

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="uv run python"
TS=$(date +"%Y%m%d_%H%M%S")
LOG="logs/step5_plot_${TS}.log"
mkdir -p logs figures

banner() { echo ""; echo "=== $1 ==="; echo ""; }

banner "Figure 1 — pedagogical partition diagram" | tee -a "$LOG"
$PYTHON scripts/plot_figure1_pedagogy.py 2>&1 | tee -a "$LOG"

banner "Calibration scatter" | tee -a "$LOG"
$PYTHON scripts/plot_calibration_scatter.py 2>&1 | tee -a "$LOG"

banner "Layerwise profile at last epoch" | tee -a "$LOG"
$PYTHON scripts/plot_layer_profile_last_epoch.py 2>&1 | tee -a "$LOG"

banner "MNIST capacity bars" | tee -a "$LOG"
$PYTHON scripts/plot_mnist_capacity_bars.py 2>&1 | tee -a "$LOG"

banner "MNIST ρ_func vs ε" | tee -a "$LOG"
$PYTHON scripts/plot_mnist_functional_pca_sweep.py --type rho 2>&1 | tee -a "$LOG"

banner "ρ_func layerwise (all three datasets)" | tee -a "$LOG"
$PYTHON scripts/plot_rho_func_layerwise.py 2>&1 | tee -a "$LOG"

banner "Composite dataset (App. B)" | tee -a "$LOG"
$PYTHON scripts/plot_composite_dataset.py 2>&1 | tee -a "$LOG"

banner "Training curves (App. F)" | tee -a "$LOG"
$PYTHON scripts/plot_training_curves.py 2>&1 | tee -a "$LOG"

banner "Summary tables and numbers (results/summary/)" | tee -a "$LOG"
$PYTHON scripts/summarize_results.py 2>&1 | tee -a "$LOG"

banner "Step 5 complete — log: $LOG"
echo "Figures written to figures/:"
ls figures/*.pdf 2>/dev/null | tee -a "$LOG"
