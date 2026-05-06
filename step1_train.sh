#!/usr/bin/env bash
# Step 1 — Generate configs and train all models.
#
# Trains 30 (composite) + 30 (WBC) + 150 (MNIST) = 210 models.
# Each model is saved as outputs/<sweep>/<experiment_name>/seed_<seed>.h5.
# Already-trained models are skipped (idempotent).
#
# Expected wall time: ~4–6 hours on a modern CPU.
#
# Usage:
#   ./step1_train.sh           # full sweep
#   ./step1_train.sh --force   # retrain even if HDF5 exists

set -euo pipefail
cd "$(dirname "$0")"

FORCE=""
for arg in "$@"; do
  [[ "$arg" == "--force" ]] && FORCE="--overwrite"
done

PYTHON="uv run python"
TS=$(date +"%Y%m%d_%H%M%S")
LOG="logs/step1_train_${TS}.log"
mkdir -p logs

banner() { echo ""; echo "=== $1 ==="; echo ""; }

# ── Generate configs ──────────────────────────────────────────────────────────
banner "Generating training configs"
$PYTHON configs/generate_composite.py | tee -a "$LOG"
$PYTHON configs/generate_wbc.py       | tee -a "$LOG"
$PYTHON configs/generate_mnist.py     | tee -a "$LOG"

# ── Composite label-noise sweep ───────────────────────────────────────────────
banner "Training composite_label_noise (30 models)"
total=$(find configs/composite_label_noise -name "*.yaml" | wc -l)
i=0
while IFS= read -r cfg; do
  i=$((i + 1))
  h5=$($PYTHON -c "
import yaml, pathlib
c = yaml.safe_load(open('$cfg'))
print(pathlib.Path(c['output_dir']) / c['experiment_name'] / f\"seed_{c['model_seed']}.h5\")
")
  if [[ -z "$FORCE" && -f "$h5" ]]; then
    echo "[$i/$total] skip (exists): $h5" | tee -a "$LOG"
    continue
  fi
  echo "[$i/$total] training: $cfg" | tee -a "$LOG"
  $PYTHON run_training.py "$cfg" $FORCE 2>&1 | tee -a "$LOG"
done < <(find configs/composite_label_noise -name "*.yaml" | sort)

# ── WBC label-noise sweep ─────────────────────────────────────────────────────
banner "Training wbc_label_noise (30 models)"
total=$(find configs/wbc_label_noise -name "*.yaml" | wc -l)
i=0
while IFS= read -r cfg; do
  i=$((i + 1))
  h5=$($PYTHON -c "
import yaml, pathlib
c = yaml.safe_load(open('$cfg'))
print(pathlib.Path(c['output_dir']) / c['experiment_name'] / f\"seed_{c['model_seed']}.h5\")
")
  if [[ -z "$FORCE" && -f "$h5" ]]; then
    echo "[$i/$total] skip (exists): $h5" | tee -a "$LOG"
    continue
  fi
  echo "[$i/$total] training: $cfg" | tee -a "$LOG"
  $PYTHON run_training.py "$cfg" $FORCE 2>&1 | tee -a "$LOG"
done < <(find configs/wbc_label_noise -name "*.yaml" | sort)

# ── MNIST capacity sweep ──────────────────────────────────────────────────────
banner "Training mnist_capacity (150 models)"
total=$(find configs/mnist_capacity -name "*.yaml" | wc -l)
i=0
while IFS= read -r cfg; do
  i=$((i + 1))
  h5=$($PYTHON -c "
import yaml, pathlib
c = yaml.safe_load(open('$cfg'))
print(pathlib.Path(c['output_dir']) / c['experiment_name'] / f\"seed_{c['model_seed']}.h5\")
")
  if [[ -z "$FORCE" && -f "$h5" ]]; then
    echo "[$i/$total] skip (exists): $h5" | tee -a "$LOG"
    continue
  fi
  echo "[$i/$total] training: $cfg" | tee -a "$LOG"
  $PYTHON run_training.py "$cfg" $FORCE 2>&1 | tee -a "$LOG"
done < <(find configs/mnist_capacity -name "*.yaml" | sort)

banner "Step 1 complete — log: $LOG"
