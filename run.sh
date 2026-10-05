#!/usr/bin/env bash
# Entry point for reproducing the paper. See README.md.
#
# Usage:
#   ./run.sh setup              # uv sync + download MNIST into data/
#   ./run.sh test [--full]      # unit tests on the smoke checkpoints (--full: on outputs/)
#   ./run.sh smoke              # whole pipeline on a tiny sweep, in smoke/ (minutes)
#   ./run.sh all [--force]      # steps 1-5 (hours; dominated by training)
#   ./run.sh step1 [--force]    # train            → outputs/
#   ./run.sh step2 [--force]    # routing MI       → results/routing_<sweep>.csv
#   ./run.sh step3 [--force]    # MI baselines     → results/baselines_<sweep>.csv
#   ./run.sh step4 [--force]    # diagnostics      → results/{region_sizes,ordering}_<sweep>.csv
#   ./run.sh step5              # figures          → figures/

set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

usage() { sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; }

cmd="${1:-}"
[[ $# -gt 0 ]] && shift

case "$cmd" in
  setup)
    uv sync
    uv run python -c "from torchvision import datasets
for train in (True, False):
    datasets.MNIST(root='./data', train=train, download=True)
print('MNIST ready in data/')"
    ;;
  test)
    # Default: the checkpoints of ./run.sh smoke (minutes). --full: the step-1 outputs/.
    if [[ "${1:-}" == "--full" ]]; then
      shift
      uv run pytest tests "$@"
    else
      if [[ ! -d smoke/outputs ]]; then
        echo "no smoke checkpoints; run ./run.sh smoke first (or ./run.sh test --full)" >&2
        exit 1
      fi
      SMOKE=1 uv run pytest tests "$@"
    fi
    ;;
  smoke)
    # Mirror the code into smoke/ and run the whole pipeline there with SMOKE=1
    # (see src_experiment/smoke.py), so the real outputs/, results/ and figures/
    # are never touched. uv finds this project's pyproject.toml from smoke/.
    SMOKE_DIR="$ROOT/smoke"
    mkdir -p "$SMOKE_DIR"
    rsync -a --delete \
      --include='/configs/generate_*.py' --exclude='/configs/*/' \
      --exclude='/.git' --exclude='/.venv' --exclude='/.cache' --exclude='/data' \
      --exclude='/outputs' --exclude='/results' --exclude='/figures' --exclude='/logs' \
      --exclude='/smoke' --exclude='/pyproject.toml' --exclude='/uv.lock' \
      --exclude='__pycache__' --exclude='.pytest_cache' \
      ./ "$SMOKE_DIR/"
    mkdir -p "$ROOT/data"
    ln -sfn "$ROOT/data" "$SMOKE_DIR/data"
    SMOKE=1 "$SMOKE_DIR/run_all.sh" "$@"
    echo ""
    echo "Smoke run finished; figures in smoke/figures/ (these check the code, not the paper)."
    ;;
  all)
    ./run_all.sh "$@"
    ;;
  step1) ./step1_train.sh "$@" ;;
  step2) ./step2_estimate.sh "$@" ;;
  step3) ./step3_baselines.sh "$@" ;;
  step4) ./step4_diagnostics.sh "$@" ;;
  step5) ./step5_plot.sh "$@" ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "unknown command: $cmd" >&2
    usage >&2
    exit 1
    ;;
esac
