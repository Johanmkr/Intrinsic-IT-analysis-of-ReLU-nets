"""Generate YAML configs for the label-permutation (memorization) control.

The training labels are randomly permuted (``permute_labels: true``, see
``src_experiment/dataset.py:permute_labels``), so they keep their marginal
but carry no information about the inputs. A narrow and a wide network per
dataset contrast low and high capacity. Test labels stay true.

Datasets: composite, wbc. Architectures: [5,5,5], [25,25,25]. Seeds: 101–105.
Total: 2 × 2 × 5 = 20 configs, written to configs/label_permutation/.

Usage:
    python configs/generate_label_permutation.py
    python configs/generate_label_permutation.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src_experiment import smoke  # noqa: E402

CONFIG_ROOT = REPO / "configs" / "label_permutation"

DATASETS = ["composite", "wbc"]
ARCHS = [[5, 5, 5], [25, 25, 25]]
SEEDS = smoke.SEEDS

EPOCHS = smoke.EPOCHS
BATCH_SIZE = 32
LR = 0.001
MOMENTUM = 0.9
GLOBAL_SEED = 42
SAVE_INTERVAL = 10
SAVE_EPOCHS = [0, 1, 2, 3, 4, 6, 8, 10]


def make_config(dataset: str, arch: list, seed: int) -> dict:
    return {
        "experiment_name": f"{dataset}_{arch}",
        "dataset": dataset,
        "output_dir": "outputs/label_permutation",
        "architecture": arch,
        "dropout": 0.0,
        "global_seed": GLOBAL_SEED,
        "model_seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LR,
        "momentum": MOMENTUM,
        "noise": 0.0,
        "permute_labels": True,
        "save_interval": SAVE_INTERVAL,
        "save_epochs": SAVE_EPOCHS,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true",
                   help="Print config paths only; do not write.")
    args = p.parse_args()

    paths = []
    for dataset in DATASETS:
        for arch in ARCHS:
            out_dir = CONFIG_ROOT / f"{dataset}_{str(arch).replace(' ', '')}"
            out_dir.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                out = out_dir / f"seed_{seed}.yaml"
                if not args.list:
                    with open(out, "w") as f:
                        yaml.dump(make_config(dataset, arch, seed), f,
                                  default_flow_style=False, sort_keys=False)
                paths.append(out)

    for p in paths:
        print(p)
    if not args.list:
        print(f"\nWrote {len(paths)} configs to {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
