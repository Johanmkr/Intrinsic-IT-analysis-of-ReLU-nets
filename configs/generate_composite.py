"""Generate per-(architecture, seed) YAML configs for the composite label-noise sweep.

Architectures used in the paper:
  - [5,5,5] and [5,5,5,5,5]           — calibration scatter (3-layer and 5-layer narrow)
  - [9,9,9] and [9,9,9,9,9]           — calibration scatter (3-layer and 5-layer medium)
  - [25,25,25] and [25,25,25,25,25]   — calibration scatter (3-layer and 5-layer wide)
  - [5,5,5,5,5], [9,9,9,9,9], [25,25,25,25,25]  — layer profile and ρ_func plots

Noise level: 0.0 only (clean training).
Seeds: 101–105.
Total: 6 archs × 1 noise × 5 seeds = 30 configs / 30 training runs.

Usage:
    python configs/generate_composite.py
    python configs/generate_composite.py --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO / "configs" / "composite_label_noise"

ARCHS = [
    [5, 5, 5],
    [5, 5, 5, 5, 5],
    [9, 9, 9],
    [9, 9, 9, 9, 9],
    [25, 25, 25],
    [25, 25, 25, 25, 25],
]
NOISE = 0.0
SEEDS = [101, 102, 103, 104, 105]

EPOCHS = 151
BATCH_SIZE = 32
LR = 0.001
MOMENTUM = 0.9
GLOBAL_SEED = 42
SAVE_INTERVAL = 10
SAVE_EPOCHS = [0, 1, 2, 3, 4, 6, 8, 10]


def arch_str(arch: list) -> str:
    return str(arch).replace(" ", "")


def make_config(arch: list, seed: int) -> dict:
    astr = str(arch)
    exp_name = f"n{NOISE}_{astr}"
    return {
        "experiment_name": exp_name,
        "dataset": "composite",
        "output_dir": "outputs/composite_label_noise",
        "architecture": arch,
        "dropout": 0.0,
        "global_seed": GLOBAL_SEED,
        "model_seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "SGD",
        "learning_rate": LR,
        "momentum": MOMENTUM,
        "noise": NOISE,
        "test_size": 0.2,
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
    for arch in ARCHS:
        astr = str(arch)
        for seed in SEEDS:
            out_dir = CONFIG_ROOT / f"n{NOISE}_{arch_str(arch)}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out = out_dir / f"seed_{seed}.yaml"
            if not args.list:
                with open(out, "w") as f:
                    yaml.dump(make_config(arch, seed), f, default_flow_style=False,
                              sort_keys=False)
            paths.append(out)

    for p in paths:
        print(p)
    if not args.list:
        print(f"\nWrote {len(paths)} configs to {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
