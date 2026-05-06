"""Generate per-(architecture, PCA-dim, seed) YAML configs for the MNIST capacity sweep.

Two architecture groups serve different figures:

  Group A — narrow 3-layer nets (calibration scatter + layer profile):
    archs: [3,3,3], [5,5,5]
    PCA dims: [10]
    → 2 × 1 × 5 = 10 configs

  Group B — wider 3-layer nets (capacity bars + ρ_func vs ε):
    archs: [7,7,7], [15,15,15], [25,25,25], [50,50,50]
    PCA dims: [2, 3, 4, 5, 10, 15, 20]
    → 4 × 7 × 5 = 140 configs

  Overlap: [7,7,7] at dim=10 appears in both figures.

Total: 10 + 140 = 150 training runs.

Usage:
    python configs/generate_mnist.py
    python configs/generate_mnist.py --list
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CONFIG_ROOT = REPO / "configs" / "mnist_capacity"

ARCHS_A = [[3, 3, 3], [5, 5, 5]]
DIMS_A = [10]

ARCHS_B = [[7, 7, 7], [15, 15, 15], [25, 25, 25], [50, 50, 50]]
DIMS_B = [2, 3, 4, 5, 10, 15, 20]

SEEDS = [101, 102, 103, 104, 105]

EPOCHS = 151
BATCH_SIZE = 32
LR = 0.001
MOMENTUM = 0.9
GLOBAL_SEED = 42
SAVE_INTERVAL = 10
SAVE_EPOCHS = [0, 1, 2, 3, 4, 6, 8, 10]


def make_config(arch: list, target_dim: int, seed: int) -> dict:
    astr = str(arch)
    exp_name = f"{target_dim}_dim_{astr}"
    return {
        "experiment_name": exp_name,
        "dataset": "mnist",
        "output_dir": "outputs/mnist_capacity",
        "architecture": arch,
        "target_dim": target_dim,
        "dropout": 0.0,
        "global_seed": GLOBAL_SEED,
        "model_seed": seed,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "SGD",
        "learning_rate": LR,
        "momentum": MOMENTUM,
        "noise": 0.0,
        "test_size": 0.2,
        "save_interval": SAVE_INTERVAL,
        "save_epochs": SAVE_EPOCHS,
    }


def _all_jobs():
    jobs = []
    for arch in ARCHS_A:
        for dim in DIMS_A:
            for seed in SEEDS:
                jobs.append((arch, dim, seed))
    seen = {(str(a), d) for a in ARCHS_A for d in DIMS_A}
    for arch in ARCHS_B:
        for dim in DIMS_B:
            key = (str(arch), dim)
            if key in seen:
                continue
            seen.add(key)
            for seed in SEEDS:
                jobs.append((arch, dim, seed))
    return jobs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true",
                   help="Print config paths only; do not write.")
    args = p.parse_args()

    paths = []
    for arch, dim, seed in _all_jobs():
        astr = str(arch)
        exp_name = f"{dim}_dim_{astr}"
        out_dir = CONFIG_ROOT / exp_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"seed_{seed}.yaml"
        if not args.list:
            with open(out, "w") as f:
                yaml.dump(make_config(arch, dim, seed), f, default_flow_style=False,
                          sort_keys=False)
        paths.append(out)

    for p in paths:
        print(p)
    if not args.list:
        print(f"\nWrote {len(paths)} configs to {CONFIG_ROOT}")


if __name__ == "__main__":
    main()
