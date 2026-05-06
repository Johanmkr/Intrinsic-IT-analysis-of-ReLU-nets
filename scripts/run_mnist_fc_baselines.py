"""Compute MI baselines for MNIST-capacity FC networks at the last epoch.

The MNIST capacity sweep (`outputs/mnist_capacity/10_dim_*/`) trained FC
networks on PCA-10 MNIST. This script taps the deepest hidden-layer
activation (layer 3, pre-ReLU) at epoch 150 and evaluates the same three
baselines used for Composite and WBC, so the routing estimator calibration
scatter can include a third MNIST dataset.

Only the three narrowest architectures ([3,3,3], [5,5,5], [7,7,7]) are
included; wider nets hit the fine-resolution regime and have negative M-M
routing bits, making them uninformative for calibration.

Output:
    results/mnist_fc_baselines.csv   — one row per (arch, seed)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src_experiment.baselines.activations import load_layer_activations
from src_experiment.baselines.mi_baselines import binning_mi, kmeans_mi, ksg_mi

TARGET_DIM = 10
EPOCH = 150
LAYERS = [1, 2, 3]   # all hidden layers for 3-hidden-layer nets
NUM_CLASSES = 10      # MNIST
ARCHS = ["[3, 3, 3]", "[5, 5, 5]", "[7, 7, 7]"]
SEEDS = [101, 102, 103, 104, 105]
BASE_DIR = REPO / "outputs" / "mnist_capacity"
OUT_CSV = REPO / "results" / "mnist_fc_baselines.csv"


def _h5_path(arch: str, seed: int) -> Path:
    subdir = f"{TARGET_DIM}_dim_{arch}"
    return BASE_DIR / subdir / f"seed_{seed}.h5"


def compute_rows(arch: str, seed: int) -> List[dict]:
    h5 = _h5_path(arch, seed)
    if not h5.exists():
        print(f"  [skip] missing {h5}")
        return []

    import h5py
    with h5py.File(h5, "r") as f:
        X = np.asarray(f["points"][:], dtype=np.float32)
        y = np.asarray(f["labels"][:], dtype=np.int64)

    rows = []
    for layer in LAYERS:
        T = load_layer_activations(h5, EPOCH, layer, X, kind="pre")
        res_bin = binning_mi(T, y, n_bins=8, num_classes=NUM_CLASSES)
        res_km = kmeans_mi(T, y, K=NUM_CLASSES, num_classes=NUM_CLASSES)
        res_ksg = ksg_mi(T, y, k=3)
        rows.append({
            "dataset": "mnist_fc",
            "arch_str": arch,
            "seed": seed,
            "epoch": EPOCH,
            "layer": layer,
            "target_dim": TARGET_DIM,
            "N": len(y),
            "bits_binning_8": res_bin["bits"],
            "bits_kmeans_KKY": res_km["bits"],
            "bits_ksg_k3": res_ksg["bits"],
        })
    print(f"  {arch} seed={seed}: layers {LAYERS} done", flush=True)
    return rows


def main() -> None:
    all_rows: List[dict] = []
    for arch in ARCHS:
        for seed in SEEDS:
            print(f"computing {arch} seed={seed} …")
            all_rows.extend(compute_rows(arch, seed))

    df = pd.DataFrame(all_rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nwrote {OUT_CSV}  ({len(df)} rows)")
    print(df[["arch_str", "seed", "layer", "bits_binning_8", "bits_kmeans_KKY", "bits_ksg_k3"]].to_string(index=False))


if __name__ == "__main__":
    main()
