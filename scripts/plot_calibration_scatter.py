"""Calibration scatter: plug-in routing MI (x) vs each baseline (y).

For every (dataset, architecture, seed) at last epoch / deepest layer,
scatter plug-in routing MI against each baseline. Diagonal y=x and Pearson r
show agreement across all conditions.

Inputs:
    results/mi_baselines.csv
    results/composite_label_noise_new_estimator.csv
    results/wbc_label_noise_new_estimator.csv
    results/mnist_capacity_new_estimator.csv
    results/mnist_fc_baselines.csv

Outputs:
    figures/calibration_scatter_raw.png / .pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src_experiment.utils import savefig
from src_experiment.paths import neurips_figpath

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FIGURES = REPO / "figures"

LAST_EPOCH_BY_DATASET = {"composite": 150, "wbc": 150}
DATASETS = ["composite", "wbc"]

DATASET_COLOUR = {
    "composite": "tab:blue",
    "wbc": "tab:orange",
    "mnist_fc": "tab:green",
}
DATASET_LABEL = {
    "composite": "Composite (7 cls)",
    "wbc": "WBC (2 cls)",
    "mnist_fc": "MNIST FC (10 cls, PCA-10)",
}

ARCHS_C = [
    "[5, 5, 5]", "[5, 5, 5, 5, 5]",
    "[9, 9, 9]", "[9, 9, 9, 9, 9]",
    "[25, 25, 25]", "[25, 25, 25, 25, 25]",
]

ARCHS_MNIST_FC = ["[3, 3, 3]", "[5, 5, 5]", "[7, 7, 7]"]
MNIST_FC_TARGET_DIM = 10
MNIST_FC_EPOCH = 150
MNIST_FC_LAYER = 3
MNIST_FC_EPS = 1.0

BASELINES = [
    ("bits_binning_8",  r"Binning $K{=}8$"),
    ("bits_kmeans_KKY", r"k-means $K{=}|Y|$"),
    ("bits_ksg_k3",     r"KSG $k{=}3$"),
]

OURS_COL   = "plug_in_bits"
OURS_LABEL = r"$\hat{I}$: Ours [bits]"
OUT_STEM   = "calibration_scatter_raw"


def _arch_depth(arch_str: str) -> int:
    return arch_str.count(",") + 1


def _load_composite_wbc() -> pd.DataFrame:
    bl = pd.read_csv(RESULTS / "mi_baselines.csv")
    bl = bl[bl["dataset"].isin(DATASETS) & bl["arch_str"].isin(ARCHS_C)].copy()
    bl["depth"] = bl["arch_str"].map(_arch_depth)
    keep = []
    for ds in DATASETS:
        sub = bl[(bl["dataset"] == ds)
                 & (bl["epoch"] == LAST_EPOCH_BY_DATASET[ds])
                 & (bl["layer"] == bl["depth"])]
        keep.append(sub)
    bl = pd.concat(keep, ignore_index=True)

    frames = []
    for ds in DATASETS:
        est = pd.read_csv(RESULTS / f"{ds}_label_noise_new_estimator.csv")
        est = est[(est["noise_level"] == 0.0)
                  & (est["epoch"] == LAST_EPOCH_BY_DATASET[ds])
                  & (est["arch_str"].isin(ARCHS_C))
                  & (est["epsilon"] == 0.0)].copy()
        est["depth"] = est["arch_str"].map(_arch_depth)
        est = est[est["layer"] == est["depth"]][
            ["dataset", "arch_str", "seed", OURS_COL]
        ]
        frames.append(est)
    est_all = pd.concat(frames, ignore_index=True)

    bl = bl.drop(columns=["bits_ours_plugin", "bits_ours_raw", "bits_ours_func",
                           "bits_ours_func_plugin"], errors="ignore")
    out = bl.merge(est_all, on=["dataset", "arch_str", "seed"],
                   how="inner").reset_index(drop=True)
    return out[out[OURS_COL].notna()].copy()


def _load_mnist_fc() -> pd.DataFrame:
    bl_path = RESULTS / "mnist_fc_baselines.csv"
    est_path = RESULTS / "mnist_capacity_new_estimator.csv"
    if not bl_path.exists() or not est_path.exists():
        print("[warn] MNIST FC files missing — skipping")
        return pd.DataFrame()

    bl = pd.read_csv(bl_path)
    est = pd.read_csv(est_path)
    est = est[(est["epoch"] == MNIST_FC_EPOCH)
              & (est["layer"] == MNIST_FC_LAYER)
              & (np.isclose(est["epsilon"], MNIST_FC_EPS))
              & (est["target_dim"] == MNIST_FC_TARGET_DIM)
              & (est["arch_str"].isin(ARCHS_MNIST_FC))][
        ["arch_str", "seed", OURS_COL]
    ].copy()

    merged = bl.merge(est, on=["arch_str", "seed"], how="inner")
    merged["dataset"] = "mnist_fc"
    return merged


def load_data() -> pd.DataFrame:
    parts = [_load_composite_wbc(), _load_mnist_fc()]
    return pd.concat([p for p in parts if not p.empty], ignore_index=True)


def plot(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(BASELINES),
                             figsize=(4.0 * len(BASELINES), 4.0),
                             sharey=True)
    for ax, (bl_col, bl_label) in zip(axes, BASELINES):
        sub = df[df[bl_col].notna() & df[OURS_COL].notna()].copy()
        if sub.empty:
            ax.set_title(f"{bl_label} (no data)")
            continue
        x = sub[OURS_COL].to_numpy()
        y = sub[bl_col].to_numpy()
        for ds, sub_ds in sub.groupby("dataset"):
            ax.scatter(sub_ds[bl_col], sub_ds[OURS_COL],
                       s=22, alpha=0.7,
                       color=DATASET_COLOUR.get(ds, "tab:gray"),
                       label=DATASET_LABEL.get(ds, ds),
                       edgecolors="white", linewidths=0.4)
        ax.plot([0.0, 2.5], [0.0, 2.5], "k--", lw=0.8, alpha=0.5)
        r = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.04, 0.96, rf"$r = {r:.3f}$  ($n={len(x)}$)",
                transform=ax.transAxes, ha="left", va="top", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
        ax.set_xlim(0.0, 2.5)
        ax.set_ylim(0.0, 2.5)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$\hat{I}$: " + bl_label + " [bits]", fontsize=14)
        ax.set_ylabel(OURS_LABEL if ax is axes[0] else "", fontsize=14)
        # ax.set_title(bl_label, fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.1), frameon=False, fontsize=12)
    # fig.suptitle("Calibration: plug-in routing MI vs baselines (last epoch, deepest layer)",
    #              y=1.08, fontsize=14)
    fig.tight_layout()
    savefig(fig, neurips_figpath / OUT_STEM)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    plot(load_data())


if __name__ == "__main__":
    main()
