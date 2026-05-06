"""Panel B — per-layer profile at the last epoch.

For each dataset we plot bits as a function of layer index for the four
routing-estimator variants and the three cheap baselines (where available),
averaged across seeds and architectures of the same depth.

Composite and WBC use 5-hidden-layer architectures (layers 1-5).
MNIST uses 3-hidden-layer FC networks on PCA-10 inputs (layers 1-3);
baselines are not computed for this configuration.

Inputs:
    results/mi_baselines.csv
    results/composite_label_noise_new_estimator.csv
    results/wbc_label_noise_new_estimator.csv
    results/mnist_capacity_new_estimator.csv

Outputs:
    figures/layer_profile_last_epoch.png  (also written to neurips_figpath)
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

NOISE = 0.0
LAST_EPOCH = 150
EPS_FUNC = 1.0
LAYERS_FIVE = [1, 2, 3, 4, 5]
LAYERS_THREE = [1, 2, 3]

# 5-layer architectures for Composite / WBC.
ARCHS_FIVE = [
    "[5, 5, 5, 5, 5]",
    "[9, 9, 9, 9, 9]",
    "[25, 25, 25, 25, 25]",
]

# 3-layer FC architectures for MNIST (widths chosen to avoid fine-resolution
# regime where M-M correction turns negative).
ARCHS_MNIST = ["[5, 5, 5]", "[7, 7, 7]"]
MNIST_TARGET_DIM = 10  # PCA components fed to the FC networks

DATASETS = ["composite", "wbc", "mnist"]
DATASET_TITLE = {"composite": "Composite", "wbc": "WBC", "mnist": "MNIST (PCA-10, 3 layers)"}

OURS = [
    ("plug_in_bits", r"$\hat I$", "tab:blue", "-", "o"),
]

BASELINES = [
    ("bits_binning_8", r"binning $K{=}8$", "tab:gray"),
    ("bits_kmeans_KKY", r"k-means $K{=}|Y|$", "tab:olive"),
    ("bits_ksg_k3", r"KSG $k{=}3$", "tab:cyan"),
]


def load_ours(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"{dataset}_label_noise_new_estimator.csv")
    return df[
        (df["noise_level"] == NOISE)
        & (df["epoch"] == LAST_EPOCH)
        & (df["arch_str"].isin(ARCHS_FIVE))
        & (df["layer"].isin(LAYERS_FIVE))
        & (np.isclose(df["epsilon"], EPS_FUNC))
    ].copy()


def load_ours_mnist() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "mnist_capacity_new_estimator.csv")
    return df[
        (df["epoch"] == LAST_EPOCH)
        & (df["target_dim"] == MNIST_TARGET_DIM)
        & (df["arch_str"].isin(ARCHS_MNIST))
        & (df["layer"].isin(LAYERS_THREE))
        & (np.isclose(df["epsilon"], EPS_FUNC))
    ].copy()


def load_baselines(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "mi_baselines.csv")
    return df[
        (df["dataset"] == dataset)
        & (df["noise_level"] == NOISE)
        & (df["epoch"] == LAST_EPOCH)
        & (df["arch_str"].isin(ARCHS_FIVE))
        & (df["layer"].isin(LAYERS_FIVE))
    ].copy()


def load_baselines_mnist() -> pd.DataFrame:
    path = RESULTS / "mnist_fc_baselines.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[
        (df["arch_str"].isin(ARCHS_MNIST))
        & (df["layer"].isin(LAYERS_THREE))
    ].copy()


def aggregate(df: pd.DataFrame, cols: list[tuple]) -> pd.DataFrame:
    rows = []
    for col, label, *_ in cols:
        if col not in df.columns:
            continue
        grp = df.groupby("layer")[col].agg(["mean", "std", "count"]).reset_index()
        grp["estimator"] = label
        grp["estimator_col"] = col
        rows.append(grp)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).rename(
        columns={"mean": "mean_bits", "std": "std_bits", "count": "n"}
    )


def _draw_panel(
    ax: plt.Axes,
    ds: str,
    os_df: pd.DataFrame,
    bs_df: pd.DataFrame,
    h_y: float,
    layers: list[int],
) -> None:
    for col, label, colour, ls, marker in OURS:
        row = os_df[os_df["estimator_col"] == col].sort_values("layer")
        if row.empty:
            continue
        xs = row["layer"].to_numpy()
        ys = row["mean_bits"].to_numpy()
        es = row["std_bits"].to_numpy()
        ax.plot(xs, ys, marker=marker, color=colour, linestyle=ls, lw=1.5,
                markersize=5, label=label)
        ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=0.12, lw=0)

    for col, label, colour in BASELINES:
        if bs_df.empty:
            break
        row = bs_df[bs_df["estimator_col"] == col].sort_values("layer")
        if row.empty:
            continue
        xs = row["layer"].to_numpy()
        ys = row["mean_bits"].to_numpy()
        es = row["std_bits"].to_numpy()
        ax.plot(xs, ys, marker="x", color=colour, linestyle="--", lw=1.0,
                markersize=5, alpha=0.85, label=label)
        ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=0.07, lw=0)

    ax.axhline(h_y, color="k", linestyle="-.", lw=0.8, alpha=0.5)
    ax.text(
        layers[-1]-1,
        h_y+0.1,
        r"$H(Y)={h_y:.2f}$".format(h_y=h_y),
        va="center",
        ha="left",
        fontsize=12,
        alpha=0.7,
    )
    ax.set_xticks(layers)
    ax.set_xticklabels(layers, fontsize=12)
    ax.set_xlabel("layer", fontsize=14)
    ax.set_title(DATASET_TITLE.get(ds, ds), fontsize=14)
    ax.set_ylim(bottom=0, top=3.5)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.25)


def plot(
    ours_summary: dict[str, pd.DataFrame],
    baseline_summary: dict[str, pd.DataFrame],
    H_Y: dict[str, float],
    out_path: Path,
) -> None:
    n = len(ours_summary)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.0), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, [d for d in DATASETS if d in ours_summary]):
        layers = LAYERS_THREE if ds == "mnist" else LAYERS_FIVE
        _draw_panel(
            ax, ds,
            ours_summary.get(ds, pd.DataFrame()),
            baseline_summary.get(ds, pd.DataFrame()),
            H_Y[ds],
            layers,
        )

    axes[0].set_ylabel("bits", fontsize=14)

    handles, labels = [], []
    seen = set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h)
                labels.append(l)
                seen.add(l)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
        fontsize=12,
    )
    # fig.suptitle(
    #     rf"Layerwise bits at last epoch ($\eta=0$, $\varepsilon={EPS_FUNC:g}$)",
    #     y=1.13,
    #     fontsize=18,
    # )
    savefig(fig, neurips_figpath / "layer_profile_last_epoch")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    ours_summary, baseline_summary, H_Y = {}, {}, {}

    for ds in ["composite", "wbc"]:
        ours = load_ours(ds)
        if ours.empty:
            print(f"[warn] no routing-estimator rows for {ds}")
            continue
        ours_summary[ds] = aggregate(ours, OURS)
        bs = load_baselines(ds)
        if bs.empty:
            print(f"[warn] no baseline rows for {ds} — Phase 1a may not be done")
        baseline_summary[ds] = aggregate(bs, BASELINES) if not bs.empty else pd.DataFrame()
        H_Y[ds] = float(ours["H_Y_bits"].mean())

    mnist_ours = load_ours_mnist()
    if mnist_ours.empty:
        print("[warn] no MNIST routing rows — skipping MNIST panel")
    else:
        ours_summary["mnist"] = aggregate(mnist_ours, OURS)
        mnist_bs = load_baselines_mnist()
        baseline_summary["mnist"] = aggregate(mnist_bs, BASELINES) if not mnist_bs.empty else pd.DataFrame()
        H_Y["mnist"] = float(mnist_ours["H_Y_bits"].mean())

    if not ours_summary:
        raise SystemExit("no datasets to plot")
    plot(ours_summary, baseline_summary, H_Y, FIGURES / "layer_profile_last_epoch.png")


if __name__ == "__main__":
    main()
