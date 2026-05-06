"""Grouped bar chart: plug-in I_raw and I_func vs PCA dimension (MNIST capacity).

For each PCA dim, one bar group containing:
  - I_raw  (plug_in_bits,      epsilon-independent)
  - I_func at several key epsilons (plug_in_func_bits)

Two output figures:
  1. Per-architecture: 2×2 grid, one panel per arch.
  2. Single-panel:     averaged over the four selected architectures.

Inputs:
    results/mnist_capacity_new_estimator.csv

Outputs:
    figures/mnist_capacity_bars_per_arch.png / .pdf
    figures/mnist_capacity_bars_pooled.png   / .pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src_experiment.utils import savefig
from src_experiment.paths import neurips_figpath

REPO        = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
FIGURES_DIR = REPO / "figures"

LAST_EPOCH       = 150
TARGET_LAST_LAYER = 3
H_Y_MNIST        = 3.319

ARCHS = ["[7, 7, 7]", "[15, 15, 15]", "[25, 25, 25]", "[50, 50, 50]"]
ARCH_LABELS = {a: f"width {a.strip()[1:-1].split(',')[0].strip()}" for a in ARCHS}

EPSILONS_FUNC = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0]

# Bar colours: raw = dark gray; func epsilons = sequential Blues colormap
RAW_COLOUR   = "#555555"
_cmap        = plt.get_cmap("Blues")
FUNC_COLOURS = [_cmap(0.3 + 0.7 * i / (len(EPSILONS_FUNC) - 1))
                for i in range(len(EPSILONS_FUNC))]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "mnist_capacity_new_estimator.csv")
    df = df[(df["layer"] == TARGET_LAST_LAYER)
            & (df["epoch"] == LAST_EPOCH)
            & (df["arch_str"].isin(ARCHS))].copy()
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Return mean/std of plug_in_bits and plug_in_func_bits per (arch, target_dim, epsilon)."""
    agg_func = (
        df.groupby(["arch_str", "target_dim", "epsilon"])[["plug_in_func_bits"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_func.columns = ["arch_str", "target_dim", "epsilon",
                        "func_mean", "func_std"]

    agg_raw = (
        df[df["epsilon"] == 0.0]
        .groupby(["arch_str", "target_dim"])[["plug_in_bits"]]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg_raw.columns = ["arch_str", "target_dim", "raw_mean", "raw_std"]

    return agg_func, agg_raw


def _draw_panel(ax: plt.Axes, agg_func: pd.DataFrame, agg_raw: pd.DataFrame,
                arch: str | None, title: str) -> None:
    """Draw grouped bars for one panel (arch=None means pooled over archs)."""
    if arch is not None:
        f = agg_func[agg_func["arch_str"] == arch]
        r = agg_raw[agg_raw["arch_str"] == arch]
    else:
        f = agg_func.groupby(["target_dim", "epsilon"])[["func_mean", "func_std"]].mean().reset_index()
        r = agg_raw.groupby("target_dim")[["raw_mean", "raw_std"]].mean().reset_index()

    pca_dims = sorted(f["target_dim"].unique())
    n_bars   = 1 + len(EPSILONS_FUNC)   # raw + func epsilons
    width    = 0.8 / n_bars
    offsets  = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * width
    xs       = np.arange(len(pca_dims))

    # Raw bars
    for xi, dim in enumerate(pca_dims):
        row = r[r["target_dim"] == dim]
        if row.empty:
            continue
        ym  = float(row["raw_mean"].iloc[0])
        ye  = float(row["raw_std"].fillna(0).iloc[0])
        ax.bar(xs[xi] + offsets[0], ym, width=width * 0.9,
               color=RAW_COLOUR, alpha=0.85)
        ax.errorbar(xs[xi] + offsets[0], ym, yerr=ye,
                    fmt="none", color="k", lw=1.0, capsize=2)

    # Func bars per epsilon
    for ei, eps in enumerate(EPSILONS_FUNC):
        colour = FUNC_COLOURS[ei]
        fe = f[np.isclose(f["epsilon"], eps)]
        for xi, dim in enumerate(pca_dims):
            row = fe[fe["target_dim"] == dim]
            if row.empty:
                continue
            ym = float(row["func_mean"].iloc[0])
            ye = float(row["func_std"].fillna(0).iloc[0])
            ax.bar(xs[xi] + offsets[1 + ei], ym, width=width * 0.9,
                   color=colour, alpha=0.85)
            ax.errorbar(xs[xi] + offsets[1 + ei], ym, yerr=ye,
                        fmt="none", color="k", lw=1.0, capsize=2)

    ax.axhline(H_Y_MNIST, color="k", linestyle="--", lw=1.0, alpha=0.6)
    ax.text(xs[-1] + 0.2, H_Y_MNIST + 0.05, r"$H(Y)$", fontsize=14, alpha=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(pca_dims, fontsize=16)
    ax.set_xlabel("PCA dim", fontsize=18)
    ax.set_ylabel("MI (bits)", fontsize=18)
    ax.set_ylim(0, H_Y_MNIST + 0.4)
    ax.set_title(title, fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.2, axis="y")


def _make_legend_handles() -> list:
    handles = [mpatches.Patch(color=RAW_COLOUR, label=r"$\hat{I}$")]
    for eps, colour in zip(EPSILONS_FUNC, FUNC_COLOURS):
        handles.append(mpatches.Patch(
            color=colour, label=rf"$\hat{{I}}_{{\mathrm{{func}}}}$, $\varepsilon={eps:g}$"))
    return handles


def plot_per_arch(agg_func: pd.DataFrame, agg_raw: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 5.5), sharey=True, sharex=True)
    for ax, arch in zip(axes.flat, ARCHS):
        _draw_panel(ax, agg_func, agg_raw, arch, ARCH_LABELS[arch])
    for ax in axes[0, :]:
        ax.set_xlabel("")
    for ax in axes[:, 1]:
        ax.set_ylabel("")
    fig.legend(handles=_make_legend_handles(), loc="upper center",
               ncol=len(EPSILONS_FUNC) + 1, bbox_to_anchor=(0.5, 1.1),
               frameon=False, fontsize=14)
    fig.tight_layout()
    savefig(fig, neurips_figpath / "mnist_capacity_bars_per_arch")


def plot_pooled(agg_func: pd.DataFrame, agg_raw: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    _draw_panel(ax, agg_func, agg_raw, arch=None,
                title=f"Averaged over widths {[ARCH_LABELS[a] for a in ARCHS]}")
    fig.legend(handles=_make_legend_handles(), loc="upper center",
               ncol=len(EPSILONS_FUNC) + 1, bbox_to_anchor=(0.5, 1.01),
               frameon=False, fontsize=16)
    fig.tight_layout()
    savefig(fig, neurips_figpath / "mnist_capacity_bars_pooled")


def main() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    df = load_data()
    if df.empty:
        raise SystemExit("No data found.")
    agg_func, agg_raw = _aggregate(df)
    plot_per_arch(agg_func, agg_raw)
    # plot_pooled(agg_func, agg_raw)


if __name__ == "__main__":
    main()
