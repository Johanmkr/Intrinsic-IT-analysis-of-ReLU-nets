"""MI vs epsilon at last epoch sweeping PCA dimension, per architecture.

The key story: raw plug-in MI (epsilon=0) saturates toward H(Y) as PCA dim
grows; functional equivalence (func) pulls it back down as epsilon increases.

Usage
-----
    uv run python scripts/plot_mnist_functional_pca_sweep.py
    uv run python scripts/plot_mnist_functional_pca_sweep.py --epoch 100

--type choices
    raw     plug-in routing MI vs epsilon   (horizontal lines — reference)
    func    plug-in functional MI vs epsilon (main story plot)
    rawmm   raw Miller-Madow vs epsilon      (reference)
    funcmm  functional Miller-Madow vs eps   (secondary)

Inputs:
    results/mnist_capacity_new_estimator.csv

Outputs:
    figures/mnist_{type}_vs_eps.png / .pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src_experiment.utils import savefig
from src_experiment.paths import neurips_figpath

REPO = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO / "results"
FIGURES_DIR = REPO / "figures"

H_Y_MNIST = 3.319
TARGET_LAST_LAYER = 3
DEFAULT_LAST_EPOCH = 150
EXCLUDE_ARCHS  = {"[5, 5, 5]"}                                        # func/raw plots: 6 panels → 2×3
ARCHS_CAPACITY = {"[7, 7, 7]", "[15, 15, 15]", "[25, 25, 25]", "[50, 50, 50]"}  # rho plot: matches capacity bars

TYPE_META = {
    "raw":    ("plug_in_bits",           r"$\hat{I}_{\mathrm{raw}}$",    False),
    "rawmm":  ("miller_madow_bits",      r"$\tilde{I}_{\mathrm{raw}}$",  False),
    "func":   ("plug_in_func_bits",      r"$\hat{I}_{\mathrm{func}}$",   True),
    "funcmm": ("miller_madow_func_bits", r"$\tilde{I}_{\mathrm{func}}$", True),
    "rho":    ("rho_func",               r"$\rho_{\mathrm{func}}$",      True),
}


def load_data(epoch: int) -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "mnist_capacity_new_estimator.csv")
    return df[(df["layer"] == TARGET_LAST_LAYER) & (df["epoch"] == epoch)].copy()


def _draw_vs_epsilon(
    df: pd.DataFrame,
    func_col: str,
    raw_col: str,
    func_label: str,
    raw_label: str,
    out_stem: str,
    epoch: int,
) -> None:
    """One panel per architecture; x=epsilon, y=MI; lines per PCA dim.

    Each line starts at raw MI (epsilon=0) and shows how functional equivalence
    compresses the estimate as epsilon grows.
    """
    agg_func = (
        df.groupby(["arch_str", "epsilon", "target_dim"])[func_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    # Raw MI is epsilon-independent — read from epsilon=0 rows
    agg_raw = (
        df[df["epsilon"] == 0.0]
        .groupby(["arch_str", "target_dim"])[raw_col]
        .mean()
        .reset_index()
        .rename(columns={raw_col: "raw_mean"})
    )

    archs = sorted(
        [a for a in agg_func["arch_str"].unique() if a not in EXCLUDE_ARCHS],
        key=lambda s: int(s.strip()[1:-1].split(",")[0]),
    )
    pca_dims = sorted(agg_func["target_dim"].unique())
    epsilons = sorted(agg_func["epsilon"].unique())

    palette = plt.get_cmap("tab10")
    colours = {d: palette(i % 10) for i, d in enumerate(pca_dims)}

    ncols = 3
    nrows = (len(archs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             sharey=True, sharex=True, squeeze=False)

    for idx, arch in enumerate(archs):
        ax = axes[idx // ncols][idx % ncols]
        sub_func = agg_func[agg_func["arch_str"] == arch]
        sub_raw = agg_raw[agg_raw["arch_str"] == arch].set_index("target_dim")["raw_mean"]

        for d in pca_dims:
            colour = colours[d]
            row = sub_func[sub_func["target_dim"] == d].sort_values("epsilon")
            if row.empty:
                continue
            xs = row["epsilon"].to_numpy()
            ys = row["mean"].to_numpy()
            es = row["std"].fillna(0).to_numpy()
            ax.plot(xs, ys, color=colour, lw=2.0)
            ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=0.12, lw=0)
            raw_val = sub_raw.get(d, np.nan)
            if np.isfinite(raw_val):
                ax.axhline(raw_val, color=colour, lw=0.9, linestyle=":", alpha=0.6)

        ax.axhline(H_Y_MNIST, color="k", linestyle="--", lw=1.0, alpha=0.5)
        ax.text(epsilons[-1] * 0.02, H_Y_MNIST + 0.05, "H(Y)", fontsize=11, alpha=0.6)
        ax.set_ylim(bottom=-0.05, top=H_Y_MNIST + 0.35)
        ax.set_xlim(left=-0.02, right=max(epsilons) + 0.05)
        ax.set_title(arch, fontsize=13)
        if idx // ncols == nrows - 1 or idx + ncols >= len(archs):
            ax.set_xlabel(r"$\varepsilon$", fontsize=13)
        if idx % ncols == 0:
            ax.set_ylabel(f"{func_label} (bits)", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.2)

    for idx in range(len(archs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color=colours[d], lw=2.0, label=f"d={d}")
        for d in pca_dims
    ]
    fig.legend(handles=legend_handles, title="PCA dim", title_fontsize=12,
               loc="upper center", ncol=len(pca_dims),
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=11)

    fig.suptitle(
        f"MNIST: {func_label} vs $\\varepsilon$ by PCA dim, per architecture"
        f" (last layer, epoch {epoch})\n"
        r"Dotted lines: raw $\hat{I}$ (no merging)",
        fontsize=13, y=1.04,
    )
    fig.tight_layout()
    savefig(fig, neurips_figpath / out_stem)


def _draw_rho_vs_epsilon(df: pd.DataFrame, out_stem: str, epoch: int) -> None:
    """ρ_func vs ε, one panel per architecture, lines per PCA dim."""
    agg = (
        df.groupby(["arch_str", "epsilon", "target_dim"])["rho_func"]
        .agg(["mean", "std"])
        .reset_index()
    )
    archs = sorted(
        [a for a in agg["arch_str"].unique() if a in ARCHS_CAPACITY],
        key=lambda s: int(s.strip()[1:-1].split(",")[0]),
    )
    pca_dims = sorted(agg["target_dim"].unique())
    epsilons = sorted(agg["epsilon"].unique())
    palette = plt.get_cmap("tab10")
    colours = {d: palette(i % 10) for i, d in enumerate(pca_dims)}

    ncols = 4
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows),
                             sharey=True, sharex=True, squeeze=False)

    for idx, arch in enumerate(archs):
        ax = axes[idx // ncols][idx % ncols]
        sub = agg[agg["arch_str"] == arch]
        for d in pca_dims:
            row = sub[sub["target_dim"] == d].sort_values("epsilon")
            if row.empty:
                continue
            xs = row["epsilon"].to_numpy()
            ys = row["mean"].to_numpy()
            es = row["std"].fillna(0).to_numpy()
            ax.plot(xs, ys, color=colours[d], lw=2.0)
            ax.fill_between(xs, ys - es, ys + es, color=colours[d], alpha=0.12, lw=0)

        ax.axhline(1.0, color="k", linestyle="--", lw=1.0, alpha=0.5)
        ax.text(epsilons[-1] * 0.02, 1.01, r"$\rho=1$", fontsize=16, alpha=0.6)
        ax.set_ylim(bottom=-0.02, top=1.12)
        ax.set_xlim(left=-0.02, right=max(epsilons) + 0.05)
        width = int(arch.strip()[1:-1].split(",")[0])
        ax.set_title(f"width {width}", fontsize=18)
        if idx // ncols == nrows - 1 or idx + ncols >= len(archs):
            ax.set_xlabel(r"$\varepsilon$", fontsize=18)
        if idx % ncols == 0:
            ax.set_ylabel(r"$\rho_{\mathrm{func}}$", fontsize=18)
        ax.tick_params(labelsize=16)
        ax.grid(alpha=0.2)

    for idx in range(len(archs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_handles = [
        plt.Line2D([0], [0], color=colours[d], lw=2.0, label=r"$d$={dd}".format(dd=d))
        for d in pca_dims
    ]
    fig.legend(handles=legend_handles,
               loc="upper center", ncol=len(pca_dims),
               bbox_to_anchor=(0.5, 1.08), frameon=False, fontsize=16)

    # fig.suptitle(
    #     r"MNIST: $\rho_{\mathrm{func}}$ vs $\varepsilon$ by PCA dim, per architecture"
    #     f" (last layer, epoch {epoch})",
    #     fontsize=18, y=1.04,
    # )
    fig.tight_layout()
    savefig(fig, neurips_figpath / out_stem)


def _draw_eps_independent(
    df: pd.DataFrame, col: str, ylabel: str, out_stem: str,
) -> None:
    """For raw/rawmm: one panel per architecture, x=epoch, y=MI, lines per PCA dim."""
    df = df[df["epsilon"] == 0.0]
    agg = (
        df.groupby(["arch_str", "target_dim"])[col]
        .agg(["mean", "std"])
        .reset_index()
    )
    archs = sorted(agg["arch_str"].unique(),
                   key=lambda s: int(s.strip()[1:-1].split(",")[0]))
    pca_dims = sorted(agg["target_dim"].unique())
    palette = plt.get_cmap("tab10")
    colours = {d: palette(i % 10) for i, d in enumerate(pca_dims)}

    ncols = min(len(archs), 4)
    nrows = (len(archs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, arch in enumerate(archs):
        ax = axes[idx // ncols][idx % ncols]
        sub = agg[agg["arch_str"] == arch].sort_values("target_dim")
        xs = sub["target_dim"].to_numpy()
        ys = sub["mean"].to_numpy()
        es = sub["std"].fillna(0).to_numpy()
        colours_list = [colours[d] for d in xs]
        ax.bar(xs, ys, color=colours_list, alpha=0.8, width=0.8)
        ax.errorbar(xs, ys, yerr=es, fmt="none", color="k", lw=1, capsize=3)
        ax.axhline(H_Y_MNIST, color="k", linestyle="--", lw=0.9, alpha=0.5)
        ax.text(xs[0], H_Y_MNIST + 0.04, "H(Y)", fontsize=7, alpha=0.6)
        ax.set_ylim(bottom=0, top=H_Y_MNIST + 0.3)
        ax.set_xticks(xs)
        ax.set_title(arch)
        ax.set_xlabel("PCA dim")
        ax.set_ylabel(f"{ylabel} (bits)")
        ax.grid(alpha=0.2, axis="y")

    for idx in range(len(archs), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=colours[d], label=f"d={d}")
        for d in pca_dims
    ]
    fig.legend(handles=legend_handles, title="PCA dim", title_fontsize=11,
               loc="upper center", ncol=len(pca_dims),
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=10)

    fig.suptitle(
        f"MNIST: {ylabel} at last epoch by PCA dim, per architecture (last layer)",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    savefig(fig, neurips_figpath / out_stem)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--type", choices=list(TYPE_META), default="func",
                   help="which estimator to plot (default: func)")
    p.add_argument("--epoch", type=int, default=DEFAULT_LAST_EPOCH,
                   help=f"epoch to use (default: {DEFAULT_LAST_EPOCH})")
    args = p.parse_args()

    FIGURES_DIR.mkdir(exist_ok=True)
    df = load_data(args.epoch)
    if df.empty:
        raise SystemExit("No data — run mnist_capacity experiments first.")

    col, ylabel, eps_dependent = TYPE_META[args.type]
    out_stem = f"mnist_{args.type}_vs_eps"

    if args.type == "rho":
        _draw_rho_vs_epsilon(df, out_stem, args.epoch)
    elif eps_dependent:
        raw_col = "plug_in_bits" if args.type == "func" else "miller_madow_bits"
        raw_label = (r"$\hat{I}_{\mathrm{raw}}$" if args.type == "func"
                     else r"$\tilde{I}_{\mathrm{raw}}$")
        _draw_vs_epsilon(df, col, raw_col, ylabel, raw_label, out_stem, args.epoch)
    else:
        _draw_eps_independent(df, col, ylabel, out_stem)


if __name__ == "__main__":
    main()
