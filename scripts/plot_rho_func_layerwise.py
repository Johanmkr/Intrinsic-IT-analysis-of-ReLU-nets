"""ρ_func as a function of layer depth across all three datasets.

ρ_func = |Ω_{D,func}| / |Ω_D| ∈ (0,1] measures how much combinatorial
capacity is genuinely used vs. collapsed by functional equivalence.  Values
near 1 mean almost every activation pattern implements a distinct affine map;
values near 0 mean the network reuses a small set of affine maps across many
distinct routing paths.

One panel per dataset; within each panel one line per architecture width,
mean ± std across seeds.

Inputs:
    results/composite_label_noise_new_estimator.csv
    results/wbc_label_noise_new_estimator.csv
    results/mnist_capacity_new_estimator.csv

Outputs:
    figures/rho_func_layerwise.png  (also written to neurips_figpath)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

ARCHS_FIVE = ["[5, 5, 5, 5, 5]", "[9, 9, 9, 9, 9]", "[25, 25, 25, 25, 25]"]
ARCHS_MNIST = ["[5, 5, 5]", "[7, 7, 7]"]
MNIST_TARGET_DIM = 10

EPSILONS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

DATASET_TITLE = {
    "composite": "Composite (7 cls, $N$=10k)",
    "wbc": "WBC (2 cls, $N$=569)",
    "mnist": "MNIST (10 cls, PCA-10)",
}


def load_composite_wbc(dataset: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"{dataset}_label_noise_new_estimator.csv")
    return df[
        (df["noise_level"] == NOISE)
        & (df["epoch"] == LAST_EPOCH)
        & (df["arch_str"].isin(ARCHS_FIVE))
    ][["arch_str", "layer", "epsilon", "rho_func", "seed"]].copy()


def load_mnist() -> pd.DataFrame:
    df = pd.read_csv(RESULTS / "mnist_capacity_new_estimator.csv")
    return df[
        (df["epoch"] == LAST_EPOCH)
        & (df["target_dim"] == MNIST_TARGET_DIM)
        & (df["arch_str"].isin(ARCHS_MNIST))
    ][["arch_str", "layer", "epsilon", "rho_func", "seed"]].copy()


def _plot_panel(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    palette = plt.get_cmap("tab10")
    colours = {eps: palette(i) for i, eps in enumerate(EPSILONS)}

    for eps in EPSILONS:
        sub = df[np.isclose(df["epsilon"], eps)].groupby("layer")["rho_func"].agg(
            ["mean", "std"]
        ).reset_index()
        if sub.empty:
            continue
        xs = sub["layer"].to_numpy()
        ys = sub["mean"].to_numpy()
        es = sub["std"].to_numpy()
        colour = colours[eps]
        ax.plot(xs, ys, marker="o", color=colour, lw=1.8, markersize=5,
                label=rf"$\varepsilon={eps:g}$")
        ax.fill_between(xs, ys - es, ys + es, color=colour, alpha=0.13, lw=0)

    ax.axhline(1.0, color="k", linestyle="--", lw=0.7, alpha=0.4)
    layers = sorted(df["layer"].unique())
    ax.set_xticks(layers)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel(r"$\rho_{\mathrm{func}}$", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, frameon=False)


def main() -> None:
    FIGURES.mkdir(exist_ok=True)

    data = {
        "composite": load_composite_wbc("composite"),
        "wbc":       load_composite_wbc("wbc"),
        "mnist":     load_mnist(),
    }

    n = len(data)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 3.8), sharey=True)

    for ax, (ds, df) in zip(axes, data.items()):
        if df.empty:
            print(f"[warn] no data for {ds}")
            ax.set_visible(False)
            continue
        _plot_panel(ax, df, DATASET_TITLE[ds])
        if ax is not axes[0]:
            ax.set_ylabel("")

    fig.suptitle(
        rf"$\rho_{{\mathrm{{func}}}}$ by layer for multiple $\varepsilon$"
        rf" (epoch {LAST_EPOCH}, averaged over architectures and seeds)",
        fontsize=11, y=1.03,
    )
    fig.tight_layout()
    savefig(fig, neurips_figpath / "rho_func_layerwise")
    print("wrote rho_func_layerwise")


if __name__ == "__main__":
    main()
