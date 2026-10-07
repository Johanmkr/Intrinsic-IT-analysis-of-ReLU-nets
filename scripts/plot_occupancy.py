"""Region-size distribution at the deepest layer (App. F, occupancy).

For each architecture, the fraction of points that lie in a region holding at
most n points, as a function of the region size n (log axis). The curve starts
at the fraction of points alone in their region (the singleton-sample fraction
of tables/occupancy.tex) and reaches 1 at the largest region. Curves that rise
early belong to saturated partitions (occupancy rho near 1).

Held-out points, clean labels, last epoch, deepest layer; the five seeds are
pooled (every seed sees the same N points, so pooling equals averaging the
per-seed curves). Colour encodes width, line style depth.

Inputs:
    results/region_sizes_composite_label_noise.csv.gz
    results/region_sizes_wbc_label_noise.csv.gz
    results/region_sizes_mnist_capacity.csv.gz

Outputs:
    figures/occupancy_cdf.png / .pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src_experiment.paths import neurips_figpath
from src_experiment.results import DEFAULT_PROTOCOL, SUFFIX
from src_experiment.smoke import LAST_EPOCH
from src_experiment.utils import savefig

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"

MNIST_TARGET_DIM = 10
WIDTHS = [3, 5, 7, 9, 15, 25, 50]
# One colour per width, shared by all panels (sequential: wider = darker).
CMAP = plt.get_cmap("viridis")
COLOUR = {w: CMAP(0.88 - 0.88 * i / (len(WIDTHS) - 1)) for i, w in enumerate(WIDTHS)}
STYLE = {3: "-", 5: (0, (4, 1.5))}  # depth -> line style

PANELS = [
    ("composite_label_noise", "Composite"),
    ("wbc_label_noise", "WBC"),
    ("mnist_capacity", f"MNIST, PCA-{MNIST_TARGET_DIM}"),
]


def load_sizes(sweep: str) -> pd.DataFrame:
    df = pd.read_csv(RESULTS / f"region_sizes_{sweep}{SUFFIX}")
    keep = ((df["protocol"] == DEFAULT_PROTOCOL) & (df["labels"] == "true_labels")
            & (df["noise_level"] == 0.0) & (df["epoch"] == LAST_EPOCH))
    if sweep == "mnist_capacity":
        keep &= df["target_dim"] == MNIST_TARGET_DIM
    df = df[keep]
    deepest = df.groupby("arch_str")["layer"].transform("max")
    return df[df["layer"] == deepest]


def point_cdf(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Region sizes n and the fraction of points in regions of size <= n."""
    mass = (g["region_size"] * g["num_regions"]).groupby(g["region_size"]).sum().sort_index()
    return mass.index.to_numpy(), mass.cumsum().to_numpy() / mass.sum()


def arch_key(arch_str: str) -> tuple[int, int]:
    widths = [int(w) for w in arch_str.strip("[]").split(",")]
    return widths[0], len(widths)


def main() -> None:
    fig, axes = plt.subplots(1, len(PANELS), figsize=(7.0, 2.6), sharey=True)
    used: set[tuple[int, int]] = set()

    for ax, (sweep, title) in zip(axes, PANELS):
        df = load_sizes(sweep)
        (n_points,) = df["N"].unique()
        for arch in sorted(df["arch_str"].unique(), key=arch_key):
            width, depth = arch_key(arch)
            x, y = point_cdf(df[df["arch_str"] == arch])
            # Step curve starting from 0 at n = 1.
            x = np.concatenate([[1], x])
            y = np.concatenate([[0], y])
            ax.step(x, y, where="post", color=COLOUR[width], ls=STYLE[depth], lw=1.6)
            used.add((width, depth))
        ax.set_xscale("log")
        ax.set_xlim(0.75, n_points)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{title} ($N$={n_points:,})", fontsize=11)
        ax.set_xlabel(r"Region size $n_\omega$", fontsize=11)
        ax.tick_params(labelsize=10)
        ax.grid(alpha=0.25, lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Fraction of points\nin regions of size $\\leq n_\\omega$", fontsize=11)

    widths = sorted({w for w, _ in used})
    handles = [Line2D([], [], ls="none", label="Width")]
    handles += [Line2D([], [], color=COLOUR[w], lw=2.2, label=str(w)) for w in widths]
    handles += [Line2D([], [], color="0.35", lw=1.6, ls=STYLE[d], label=f"{d} layers")
                for d in sorted({d for _, d in used})]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.97),
               ncol=len(handles), frameon=False, fontsize=9.5,
               handlelength=1.5, columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout(w_pad=0.8)
    savefig(fig, neurips_figpath / "occupancy_cdf")


if __name__ == "__main__":
    main()
