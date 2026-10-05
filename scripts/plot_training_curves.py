"""Training curves for all evaluated models.

Reads the per-epoch curves stored in each HDF5 (``training_results``, written
by step 1) of the clean networks and produces three figures (App. F):
  figures/training_curves_composite.{pdf,png}
  figures/training_curves_wbc.{pdf,png}
  figures/training_curves_mnist.{pdf,png}

Each panel shows mean ± std test accuracy (left y-axis, blue) and test
loss (right y-axis, orange) over seeds.  MNIST panels show one line per
PCA dimensionality with a shared legend.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))

from src_experiment.paths import neurips_figpath  # noqa: E402
from src_experiment.smoke import LAST_EPOCH  # noqa: E402

OUTPUTS = REPO / "outputs"
FIGURES = neurips_figpath

NOISE = 0.0
DIM_CMAP = plt.get_cmap("tab10")

ACC_COLOR = "tab:blue"
LOSS_COLOR = "tab:orange"

# ─── helpers ──────────────────────────────────────────────────────────────────


def _parse_arch(name: str) -> tuple[int, int]:
    vals = ast.literal_eval(name)
    return len(vals), vals[0]


def _read_curves(h5_path: Path) -> dict[str, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        tr = f["training_results"]
        return {
            "test_acc": tr["test_accuracy"][:],
            "test_loss": tr["test_loss"][:],
        }


def _collect(output_dir: Path) -> pd.DataFrame:
    rows = []
    for seed_path in sorted(output_dir.glob("seed_*.h5")):
        seed = int(re.search(r"seed_(\d+)", seed_path.name).group(1))
        rows.append({"seed": seed, **_read_curves(seed_path)})
    return pd.DataFrame(rows)


def _twin_panel(
    ax: plt.Axes,
    epochs: np.ndarray,
    acc_mean: np.ndarray,
    acc_std: np.ndarray,
    loss_mean: np.ndarray,
    loss_std: np.ndarray,
    acc_color: str = ACC_COLOR,
    loss_color: str = LOSS_COLOR,
    acc_label: str | None = None,
    loss_label: str | None = None,
    lw: float = 1.4,
) -> plt.Axes:
    """Plot accuracy on ax (left) and loss on a twin right axis."""
    ax.plot(epochs, acc_mean, color=acc_color, lw=lw, label=acc_label)
    ax.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std,
                    color=acc_color, alpha=0.18, lw=0)
    ax.set_xlim(0, LAST_EPOCH)
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="y", labelcolor=acc_color, labelsize=7)

    ax2 = ax.twinx()
    ax2.plot(epochs, loss_mean, color=loss_color, lw=lw, linestyle="--",
             label=loss_label)
    ax2.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std,
                     color=loss_color, alpha=0.15, lw=0)
    ax2.tick_params(axis="y", labelcolor=loss_color, labelsize=7)
    ax2.set_ylim(bottom=0)
    return ax2


# ─── composite / WBC ──────────────────────────────────────────────────────────


def _load_noisy_dataset(output_root: Path) -> dict:
    data: dict[float, dict[str, pd.DataFrame]] = {}
    for exp_dir in sorted(output_root.iterdir()):
        m = re.match(r"n([\d.]+)_(\[.+\])", exp_dir.name)
        if not m:
            continue
        noise, arch = float(m.group(1)), m.group(2)
        df = _collect(exp_dir)
        if not df.empty:
            data.setdefault(noise, {})[arch] = df
    return data


def _plot_noisy_dataset(
    data: dict[float, dict[str, pd.DataFrame]],
    dataset_label: str,
    out_path: Path,
) -> None:
    arch_data = data.get(NOISE, {})
    if not arch_data:
        print(f"  [warn] no data for noise={NOISE} in {dataset_label}")
        return

    all_archs = sorted(arch_data.keys(), key=_parse_arch)
    depths = sorted({_parse_arch(a)[0] for a in all_archs})
    widths = sorted({_parse_arch(a)[1] for a in all_archs})
    n_rows, n_cols = len(depths), len(widths)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 2.8 * n_rows),
        squeeze=False,
    )
    fig.suptitle(f"{dataset_label} — test accuracy (solid) / loss (dashed)", fontsize=11)

    epochs = np.arange(LAST_EPOCH + 1)

    for arch in all_archs:
        depth, width = _parse_arch(arch)
        row, col = depths.index(depth), widths.index(width)
        ax = axes[row][col]
        ax.set_title(f"[{', '.join([str(width)]*depth)}]", fontsize=8)

        df = arch_data[arch]
        acc = np.stack(df["test_acc"].values)
        loss = np.stack(df["test_loss"].values)
        _twin_panel(ax, epochs, acc.mean(0), acc.std(0), loss.mean(0), loss.std(0))

        ax.grid(alpha=0.2)

    for i, depth in enumerate(depths):
        for j, width in enumerate(widths):
            if f"[{', '.join([str(width)]*depth)}]" not in all_archs:
                axes[i][j].set_visible(False)

    for ax in axes[-1]:
        ax.set_xlabel("epoch", fontsize=8)
    for i in range(n_rows):
        axes[i][0].set_ylabel(f"depth {depths[i]}\ntest acc", fontsize=8,
                               color=ACC_COLOR)

    # Shared dummy legend
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=ACC_COLOR, lw=1.5, label="test accuracy"),
            plt.Line2D([0], [0], color=LOSS_COLOR, lw=1.5, linestyle="--", label="test loss"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.01),
        frameon=False, fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    _savefig(fig, out_path)


# ─── MNIST capacity ───────────────────────────────────────────────────────────


def _load_mnist(output_root: Path) -> dict:
    data: dict[int, dict[str, pd.DataFrame]] = {}
    for exp_dir in sorted(output_root.iterdir()):
        m = re.match(r"(\d+)_dim_(\[.+\])", exp_dir.name)
        if not m:
            continue
        dim, arch = int(m.group(1)), m.group(2)
        df = _collect(exp_dir)
        if not df.empty:
            data.setdefault(dim, {})[arch] = df
    return data


def _plot_mnist(data: dict[int, dict[str, pd.DataFrame]], out_path: Path) -> None:
    all_archs = sorted(
        {arch for dd in data.values() for arch in dd}, key=_parse_arch
    )
    all_dims = sorted(data.keys())
    n_arch = len(all_archs)
    n_cols = min(4, n_arch)
    n_arch_rows = (n_arch + n_cols - 1) // n_cols

    colors = {dim: DIM_CMAP(i / max(len(all_dims) - 1, 1)) for i, dim in enumerate(all_dims)}

    # Two blocks of rows: top = accuracy, bottom = loss
    n_rows = n_arch_rows * 2
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.5 * n_cols, 2.6 * n_rows),
        squeeze=False,
    )
    fig.suptitle("MNIST capacity", fontsize=12, y=1.01)

    epochs = np.arange(LAST_EPOCH + 1)

    for idx, arch in enumerate(all_archs):
        arch_row, col = divmod(idx, n_cols)
        ax_acc = axes[arch_row][col]
        ax_loss = axes[arch_row + n_arch_rows][col]
        depth, width = _parse_arch(arch)
        title = f"[{', '.join([str(width)]*depth)}]"
        ax_acc.set_title(title, fontsize=8)
        ax_loss.set_title(title, fontsize=8)

        for dim in all_dims:
            if arch not in data.get(dim, {}):
                continue
            df = data[dim][arch]
            acc = np.stack(df["test_acc"].values)
            loss = np.stack(df["test_loss"].values)
            c = colors[dim]
            ax_acc.plot(epochs, acc.mean(0), color=c, lw=1.2, label=f"PCA-{dim}")
            ax_acc.fill_between(epochs, acc.mean(0) - acc.std(0),
                                acc.mean(0) + acc.std(0), color=c, alpha=0.12, lw=0)
            ax_loss.plot(epochs, loss.mean(0), color=c, lw=1.2)
            ax_loss.fill_between(epochs, loss.mean(0) - loss.std(0),
                                 loss.mean(0) + loss.std(0), color=c, alpha=0.12, lw=0)

        for ax in (ax_acc, ax_loss):
            ax.set_xlim(0, LAST_EPOCH)
            ax.grid(alpha=0.2)
        ax_acc.set_ylim(0, 1.05)
        ax_loss.set_ylim(bottom=0)

    # Hide unused slots in both blocks
    for idx in range(len(all_archs), n_arch_rows * n_cols):
        arch_row, col = divmod(idx, n_cols)
        axes[arch_row][col].set_visible(False)
        axes[arch_row + n_arch_rows][col].set_visible(False)

    # Row labels and x-labels
    for col in range(n_cols):
        axes[n_arch_rows - 1][col].set_xlabel("epoch", fontsize=8)
        axes[n_rows - 1][col].set_xlabel("epoch", fontsize=8)
    axes[0][0].set_ylabel("test accuracy", fontsize=8)
    axes[n_arch_rows][0].set_ylabel("test loss", fontsize=8)

    # Divider label between the two blocks
    fig.text(0.01, 0.75, "accuracy", va="center", ha="left",
             fontsize=9, fontweight="bold", rotation=90)
    fig.text(0.01, 0.27, "loss", va="center", ha="left",
             fontsize=9, fontweight="bold", rotation=90)

    # Shared legend for PCA dims
    legend_handles = [
        plt.Line2D([0], [0], color=colors[d], lw=1.5, label=f"PCA-{d}")
        for d in all_dims
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(all_dims),
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout(rect=[0.03, 0.04, 1, 1])
    _savefig(fig, out_path)


# ─── I/O ──────────────────────────────────────────────────────────────────────


def _savefig(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=150, bbox_inches="tight")
    print(f"  saved {path.with_suffix('.pdf')}")
    plt.close(fig)


# ─── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)

    print("Loading composite...")
    comp_data = _load_noisy_dataset(OUTPUTS / "composite_label_noise")
    _plot_noisy_dataset(comp_data, "Composite", FIGURES / "training_curves_composite")

    print("Loading WBC...")
    wbc_data = _load_noisy_dataset(OUTPUTS / "wbc_label_noise")
    _plot_noisy_dataset(wbc_data, "WBC", FIGURES / "training_curves_wbc")

    print("Loading MNIST capacity...")
    mnist_data = _load_mnist(OUTPUTS / "mnist_capacity")
    _plot_mnist(mnist_data, FIGURES / "training_curves_mnist")

    print("Done.")


if __name__ == "__main__":
    main()
