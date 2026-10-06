"""The 2D Composite dataset as the networks see it (App. B, Fig. [fig:composite_dataset]).

Loads the training split through ``get_new_data`` (same generation, split and
MinMax scaling to [-1, 1] as in training) and scatters it by class.
Output: figures/composite_dataset.{pdf,png}
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src_experiment.dataset import get_new_data  # noqa: E402
from src_experiment.paths import neurips_figpath  # noqa: E402
from src_experiment.utils import savefig  # noqa: E402


def plot_composite_dataset():
    train_loader, _ = get_new_data(dataset_name="composite", split_seed=42)
    # The stored tensors, not a batch from the shuffling loader, so the points
    # are drawn in the same order (and overlap the same way) on every run.
    X_train, y_train = train_loader.dataset.tensors
    X, y = X_train.numpy(), y_train.numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="tab10", alpha=0.7, s=15, edgecolors="none")
    ax.set_title("2D Composite Dataset (Scaled)", fontsize=16, pad=15)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal", "box")
    legend = ax.legend(*scatter.legend_elements(), title="Classes",
                       loc="upper right", bbox_to_anchor=(1.15, 1))
    ax.add_artist(legend)
    savefig(fig, neurips_figpath / "composite_dataset")


if __name__ == "__main__":
    plot_composite_dataset()
