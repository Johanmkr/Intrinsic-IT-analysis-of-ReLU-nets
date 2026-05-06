"""
Pedagogical Figure 1: how ReLU networks partition input space and how
activation patterns uniquely identify each linear region.

Trains a small 2→4→4→2 network on the moons dataset, then visualises:
  (a) After layer 1 – 4 hyperplanes create convex regions
  (b) After layers 1+2 – piecewise-linear refinement of the partition
  (c) Network diagram – one region ω identified by its activation pattern π_ω

Output:
    figures/figure1_pedagogy.pdf / .png

Usage:
    uv run python scripts/plot_figure1_pedagogy.py [--retrain]

    Use --retrain to force retraining (otherwise loads saved weights if available)
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.gridspec import GridSpec
from matplotlib.patches import ConnectionPatch
from sklearn.datasets import make_moons
from sklearn.preprocessing import MinMaxScaler

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src_experiment.utils import savefig
from src_experiment.paths import neurips_figpath

REPO = Path(__file__).resolve().parents[1]
FIGURES = REPO / "figures"
FIGURES.mkdir(exist_ok=True)
WEIGHTS_FILE = REPO / ".cache" / "figure1_pedagogy_weights.pt"
WEIGHTS_FILE.parent.mkdir(exist_ok=True)

SEED   = 7
HIDDEN = 4   # neurons per hidden layer

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
X_raw, y = make_moons(n_samples=600, noise=0.12, random_state=SEED)
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X_raw).astype(np.float32)
X_t = torch.tensor(X)
y_t = torch.tensor(y, dtype=torch.long)

# ── Train ─────────────────────────────────────────────────────────────────────
class SmallNet(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        torch.manual_seed(SEED)
        self.l1, self.relu1 = nn.Linear(2, h), nn.ReLU()
        self.l2, self.relu2 = nn.Linear(h, h), nn.ReLU()
        self.out = nn.Linear(h, 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.out(self.relu2(self.l2(self.relu1(self.l1(x)))))


def train_or_load_model(retrain=False):
    """Train model or load pre-trained weights if available."""
    net = SmallNet(HIDDEN)

    if WEIGHTS_FILE.exists() and not retrain:
        print(f"Loading pre-trained weights from {WEIGHTS_FILE}")
        net.load_state_dict(torch.load(WEIGHTS_FILE))
        return net

    print("Training network...")
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(2000):
        opt.zero_grad()
        nn.CrossEntropyLoss()(net(X_t), y_t).backward()
        opt.step()

    with torch.no_grad():
        acc = (net(X_t).argmax(1) == y_t).float().mean().item()
    print(f"Train accuracy: {acc:.3f}")

    torch.save(net.state_dict(), WEIGHTS_FILE)
    print(f"Saved weights to {WEIGHTS_FILE}")

    return net


# Parse command-line arguments
parser = argparse.ArgumentParser(description="Generate pedagogical Figure 1")
parser.add_argument("--retrain", action="store_true", help="Force retraining of the network")
args = parser.parse_args()

net = train_or_load_model(retrain=args.retrain)

# ── Extract weights ───────────────────────────────────────────────────────────
W1    = net.l1.weight.detach().numpy()    # (H, 2)
b1    = net.l1.bias.detach().numpy()
W2    = net.l2.weight.detach().numpy()    # (H, H)
b2    = net.l2.bias.detach().numpy()
W_out = net.out.weight.detach().numpy()
b_out = net.out.bias.detach().numpy()

# ── Grid forward pass ─────────────────────────────────────────────────────────
N   = 700
lim = 1.08
xs  = np.linspace(-lim, lim, N)
XX, YY = np.meshgrid(xs, xs)
pts = np.stack([XX.ravel(), YY.ravel()], axis=1)

pre1   = pts @ W1.T + b1
h1     = np.maximum(pre1, 0.0)
pre2   = h1 @ W2.T + b2
h2     = np.maximum(pre2, 0.0)
logits = h2 @ W_out.T + b_out

pw    = 2 ** np.arange(HIDDEN - 1, -1, -1)
code1 = (pre1 > 0).astype(int) @ pw
code2 = (pre2 > 0).astype(int) @ pw
code_full = code1 * (2 ** HIDDEN) + code2

full_grid = code_full.reshape(N, N)

# ── Pick highlighted region: largest, with non-trivial activation pattern ─────
def decode(code, n):
    return [(code >> (n - 1 - i)) & 1 for i in range(n)]

top = sorted(Counter(code_full.tolist()).items(), key=lambda x: -x[1])
for best_full, _ in top:
    c1 = best_full >> HIDDEN
    c2 = best_full & ((1 << HIDDEN) - 1)
    a1, a2 = decode(c1, HIDDEN), decode(c2, HIDDEN)
    if 0 < sum(a1) < HIDDEN and 0 < sum(a2) < HIDDEN:  # non-trivial
        break

ACT1, ACT2 = a1, a2
print(f"Highlighted region: π¹={ACT1}  π²={ACT2}")

# ── Colour palette ────────────────────────────────────────────────────────────
YELLOW_HL   = np.array([1.00, 0.85, 0.05, 1.0])
HL_EDGE     = "#b8960a"
NODE_ON     = "#27ae60"
NODE_OFF    = "#ecf0f1"
NODE_IN     = "#95a5a6"
CLASS_C     = ["#003cff", "#ff1900"]   # class 0 = blue, class 1 = red

# Soft qualitative palette – 16 distinct pastel colours
_RAW_PALETTE = [
    "#a8ddb5", "#fee8c8", "#b3cde3", "#fbb4ae",
    "#ccebc5", "#decbe4", "#fed9a6", "#e5d8bd",
    "#b2e2e2", "#f2f0f7", "#d4b9da", "#c7e9b4",
    "#fdcc8a", "#bae4bc", "#f1eef6", "#d0d1e6",
]
rng_pal = np.random.RandomState(3)
_RAW_PALETTE = [_RAW_PALETTE[i] for i in rng_pal.permutation(len(_RAW_PALETTE))]

def hex_to_rgba(h, alpha=0.88):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return np.array([r / 255, g / 255, b / 255, alpha])

PALETTE = np.array([hex_to_rgba(c) for c in _RAW_PALETTE])


def region_image(codes_flat, highlight_code):
    unique = np.unique(codes_flat)
    img = np.zeros((N * N, 4))
    for i, c in enumerate(unique):
        mask = codes_flat == c
        if c == highlight_code:
            img[mask] = YELLOW_HL
        else:
            img[mask] = [1, 1, 1, 0]  # White with 0 alpha (transparent)
    return img.reshape(N, N, 4)


# ── Figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 5))
gs  = GridSpec(1, 3, figure=fig, wspace=-0.10,
               left=0.01, right=0.99, top=0.92, bottom=0.10)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

EXTENT = [-lim, lim, -lim, lim]
scatter_kw = dict(s=6, linewidths=0, alpha=0.55, zorder=3)


def plot_data(ax):
    for cls in [0, 1]:
        m = y == cls
        ax.scatter(X[m, 0], X[m, 1], c=CLASS_C[cls], **scatter_kw)


def label_region(ax, mask, label):
    cx, cy = pts[mask].mean(axis=0)
    ax.text(cx, cy, label, ha="center", va="center", fontsize=9.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=YELLOW_HL[:3],
                      alpha=0.95, ec=HL_EDGE, lw=1.6))
    return cx, cy


# ── Panel (a): layer-1 regions ────────────────────────────────────────────────
ax_a.imshow(region_image(code1, c1), origin="lower",
            extent=EXTENT, aspect="equal", interpolation="nearest")

line_cols = plt.cm.Set1(np.linspace(0.0, 0.75, HIDDEN))
for i in range(HIDDEN):
    ax_a.contour(XX, YY, pre1[:, i].reshape(N, N), levels=[0],
                 colors="black", linewidths=1.5, alpha=0.9, zorder=2)

plot_data(ax_a)
mask_a = code1 == c1
cx_a, cy_a = label_region(ax_a, mask_a, r"region $\omega$")

ax_a.set_xlim(-lim, lim); ax_a.set_ylim(-lim, lim)
ax_a.set_xticks([]); ax_a.set_yticks([])
ax_a.set_title("(a)  After layer 1", fontsize=12, pad=6)

# ── Panel (b): full 2-layer partition ─────────────────────────────────────────
ax_b.imshow(region_image(code_full, best_full), origin="lower",
            extent=EXTENT, aspect="equal", interpolation="nearest")

bnd = np.zeros((N, N), bool)
bnd[:-1, :] |= (full_grid[:-1, :] != full_grid[1:, :])
bnd[:, :-1] |= (full_grid[:, :-1] != full_grid[:, 1:])

# Draw boundaries as contours with explicit linewidth
for code_val in np.unique(full_grid):
    ax_b.contour(XX, YY, full_grid, levels=[code_val - 0.5],
                 colors="black", linewidths=1, alpha=0.8, zorder=2)

plot_data(ax_b)
mask_b = code_full == best_full
cx_b, cy_b = label_region(ax_b, mask_b, r"region $\omega$")

ax_b.set_xlim(-lim, lim); ax_b.set_ylim(-lim, lim)
ax_b.set_xticks([]); ax_b.set_yticks([])
ax_b.set_title("(b)  After layer 2", fontsize=12, pad=6)

# Add legend between panels (a) and (b), below titles
c0_h = mpatches.Patch(color=CLASS_C[0], label="Class 0", alpha=0.8)
c1_h = mpatches.Patch(color=CLASS_C[1], label="Class 1", alpha=0.8)
legend_y = 0.97  # Below title, above plot
fig.legend(handles=[c0_h, c1_h], loc="upper center", fontsize=9, framealpha=0.9,
           bbox_to_anchor=(0.35, legend_y), ncol=2, frameon=False)

# ── Panel (c): network diagram ────────────────────────────────────────────────
y_top  = (HIDDEN - 1) * 1.2 + 0.5
y_mid  = (HIDDEN - 1) * 1.2 / 2

ax_c.set_xlim(-0.1, 4.5)
ax_c.set_ylim(-1.4, y_top + 0.6)
ax_c.axis("off")
ax_c.set_title(r"(c)  Region $\omega$ identified by $\pi_\omega$", fontsize=12, pad=6)

R = 0.25

inp_xy = [(0.7, y_mid - 0.6), (0.7, y_mid + 0.6)]
h1_xy  = [(2.0, i * 1.2) for i in range(HIDDEN)]
h2_xy  = [(3.5, i * 1.2) for i in range(HIDDEN)]


def draw_node(ax, pos, fc, label="", fs=10):
    ax.add_patch(plt.Circle(pos, R, color=fc, ec="#2c3e50", lw=1.5, zorder=4))
    ax.text(*pos, label, ha="center", va="center",
            fontsize=fs, fontweight="bold", zorder=5)


for ip in inp_xy:
    for hp in h1_xy:
        ax_c.plot([ip[0] + R, hp[0] - R], [ip[1], hp[1]],
                  color="#bdc3c7", lw=0.7, alpha=0.5, zorder=1)

for j, hp1 in enumerate(h1_xy):
    col = NODE_ON if ACT1[j] else "#d5d8dc"
    lw  = 1.3 if ACT1[j] else 0.5
    for hp2 in h2_xy:
        ax_c.plot([hp1[0] + R, hp2[0] - R], [hp1[1], hp2[1]],
                  color=col, lw=lw, alpha=0.6, zorder=1)

for i, pos in enumerate(inp_xy):
    draw_node(ax_c, pos, NODE_IN, f"$x_{i+1}$", fs=9)

for j, pos in enumerate(h1_xy):
    draw_node(ax_c, pos, NODE_ON if ACT1[j] else NODE_OFF, str(ACT1[j]))

for k, pos in enumerate(h2_xy):
    draw_node(ax_c, pos, NODE_ON if ACT2[k] else NODE_OFF, str(ACT2[k]))

ax_c.text(0.7, y_top, "Input",   ha="center", fontsize=9, color="#444")
ax_c.text(2.0, y_top, "Layer 1", ha="center", fontsize=9, color="#444")
ax_c.text(3.5, y_top, "Layer 2", ha="center", fontsize=9, color="#444")

def fmt(bits):
    return "(" + ", ".join(str(b) for b in bits) + ")"

lbl_kw = dict(ha="center", fontsize=9, color="#111",
              bbox=dict(boxstyle="round,pad=0.25", fc=YELLOW_HL[:3],
                        alpha=0.88, ec=HL_EDGE, lw=1.2))
ax_c.text(2.0, -1.20, r"$\pi^{(1)}_\omega = $" + fmt(ACT1), **lbl_kw)
ax_c.text(3.5, -1.20, r"$\pi^{(2)}_\omega = $" + fmt(ACT2), **lbl_kw)

# ── Save ──────────────────────────────────────────────────────────────────────
# for ext in ("pdf", "png"):
#     out = FIGURES / f"figure1_pedagogy.{ext}"
#     fig.savefig(out, bbox_inches="tight", dpi=200)
#     print(f"Saved {out}")
savefig(fig, neurips_figpath / "pedagogical_figure1")

plt.show()
