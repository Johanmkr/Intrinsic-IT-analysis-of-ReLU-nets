"""Export the data of the project-page demos to site/data/ (see site/README.md).

Reads the trained networks in outputs/ (step 1) and the stored results in
results/, and writes small JSON files that the browser demos compute from:

  points.json      the 2,000 held-out Composite points and labels
  networks.json    weights of the demo networks (all saved epochs or the last)
  reference.json   the pipeline's region counts, quotient sizes and MI for the
                   same networks, used by the JavaScript tests (site/tests/)
  bias_cells.json  ρ, plug-in, Miller–Madow and H(Y) of the 138 network cells

Weights and points are float32 in the pipeline; they are written with 9
significant digits, which converts back to the exact float32 value
(Math.fround in the browser).

Usage:
    uv run python scripts/export_site_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "site" / "data"
SEED = 101
LAST_EPOCH = 150

# (key, sweep, experiment directory, condition, export every saved epoch?)
NETWORKS = [
    ("c555", "composite_label_noise", "n0.0_[5, 5, 5]", "clean", True),
    ("c999", "composite_label_noise", "n0.0_[9, 9, 9]", "clean", True),
    ("c252525", "composite_label_noise", "n0.0_[25, 25, 25]", "clean", True),
    ("p555", "label_permutation", "composite_[5, 5, 5]", "permuted", False),
    ("p252525", "label_permutation", "composite_[25, 25, 25]", "permuted", False),
]
REFERENCE_EPSILONS = (0.0, 0.1, 0.3, 0.5, 1.0, 2.0)


def f32(a: np.ndarray):
    """Nested lists of 9-significant-digit floats (round-trips to float32)."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 0:
        return float(f"{float(a):.9g}")
    return [f32(x) for x in a]


def write(name: str, obj) -> None:
    path = OUT / name
    path.write_text(json.dumps(obj, separators=(",", ":")) + "\n")
    print(f"  {name} ({path.stat().st_size / 1024:.0f} kB)")


def load_network(sweep: str, exp: str, all_epochs: bool) -> tuple[dict, np.ndarray, np.ndarray]:
    h5 = REPO / "outputs" / sweep / exp / f"seed_{SEED}.h5"
    with h5py.File(h5, "r") as f:
        arch = [int(w) for w in f["metadata"].attrs["architecture"]]
        epochs = sorted(int(k.split("_")[1]) for k in f["epochs"])
        weights = {}
        for epoch in (epochs if all_epochs else [LAST_EPOCH]):
            g = f[f"epochs/epoch_{epoch}"]
            weights[str(epoch)] = [{"W": f32(g[f"l{i}.weight"][:]), "b": f32(g[f"l{i}.bias"][:])}
                                   for i in range(1, len(arch) + 2)]
            acc = float(g.attrs["test_accuracy"])
        points, labels = f["points"][:], f["labels"][:]
    return {"architecture": arch, "weights": weights, "test_accuracy": acc}, points, labels


def reference(sweep: str, arch: list[int], condition: str) -> list[dict]:
    df = pd.read_csv(REPO / "results" / f"routing_{sweep}.csv.gz")
    df = df[(df["dataset"] == "composite") & (df["arch_str"] == str(arch)) & (df["seed"] == SEED)
            & (df["protocol"] == "heldout") & (df["labels"] == "true_labels") & (df["noise_level"] == 0.0)]
    df = df[(df["epsilon"] == 0.0) | ((df["epoch"] == LAST_EPOCH) & df["epsilon"].isin(REFERENCE_EPSILONS))]
    cols = ["epoch", "layer", "epsilon", "N", "num_regions", "num_quotient", "H_Y_bits", "accuracy",
            "plug_in_bits", "miller_madow_bits", "plug_in_func_bits", "miller_madow_func_bits"]
    return df[cols].sort_values(["epoch", "layer", "epsilon"]).to_dict("records")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Writing site/data/:")
    networks, refs, points = {}, {}, None
    for key, sweep, exp, condition, all_epochs in NETWORKS:
        net, X, y = load_network(sweep, exp, all_epochs)
        if points is None:
            points = {"X": f32(X), "y": [int(v) for v in y]}
        elif not np.array_equal(np.asarray(points["X"], dtype=np.float32), X):
            raise SystemExit(f"{key}: held-out points differ from the other networks")
        networks[key] = {"label": f"{condition} {exp.split('_', 1)[1]}", "condition": condition,
                         "seed": SEED, **net}
        refs[key] = reference(sweep, net["architecture"], condition)
    write("points.json", {"dataset": "composite", "protocol": "heldout", "N": len(points["y"]),
                          "num_classes": 7, **points})
    write("networks.json", networks)
    write("reference.json", refs)

    cells = pd.read_csv(REPO / "results" / "summary" / "bias_corrections_by_cell.csv")
    write("bias_cells.json", [
        {"dataset": r.dataset, "arch": r.arch_str, "dim": int(r.target_dim), "layer": int(r.layer),
         "N": int(r.N), "rho": round(r.rho, 5), "H_Y": round(r.H_Y_bits, 4),
         "plug_in": round(r.plug_in_mean, 4), "miller_madow": round(r.miller_madow_mean, 4)}
        for r in cells.itertuples()])


if __name__ == "__main__":
    main()
