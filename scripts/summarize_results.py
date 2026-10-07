"""Summary tables and numbers for the camera-ready appendix (step 5).

Reads only results/ and writes results/summary/:

  calibration.csv              Fig. [fig:calibration-scatter]: n and Pearson r per baseline
  bias_corrections_by_cell.csv Exp 1 + Exp 10: every estimator per (dataset, arch, dim, layer),
                               mean and seed-std over seeds, correction = plug-in − estimator
  bias_corrections_by_rho.csv  the same, averaged over cells in ρ ranges (headline numbers)
  heldout_vs_insample.csv      Exp 5: held-out vs in-sample routing MI and ρ (Composite, WBC)
  baseline_sensitivity.csv     Exp 6: Pearson r of plug-in routing MI vs every baseline setting
  occupancy.csv                Exp 7: ρ and singleton fractions per configuration
  label_permutation.csv        Exp 2: clean vs label-permuted networks
  ordering_by_cell.csv         Exp 4: quotient spread over 16 visiting orders, per cell
  ordering_by_epsilon.csv      the same, summarised per ε
  capacity.csv                 Study 2: raw and quotient MI, ρ, ρ_func per (width, d, ε)
  numbers.json                 every number quoted in the text, under a stable key
  SUMMARY.md                   the same numbers, readable, with the selection behind each

Unless stated otherwise: clean networks, last epoch, held-out points with true
labels (the protocol of every figure), means over the 5 seeds, and the raw
(ε-independent) estimates taken from the ε = 0 rows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO))
sys.path.append(str(REPO / "scripts"))

from src_experiment.estimators import ESTIMATORS  # noqa: E402
from src_experiment.results import SUFFIX, load_baselines, load_routing  # noqa: E402
from src_experiment.smoke import LAST_EPOCH, MNIST_DIMS_B  # noqa: E402

RESULTS = REPO / "results"
OUT = RESULTS / "summary"

CW_ARCHS = ["[5, 5, 5]", "[5, 5, 5, 5, 5]", "[9, 9, 9]", "[9, 9, 9, 9, 9]",
            "[25, 25, 25]", "[25, 25, 25, 25, 25]"]
MNIST_STUDY1 = ["[3, 3, 3]", "[5, 5, 5]", "[7, 7, 7]"]          # at PCA-10 (calibration)
MNIST_STUDY2 = ["[7, 7, 7]", "[15, 15, 15]", "[25, 25, 25]", "[50, 50, 50]"]  # capacity sweep
EPS_QUOTIENT = 0.3                                              # the paper's operating point
PERMUTATION_ARCHS = ["[5, 5, 5]", "[25, 25, 25]"]
RHO_RANGES = {"rho<0.05": (0.0, 0.05), "rho>0.3": (0.3, np.inf), "rho>0.9": (0.9, np.inf)}

# Baseline settings of step 3: family, column prefix, submitted in the paper?
BASELINE_SETTINGS = (
    [("binning", f"binning{k}", k == 8) for k in (2, 4, 8, 16, 30)]
    + [("kmeans", f"kmeans{k}", k == "KY") for k in ("KY", "2KY", "4KY", "16", "64", "256")]
    + [("ksg", f"ksg{k}", k == 3) for k in (3, 5, 10)]
)

numbers: dict[str, float | int | str] = {}


def depth(arch_str: str) -> int:
    return arch_str.count(",") + 1


def deepest(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["layer"] == df["arch_str"].map(depth)]


def routing(sweep: str, protocol: str = "heldout", labels: str = "true_labels") -> pd.DataFrame:
    df = load_routing(RESULTS, sweep, protocol=protocol, labels=labels)
    return df[df["epoch"] == LAST_EPOCH]


def clean_cells(protocol: str = "heldout") -> pd.DataFrame:
    """Raw-estimate rows (ε = 0) of the paper's clean configurations, all layers."""
    frames = []
    for sweep in ("composite_label_noise", "wbc_label_noise"):
        df = routing(sweep, protocol)
        frames.append(df[(df["noise_level"] == 0.0) & df["arch_str"].isin(CW_ARCHS)])
    if protocol == "heldout":
        df = routing("mnist_capacity")
        s1 = df["arch_str"].isin(MNIST_STUDY1) & (df["target_dim"] == 10)
        s2 = df["arch_str"].isin(MNIST_STUDY2) & df["target_dim"].isin(MNIST_DIMS_B)
        frames.append(df[s1 | s2])
    df = pd.concat(frames, ignore_index=True)
    return df[df["epsilon"] == 0.0].copy()


def pearson(x, y) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() >= 3 else float("nan")


def write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.6g")
    print(f"  {name}.csv ({len(df)} rows)")


# ---------------------------------------------------------------------------
# Fig. 2: calibration scatter (the figure's own pairing)
# ---------------------------------------------------------------------------
def calibration() -> None:
    import plot_calibration_scatter as fig2

    df = fig2.load_data()
    rows = []
    for col, label in fig2.BASELINES:
        r = pearson(df[fig2.OURS_COL], df[col])
        rows.append({"baseline": col, "n": len(df), "pearson_r": r,
                     **{f"n_{ds}": int((df["dataset"] == ds).sum()) for ds in sorted(df["dataset"].unique())}})
        numbers[f"fig2.r.{col.split('_')[0]}"] = r
    numbers["fig2.n"] = len(df)
    write(pd.DataFrame(rows), "calibration")

    # Label entropy of the evaluated points (held-out) and of all points (in-sample)
    for protocol in ("heldout", "insample"):
        cells = clean_cells(protocol)
        for ds, h in cells.groupby("dataset")["H_Y_bits"].mean().items():
            numbers[f"hy.{ds}.{protocol}"] = float(h)

    # Range of the plotted values, and KSG at the H(Y) ceiling on Composite
    cols = [fig2.OURS_COL] + [c for c, _ in fig2.BASELINES]
    numbers["fig2.max_bits"] = float(df[cols].max().max())
    comp = df[df["dataset"] == "composite"]
    h = numbers["hy.composite.heldout"]
    numbers["fig2.composite.n"] = len(comp)
    numbers["fig2.composite.ksg_within_0.02_of_H_Y"] = int(((h - comp["ksg3_bits"]).abs() < 0.02).sum())
    numbers["fig2.composite.ksg_median"] = float(comp["ksg3_bits"].median())
    numbers["fig2.composite.routing_min"] = float(comp[fig2.OURS_COL].min())
    numbers["fig2.composite.routing_max"] = float(comp[fig2.OURS_COL].max())


# ---------------------------------------------------------------------------
# Exp 1 + Exp 10: bias corrections and seed variance
# ---------------------------------------------------------------------------
def bias_corrections() -> pd.DataFrame:
    df = clean_cells()
    keys = ["dataset", "arch_str", "target_dim", "layer"]
    agg = {"N": ("N", "first"), "n_seeds": ("seed", "nunique"), "H_Y_bits": ("H_Y_bits", "mean"),
           "rho": ("rho", "mean"), "num_regions": ("num_regions", "mean"),
           "singleton_region_frac": ("singleton_region_frac", "mean"),
           "singleton_sample_frac": ("singleton_sample_frac", "mean")}
    for est in ESTIMATORS:
        agg[f"{est}_mean"] = (f"{est}_bits", "mean")
        agg[f"{est}_std"] = (f"{est}_bits", "std")
    cells = df.groupby(keys).agg(**agg).reset_index()
    for est in ESTIMATORS[1:]:
        cells[f"{est}_correction"] = cells["plug_in_mean"] - cells[f"{est}_mean"]
    mnist = cells["dataset"] == "mnist"
    in1 = cells["arch_str"].isin(MNIST_STUDY1) & (cells["target_dim"] == 10)
    in2 = cells["arch_str"].isin(MNIST_STUDY2)
    cells.insert(1, "study", np.where(~mnist, "calibration",
                                      np.where(in1 & in2, "calibration+capacity",
                                               np.where(in1, "calibration", "capacity"))))
    write(cells, "bias_corrections_by_cell")

    rows = []
    for rng, (lo, hi) in RHO_RANGES.items():
        sub = cells[(cells["rho"] > lo) & (cells["rho"] <= hi)] if lo > 0 else cells[cells["rho"] < hi]
        for est in ESTIMATORS:
            rows.append({
                "rho_range": rng, "estimator": est, "n_cells": len(sub),
                "mean_value": sub[f"{est}_mean"].mean(),
                "mean_correction": (sub["plug_in_mean"] - sub[f"{est}_mean"]).mean(),
                "mean_seed_std": sub[f"{est}_std"].mean(),
                "mean_gap_to_H_Y": (sub["H_Y_bits"] - sub[f"{est}_mean"]).mean(),
                "nan_cells": int(sub[f"{est}_mean"].isna().sum()),
            })
    by_rho = pd.DataFrame(rows)
    write(by_rho, "bias_corrections_by_rho")

    numbers["mm.n_cells"] = len(cells)
    numbers["mm.max_correction"] = float(cells["miller_madow_correction"].max())
    numbers["mm.min_value"] = float(cells["miller_madow_mean"].min())
    for rng in ("rho<0.05", "rho>0.3"):
        r = by_rho[by_rho["rho_range"] == rng].set_index("estimator")
        numbers[f"mm.{rng}.n_cells"] = int(r.loc["plug_in", "n_cells"])
        numbers[f"mm.{rng}.correction"] = r.loc["miller_madow", "mean_correction"]
        numbers[f"mm.{rng}.plug_in_std"] = r.loc["plug_in", "mean_seed_std"]
        numbers[f"mm.{rng}.mm_std"] = r.loc["miller_madow", "mean_seed_std"]
    r = by_rho[by_rho["rho_range"] == "rho>0.9"].set_index("estimator")
    numbers["beyond_mm.rho>0.9.n_cells"] = int(r.loc["plug_in", "n_cells"])
    for est in ESTIMATORS:
        numbers[f"beyond_mm.rho>0.9.gap_to_H_Y.{est}"] = r.loc[est, "mean_gap_to_H_Y"]
        numbers[f"beyond_mm.rho>0.9.correction.{est}"] = r.loc[est, "mean_correction"]
    numbers["beyond_mm.ansb_nan_cells"] = int(cells["ansb_mean"].isna().sum())
    # ANSB needs N/K << 1 (strong undersampling); here N/K = 1/ρ ≥ 1 always.
    numbers["beyond_mm.min_N_over_K"] = float((1 / cells["rho"]).min())
    return cells


# ---------------------------------------------------------------------------
# Exp 5: held-out vs in-sample (Composite, WBC)
# ---------------------------------------------------------------------------
def heldout_vs_insample() -> None:
    keys = ["dataset", "arch_str", "layer"]
    cols = ["N", "rho", "plug_in_bits", "miller_madow_bits"]
    h = clean_cells("heldout")
    h = h[h["dataset"] != "mnist"].groupby(keys)[cols].mean()
    i = clean_cells("insample").groupby(keys)[cols].mean()
    df = h.join(i, lsuffix="_heldout", rsuffix="_insample", how="inner").reset_index()
    df["delta_plug_in"] = df["plug_in_bits_insample"] - df["plug_in_bits_heldout"]
    df["delta_miller_madow"] = df["miller_madow_bits_insample"] - df["miller_madow_bits_heldout"]
    write(df, "heldout_vs_insample")

    last = deepest(df)
    for ds, g in df.groupby("dataset"):
        numbers[f"heldout.{ds}.N_heldout"] = int(g["N_heldout"].iloc[0])
        numbers[f"heldout.{ds}.N_insample"] = int(g["N_insample"].iloc[0])
        numbers[f"heldout.{ds}.max_abs_delta_plug_in.deepest"] = float(last[last.dataset == ds]["delta_plug_in"].abs().max())
        numbers[f"heldout.{ds}.max_abs_delta_plug_in.all_layers"] = float(g["delta_plug_in"].abs().max())
        numbers[f"heldout.{ds}.max_abs_delta_mm.deepest"] = float(last[last.dataset == ds]["delta_miller_madow"].abs().max())
    for arch in ("[25, 25, 25]", "[25, 25, 25, 25, 25]"):
        row = last[(last["dataset"] == "wbc") & (last["arch_str"] == arch)]
        for p in ("insample", "heldout"):
            numbers[f"wbc_rho.{arch}.{p}"] = float(row[f"rho_{p}"].iloc[0]) if len(row) else float("nan")


# ---------------------------------------------------------------------------
# Exp 6: baseline hyperparameter sensitivity
# ---------------------------------------------------------------------------
def paired_baselines() -> pd.DataFrame:
    """Plug-in routing MI and every baseline on the same network, layer and points
    (deepest layer; MNIST: the calibration architectures at PCA-10), as Fig. 2."""
    frames = []
    for sweep, archs, dim in (("composite_label_noise", CW_ARCHS, None),
                              ("wbc_label_noise", CW_ARCHS, None),
                              ("mnist_capacity", MNIST_STUDY1, 10)):
        r = routing(sweep)
        r = deepest(r[(r["epsilon"] == 0.0) & (r["noise_level"] == 0.0) & r["arch_str"].isin(archs)])
        b = load_baselines(RESULTS, sweep)
        b = deepest(b[(b["epoch"] == LAST_EPOCH) & (b["noise_level"] == 0.0) & b["arch_str"].isin(archs)])
        if dim is not None:
            r, b = r[r["target_dim"] == dim], b[b["target_dim"] == dim]
        b = b.drop(columns=[c for c in ("N", "network_id") if c in b.columns])
        frames.append(r[["dataset", "arch_str", "target_dim", "seed", "layer", "plug_in_bits"]].merge(
            b, on=["dataset", "arch_str", "target_dim", "seed", "layer"], validate="one_to_one"))
    return pd.concat(frames, ignore_index=True)


def baseline_sensitivity() -> None:
    df = paired_baselines()
    groups = {"composite+wbc": df["dataset"] != "mnist", "mnist": df["dataset"] == "mnist",
              "all": df["dataset"].notna()}
    rows = []
    for family, prefix, submitted in BASELINE_SETTINGS:
        variants = ["bits"] if family == "ksg" else ["miller_madow_bits", "plug_in_bits"]
        for variant in variants:
            col = f"{prefix}_{variant}"
            for group, mask in groups.items():
                rows.append({"group": group, "family": family, "setting": prefix,
                             "estimator": variant.replace("_bits", "") if family != "ksg" else "ksg (clipped at 0)",
                             "submitted": submitted, "n": int(mask.sum()),
                             "pearson_r": pearson(df.loc[mask, "plug_in_bits"], df.loc[mask, col])})
    out = pd.DataFrame(rows)
    write(out, "baseline_sensitivity")

    # The rebuttal's grid: binning/k-means Miller–Madow corrected, KSG clipped.
    main = out[out["estimator"].isin(["miller_madow", "ksg (clipped at 0)"])]
    for group in ("composite+wbc", "mnist"):
        g = main[main["group"] == group]
        numbers[f"sensitivity.{group}.n"] = int(g["n"].iloc[0])
        numbers[f"sensitivity.{group}.r_min"] = float(g["pearson_r"].min())
        numbers[f"sensitivity.{group}.r_max"] = float(g["pearson_r"].max())
        neg = g[g["pearson_r"] < 0]
        numbers[f"sensitivity.{group}.negative_r"] = ", ".join(
            f"{s} (r={r:.3f})" for s, r in zip(neg["setting"], neg["pearson_r"])) or "none"
        b = g[g["family"] == "binning"]
        numbers[f"sensitivity.{group}.best_binning"] = str(b.loc[b["pearson_r"].idxmax(), "setting"])


# ---------------------------------------------------------------------------
# Exp 7: occupancy
# ---------------------------------------------------------------------------
def occupancy(cells: pd.DataFrame) -> None:
    keys = ["dataset", "arch_str", "target_dim", "layer"]
    cols = ["N", "num_regions", "rho", "singleton_region_frac", "singleton_sample_frac"]
    frames = []
    for protocol in ("heldout", "insample"):
        d = deepest(clean_cells(protocol))
        g = d.groupby(keys)[cols].mean()
        g["rho_std"] = d.groupby(keys)["rho"].std()
        g.insert(0, "protocol", protocol)
        frames.append(g.reset_index())
    df = pd.concat(frames, ignore_index=True)
    write(df, "occupancy")
    h = df[df["protocol"] == "heldout"]
    for ds, g in h.groupby("dataset"):
        numbers[f"occupancy.{ds}.configs"] = len(g)
        numbers[f"occupancy.{ds}.rho>0.3"] = int((g["rho"] > 0.3).sum())
        numbers[f"occupancy.{ds}.rho_max"] = float(g["rho"].max())


# ---------------------------------------------------------------------------
# Exp 2: label permutation
# ---------------------------------------------------------------------------
def label_permutation() -> None:
    cols = ["N", "H_Y_bits", "accuracy", "rho", "plug_in_bits", "miller_madow_bits",
            "plug_in_func_bits", "miller_madow_func_bits", "rho_func"]
    frames = []
    for condition, sweeps, jobs in (
        ("clean", ("composite_label_noise", "wbc_label_noise"),
         (("heldout", "true_labels"), ("insample", "true_labels"))),
        ("permuted", ("label_permutation",),
         (("heldout", "true_labels"), ("insample", "true_labels"),
          ("train", "true_labels"), ("train", "train_labels"))),
    ):
        for sweep in sweeps:
            for protocol, labels in jobs:
                d = routing(sweep, protocol, labels)
                d = deepest(d[np.isclose(d["epsilon"], EPS_QUOTIENT) & (d["noise_level"] == 0.0)
                              & d["arch_str"].isin(PERMUTATION_ARCHS)])
                g = d.groupby(["dataset", "arch_str"])[cols].mean()
                g["n_seeds"] = d.groupby(["dataset", "arch_str"])["seed"].nunique()
                g.insert(0, "labels", labels)
                g.insert(0, "protocol", protocol)
                g.insert(0, "condition", condition)
                frames.append(g.reset_index())
    df = pd.concat(frames, ignore_index=True)
    df.insert(df.columns.get_loc("plug_in_func_bits"), "epsilon", EPS_QUOTIENT)
    write(df, "label_permutation")
    h = df[df["protocol"] == "heldout"].set_index(["dataset", "arch_str", "condition"])
    mem = df[(df["protocol"] == "train") & (df["labels"] == "train_labels")].set_index(["dataset", "arch_str"])
    for (ds, arch, cond), r in h.iterrows():
        k = f"permutation.{ds}.{arch}.{cond}"
        numbers[f"{k}.raw"] = r["plug_in_bits"]
        numbers[f"{k}.quotient"] = r["plug_in_func_bits"]
        numbers[f"{k}.test_accuracy"] = r["accuracy"]
        numbers[f"{k}.H_Y"] = r["H_Y_bits"]
    for (ds, arch), r in mem.iterrows():
        numbers[f"permutation.{ds}.{arch}.permuted.train_label_accuracy"] = r["accuracy"]


# ---------------------------------------------------------------------------
# Exp 4: functional-quotient ordering sensitivity
# ---------------------------------------------------------------------------
def ordering() -> None:
    df = pd.concat([pd.read_csv(RESULTS / f"ordering_{s}{SUFFIX}")
                    for s in ("composite_label_noise", "wbc_label_noise", "mnist_capacity")],
                   ignore_index=True)
    keys = ["dataset", "arch_str", "target_dim", "seed", "layer", "epsilon"]
    g = df.groupby(keys)
    cells = g.agg(num_regions=("num_regions", "first"), n_orders=("order", "nunique"),
                  num_quotient_min=("num_quotient", "min"),
                  num_quotient_max=("num_quotient", "max")).reset_index()
    cells["num_quotient_spread"] = cells["num_quotient_max"] - cells["num_quotient_min"]
    cells["relative_spread"] = cells["num_quotient_spread"] / cells["num_regions"]
    for col in ("plug_in_func_bits", "miller_madow_func_bits"):
        cells[f"{col}_spread"] = (g[col].max() - g[col].min()).to_numpy()
    write(cells, "ordering_by_cell")

    rows = []
    for (eps, scope), sub in [((e, "all layers"), s) for e, s in cells.groupby("epsilon")] + \
                             [((e, "deepest layer"), deepest(s)) for e, s in cells.groupby("epsilon")]:
        top = sub.loc[sub["num_quotient_spread"].idxmax()]
        rows.append({"epsilon": eps, "scope": scope, "n_cells": len(sub),
                     "regions_min": int(sub["num_regions"].min()), "regions_max": int(sub["num_regions"].max()),
                     "spread_median": sub["num_quotient_spread"].median(),
                     "spread_p95": sub["num_quotient_spread"].quantile(0.95),
                     "spread_max": int(top["num_quotient_spread"]), "regions_at_spread_max": int(top["num_regions"]),
                     "relative_spread_max": sub["relative_spread"].max(),
                     "plug_in_func_spread_max": sub["plug_in_func_bits_spread"].max(),
                     "miller_madow_func_spread_max": sub["miller_madow_func_bits_spread"].max()})
    summ = pd.DataFrame(rows)
    write(summ, "ordering_by_epsilon")
    for scope, tag in (("all layers", "all_layers"), ("deepest layer", "deepest")):
        r = summ[(np.isclose(summ["epsilon"], EPS_QUOTIENT)) & (summ["scope"] == scope)].iloc[0]
        for k in ("n_cells", "regions_min", "regions_max", "spread_median", "spread_max", "regions_at_spread_max",
                  "relative_spread_max", "miller_madow_func_spread_max", "plug_in_func_spread_max"):
            numbers[f"ordering.eps0.3.{tag}.{k}"] = r[k].item() if hasattr(r[k], "item") else r[k]
    numbers["ordering.n_random_orders"] = int(df["order"].max())
    numbers["ordering.all_eps.miller_madow_func_spread_max"] = float(summ["miller_madow_func_spread_max"].max())
    numbers["ordering.all_eps.relative_spread_max"] = float(summ["relative_spread_max"].max())


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Study 2 and Figs. [fig:mnist-capacity], [fig:rho-func-layerwise]: the ε window
# ---------------------------------------------------------------------------
def capacity() -> None:
    df = deepest(routing("mnist_capacity"))
    df = df[df["arch_str"].isin(MNIST_STUDY2) & df["target_dim"].isin(MNIST_DIMS_B)]
    agg = (df.groupby(["arch_str", "target_dim", "epsilon"])
             .agg(H_Y_bits=("H_Y_bits", "mean"), rho=("rho", "mean"), rho_func=("rho_func", "mean"),
                  raw_mean=("plug_in_bits", "mean"), raw_std=("plug_in_bits", "std"),
                  func_mean=("plug_in_func_bits", "mean"), func_std=("plug_in_func_bits", "std"))
             .reset_index())
    agg.insert(1, "width", agg["arch_str"].map(lambda a: int(a.strip("[]").split(",")[0])))
    agg["func_below_raw"] = agg["raw_mean"] - agg["func_mean"]
    agg["func_gap_to_H_Y"] = agg["H_Y_bits"] - agg["func_mean"]
    write(agg, "capacity")

    h = float(agg["H_Y_bits"].mean())
    numbers["capacity.H_Y"] = h
    raw = agg[agg["epsilon"] == 0.0]
    for w, sub in raw.groupby("width"):
        small = sub[sub["target_dim"] <= 5]
        numbers[f"capacity.w{w}.d<=5.raw_max"] = float(small["raw_mean"].max())
        numbers[f"capacity.w{w}.d<=5.gap_min"] = float(h - small["raw_mean"].max())
    for eps in (0.1, 0.2):
        sub = agg[np.isclose(agg["epsilon"], eps)]
        numbers[f"capacity.eps{eps}.below_raw_max"] = float(sub["func_below_raw"].max())
        numbers[f"capacity.eps{eps}.rho_func_min"] = float(sub["rho_func"].min())
    for eps in (0.3, 0.5):
        sub = agg[np.isclose(agg["epsilon"], eps) & (agg["target_dim"] >= 10)]
        for w, s in sub.groupby("width"):
            numbers[f"capacity.eps{eps}.d>=10.w{w}.gap_min"] = float(s["func_gap_to_H_Y"].min())
            numbers[f"capacity.eps{eps}.d>=10.w{w}.gap_max"] = float(s["func_gap_to_H_Y"].max())
    d2 = agg[(agg["target_dim"] == 2)].pivot_table(index="width", columns="epsilon", values="func_mean")
    ratio = d2[2.0] / d2[0.0]
    numbers["capacity.eps2.d2.func_over_raw_min"] = float(ratio.min())
    numbers["capacity.eps2.d2.func_over_raw_max"] = float(ratio.max())
    win = agg.pivot_table(index=["width", "target_dim"], columns="epsilon", values="func_mean")
    numbers["capacity.window.max_abs_diff_eps0.3_eps0.5"] = float((win[0.3] - win[0.5]).abs().max())

    # Per-network ρ_func curves over depth (the networks of [fig:rho-func-layerwise])
    five = ["[5, 5, 5, 5, 5]", "[9, 9, 9, 9, 9]", "[25, 25, 25, 25, 25]"]
    frames = [routing(s) for s in ("composite_label_noise", "wbc_label_noise")]
    frames = [f[(f["noise_level"] == 0.0) & f["arch_str"].isin(five)] for f in frames]
    m = routing("mnist_capacity")
    frames.append(m[(m["target_dim"] == 10) & m["arch_str"].isin(["[5, 5, 5]", "[7, 7, 7]"])])
    lw = pd.concat(frames, ignore_index=True)
    lw = lw[lw["epsilon"] > 0].sort_values("layer")
    rise = lw.groupby(["dataset", "arch_str", "seed", "epsilon"])["rho_func"].agg(lambda r: np.diff(r).max())
    numbers["rho_func_layerwise.n_networks"] = int(lw.groupby(["dataset", "arch_str", "seed"]).ngroups)
    numbers["rho_func_layerwise.n_curves"] = len(rise)
    numbers["rho_func_layerwise.n_curves_rising"] = int((rise > 0).sum())
    numbers["rho_func_layerwise.max_rise"] = float(rise.max())
    numbers["rho_func_layerwise.layer1_all_one"] = bool(np.allclose(lw.loc[lw["layer"] == 1, "rho_func"], 1.0))


def summary_markdown() -> str:
    prov = json.loads((RESULTS / "provenance.json").read_text())
    commit = prov.get("step2_estimate", {}).get("git_commit", "unknown")
    lines = [
        "# Summary of the results (generated by scripts/summarize_results.py; do not edit)",
        "",
        f"Results from commit `{commit}`. Unless stated otherwise: clean networks, last epoch, "
        "held-out points with true labels, mean over seeds, raw estimates from the ε = 0 rows. "
        "Tables are the CSVs next to this file; every number below is also in `numbers.json`.",
        "",
    ]
    sections = {
        "fig2": "Fig. 2 calibration (plug-in routing MI vs baseline; same network, layer and points)",
        "hy": "Label entropy H(Y) in bits (held-out points; insample = all points)",
        "mm": "Miller–Madow correction and seed variance (Exp 1; `bias_corrections_by_*.csv`)",
        "beyond_mm": "Other bias corrections at ρ > 0.9 (Exp 10; gap to H(Y) of the corrected value)",
        "heldout": "Held-out vs in-sample (Exp 5; `heldout_vs_insample.csv`)",
        "wbc_rho": "WBC ρ at the deepest layer (Exp 5)",
        "sensitivity": "Baseline hyperparameter sensitivity (Exp 6; MM-corrected binning/k-means, KSG; `baseline_sensitivity.csv`)",
        "occupancy": "Occupancy, deepest layer (Exp 7; `occupancy.csv`)",
        "permutation": f"Label permutation (Exp 2; deepest layer, quotient at ε = {EPS_QUOTIENT}; `label_permutation.csv`)",
        "capacity": "Study 2: MNIST capacity sweep, last layer (`capacity.csv`; raw = ε 0, func = plug-in quotient)",
        "rho_func_layerwise": "Per-network ρ_func over depth (networks of Fig. 6, ε > 0)",
        "ordering": f"Quotient ordering sensitivity (Exp 4; first-encounter order + random orders; `ordering_by_*.csv`)",
    }
    for prefix, title in sections.items():
        lines += [f"## {title}", ""]
        for k, v in numbers.items():
            if k.split(".")[0] == prefix:
                val = f"{v:.4g}" if isinstance(v, float) else str(v)
                lines.append(f"- `{k}` = {val}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("Writing results/summary/:")
    calibration()
    cells = bias_corrections()
    heldout_vs_insample()
    baseline_sensitivity()
    occupancy(cells)
    label_permutation()
    ordering()
    capacity()
    clean = {k: (v.item() if hasattr(v, "item") else v) for k, v in numbers.items()}
    (OUT / "numbers.json").write_text(json.dumps(clean, indent=2, allow_nan=True) + "\n")
    (OUT / "SUMMARY.md").write_text(summary_markdown())
    print(f"  numbers.json ({len(clean)} numbers), SUMMARY.md")


if __name__ == "__main__":
    main()
