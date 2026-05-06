"""
Phase A.1 sweep runner: evaluate the MI baselines (binning, k-means, KSG,
InfoNCE, MINE-f) across **all saved epochs and all hidden layers** of every
label-noise HDF5.  One CSV row per (epoch, layer) cell; per-HDF5 CSVs land at
``mi_baselines_seed_<seed>.csv`` next to the HDF5.

Resumable: already-computed (epoch, layer) cells are skipped automatically.
New cells are appended and flushed to disk after each cell so a crash only
loses the cell in progress.  ``--aggregate`` concatenates all per-HDF5 CSVs
into ``results/mi_baselines.csv``.

Examples
--------
Smoke run — one HDF5, cheap baselines only::

    uv run python scripts/run_mi_baselines.py --datasets composite --limit 1 \\
        --skip-mine --skip-infonce

Sweep all epochs/layers, skip neural estimators (fast)::

    uv run python scripts/run_mi_baselines.py --datasets both \\
        --skip-mine --skip-infonce

Target a single layer across all epochs::

    uv run python scripts/run_mi_baselines.py --datasets composite \\
        --layer-filter 3 --skip-mine --skip-infonce

Aggregate into results/mi_baselines.csv::

    uv run python scripts/run_mi_baselines.py --aggregate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from run_label_noise_estimator import DATASETS, discover_jobs  # noqa: E402
from src_experiment.baselines.activations import load_activations_dispatch  # noqa: E402
from src_experiment.baselines.mi_baselines import (  # noqa: E402
    InfoNCEEstimator,
    MINEEstimator,
    binning_mi,
    kmeans_mi,
    ksg_mi,
)
from src_experiment.probe_loader import (  # noqa: E402
    ProbeBundle,
    make_composite_probe,
    make_mnist_full_lenet_probe,
    make_wbc_probe,
)


# Sweep grids (planning §A.1.1).
BINNING_BINS: Tuple[int, ...] = (2, 4, 8, 16, 30)
KSG_KS: Tuple[int, ...] = (3, 5, 10)
KMEANS_FIXED_K: Tuple[int, ...] = (16, 64, 256)
MINE_SEEDS_DEFAULT = 5
INFONCE_SEEDS_DEFAULT = 3
SUBSAMPLE_N_DEFAULT = 5_000


# ---------------------------------------------------------------------------
# Probe resolution
# ---------------------------------------------------------------------------
def _resolve_probe(dataset: str) -> ProbeBundle:
    if dataset == "composite":
        return make_composite_probe()  # N=20k, cached
    if dataset == "wbc":
        return make_wbc_probe(mode="full")  # N=569
    if dataset == "mnist_full_lenet":
        return make_mnist_full_lenet_probe()  # N=10k, cached
    raise ValueError(f"unknown dataset {dataset!r}")


# ---------------------------------------------------------------------------
# HDF5 introspection
# ---------------------------------------------------------------------------
def _h5_summary(h5_path: Path) -> dict:
    """Read architecture, all epochs, all layer indices, num_classes from one HDF5.

    Handles both MLP HDF5s (``architecture`` attr is the hidden-width list)
    and LeNet5 HDF5s (``arch_type=lenet5``; hidden layers are
    n_conv conv-ReLU steps + n_fc_hidden FC-ReLU steps).
    """
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f["metadata"].attrs)
        all_epochs = sorted(
            int(k.split("_")[1]) for k in f["epochs"].keys() if k.startswith("epoch_")
        )
        arch_type = str(attrs.get("arch_type", "mlp"))
        if arch_type == "lenet5":
            conv_channels = list(attrs.get("conv_channels", []))
            fc_widths = list(attrs.get("fc_widths", []))
            arch = list(conv_channels) + list(fc_widths)
            n_hidden = len(conv_channels) + len(fc_widths)
        else:
            arch = list(attrs.get("architecture", []))
            n_hidden = len(arch)
    return {
        "architecture": arch,
        "arch_type": arch_type,
        "all_epochs": all_epochs,
        "all_layers": list(range(1, n_hidden + 1)),  # 1-indexed hidden ReLU layers
        "num_classes": int(attrs.get("inferred_num_classes", 0)),
        "network_id": str(attrs.get("experiment_name", h5_path.stem)),
    }


def _crossref_existing(h5_path: Path, epoch: int, layer: int) -> dict:
    """Pick the routing-information estimator outputs from the existing
    ``new_estimator_<seed>.csv`` if one exists. Returns empty dict if not.

    Surfaces both the **plug-in** (``bits_ours_plugin`` / ``bits_ours_func_plugin``)
    and the **Miller-Madow corrected** (``bits_ours_raw`` /
    ``bits_ours_func``) variants, so the baseline figure can compare both.
    Joins on (epoch, layer); for the functional column we take the smallest
    non-zero ε row.
    """
    csv = h5_path.with_name(f"new_estimator_{h5_path.stem}.csv")
    if not csv.exists():
        return {}
    df = pd.read_csv(csv)
    sub = df[(df["epoch"] == epoch) & (df["layer"] == layer)]
    if sub.empty:
        return {}
    raw = sub.iloc[0]
    out = {
        "bits_ours_plugin": float(raw["plug_in_bits"]),
        "bits_ours_raw": float(raw["miller_madow_bits"]),
        "rho": float(raw["rho"]),
    }
    func_rows = sub[sub["epsilon"] > 0]
    if not func_rows.empty:
        f = func_rows.sort_values("epsilon").iloc[0]
        out["bits_ours_func_plugin"] = float(f["plug_in_func_bits"])
        out["bits_ours_func"] = float(f["miller_madow_func_bits"])
        out["rho_func"] = float(f["rho_func"])
        out["epsilon_for_ours_func"] = float(f["epsilon"])
    return out


# ---------------------------------------------------------------------------
# Per-HDF5 evaluation
# ---------------------------------------------------------------------------
def _evaluate_cell(
    h5_path: Path,
    epoch: int,
    layer: int,
    X: np.ndarray,
    Y: np.ndarray,
    N_orig: int,
    N_eff: int,
    meta: dict,
    args: argparse.Namespace,
) -> dict:
    """Evaluate all baseline estimators at a single (epoch, layer) cell."""
    num_classes = meta["num_classes"]
    T = load_activations_dispatch(h5_path, epoch, layer, X, kind="pre")
    d_T = T.shape[1]

    row: dict = {
        "network_id": meta["network_id"],
        "dataset": meta["dataset"],
        "noise_level": meta["noise"],
        "arch_str": meta["arch"],
        "seed": int(meta["seed"]),
        "epoch": int(epoch),
        "layer": int(layer),
        "d_T": int(d_T),
        "num_classes": int(num_classes),
        "probe_N": int(N_orig),
        "N_used": int(N_eff),
    }

    # Binning sweep
    t0 = time.perf_counter()
    for nb in BINNING_BINS:
        out = binning_mi(T, Y, n_bins=nb, num_classes=num_classes)
        row[f"bits_binning_{nb}"] = out["bits"]
        row[f"R_binning_{nb}"] = out["num_regions"]
    row["wall_seconds_binning"] = time.perf_counter() - t0

    # K-means sweep — class-relative settings + fixed grid
    kmeans_settings: List[Tuple[str, int]] = [
        ("KY", num_classes),
        ("2KY", 2 * num_classes),
        ("4KY", 4 * num_classes),
    ]
    seen = {num_classes, 2 * num_classes, 4 * num_classes}
    for K in KMEANS_FIXED_K:
        if K not in seen:
            kmeans_settings.append((str(K), K))
            seen.add(K)
    t0 = time.perf_counter()
    for label, K in kmeans_settings:
        out = kmeans_mi(T, Y, K=K, seed=int(meta["seed"]), num_classes=num_classes)
        row[f"bits_kmeans_K{label}"] = out["bits"]
        row[f"R_kmeans_K{label}"] = out["num_regions"]
    row["wall_seconds_kmeans"] = time.perf_counter() - t0

    # KSG sweep
    t0 = time.perf_counter()
    for k in KSG_KS:
        out = ksg_mi(T, Y, k=k)
        row[f"bits_ksg_k{k}"] = out["bits"]
    row["wall_seconds_ksg"] = time.perf_counter() - t0

    # InfoNCE
    if not args.skip_infonce:
        t0 = time.perf_counter()
        nce_bits = []
        nce = InfoNCEEstimator(
            n_iter=args.infonce_iter, batch=args.batch, device=args.device
        )
        for s in range(args.infonce_seeds):
            nce_bits.append(nce.estimate(T, Y, num_classes, seed=s)["bits"])
        row["bits_infonce_mean"] = float(np.mean(nce_bits))
        row["bits_infonce_std"] = float(np.std(nce_bits))
        row["wall_seconds_infonce"] = time.perf_counter() - t0
        row["infonce_seeds"] = args.infonce_seeds

    # MINE
    if not args.skip_mine:
        t0 = time.perf_counter()
        mine_bits = []
        mine = MINEEstimator(
            n_iter=args.mine_iter,
            batch=args.batch,
            device=args.device,
        )
        for s in range(args.mine_seeds):
            mine_bits.append(mine.estimate(T, Y, num_classes, seed=s)["bits"])
        row["bits_mine_mean"] = float(np.mean(mine_bits))
        row["bits_mine_std"] = float(np.std(mine_bits))
        row["wall_seconds_mine"] = time.perf_counter() - t0
        row["mine_seeds"] = args.mine_seeds

    # Cross-ref to existing new-estimator CSV (no-op if missing).
    row.update(_crossref_existing(h5_path, epoch, layer))

    return row


def evaluate_one(
    job: dict,
    probe: ProbeBundle,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Sweep all (epoch, layer) cells for one HDF5; skip already-computed pairs."""
    h5_path: Path = job["h5"]
    info = _h5_summary(h5_path)
    num_classes = info["num_classes"]

    all_epochs: List[int] = info["all_epochs"]
    all_layers: List[int] = info["all_layers"]

    # Apply CLI epoch/layer filters if provided.
    if args.epoch_filter:
        all_epochs = [e for e in all_epochs if e in set(args.epoch_filter)]
    if args.layer_filter:
        all_layers = [l for l in all_layers if l in set(args.layer_filter)]

    # Load existing rows so we can resume partial runs.
    out_csv = _csv_path_for(h5_path)
    done_cells: set = set()
    existing_rows: List[pd.DataFrame] = []
    if out_csv.exists() and not args.force:
        try:
            prev = pd.read_csv(out_csv)
            done_cells = set(zip(prev["epoch"].tolist(), prev["layer"].tolist()))
            existing_rows.append(prev)
        except Exception:
            pass

    todo = [
        (e, l) for e in all_epochs for l in all_layers if (e, l) not in done_cells
    ]
    if not todo:
        # Signal to run_jobs that nothing was computed.
        return pd.DataFrame()

    X = probe.X_probe.astype(np.float32, copy=False)
    Y = probe.y_probe.astype(np.int64, copy=False)
    N = X.shape[0]

    # Subsample once (deterministic per HDF5) so all cells use the same subset.
    if N > args.subsample_n:
        rng = np.random.default_rng(int(job["seed"]))
        idx = rng.choice(N, size=args.subsample_n, replace=False)
        X = X[idx]
        Y = Y[idx]
        N_eff = args.subsample_n
    else:
        N_eff = N

    meta = {
        "network_id": info["network_id"],
        "dataset": job["dataset"],
        "noise": job["noise"],
        "arch": job["arch"],
        "seed": job["seed"],
        "num_classes": num_classes,
    }

    new_rows: List[dict] = []
    for cell_idx, (epoch, layer) in enumerate(todo, start=1):
        print(f"    cell {cell_idx}/{len(todo)}: epoch={epoch} layer={layer}", flush=True)
        try:
            row = _evaluate_cell(h5_path, epoch, layer, X, Y, N, N_eff, meta, args)
            new_rows.append(row)
        except Exception as exc:
            print(f"    -> FAILED epoch={epoch} layer={layer}: {exc!r}", flush=True)

        # Write incrementally so a crash only loses the current cell.
        if new_rows:
            combined = pd.concat(
                existing_rows + [pd.DataFrame(new_rows)], ignore_index=True
            )
            combined.to_csv(out_csv, index=False)

    all_frames = existing_rows + ([pd.DataFrame(new_rows)] if new_rows else [])
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def _csv_path_for(h5_path: Path) -> Path:
    return h5_path.with_name(f"mi_baselines_{h5_path.stem}.csv")


def run_jobs(jobs: List[dict], args: argparse.Namespace) -> None:
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        print("nothing to do")
        return

    # Build probes once per dataset (cached).
    probes: dict[str, ProbeBundle] = {}

    total = len(jobs)
    cum = 0.0
    done = 0
    skipped = 0
    for i, job in enumerate(jobs, start=1):
        tag = f"{job['dataset']}/n{job['noise']}/{job['arch']}/seed_{job['seed']}"
        if job["dataset"] not in probes:
            probes[job["dataset"]] = _resolve_probe(job["dataset"])
            print(f"  probe[{job['dataset']}]: {probes[job['dataset']].note}")

        print(f"[{i}/{total}] {tag}", flush=True)
        t0 = time.perf_counter()
        try:
            df = evaluate_one(job, probes[job["dataset"]], args)
        except Exception as exc:
            print(f"  -> FAILED: {exc!r}", file=sys.stderr)
            continue
        if df.empty:
            skipped += 1
            print(f"  -> all cells already done, skipped")
            continue
        dt = time.perf_counter() - t0
        cum += dt
        done += 1
        avg = cum / done
        eta = avg * (total - i)
        print(
            f"  -> {len(df)} rows   {dt:6.2f}s   cum={cum/60:.1f}min   "
            f"ETA {eta/60:.1f} min over {total - i} remaining"
        )

    print(f"\nfinished: {done} ran, {skipped} fully-skipped, {total - done - skipped} failed.")


def aggregate(output: Path) -> None:
    frames: List[pd.DataFrame] = []
    for ds_root in DATASETS.values():
        for csv in ds_root.glob("*/mi_baselines_seed_*.csv"):
            try:
                frames.append(pd.read_csv(csv))
            except Exception as exc:
                print(f"[warn] failed to read {csv}: {exc}", file=sys.stderr)
    if not frames:
        print("no per-HDF5 CSVs found; nothing to aggregate")
        return
    out = pd.concat(frames, ignore_index=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    print(f"wrote {len(out)} rows to {output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS) + ["both"],
        default=["both"],
    )
    p.add_argument("--noise", nargs="+", type=float, default=None)
    p.add_argument("--archs", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "mi_baselines.csv",
    )

    p.add_argument("--subsample-n", type=int, default=SUBSAMPLE_N_DEFAULT,
                   help="cap probe size when running estimators (default 5000)")
    p.add_argument("--epoch-filter", nargs="+", type=int, default=None,
                   help="restrict to these epoch indices (default: all saved epochs)")
    p.add_argument("--layer-filter", nargs="+", type=int, default=None,
                   help="restrict to these layer indices (default: all hidden layers)")
    p.add_argument("--skip-mine", action="store_true")
    p.add_argument("--skip-infonce", action="store_true")
    p.add_argument("--mine-seeds", type=int, default=MINE_SEEDS_DEFAULT)
    p.add_argument("--mine-iter", type=int, default=2000)
    p.add_argument("--infonce-seeds", type=int, default=INFONCE_SEEDS_DEFAULT)
    p.add_argument("--infonce-iter", type=int, default=1000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    args = p.parse_args(argv)

    if args.aggregate:
        aggregate(args.output)
        return 0

    datasets = sorted(DATASETS) if "both" in args.datasets else sorted(set(args.datasets))
    jobs = discover_jobs(
        datasets=datasets,
        noise_filter=args.noise,
        arch_filter=args.archs,
        seed_filter=args.seeds,
    )
    print(f"discovered {len(jobs)} jobs across {datasets}")
    if args.list:
        for j in jobs:
            print(f"  {j['dataset']}/n{j['noise']}/{j['arch']}/seed_{j['seed']}")
        return 0

    run_jobs(jobs, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
