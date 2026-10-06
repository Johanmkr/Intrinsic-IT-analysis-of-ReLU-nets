"""Step 3: MI baselines on the hidden-layer pre-activations T of every clean network.

For each clean network (Composite, WBC, MNIST) and each protocol of step 2 —
so the baselines see exactly the points the routing estimate sees — and every
hidden layer at the last epoch, computes the grid

  binning   uniform per-neuron bins over [−max|T|, max|T|], K ∈ {2, 4, 8, 16, 30}
  k-means   K ∈ {|Y|, 2|Y|, 4|Y|, 16, 64, 256} (n_init=10, seeded by the network seed)
  KSG       Ross (2014) mixed continuous–discrete kNN estimator, k ∈ {3, 5, 10}

Binning and k-means give discrete partitions of T, so every estimator of
src_experiment/estimators.py is applied to them (columns
``binning<K>_<estimator>_bits``, ``kmeans<K>_<estimator>_bits``, plus the
number of occupied cells). KSG gives ``ksg<k>_bits`` (clipped at 0, as in the
paper) and ``ksg<k>_signed_bits``.

One CSV per (network, protocol) next to the HDF5 (``baselines_seed_<s>_<protocol>.csv``),
aggregated by ``--aggregate`` into ``results/baselines_<sweep>.csv.gz``.

Usage:
    python run_baselines.py [--sweeps ...] [--workers N] [--all-epochs] [--force]
    python run_baselines.py --aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import pandas as pd

from run_estimate import RESULTS, _git, _git_dirty, _insample, _probe, _tag, discover_jobs
from src_experiment.baselines.activations import load_layer_activations
from src_experiment.baselines.mi_baselines import ksg_mi, quantize_per_layer
from src_experiment.estimators import all_mutual_information_bits, contingency_table
from src_experiment.results import SUFFIX, write_table
from src_experiment.routing_estimator import RoutingEstimator

SWEEPS = ("composite_label_noise", "wbc_label_noise", "mnist_capacity")
BINNING_BINS = (2, 4, 8, 16, 30)
KMEANS_FIXED_K = (16, 64, 256)
KSG_KS = (3, 5, 10)


def _discrete(prefix: str, cells: np.ndarray, y: np.ndarray) -> dict:
    table = contingency_table(cells, y)
    out = {f"{prefix}_{k}": v for k, v in all_mutual_information_bits(table).items()}
    out[f"{prefix}_num_cells"] = table.shape[0]
    return out


def baselines_row(T: np.ndarray, y: np.ndarray, num_classes: int, seed: int) -> dict:
    from sklearn.cluster import KMeans

    row: dict = {"d_T": T.shape[1]}
    for nb in BINNING_BINS:
        binned = quantize_per_layer(T, nb)
        row.update(_discrete(f"binning{nb}", np.unique(binned, axis=0, return_inverse=True)[1].ravel(), y))
    kmeans_K = {"KY": num_classes, "2KY": 2 * num_classes, "4KY": 4 * num_classes}
    for K in KMEANS_FIXED_K:
        if K not in kmeans_K.values():
            kmeans_K[str(K)] = K
    for label, K in kmeans_K.items():
        km = KMeans(n_clusters=min(K, len(T)), n_init=10, random_state=seed).fit(T)
        row.update(_discrete(f"kmeans{label}", km.labels_, y))
    for k in KSG_KS:
        out = ksg_mi(T, y, k=k)
        row[f"ksg{k}_bits"] = out["bits"]
        row[f"ksg{k}_signed_bits"] = out["bits_signed"]
    return row


def run_one(job: dict, all_epochs: bool) -> float:
    t0 = time.perf_counter()
    est = RoutingEstimator(job["h5"])
    X, y = _probe(job["dataset"], job["protocol"], job["labels"], job["global_seed"], len(est.labels))
    if X is None:
        X, y = est.points, est.labels
    num_classes = int(max(y.max(), est.labels.max())) + 1
    rows = []
    for epoch in (est.epochs if all_epochs else est.epochs[-1:]):
        for layer in range(1, est.num_hidden_layers + 1):
            T = load_layer_activations(job["h5"], epoch, layer, X, kind="pre").astype(np.float32)
            rows.append({"network_id": est.network_id, "seed": est.seed, "epoch": epoch,
                         "layer": layer, "N": len(y), "num_classes": num_classes,
                         **baselines_row(T, y, num_classes, est.seed)})
    df = pd.DataFrame(rows)
    for key in ("sweep", "dataset", "arch_str", "target_dim", "noise_level", "protocol", "labels"):
        df.insert(0, key, job[key])
    tmp = job["csv"].with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(job["csv"])
    return time.perf_counter() - t0


def jobs_for(sweeps: List[str]) -> List[dict]:
    jobs = discover_jobs(sweeps)
    for j in jobs:
        j["csv"] = j["csv"].with_name(j["csv"].name.replace("routing_", "baselines_", 1))
    return jobs


def run_jobs(jobs: List[dict], workers: int, force: bool, all_epochs: bool) -> int:
    todo = [j for j in jobs if force or not j["csv"].exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, running {len(todo)} "
          f"on {workers} worker(s)", flush=True)
    for dataset, seed in {(j["dataset"], j["global_seed"]) for j in todo if j["protocol"] != "heldout"}:
        _insample(dataset, seed)
    t0, failed = time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, j, all_epochs): j for j in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            job = futures[fut]
            try:
                print(f"[{i}/{len(todo)}] {_tag(job)}: {fut.result():.1f}s", flush=True)
            except Exception as exc:
                failed += 1
                print(f"[{i}/{len(todo)}] FAILED {_tag(job)}: {exc!r}", file=sys.stderr, flush=True)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min, {failed} failed", flush=True)
    return failed


def aggregate(sweeps: List[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    summary = {}
    for sweep in sweeps:
        jobs = jobs_for([sweep])
        missing = [_tag(j) for j in jobs if not j["csv"].exists()]
        if missing:
            print(f"[warn] {sweep}: {len(missing)} of {len(jobs)} jobs have no CSV, e.g. {missing[0]}",
                  file=sys.stderr)
        frames = [pd.read_csv(j["csv"]) for j in jobs if j["csv"].exists()]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        out = RESULTS / f"baselines_{sweep}{SUFFIX}"
        write_table(df, out)
        summary[sweep] = {"rows": len(df), "protocols": sorted(df["protocol"].unique()),
                          "epochs": sorted(int(e) for e in df["epoch"].unique()),
                          "missing_jobs": len(missing)}
        print(f"wrote {len(df)} rows to {out.relative_to(RESULTS.parent)}")
    path = RESULTS / "provenance.json"
    prov = json.loads(path.read_text()) if path.exists() else {}
    prov["step3_baselines"] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git_dirty(),
        "binning_bins": list(BINNING_BINS), "kmeans_fixed_K": list(KMEANS_FIXED_K),
        "ksg_k": list(KSG_KS), "sweeps": summary,
    }
    path.write_text(json.dumps(prov, indent=2) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweeps", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    p.add_argument("--all-epochs", action="store_true", help="every saved epoch, not just the last")
    p.add_argument("--force", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args(argv)
    if args.aggregate:
        aggregate(args.sweeps)
        return 0
    return 1 if run_jobs(jobs_for(args.sweeps), args.workers, args.force, args.all_epochs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
