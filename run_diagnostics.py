"""Step 4: diagnostics that need the regions themselves, at the last epoch.

region sizes   For every network (all sweeps), every step-2 protocol and every
               hidden layer: how many regions hold 1, 2, 3, ... points.
               → results/region_sizes_<sweep>.csv.gz  (columns ..., region_size, num_regions)
ordering       The functional quotient visits regions in first-encounter order
               (order 0). For every clean network, on the held-out points, every
               hidden layer and ε ∈ ORDER_EPSILONS, the quotient is recomputed
               under N_ORDERS random visiting orders (order k uses random seed k)
               and scored with every estimator.
               → results/ordering_<sweep>.csv.gz

One CSV per (network, protocol, kind) next to the HDF5; ``--aggregate``
concatenates them.

Usage:
    python run_diagnostics.py [--sweeps ...] [--workers N] [--force]
    python run_diagnostics.py --aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from run_estimate import RESULTS, SWEEP_PROTOCOLS, _git, _git_dirty, _insample, _probe, _tag, discover_jobs
from src_experiment.estimators import all_mutual_information_bits, contingency_table
from src_experiment.functional_quotient import (
    _build_active_data,
    cluster_functional,
    collect_unique_region_patterns,
)
from src_experiment.results import SUFFIX, write_table
from src_experiment.routing_estimator import (
    RoutingEstimator,
    cumulative_pattern_hashes,
    forward_activation_patterns,
)

CLEAN_SWEEPS = ("composite_label_noise", "wbc_label_noise", "mnist_capacity")
ORDER_EPSILONS = (0.1, 0.3, 0.5, 1.0)
N_ORDERS = 15
META = ("sweep", "dataset", "arch_str", "target_dim", "noise_level", "protocol", "labels")


def _write(df: pd.DataFrame, job: dict, path) -> None:
    for key in META:
        df.insert(0, key, job[key])
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def run_one(job: dict) -> float:
    t0 = time.perf_counter()
    est = RoutingEstimator(job["h5"])
    X, y = _probe(job["dataset"], job["protocol"], job["labels"], job["global_seed"], len(est.labels))
    if X is None:
        X, y = est.points, est.labels
    epoch = est.epochs[-1]
    W, b = est._load_weights(epoch)
    patterns = forward_activation_patterns(W, b, X)

    sizes, orders = [], []
    for layer in range(1, est.num_hidden_layers + 1):
        omega = cumulative_pattern_hashes(patterns, layer)
        _, region_sizes = np.unique(omega, return_counts=True)
        for size, count in zip(*np.unique(region_sizes, return_counts=True)):
            sizes.append({"seed": est.seed, "epoch": epoch, "layer": layer, "N": len(y),
                          "region_size": int(size), "num_regions": int(count)})
        if not job["ordering"]:
            continue
        active = list(_build_active_data(W, b, collect_unique_region_patterns(patterns, omega, layer), layer).items())
        for order in range(N_ORDERS + 1):
            items = active[:]
            if order > 0:
                random.Random(order).shuffle(items)
            for eps in ORDER_EPSILONS:
                qmap, num_q = cluster_functional(dict(items), eps)
                qids = np.fromiter((qmap[w] for w in omega), dtype=np.int64, count=len(y))
                orders.append({"seed": est.seed, "epoch": epoch, "layer": layer, "epsilon": eps,
                               "order": order, "num_regions": len(active), "num_quotient": num_q,
                               **{k.replace("_bits", "_func_bits"): v for k, v in
                                  all_mutual_information_bits(contingency_table(qids, y)).items()}})
    _write(pd.DataFrame(sizes), job, job["sizes_csv"])
    if job["ordering"]:
        _write(pd.DataFrame(orders), job, job["ordering_csv"])
    return time.perf_counter() - t0


def jobs_for(sweeps: List[str]) -> List[dict]:
    jobs = discover_jobs(sweeps)
    for j in jobs:
        name = j["csv"].name
        j["sizes_csv"] = j["csv"].with_name(name.replace("routing_", "regionsizes_", 1))
        j["ordering_csv"] = j["csv"].with_name(name.replace("routing_", "ordering_", 1))
        j["ordering"] = j["sweep"] in CLEAN_SWEEPS and j["protocol"] == "heldout"
    return jobs


def _done(job: dict) -> bool:
    return job["sizes_csv"].exists() and (not job["ordering"] or job["ordering_csv"].exists())


def run_jobs(jobs: List[dict], workers: int, force: bool) -> int:
    todo = [j for j in jobs if force or not _done(j)]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, running {len(todo)} "
          f"on {workers} worker(s)", flush=True)
    for dataset, seed in {(j["dataset"], j["global_seed"]) for j in todo if j["protocol"] != "heldout"}:
        _insample(dataset, seed)
    t0, failed = time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, j): j for j in todo}
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
        for kind, key in (("region_sizes", "sizes_csv"), ("ordering", "ordering_csv")):
            paths = [j[key] for j in jobs if kind == "region_sizes" or j["ordering"]]
            if not paths:
                continue
            missing = [p for p in paths if not p.exists()]
            if missing:
                print(f"[warn] {sweep}/{kind}: {len(missing)} of {len(paths)} CSVs missing",
                      file=sys.stderr)
            frames = [pd.read_csv(p) for p in paths if p.exists()]
            if not frames:
                continue
            df = pd.concat(frames, ignore_index=True)
            out = RESULTS / f"{kind}_{sweep}{SUFFIX}"
            write_table(df, out)
            summary[f"{kind}_{sweep}"] = {"rows": len(df), "missing_jobs": len(missing)}
            print(f"wrote {len(df)} rows to {out.relative_to(RESULTS.parent)}")
    path = RESULTS / "provenance.json"
    prov = json.loads(path.read_text()) if path.exists() else {}
    prov["step4_diagnostics"] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": _git_dirty(),
        "ordering_epsilons": list(ORDER_EPSILONS), "random_orders": N_ORDERS,
        "files": summary,
    }
    path.write_text(json.dumps(prov, indent=2) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweeps", nargs="+", default=list(SWEEP_PROTOCOLS), choices=list(SWEEP_PROTOCOLS))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    p.add_argument("--force", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args(argv)
    if args.aggregate:
        aggregate(args.sweeps)
        return 0
    return 1 if run_jobs(jobs_for(args.sweeps), args.workers, args.force) else 0


if __name__ == "__main__":
    raise SystemExit(main())
