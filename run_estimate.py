"""Step 2: routing information of every trained network, for every protocol.

For each ``outputs/<sweep>/<experiment>/seed_<s>.h5`` and each protocol of its
sweep, :class:`FunctionalQuotientEstimator` computes the raw and ε-quotient
routing MI with every estimator in :mod:`src_experiment.estimators`, at every
saved epoch and hidden layer. One CSV per (network, protocol) is written next
to the HDF5 (``routing_seed_<s>_<protocol>.csv``); existing ones are skipped
unless ``--force``. ``--aggregate`` concatenates them into
``results/routing_<sweep>.csv`` and writes ``results/provenance.json``.

Protocols (``protocol`` column):
  heldout   the test split stored in the HDF5 (never trained on):
            Composite N=2000, WBC N=114, MNIST N=10000.
  insample  every point of the dataset, train split followed by the test
            split (Composite N=10000, WBC N=569). Not run for MNIST.
  train     the train split only (label-permutation sweep).

Labels (``labels`` column): ``true_labels`` everywhere; for the
label-permutation sweep also ``train_labels``, the permuted labels the network
was trained on (train protocol only), which shows how much of them the routing
memorised.

Usage:
    python run_estimate.py [--sweeps ...] [--workers N] [--force]
    python run_estimate.py --aggregate
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import pandas as pd

from src_experiment import smoke
from src_experiment.dataset import permute_labels
from src_experiment.functional_quotient import DEFAULT_EPSILONS, FunctionalQuotientEstimator
from src_experiment.probe_loader import make_composite_insample, make_wbc_insample

REPO = Path(__file__).resolve().parent
OUTPUTS = REPO / "outputs"
RESULTS = REPO / "results"

# (protocol, labels) jobs per network of each sweep.
SWEEP_PROTOCOLS = {
    "composite_label_noise": (("heldout", "true_labels"), ("insample", "true_labels")),
    "wbc_label_noise": (("heldout", "true_labels"), ("insample", "true_labels")),
    "mnist_capacity": (("heldout", "true_labels"),),
    "label_permutation": (("heldout", "true_labels"), ("insample", "true_labels"),
                          ("train", "true_labels"), ("train", "train_labels")),
}


def _insample(dataset: str, global_seed: int):
    if dataset == "composite":
        return make_composite_insample(global_seed)
    if dataset == "wbc":
        return make_wbc_insample(global_seed)
    raise ValueError(f"no in-sample protocol for {dataset!r}")


def _probe(dataset: str, protocol: str, labels: str, global_seed: int, n_test: int):
    """(X, y) for a job, or (None, None) for the HDF5's stored test split."""
    if protocol == "heldout":
        return None, None
    p = _insample(dataset, global_seed)
    if protocol == "insample":
        return p.X_probe, p.y_probe
    X, y = p.X_probe[:-n_test], p.y_probe[:-n_test]  # train split
    # process_and_split permutes the train labels with the data seed.
    return X, (permute_labels(y, global_seed) if labels == "train_labels" else y)


def discover_jobs(sweeps: List[str]) -> List[dict]:
    jobs = []
    for sweep in sweeps:
        for h5 in sorted((OUTPUTS / sweep).glob("*/seed_*.h5")):
            with h5py.File(h5, "r") as f:
                a = dict(f["metadata"].attrs)
            meta = {
                "sweep": sweep,
                "dataset": str(a["dataset"]),
                "arch_str": str([int(w) for w in a["architecture"]]),
                "target_dim": int(a.get("target_dim", -1)),
                "noise_level": float(a.get("noise", 0.0)),
                "global_seed": int(a.get("global_seed", 42)),
                "permute_labels": str(a.get("permute_labels", False)) == "True",
            }
            for protocol, labels in SWEEP_PROTOCOLS[sweep]:
                name = protocol if labels == "true_labels" else f"{protocol}_{labels}"
                jobs.append({**meta, "protocol": protocol, "labels": labels, "h5": h5,
                             "csv": h5.with_name(f"routing_{h5.stem}_{name}.csv")})
    return jobs


def run_one(job: dict, epsilons: List[float]) -> float:
    t0 = time.perf_counter()
    est = FunctionalQuotientEstimator(job["h5"])
    X, y = _probe(job["dataset"], job["protocol"], job["labels"], job["global_seed"], len(est.labels))
    df = est.evaluate_all(X=X, y=y, epsilons=epsilons)
    for key in ("sweep", "dataset", "arch_str", "target_dim", "noise_level", "permute_labels",
                "protocol", "labels"):
        df.insert(0, key, job[key])
    tmp = job["csv"].with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(job["csv"])  # never leave a half-written CSV that would be skipped
    return time.perf_counter() - t0


def _tag(job: dict) -> str:
    return (f"{job['h5'].parent.parent.name}/{job['h5'].parent.name}/{job['h5'].stem}/"
            f"{job['protocol']}/{job['labels']}")


def run_jobs(jobs: List[dict], epsilons: List[float], workers: int, force: bool) -> int:
    todo = [j for j in jobs if force or not j["csv"].exists()]
    print(f"{len(jobs)} jobs, {len(jobs) - len(todo)} already done, running {len(todo)} "
          f"on {workers} worker(s)", flush=True)
    # Build the in-sample sets once here; forked workers inherit the caches.
    for dataset, seed in {(j["dataset"], j["global_seed"]) for j in todo if j["protocol"] != "heldout"}:
        _insample(dataset, seed)
    t0, failed = time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, j, epsilons): j for j in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            job = futures[fut]
            try:
                print(f"[{i}/{len(todo)}] {_tag(job)}: {fut.result():.1f}s", flush=True)
            except Exception as exc:
                failed += 1
                print(f"[{i}/{len(todo)}] FAILED {_tag(job)}: {exc!r}", file=sys.stderr, flush=True)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min, {failed} failed", flush=True)
    return failed


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def aggregate(sweeps: List[str]) -> None:
    RESULTS.mkdir(exist_ok=True)
    summary = {}
    for sweep in sweeps:
        jobs = discover_jobs([sweep])
        missing = [_tag(j) for j in jobs if not j["csv"].exists()]
        if missing:
            print(f"[warn] {sweep}: {len(missing)} of {len(jobs)} jobs have no CSV, e.g. {missing[0]}",
                  file=sys.stderr)
        frames = []
        for j in jobs:
            if j["csv"].exists():
                f = pd.read_csv(j["csv"])
                # CSVs from before the labels column existed are true-label, unpermuted.
                f["labels"] = f.get("labels", j["labels"])
                f["permute_labels"] = f.get("permute_labels", j["permute_labels"])
                frames.append(f)
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        out = RESULTS / f"routing_{sweep}.csv"
        df.to_csv(out, index=False)
        summary[sweep] = {
            "rows": len(df),
            "networks": int(df.groupby(["arch_str", "target_dim", "noise_level", "seed"]).ngroups),
            "protocols": sorted(df["protocol"].unique()),
            "epsilons": sorted(df["epsilon"].unique().tolist()),
            "missing_jobs": len(missing),
        }
        print(f"wrote {len(df)} rows to {out.relative_to(REPO)}")
    provenance = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "smoke": smoke.SMOKE,
        "jobs_per_network": {s: [f"{p}/{l}" for p, l in SWEEP_PROTOCOLS[s]] for s in sweeps},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sweeps": summary,
    }
    path = RESULTS / "provenance.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old["step2_estimate"] = provenance
    path.write_text(json.dumps(old, indent=2) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweeps", nargs="+", default=list(SWEEP_PROTOCOLS), choices=list(SWEEP_PROTOCOLS))
    p.add_argument("--epsilons", nargs="+", type=float, default=list(DEFAULT_EPSILONS))
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    p.add_argument("--force", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    args = p.parse_args(argv)
    if args.aggregate:
        aggregate(args.sweeps)
        return 0
    return 1 if run_jobs(discover_jobs(args.sweeps), args.epsilons, args.workers, args.force) else 0


if __name__ == "__main__":
    raise SystemExit(main())
