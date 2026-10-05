"""
Run :class:`FunctionalQuotientEstimator` (Recipes 1-4) over the trained
MNIST capacity sweep in ``outputs/mnist_capacity/``.

The stored test points in each HDF5 are the full MNIST test set (N=10000),
already PCA-reduced to ``target_dim`` and MinMax-scaled to [-1, 1] in the
model's training-time feature space, so we use them directly as probe
(no separate probe loader needed; ``truncation_prob`` will be NaN).

Per-HDF5 results are written to ``new_estimator_seed_<seed>.csv`` next to
the HDF5. Re-running skips files whose CSV exists; resumable.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from src_experiment.functional_quotient import FunctionalQuotientEstimator

REPO = Path(__file__).resolve().parent
ROOT = REPO / "outputs" / "mnist_capacity"

DEFAULT_EPSILONS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0)

# Filename pattern: ``2_dim_[5, 5, 5]`` -> target_dim=2, arch="[5, 5, 5]"
_DIR_RE = re.compile(r"^(?P<dim>\d+)_dim_(?P<arch>\[.*\])$")
_SEED_RE = re.compile(r"^seed_(?P<seed>\d+)\.h5$")


def discover_jobs(
    dim_filter: Optional[Iterable[int]] = None,
    arch_filter: Optional[Iterable[str]] = None,
    seed_filter: Optional[Iterable[int]] = None,
) -> List[dict]:
    jobs: List[dict] = []
    if not ROOT.is_dir():
        print(f"[warn] root missing: {ROOT}", file=sys.stderr)
        return jobs
    for cfg_dir in sorted(ROOT.iterdir()):
        if not cfg_dir.is_dir():
            continue
        m = _DIR_RE.match(cfg_dir.name)
        if not m:
            continue
        target_dim = int(m.group("dim"))
        arch = m.group("arch")
        if dim_filter is not None and target_dim not in dim_filter:
            continue
        if arch_filter is not None and arch not in arch_filter:
            continue
        for h5 in sorted(cfg_dir.glob("seed_*.h5")):
            ms = _SEED_RE.match(h5.name)
            if not ms:
                continue
            seed = int(ms.group("seed"))
            if seed_filter is not None and seed not in seed_filter:
                continue
            jobs.append({
                "target_dim": target_dim,
                "arch": arch,
                "seed": seed,
                "h5": h5,
                "csv": h5.with_name(f"new_estimator_{h5.stem}.csv"),
            })
    return jobs


def run_one(job: dict, epsilons: Iterable[float]) -> float:
    t0 = time.perf_counter()
    estimator = FunctionalQuotientEstimator(job["h5"])
    df = estimator.evaluate_all(epsilons=tuple(epsilons))
    df["dataset"] = "mnist"
    df["target_dim"] = job["target_dim"]
    df["arch_str"] = job["arch"]
    df["probe_N"] = len(estimator.labels)
    df.to_csv(job["csv"], index=False)
    return time.perf_counter() - t0


def _tag(job: dict) -> str:
    return f"mnist/dim_{job['target_dim']}/{job['arch']}/seed_{job['seed']}"


def run_jobs(jobs: List[dict], args: argparse.Namespace) -> None:
    """Run the jobs on ``args.workers`` processes. Each job reads one HDF5 and
    writes its own CSV, so results do not depend on the number of workers."""
    if args.limit is not None:
        jobs = jobs[: args.limit]
    total = len(jobs)
    todo = []
    for job in jobs:
        if job["csv"].exists() and not args.force:
            print(f"skip (exists): {_tag(job)}")
        else:
            todo.append(job)
    if not todo:
        print("nothing to do")
        return
    print(f"running {len(todo)} of {total} jobs on {args.workers} worker(s)", flush=True)
    t0 = time.perf_counter()
    done = failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, job, args.epsilons): job for job in todo}
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                dt = fut.result()
            except Exception as exc:
                failed += 1
                print(f"FAILED: {_tag(job)}: {exc!r}", file=sys.stderr, flush=True)
                continue
            done += 1
            print(f"[{done + failed}/{len(todo)}] {_tag(job)}: {dt:6.1f}s", flush=True)
    print(f"\nfinished: {done} ran, {failed} failed, {total - len(todo)} skipped "
          f"in {(time.perf_counter() - t0) / 60:.1f} min.")


def aggregate(output: Path) -> None:
    frames: List[pd.DataFrame] = []
    for csv in ROOT.glob("*/new_estimator_seed_*.csv"):
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


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dims", nargs="+", type=int, default=None,
                   help="restrict to these PCA target dims (e.g. 2 5 10 20)")
    p.add_argument("--archs", nargs="+", default=None,
                   help="restrict to these architecture strings, e.g. '[25, 25, 25]'")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="restrict to these model seeds (e.g. 101 102 103)")
    p.add_argument("--epsilons", nargs="+", type=float, default=list(DEFAULT_EPSILONS),
                   help="ε grid for Recipe 2")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2),
                   help="parallel processes (default: #CPUs - 2)")
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true")
    p.add_argument("--aggregate", action="store_true")
    p.add_argument("--output", type=Path,
                   default=REPO / "results" / "mnist_capacity_new_estimator.csv")

    args = p.parse_args(argv)

    if args.aggregate:
        aggregate(args.output)
        return 0

    jobs = discover_jobs(
        dim_filter=set(args.dims) if args.dims else None,
        arch_filter=set(args.archs) if args.archs else None,
        seed_filter=set(args.seeds) if args.seeds else None,
    )

    if args.list:
        print(f"{len(jobs)} jobs:")
        for j in jobs:
            print(f"  mnist/dim_{j['target_dim']}/{j['arch']}/seed_{j['seed']}")
        return 0

    run_jobs(jobs, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
