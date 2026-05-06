"""
Run :class:`FunctionalQuotientEstimator` (Recipes 1-4) over the already-trained
label-noise sweeps in ``outputs/composite_label_noise/`` and
``outputs/wbc_label_noise/``.

Per-HDF5 results are written to ``new_estimator_seed_<seed>.csv`` next to the
HDF5 file. Re-running skips files whose CSV already exists, so the run is
resumable.

Examples
--------
Time one experiment::

    uv run python run_label_noise_estimator.py --datasets composite --limit 1

Run all composite, narrow architectures::

    uv run python run_label_noise_estimator.py --datasets composite \\
        --archs '[5, 5, 5]' '[25, 25, 25]'

Aggregate after runs::

    uv run python run_label_noise_estimator.py --aggregate \\
        --output results/label_noise_new_estimator.csv
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from src_experiment.functional_quotient import FunctionalQuotientEstimator
from src_experiment.probe_loader import (
    ProbeBundle,
    make_composite_probe,
    make_wbc_probe,
)

REPO = Path(__file__).resolve().parent
OUTPUTS = REPO / "outputs"

DATASETS = {
    "composite": OUTPUTS / "composite_label_noise",
    "wbc": OUTPUTS / "wbc_label_noise",
    "mnist_full_lenet": OUTPUTS / "mnist_full_lenet",
}

# Wider than spec default. See `new_estimator_next_steps.md` for rationale.
DEFAULT_EPSILONS = (0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0)

# Filename pattern: ``n0.2_[25, 25, 25]`` (MLP) or ``n0.0_LeNet-XS`` (CNN).
# arch group accepts either a bracketed list or a hyphenated identifier.
_DIR_RE = re.compile(r"^n(?P<noise>[0-9.]+)_(?P<arch>.+)$")
_SEED_RE = re.compile(r"^seed_(?P<seed>\d+)\.h5$")


# ---------------------------------------------------------------------------
# Job discovery
# ---------------------------------------------------------------------------
def discover_jobs(
    datasets: Iterable[str],
    noise_filter: Optional[Iterable[float]] = None,
    arch_filter: Optional[Iterable[str]] = None,
    seed_filter: Optional[Iterable[int]] = None,
) -> List[dict]:
    """Walk dataset roots and return a sorted, deterministic job list."""
    jobs: List[dict] = []
    for ds in datasets:
        root = DATASETS[ds]
        if not root.is_dir():
            print(f"[warn] dataset root missing: {root}", file=sys.stderr)
            continue
        for cfg_dir in sorted(root.iterdir()):
            if not cfg_dir.is_dir():
                continue
            m = _DIR_RE.match(cfg_dir.name)
            if not m:
                continue
            noise = float(m.group("noise"))
            arch = m.group("arch")
            if noise_filter is not None and noise not in noise_filter:
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
                jobs.append(
                    {
                        "dataset": ds,
                        "noise": noise,
                        "arch": arch,
                        "seed": seed,
                        "h5": h5,
                        "csv": h5.with_name(f"new_estimator_{h5.stem}.csv"),
                    }
                )
    return jobs


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
def run_one(
    job: dict, epsilons: Iterable[float], probe: Optional[ProbeBundle]
) -> float:
    """Evaluate one HDF5; write its CSV; return wall-clock seconds."""
    t0 = time.perf_counter()
    estimator = FunctionalQuotientEstimator(job["h5"])
    if probe is None:
        df = estimator.evaluate_all(epsilons=tuple(epsilons))
        n_probe = len(estimator.labels)
    else:
        df = estimator.evaluate_all(
            X=probe.X_probe,
            y=probe.y_probe,
            X_holdout=probe.X_holdout,
            y_holdout=probe.y_holdout,
            epsilons=tuple(epsilons),
        )
        n_probe = len(probe.y_probe)
    df["dataset"] = job["dataset"]
    df["noise_level"] = job["noise"]
    df["arch_str"] = job["arch"]
    df["probe_N"] = n_probe
    df.to_csv(job["csv"], index=False)
    return time.perf_counter() - t0


def _resolve_probe(job: dict, args: argparse.Namespace) -> Optional[ProbeBundle]:
    """Return a ProbeBundle if the user requested a non-default probe; else None."""
    if job["dataset"] == "composite":
        if args.composite_probe_size is None:
            return None
        return make_composite_probe(
            global_seed=42,
            probe_size=args.composite_probe_size,
            holdout_size=args.composite_holdout_size,
            probe_seed=args.composite_probe_seed,
            holdout_seed=args.composite_holdout_seed,
        )
    if job["dataset"] == "wbc":
        if args.wbc_mode == "test":
            return None
        return make_wbc_probe(
            global_seed=42,
            mode=args.wbc_mode,
            holdout_frac=args.wbc_holdout_frac,
        )
    return None


def run_jobs(
    jobs: List[dict],
    args: argparse.Namespace,
) -> None:
    if args.limit is not None:
        jobs = jobs[: args.limit]
    if not jobs:
        print("nothing to do")
        return

    total = len(jobs)
    cum = 0.0
    done = 0
    for i, job in enumerate(jobs, start=1):
        tag = f"{job['dataset']}/n{job['noise']}/{job['arch']}/seed_{job['seed']}"
        if job["csv"].exists() and not args.force:
            print(f"[{i}/{total}] skip (exists): {tag}")
            continue
        try:
            probe = _resolve_probe(job, args)
        except Exception as exc:
            print(f"[{i}/{total}] FAILED to build probe for {tag}: {exc!r}", file=sys.stderr)
            continue
        probe_tag = f" [{probe.note}]" if probe is not None else ""
        print(f"[{i}/{total}] running: {tag}{probe_tag}")
        try:
            dt = run_one(job, args.epsilons, probe)
        except Exception as exc:
            print(f"  -> FAILED: {exc!r}", file=sys.stderr)
            continue
        cum += dt
        done += 1
        avg = cum / done
        eta = avg * (total - i)
        print(
            f"  -> {dt:6.2f}s  (avg {avg:.2f}s/job, "
            f"ETA {eta/60:.1f} min over {total - i} remaining)"
        )

    print(f"\nfinished: {done} ran, {total - done} skipped/failed.")
    if done:
        print(f"total wall time: {cum/60:.2f} min, mean {cum/done:.2f}s/job.")


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate(output: Path) -> None:
    frames: List[pd.DataFrame] = []
    for ds_root in DATASETS.values():
        for csv in ds_root.glob("*/new_estimator_seed_*.csv"):
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS) + ["both"],
        default=["both"],
        help="which label-noise sweep(s) to evaluate",
    )
    p.add_argument(
        "--noise",
        nargs="+",
        type=float,
        default=None,
        help="restrict to these noise levels (e.g. 0.0 0.2). default: all",
    )
    p.add_argument(
        "--archs",
        nargs="+",
        default=None,
        help="restrict to these architecture strings, e.g. '[25, 25, 25]'",
    )
    p.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="restrict to these model seeds (e.g. 101 102)",
    )
    p.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=list(DEFAULT_EPSILONS),
        help="ε grid for Recipe 2",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="run at most N jobs (use --limit 1 to time a single experiment)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="recompute even if per-HDF5 CSV exists",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="list jobs and exit (no work performed)",
    )
    p.add_argument(
        "--aggregate",
        action="store_true",
        help="concatenate per-HDF5 CSVs and exit",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=REPO / "results" / "label_noise_new_estimator.csv",
        help="path for aggregated CSV",
    )

    # Composite probe options
    p.add_argument(
        "--composite-probe-size",
        type=int,
        default=None,
        help="if set, regenerate composite probe with this N (else use stored test set)",
    )
    p.add_argument(
        "--composite-holdout-size",
        type=int,
        default=10000,
        help="composite holdout N (only used when --composite-probe-size is set)",
    )
    p.add_argument("--composite-probe-seed", type=int, default=1042)
    p.add_argument("--composite-holdout-seed", type=int, default=2042)

    # WBC probe mode
    p.add_argument(
        "--wbc-mode",
        choices=["test", "full", "split"],
        default="test",
        help="test: stored test set (114); full: all 569 (no holdout); "
        "split: 70/30 split of stored test",
    )
    p.add_argument(
        "--wbc-holdout-frac",
        type=float,
        default=0.3,
        help="holdout fraction when --wbc-mode=split",
    )

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
