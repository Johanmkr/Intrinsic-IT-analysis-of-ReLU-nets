"""Train networks from YAML configs (step 1).

    python run_training.py CONFIG.yaml [--overwrite]
        Train one network.
    python run_training.py --sweeps composite_label_noise ... [--workers N] [--overwrite]
        Train every configs/<sweep>/*/*.yaml whose HDF5 does not exist yet,
        N at a time (default: #CPUs − 2), one single-threaded process each.
        Each network's output goes to logs/train/<sweep>/<experiment>_seed_<s>.log.
    python run_training.py --export-curves
        Write the per-epoch training curves of every trained network to
        results/training_curves.csv.gz (read by scripts/plot_training_curves.py).
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent
# Ensure src_experiment is importable
sys.path.append(str(REPO))

CURVES = ("train_loss", "train_accuracy", "eval_train_loss", "eval_train_accuracy",
          "test_loss", "test_accuracy")


def _h5_path(cfg_path: Path) -> Path:
    c = yaml.safe_load(cfg_path.read_text())
    return REPO / c["output_dir"] / c["experiment_name"] / f"seed_{c['model_seed']}.h5"


def _train_one(cfg_path: Path, overwrite: bool) -> float:
    t0 = time.perf_counter()
    rel = cfg_path.relative_to(REPO / "configs")
    log = REPO / "logs" / "train" / rel.parent.parent / f"{rel.parent.name}_{rel.stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    cmd = [sys.executable, str(REPO / "run_training.py"), str(cfg_path)] + (["--overwrite"] if overwrite else [])
    with open(log, "w") as f:
        subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - t0


def train_sweeps(sweeps, workers: int, overwrite: bool) -> int:
    configs = [c for s in sweeps for c in sorted((REPO / "configs" / s).glob("*/*.yaml"))]
    todo = [c for c in configs if overwrite or not _h5_path(c).exists()]
    print(f"{len(configs)} configs, {len(configs) - len(todo)} already trained, "
          f"training {len(todo)} on {workers} worker(s)", flush=True)
    t0, failed = time.perf_counter(), 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_train_one, c, overwrite): c for c in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            tag = futures[fut].relative_to(REPO / "configs")
            try:
                print(f"[{i}/{len(todo)}] {tag}: {fut.result() / 60:.1f} min", flush=True)
            except subprocess.CalledProcessError:
                failed += 1
                print(f"[{i}/{len(todo)}] FAILED {tag} (see logs/train/)", file=sys.stderr, flush=True)
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min, {failed} failed", flush=True)
    return failed


def export_curves() -> None:
    """One row per (network, epoch) with the six curves stored in each HDF5."""
    import h5py
    import pandas as pd

    from src_experiment.results import SUFFIX, write_table

    frames = []
    for h5 in sorted((REPO / "outputs").glob("*/*/seed_*.h5")):
        with h5py.File(h5, "r") as f:
            a = f["metadata"].attrs
            curves = {k: f[f"training_results/{k}"][:] for k in CURVES}
            meta = {
                "sweep": h5.parent.parent.name,
                "dataset": str(a["dataset"]),
                "arch_str": str([int(w) for w in a["architecture"]]),
                "target_dim": int(a.get("target_dim", -1)),
                "noise_level": float(a.get("noise", 0.0)),
                "permute_labels": str(a.get("permute_labels", False)) == "True",
                "seed": int(a["model_seed"]),
            }
        df = pd.DataFrame({"epoch": range(len(curves["test_accuracy"])), **curves})
        for key, value in reversed(meta.items()):
            df.insert(0, key, value)
        frames.append(df)
    if not frames:
        sys.exit("no trained networks in outputs/")
    out = REPO / "results" / f"training_curves{SUFFIX}"
    out.parent.mkdir(exist_ok=True)
    write_table(pd.concat(frames, ignore_index=True), out)
    print(f"wrote curves of {len(frames)} networks to {out.relative_to(REPO)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", type=str, help="Path to one YAML configuration file.")
    parser.add_argument("--sweeps", nargs="+", help="Train all configs of these sweeps (configs/<sweep>/).")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) - 2))
    parser.add_argument("--overwrite", action="store_true", help="Retrain even if the HDF5 exists.")
    parser.add_argument("--export-curves", action="store_true",
                        help="Write results/training_curves.csv.gz from the trained networks.")
    args = parser.parse_args()

    if args.export_curves:
        export_curves()
        sys.exit(0)
    if args.sweeps:
        sys.exit(1 if train_sweeps(args.sweeps, args.workers, args.overwrite) else 0)
    if not args.config:
        parser.error("give a config file or --sweeps")

    from src_experiment.run_experiment import run
    run(args.config, overwrite=args.overwrite)
