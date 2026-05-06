"""
Probe / holdout set builders for the label-noise evaluation.

Reproduces the model's training-time preprocessing pipeline so the
probe-set features live in exactly the same transformed space as the data the
model was trained on. Cached per-dataset: building a fresh probe is the
expensive step (UCI fetch / synthetic generation + scaler refit), so the
results are reused across all (noise, arch, seed) jobs within one dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src_experiment.dataset import (
    N_SAMPLES,
    _load_uci,
    _make_composite_data,
)


@dataclass
class ProbeBundle:
    """Probe (and optional holdout) data for one dataset/global_seed."""

    X_probe: np.ndarray
    y_probe: np.ndarray
    X_holdout: Optional[np.ndarray] = None
    y_holdout: Optional[np.ndarray] = None
    note: str = ""

    def has_holdout(self) -> bool:
        return self.X_holdout is not None


def _fit_train_scaler(
    X_full: np.ndarray, y_full: np.ndarray, *, global_seed: int, test_size: float = 0.2
) -> MinMaxScaler:
    """Replicate ``process_and_split`` to recover the trained model's scaler."""
    X_train, _, _, _ = train_test_split(
        X_full, y_full, test_size=test_size, random_state=global_seed, stratify=y_full
    )
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)
    return scaler


# ---------------------------------------------------------------------------
# Composite (synthetic, freely regenerable)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def make_composite_probe(
    global_seed: int = 42,
    probe_size: int = 20000,
    holdout_size: int = 10000,
    probe_seed: int = 1042,
    holdout_seed: int = 2042,
) -> ProbeBundle:
    """Fresh probe + holdout for the composite dataset.

    The trained model's scaler is recovered by replaying the original training
    pipeline (``_make_composite_data`` with ``n_samples=N_SAMPLES, seed=global_seed``,
    then 80/20 stratified split, ``MinMaxScaler`` fit on the train slice). Fresh
    probe / holdout samples are generated with distinct seeds and pushed through
    the same scaler so they live in the same transformed feature space.

    Labels are kept *clean* — for label-noise studies we want to measure routing
    information against the true task, not the noisy training labels.
    """
    X_orig, y_orig = _make_composite_data(n_samples=N_SAMPLES, seed=global_seed)
    scaler = _fit_train_scaler(X_orig, y_orig, global_seed=global_seed)

    def _build(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
        X, y = _make_composite_data(n_samples=n, seed=seed)
        X = scaler.transform(X)
        X = np.clip(X, -1.0, 1.0).astype(np.float32)
        return X, y.astype(np.int64)

    X_p, y_p = _build(probe_size, probe_seed)
    if holdout_size > 0:
        X_h, y_h = _build(holdout_size, holdout_seed)
    else:
        X_h, y_h = None, None

    return ProbeBundle(
        X_probe=X_p,
        y_probe=y_p,
        X_holdout=X_h,
        y_holdout=y_h,
        note=(
            f"composite probe N={probe_size} (seed={probe_seed}), "
            f"holdout N={holdout_size} (seed={holdout_seed}); "
            f"scaler refit from training-time pipeline (global_seed={global_seed})"
        ),
    )


# ---------------------------------------------------------------------------
# WBC (real, fixed N=569)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _wbc_raw() -> Tuple[np.ndarray, np.ndarray]:
    return _load_uci(id=17, target_col="Diagnosis", target_val="M")


@lru_cache(maxsize=None)
def make_wbc_probe(
    global_seed: int = 42,
    mode: str = "test",
    holdout_frac: float = 0.3,
) -> ProbeBundle:
    """Probe builder for WBC, real dataset of N=569 total.

    Modes
    -----
    ``test`` (default)
        Use the stored test set (~114). No holdout. Matches existing HDF5 layout.
    ``full``
        Use the full 569-sample WBC (train+test) as probe. No holdout. Includes
        data the model trained on — acceptable for routing analysis but worth
        flagging.
    ``split``
        Split the stored test set into probe (1-``holdout_frac``) / holdout
        (``holdout_frac``). Both very small; only useful for the smallest
        architectures.
    """
    X, y = _wbc_raw()
    # Replicate training: 80/20 stratified split, scaler fit on train.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=global_seed, stratify=y
    )
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = np.clip(scaler.transform(X_test), -1.0, 1.0).astype(np.float32)

    if mode == "test":
        return ProbeBundle(
            X_probe=X_test_s,
            y_probe=y_test.astype(np.int64),
            note=f"wbc test set (N={len(X_test_s)}); no holdout",
        )

    if mode == "full":
        X_train_s = np.clip(X_train_s, -1.0, 1.0).astype(np.float32)
        X_full = np.concatenate([X_train_s, X_test_s], axis=0)
        y_full = np.concatenate([y_train, y_test]).astype(np.int64)
        return ProbeBundle(
            X_probe=X_full,
            y_probe=y_full,
            note=(
                f"wbc full set (N={len(X_full)}); train portion was seen during training. "
                "No holdout."
            ),
        )

    if mode == "split":
        X_p, X_h, y_p, y_h = train_test_split(
            X_test_s,
            y_test,
            test_size=holdout_frac,
            random_state=global_seed + 1,
            stratify=y_test,
        )
        return ProbeBundle(
            X_probe=X_p,
            y_probe=y_p.astype(np.int64),
            X_holdout=X_h,
            y_holdout=y_h.astype(np.int64),
            note=(
                f"wbc split: probe N={len(X_p)}, holdout N={len(X_h)} "
                f"(holdout_frac={holdout_frac})"
            ),
        )

    raise ValueError(f"unknown wbc mode: {mode!r} (expected test|full|split)")


# ---------------------------------------------------------------------------
# MNIST full + LeNet
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def make_mnist_full_lenet_probe(
    canonical_h5: Optional[str] = None,
) -> ProbeBundle:
    """Probe builder for the ``mnist_full_lenet`` sweep.

    All 60 LeNet HDF5s under ``outputs/mnist_full_lenet/`` share the same
    saved test set (10 000 MNIST samples already in ``[-1, 1]``). Read
    ``points`` / ``labels`` from a canonical HDF5 (LeNet-XS / noise=0 /
    seed=101) so probe construction is deterministic and avoids re-running
    the original training-time preprocessing.
    """
    import h5py
    from pathlib import Path as _Path

    if canonical_h5 is None:
        canonical_h5 = str(
            _Path(__file__).resolve().parents[1]
            / "outputs"
            / "mnist_full_lenet"
            / "n0.0_LeNet-XS"
            / "seed_101.h5"
        )
    with h5py.File(canonical_h5, "r") as f:
        X = np.asarray(f["points"][:], dtype=np.float32)
        y = np.asarray(f["labels"][:], dtype=np.int64)
    return ProbeBundle(
        X_probe=X,
        y_probe=y,
        note=(
            f"mnist_full_lenet probe N={len(X)}; read from {canonical_h5} "
            f"(canonical: same test set is saved into every LeNet HDF5)"
        ),
    )
