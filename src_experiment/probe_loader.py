"""In-sample point sets for the ``insample`` and ``train`` protocols (steps 2 and 3).

Each builder replays the training-time preprocessing of ``get_new_data`` (80/20
stratified split with the data seed, ``MinMaxScaler`` to [-1, 1] fit on the
train slice, test slice clipped) and returns the train slice followed by the
test slice, so the last 20 % equal the ``points`` stored in each HDF5. Labels
are the true labels. Cached per process.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src_experiment.dataset import N_SAMPLES, _load_uci, _make_composite_data


@dataclass
class ProbeBundle:
    X_probe: np.ndarray
    y_probe: np.ndarray
    note: str = ""


def _split_and_scale(X: np.ndarray, y: np.ndarray, global_seed: int, clip_train: bool) -> ProbeBundle:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=global_seed, stratify=y
    )
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    if clip_train:
        X_train = np.clip(X_train, -1.0, 1.0)
    X_test = np.clip(scaler.transform(X_test), -1.0, 1.0)
    return ProbeBundle(
        X_probe=np.concatenate([X_train, X_test]).astype(np.float32),
        y_probe=np.concatenate([y_train, y_test]).astype(np.int64),
    )


@lru_cache(maxsize=None)
def make_composite_insample(global_seed: int = 42) -> ProbeBundle:
    """All N_SAMPLES = 10 000 Composite points (train split, then test split)."""
    X, y = _make_composite_data(n_samples=N_SAMPLES, seed=global_seed)
    p = _split_and_scale(X, y, global_seed, clip_train=False)
    p.note = f"composite in-sample (N={len(p.y_probe)})"
    return p


@lru_cache(maxsize=None)
def make_wbc_insample(global_seed: int = 42) -> ProbeBundle:
    """All 569 WBC points (train split, then test split)."""
    X, y = _load_uci(id=17, target_col="Diagnosis", target_val="M")
    p = _split_and_scale(X, y, global_seed, clip_train=True)
    p.note = f"wbc in-sample (N={len(p.y_probe)})"
    return p
