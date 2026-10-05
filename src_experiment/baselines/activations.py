"""
Activation extractor for the MI baselines (step 3).

Loads model weights from a training-output HDF5, runs a forward pass on a
probe matrix, and returns the pre- or post-activation at the chosen layer.
The forward-pass convention matches
``src_experiment.routing_estimator.forward_activation_patterns`` — strict
``z > 0`` ReLU gating, PyTorch weight layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Tuple, Union

import h5py
import numpy as np

PathLike = Union[str, Path]
Kind = Literal["pre", "post"]


def _load_all_weights(
    h5_path: Path, epoch: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load every stored weight/bias at ``epoch`` (hidden + output).

    Weights use PyTorch layout: ``W[i].shape == (n_{i+1}, n_i)``.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[f"epochs/epoch_{epoch}"]
        idx = 1
        Ws: List[np.ndarray] = []
        bs: List[np.ndarray] = []
        while f"l{idx}.weight" in grp:
            Ws.append(np.asarray(grp[f"l{idx}.weight"][:], dtype=np.float32))
            bs.append(np.asarray(grp[f"l{idx}.bias"][:], dtype=np.float32))
            idx += 1
    if not Ws:
        raise ValueError(f"no weights found at epoch {epoch} in {h5_path}")
    return Ws, bs


def load_layer_activations(
    h5_path: PathLike,
    epoch: int,
    layer: int,
    X: np.ndarray,
    kind: Kind = "pre",
) -> np.ndarray:
    """Forward pass on ``X`` and return one layer's activation matrix.

    Parameters
    ----------
    h5_path
        Training-output HDF5 in this project's standard layout.
    epoch
        Saved epoch index (must match a key under ``epochs/`` in the file).
    layer
        1-indexed layer. ``0`` returns the input ``X``. Hidden layers run
        ``1..L``; ``layer=L+1`` returns the output (logit) layer.
    X
        Probe set, shape ``(N, n_0)``, already in the model's training-time
        feature space (see ``src_experiment.probe_loader``).
    kind
        ``"pre"`` → ``z = a_{l-1} W_l^T + b_l`` (signed pre-activation).
        ``"post"`` → ``ReLU(z)`` on hidden layers; on the output layer post
        equals pre (no nonlinearity applied during training). At
        ``layer=0`` both kinds return ``X`` unchanged.

    Returns
    -------
    ``(N, d_layer)`` float32 array.
    """
    if layer < 0:
        raise ValueError("layer must be >= 0")
    if kind not in ("pre", "post"):
        raise ValueError(f"kind must be 'pre' or 'post', got {kind!r}")

    X = np.asarray(X, dtype=np.float32)
    if layer == 0:
        return X.copy()

    Ws, bs = _load_all_weights(Path(h5_path), epoch)
    n_layers = len(Ws)
    if layer > n_layers:
        raise ValueError(
            f"layer={layer} exceeds number of stored layers ({n_layers})"
        )

    a = X
    for idx, (W, b) in enumerate(zip(Ws, bs), start=1):
        z = a @ W.T + b
        if idx == layer:
            if kind == "pre":
                return np.asarray(z, dtype=np.float32)
            # post: ReLU on hidden layers; output layer is linear so post == pre.
            if idx < n_layers:
                return np.asarray(np.where(z > 0, z, 0.0), dtype=np.float32)
            return np.asarray(z, dtype=np.float32)
        a = np.where(z > 0, z, 0.0)

    raise RuntimeError("forward pass terminated without returning")  # unreachable
