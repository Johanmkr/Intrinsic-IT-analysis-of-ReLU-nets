"""
Recipe 1: routing-information estimator $\\tilde I(Y;\\Omega_{\\mathcal D})$
on a CPWL ReLU network.

Independent of the Julia tree: weights are loaded from HDF5, but cumulative
activation patterns are computed from a fresh forward pass on whatever probe
set is supplied. Region identity is the deterministic md5 of the bit-packed
cumulative pattern $\\pi^{\\le l}(x) = (\\pi^1, \\ldots, \\pi^l)$.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Forward pass / pattern extraction
# ---------------------------------------------------------------------------
def forward_activation_patterns(
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    X: np.ndarray,
) -> List[np.ndarray]:
    """Run a ReLU forward pass and return per-hidden-layer activation patterns.

    Strict `z > 0` matches the mathematical definition (dead neurons → 0).

    Parameters
    ----------
    weights, biases
        L hidden-layer weights/biases. ``weights[i]`` has shape ``(n_{i+1}, n_i)``
        (PyTorch convention), ``biases[i]`` has shape ``(n_{i+1},)``.
    X
        Probe set, shape ``(N, n_0)``.

    Returns
    -------
    list of length L; entry ``i`` is a ``(N, n_{i+1})`` bool array of $\\pi^{i+1}$.
    """
    if len(weights) != len(biases):
        raise ValueError("weights and biases must have equal length")
    if not weights:
        raise ValueError("at least one hidden layer required")

    a = np.asarray(X, dtype=np.float32)
    patterns: List[np.ndarray] = []
    for W, b in zip(weights, biases):
        z = a @ W.T + b
        pi = z > 0
        patterns.append(pi)
        a = np.where(pi, z, 0.0)
    return patterns


def cumulative_pattern_hashes(
    per_layer_patterns: Sequence[np.ndarray],
    layer: int,
) -> np.ndarray:
    """Deterministic md5 hash of $\\pi^{\\le l}(x_i)$ for each sample.

    ``layer`` is 1-indexed: ``layer=l`` hashes the concatenation of layers
    1..l. Returns a length-N object array of 16-byte ``bytes`` digests.
    """
    if layer < 1 or layer > len(per_layer_patterns):
        raise ValueError(f"layer must be in 1..{len(per_layer_patterns)}")
    cumulative = np.concatenate(
        [p.astype(bool, copy=False) for p in per_layer_patterns[:layer]],
        axis=1,
    )
    N = cumulative.shape[0]
    packed = np.packbits(cumulative, axis=1)
    digests = np.empty(N, dtype=object)
    for i in range(N):
        digests[i] = hashlib.md5(packed[i].tobytes()).digest()
    return digests


# ---------------------------------------------------------------------------
# Plug-in MI + Miller-Madow
# ---------------------------------------------------------------------------
@dataclass
class RoutingResult:
    layer: int
    N: int
    num_regions: int
    plug_in_bits: float
    miller_madow_bits: float
    H_Y_bits: float
    rho: float


def routing_information(
    omega_ids: np.ndarray,
    y: np.ndarray,
    num_classes: Optional[int] = None,
) -> Tuple[float, float, int, float]:
    """Plug-in MI and Miller-Madow corrected MI between region IDs and labels.

    Returns ``(plug_in_bits, miller_madow_bits, num_regions, H_Y_bits)``.
    """
    N = len(y)
    if len(omega_ids) != N:
        raise ValueError("omega_ids and y length mismatch")

    contingency: Dict[Tuple[bytes, int], int] = Counter()
    for w, label in zip(omega_ids, y):
        contingency[(w, int(label))] += 1

    region_ids = sorted({k[0] for k in contingency})
    class_ids = sorted({int(c) for c in np.unique(y)})
    if num_classes is None:
        num_classes = len(class_ids)

    R = len(region_ids)
    C = len(class_ids)
    region_idx = {r: i for i, r in enumerate(region_ids)}
    class_idx = {c: i for i, c in enumerate(class_ids)}

    table = np.zeros((R, C), dtype=np.int64)
    for (w, c), n in contingency.items():
        table[region_idx[w], class_idx[c]] = n

    n_omega = table.sum(axis=1)
    n_y = table.sum(axis=0)

    P_yw = table / N
    P_w = n_omega / N
    P_y = n_y / N

    denom = P_w[:, None] * P_y[None, :]
    mask = table > 0
    plug_in_bits = float(
        (P_yw[mask] * np.log2(P_yw[mask] / denom[mask])).sum()
    )

    mm_bits = plug_in_bits - (R - 1) * (num_classes - 1) / (2 * N * np.log(2))

    H_Y = float(-(P_y[P_y > 0] * np.log2(P_y[P_y > 0])).sum())
    return plug_in_bits, mm_bits, R, H_Y


# ---------------------------------------------------------------------------
# HDF5-backed estimator
# ---------------------------------------------------------------------------
class RoutingEstimator:
    """Loads a model checkpoint from the project's HDF5 layout and computes
    Recipe 1 across hidden layers.

    HDF5 layout (training output):
        metadata.attrs['architecture']          # list of hidden widths
        epochs/epoch_<N>/l<i>.weight, l<i>.bias # i = 1..L+1; output layer is L+1
        points, labels                          # stored test set
    """

    def __init__(self, h5_path: PathLike):
        self.h5_path = Path(h5_path)
        if not self.h5_path.exists():
            raise FileNotFoundError(self.h5_path)

        with h5py.File(self.h5_path, "r") as f:
            attrs = dict(f["metadata"].attrs)
            self.architecture: List[int] = list(attrs.get("architecture", []))
            self.num_hidden_layers = len(self.architecture)
            self.network_id: str = str(attrs.get("experiment_name", self.h5_path.stem))
            self.seed: int = int(attrs.get("model_seed", -1))

            points = f["points"][:]
            if points.ndim == 2 and points.shape[0] < points.shape[1]:
                points = points.T
            self.points = np.asarray(points, dtype=np.float32)
            self.labels = np.asarray(f["labels"][:], dtype=np.int64)

            self.epochs = sorted(
                int(k.split("_")[1]) for k in f["epochs"].keys() if k.startswith("epoch_")
            )

    def _load_weights(
        self, epoch: int, include_output: bool = False
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Return hidden-layer (W, b) lists, plus the output layer if requested."""
        n_layers = self.num_hidden_layers + int(include_output)
        with h5py.File(self.h5_path, "r") as f:
            grp = f[f"epochs/epoch_{epoch}"]
            W: List[np.ndarray] = []
            b: List[np.ndarray] = []
            for i in range(1, n_layers + 1):
                W.append(np.asarray(grp[f"l{i}.weight"][:], dtype=np.float32))
                b.append(np.asarray(grp[f"l{i}.bias"][:], dtype=np.float32))
        return W, b

    def evaluate_epoch(
        self,
        epoch: int,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> List[RoutingResult]:
        """Plug-in and Miller–Madow routing MI at every hidden layer of one checkpoint.

        Defaults to the test set stored in the HDF5 file; pass ``X``/``y`` to
        score a different set.
        """
        if X is None:
            X, y = self.points, self.labels
        if y is None:
            raise ValueError("y must be provided when X is provided")

        W, b = self._load_weights(epoch)
        patterns = forward_activation_patterns(W, b, X)

        N = len(y)
        num_classes = int(np.max(y)) + 1
        out: List[RoutingResult] = []
        for layer in range(1, self.num_hidden_layers + 1):
            omega = cumulative_pattern_hashes(patterns, layer)
            plug_in, mm, R, H_Y = routing_information(
                omega, y, num_classes=num_classes
            )
            out.append(
                RoutingResult(
                    layer=layer,
                    N=N,
                    num_regions=R,
                    plug_in_bits=plug_in,
                    miller_madow_bits=mm,
                    H_Y_bits=H_Y,
                    rho=R / N,
                )
            )
        return out


# ---------------------------------------------------------------------------
# Smoke-test / CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m src_experiment.routing_estimator <path/to/file.h5>")
        sys.exit(1)
    estimator = RoutingEstimator(sys.argv[1])
    for r in estimator.evaluate_epoch(estimator.epochs[-1]):
        print(r)
