"""
Recipes 2 & 3:

- Recipe 2: functional-equivalence quotient via the active subnetwork matrix
  $\\tilde A^l_\\omega$ and ε-tolerance clustering.
- Recipe 3: quotient MI estimator $\\tilde I_{\\mathrm{func}}$ -- Recipe 1 applied
  to merged contingency rows where regions sharing a quotient class are pooled.

Reuses :mod:`src_experiment.routing_estimator` (Recipe 1).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src_experiment.estimators import (
    all_mutual_information_bits,
    contingency_table,
)
from src_experiment.routing_estimator import (
    RoutingEstimator,
    cumulative_pattern_hashes,
    forward_activation_patterns,
    routing_information,
)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Active subnetwork matrix
# ---------------------------------------------------------------------------
def compute_active_subnetwork(
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    per_layer_patterns: Sequence[np.ndarray],
    layer: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Active subnetwork matrix and bias for one region at depth ``layer``.

    Recursion: $\\tilde A_i = W^i[S_i, S_{i-1}]\\,\\tilde A_{i-1}$,
    $\\tilde c_i = W^i[S_i, S_{i-1}]\\,\\tilde c_{i-1} + b^i[S_i]$,
    initialized $\\tilde A_0 = I_{n_0}$, $\\tilde c_0 = 0$.

    Parameters
    ----------
    weights, biases
        Hidden-layer weight/bias arrays (PyTorch convention, ``W[i].shape == (n_{i+1}, n_i)``).
    per_layer_patterns
        Per-layer activation patterns for *one* region; ``len >= layer``.
    layer
        1-indexed depth.

    Returns
    -------
    tilde_A : ``(|S_l|, n_0)`` float array.
    tilde_c : ``(|S_l|,)``    float array.
    S_l     : indices of active output neurons at layer ``layer``.
    """
    if layer < 1:
        raise ValueError("layer must be >= 1")
    if layer > len(weights):
        raise ValueError(f"layer {layer} exceeds depth {len(weights)}")

    n_0 = weights[0].shape[1]
    dtype = weights[0].dtype
    S_prev = np.arange(n_0)
    tilde_A = np.eye(n_0, dtype=dtype)
    tilde_c = np.zeros(n_0, dtype=dtype)

    for i in range(1, layer + 1):
        pi = np.asarray(per_layer_patterns[i - 1], dtype=bool)
        S_curr = np.where(pi)[0]
        W_step = weights[i - 1][np.ix_(S_curr, S_prev)]
        b_step = biases[i - 1][S_curr]
        tilde_A = W_step @ tilde_A
        tilde_c = W_step @ tilde_c + b_step
        S_prev = S_curr

    return tilde_A, tilde_c, S_prev


def collect_unique_region_patterns(
    per_layer_patterns: Sequence[np.ndarray],
    omega_ids: np.ndarray,
    layer: int,
) -> Dict[bytes, List[np.ndarray]]:
    """Map each unique region ID to a representative's per-layer patterns up to ``layer``.

    All samples sharing a hash share the cumulative pattern by construction, so any
    sample's per-layer slices serve as the representative.
    """
    first_idx: Dict[bytes, int] = {}
    for i, w in enumerate(omega_ids):
        if w not in first_idx:
            first_idx[w] = i

    out: Dict[bytes, List[np.ndarray]] = {}
    for rid, i in first_idx.items():
        out[rid] = [
            np.asarray(per_layer_patterns[k][i], dtype=bool) for k in range(layer)
        ]
    return out


# ---------------------------------------------------------------------------
# ε-tolerance clustering
# ---------------------------------------------------------------------------
@dataclass
class _RegionActive:
    tilde_A: np.ndarray
    tilde_c: np.ndarray
    S_l: np.ndarray


def _build_active_data(
    weights: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
    region_patterns: Dict[bytes, List[np.ndarray]],
    layer: int,
) -> Dict[bytes, _RegionActive]:
    out: Dict[bytes, _RegionActive] = {}
    for rid, patterns in region_patterns.items():
        tA, tc, S_l = compute_active_subnetwork(weights, biases, patterns, layer)
        out[rid] = _RegionActive(tA, tc, S_l)
    return out


def _relative_frob(A1: np.ndarray, A2: np.ndarray) -> float:
    """Relative Frobenius distance between two matrices."""
    denom = 0.5 * (np.linalg.norm(A1, "fro") + np.linalg.norm(A2, "fro"))
    if denom < 1e-12:
        return 0.0
    return float(np.linalg.norm(A1 - A2, "fro") / denom)


def are_functionally_equivalent(A1: np.ndarray, A2: np.ndarray, eps: float) -> bool:
    """True if A1 and A2 implement the same linear transformation up to relative
    tolerance eps.  Bias is excluded by design."""
    return _relative_frob(A1, A2) <= eps


def cluster_functional(
    active_data: Dict[bytes, _RegionActive],
    epsilon: float,
) -> Tuple[Dict[bytes, int], int]:
    """ε-tolerance functional-equivalence clustering.

    Buckets first by ``S_l`` (matching active output set is necessary), then
    naive O(k²) relative-Frobenius comparison of tilde_A within each bucket.
    Bias tilde_c is excluded from the criterion (scale-invariant definition).
    Uses ``<=`` so ``epsilon=0`` recovers numerical equality.
    """
    buckets: Dict[Tuple[int, bytes], List[bytes]] = defaultdict(list)
    for rid, ra in active_data.items():
        key = (len(ra.S_l), ra.S_l.tobytes())
        buckets[key].append(rid)

    quotient_map: Dict[bytes, int] = {}
    next_qid = 0
    for rids in buckets.values():
        reps: List[Tuple[int, np.ndarray]] = []
        for rid in rids:
            ra = active_data[rid]
            assigned = False
            for ref_qid, ref_A in reps:
                if are_functionally_equivalent(ra.tilde_A, ref_A, epsilon):
                    quotient_map[rid] = ref_qid
                    assigned = True
                    break
            if not assigned:
                quotient_map[rid] = next_qid
                reps.append((next_qid, ra.tilde_A))
                next_qid += 1

    return quotient_map, next_qid


# ---------------------------------------------------------------------------
# Recipe 3: quotient MI
# ---------------------------------------------------------------------------
def routing_information_quotient(
    omega_ids: np.ndarray,
    y: np.ndarray,
    quotient_map: Dict[bytes, int],
    num_classes: Optional[int] = None,
) -> Tuple[float, float, int, float]:
    """Recipe 3: replace ``omega_ids`` with their quotient class IDs and apply Recipe 1."""
    qids = np.fromiter(
        (quotient_map[w] for w in omega_ids), dtype=np.int64, count=len(omega_ids)
    )
    return routing_information(qids, y, num_classes=num_classes)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
DEFAULT_EPSILONS: Tuple[float, ...] = (
    0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0
)


def network_accuracy(
    weights: Sequence[np.ndarray], biases: Sequence[np.ndarray], X: np.ndarray, y: np.ndarray
) -> float:
    """Fraction of ``X`` whose argmax output equals ``y`` (all layers incl. output)."""
    a = np.asarray(X, dtype=np.float32)
    for W, b in zip(weights[:-1], biases[:-1]):
        a = np.maximum(a @ W.T + b, 0.0)
    logits = a @ weights[-1].T + biases[-1]
    return float((logits.argmax(axis=1) == y).mean())


def _estimates(omega_ids: np.ndarray, y: np.ndarray, suffix: str) -> Dict[str, float]:
    table = contingency_table(omega_ids, y)
    return {k.replace("_bits", f"{suffix}_bits"): v
            for k, v in all_mutual_information_bits(table).items()}


class FunctionalQuotientEstimator:
    """Routing MI (raw and ε-quotient) of one trained network, all estimators.

    Composes :class:`RoutingEstimator` for HDF5 I/O. One output row per
    (epoch, hidden layer, ε) with, for every estimator in
    :data:`src_experiment.estimators.ESTIMATORS`, the raw ``<name>_bits`` and
    the quotient ``<name>_func_bits``.
    """

    def __init__(self, h5_path: PathLike):
        self.routing = RoutingEstimator(h5_path)
        self.h5_path = self.routing.h5_path
        self.architecture = self.routing.architecture
        self.num_hidden_layers = self.routing.num_hidden_layers
        self.network_id = self.routing.network_id
        self.seed = self.routing.seed
        self.epochs = self.routing.epochs
        self.points = self.routing.points
        self.labels = self.routing.labels

    def evaluate_epoch(
        self,
        epoch: int,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        epsilons: Sequence[float] = DEFAULT_EPSILONS,
    ) -> List[dict]:
        if X is None:
            X, y = self.points, self.labels
        if y is None:
            raise ValueError("y must be provided when X is provided")

        W, b = self.routing._load_weights(epoch, include_output=True)
        accuracy = network_accuracy(W, b, X, y)
        W, b = W[:-1], b[:-1]
        patterns = forward_activation_patterns(W, b, X)
        N = len(y)
        H_Y = routing_information(np.zeros(N, dtype=np.int64), y)[3]

        rows: List[dict] = []
        for layer in range(1, self.num_hidden_layers + 1):
            omega = cumulative_pattern_hashes(patterns, layer)
            raw = _estimates(omega, y, "")
            _, region_sizes = np.unique(omega, return_counts=True)
            R = len(region_sizes)

            region_patterns = collect_unique_region_patterns(patterns, omega, layer)
            active_data = _build_active_data(W, b, region_patterns, layer)

            for eps in epsilons:
                quotient_map, num_q = cluster_functional(active_data, eps)
                qids = np.fromiter((quotient_map[w] for w in omega), dtype=np.int64, count=N)
                rows.append({
                    "epoch": epoch,
                    "layer": layer,
                    "epsilon": float(eps),
                    "N": N,
                    "H_Y_bits": H_Y,
                    "accuracy": accuracy,
                    "num_regions": R,
                    "rho": R / N,
                    "singleton_region_frac": float((region_sizes == 1).mean()),
                    "singleton_sample_frac": float((region_sizes == 1).sum() / N),
                    "num_quotient": num_q,
                    "rho_func": num_q / R,
                    **raw,
                    **_estimates(qids, y, "_func"),
                })
        return rows

    def evaluate_all(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        epsilons: Sequence[float] = DEFAULT_EPSILONS,
    ) -> pd.DataFrame:
        rows = []
        for ep in self.epochs:
            for r in self.evaluate_epoch(ep, X=X, y=y, epsilons=epsilons):
                rows.append({"network_id": self.network_id, "seed": self.seed, **r})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(
            "usage: python -m src_experiment.functional_quotient <path/to/file.h5>"
        )
        sys.exit(1)
    estimator = FunctionalQuotientEstimator(sys.argv[1])
    df = estimator.evaluate_all()
    print(df.to_string(index=False))
