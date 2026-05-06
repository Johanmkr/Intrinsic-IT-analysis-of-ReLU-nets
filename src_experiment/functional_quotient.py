"""
Recipes 2 & 3 from `claude_new_estimator_instructions.md`:

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

from src_experiment.routing_estimator import (
    RoutingEstimator,
    cumulative_pattern_hashes,
    forward_activation_patterns,
    routing_information,
    truncation_probability,
)
from src_experiment.rtg_analyzer import (
    cumulative_patterns_per_region,
    hamming1_adjacency,
    rtg_diagnostics,
)
from src_experiment.rtg_overlap import (
    region_dominant_class,
    routing_loss_proxy,
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
@dataclass
class QuotientResult:
    layer: int
    epsilon: float
    N: int
    num_regions: int
    num_quotient: int
    rho: float
    rho_func: float
    plug_in_bits: float
    miller_madow_bits: float
    plug_in_func_bits: float
    miller_madow_func_bits: float
    H_Y_bits: float
    truncation_prob: float
    num_rtg_components: int
    rtg_largest_component_frac: float
    rtg_isolated_frac: float
    rl_proxy: float


DEFAULT_EPSILONS: Tuple[float, ...] = (0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1e-1)


class FunctionalQuotientEstimator:
    """Recipe 2 + Recipe 3 driver. Composes :class:`RoutingEstimator` for HDF5 I/O."""

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
        X_holdout: Optional[np.ndarray] = None,
        y_holdout: Optional[np.ndarray] = None,
        epsilons: Sequence[float] = DEFAULT_EPSILONS,
    ) -> List[QuotientResult]:
        if X is None:
            X, y = self.points, self.labels
        if y is None:
            raise ValueError("y must be provided when X is provided")

        W, b = self.routing._load_weights(epoch)
        patterns = forward_activation_patterns(W, b, X)

        holdout_patterns = None
        if X_holdout is not None:
            if y_holdout is None:
                raise ValueError("y_holdout must accompany X_holdout")
            holdout_patterns = forward_activation_patterns(W, b, X_holdout)

        N = len(y)
        num_classes = int(np.max(y)) + 1
        out: List[QuotientResult] = []

        for layer in range(1, self.num_hidden_layers + 1):
            omega = cumulative_pattern_hashes(patterns, layer)
            plug_in, mm, R, H_Y = routing_information(omega, y, num_classes=num_classes)
            rho = R / N

            tp = float("nan")
            if holdout_patterns is not None:
                omega_h = cumulative_pattern_hashes(holdout_patterns, layer)
                tp = truncation_probability(omega, omega_h)

            region_patterns = collect_unique_region_patterns(patterns, omega, layer)
            active_data = _build_active_data(W, b, region_patterns, layer)

            # Recipe 4: ε-independent, computed once per (epoch, layer)
            cum_patterns = cumulative_patterns_per_region(patterns, omega, layer)
            adjacency = hamming1_adjacency(cum_patterns)
            rtg = rtg_diagnostics(adjacency)

            # Experiment 3: routing-loss proxy (also ε-independent)
            dominant = region_dominant_class(omega, y)
            rl = routing_loss_proxy(adjacency, dominant)

            for eps in epsilons:
                quotient_map, num_q = cluster_functional(active_data, eps)
                pi_func, mm_func, _, _ = routing_information_quotient(
                    omega, y, quotient_map, num_classes=num_classes
                )
                out.append(
                    QuotientResult(
                        layer=layer,
                        epsilon=float(eps),
                        N=N,
                        num_regions=R,
                        num_quotient=num_q,
                        rho=rho,
                        rho_func=num_q / R if R > 0 else float("nan"),
                        plug_in_bits=plug_in,
                        miller_madow_bits=mm,
                        plug_in_func_bits=pi_func,
                        miller_madow_func_bits=mm_func,
                        H_Y_bits=H_Y,
                        truncation_prob=tp,
                        num_rtg_components=rtg.num_components,
                        rtg_largest_component_frac=rtg.largest_component_frac,
                        rtg_isolated_frac=rtg.isolated_frac,
                        rl_proxy=rl,
                    )
                )
        return out

    def evaluate_all(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        X_holdout: Optional[np.ndarray] = None,
        y_holdout: Optional[np.ndarray] = None,
        epsilons: Sequence[float] = DEFAULT_EPSILONS,
    ) -> pd.DataFrame:
        rows = []
        for ep in self.epochs:
            for r in self.evaluate_epoch(
                ep,
                X=X,
                y=y,
                X_holdout=X_holdout,
                y_holdout=y_holdout,
                epsilons=epsilons,
            ):
                rows.append(
                    {
                        "network_id": self.network_id,
                        "epoch": ep,
                        "layer": r.layer,
                        "epsilon": r.epsilon,
                        "seed": self.seed,
                        "N": r.N,
                        "num_regions": r.num_regions,
                        "num_quotient": r.num_quotient,
                        "rho": r.rho,
                        "rho_func": r.rho_func,
                        "plug_in_bits": r.plug_in_bits,
                        "miller_madow_bits": r.miller_madow_bits,
                        "plug_in_func_bits": r.plug_in_func_bits,
                        "miller_madow_func_bits": r.miller_madow_func_bits,
                        "H_Y_bits": r.H_Y_bits,
                        "truncation_prob": r.truncation_prob,
                        "num_rtg_components": r.num_rtg_components,
                        "rtg_largest_component_frac": r.rtg_largest_component_frac,
                        "rtg_isolated_frac": r.rtg_isolated_frac,
                        "rl_proxy": r.rl_proxy,
                    }
                )
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
