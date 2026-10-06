"""MI baselines on continuous hidden-layer activations T (step 3).

- :func:`quantize_per_layer` — Tishby/Saxe-style per-neuron uniform binning;
  the resulting cells (and k-means clusters, see run_baselines.py) are scored
  with the discrete estimators of :mod:`src_experiment.estimators`.
- :func:`ksg_mi` — KSG / Ross (2014) mixed continuous–discrete kNN estimator.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

LOG2 = float(np.log(2.0))


def quantize_per_layer(T: np.ndarray, n_bins: int) -> np.ndarray:
    """Uniform-bin each neuron over ``[-c_l, c_l]`` with ``c_l = max |T|``."""
    T = np.asarray(T, dtype=np.float32)
    c_l = float(np.max(np.abs(T)))
    if c_l == 0.0:
        return np.zeros(T.shape, dtype=np.uint8)
    edges = np.linspace(-c_l, c_l, n_bins + 1)[1:-1]
    binned = np.digitize(T, edges)
    return np.clip(binned, 0, n_bins - 1).astype(np.uint8)


def ksg_mi(
    T: np.ndarray,
    Y: np.ndarray,
    k: int = 3,
) -> Dict[str, float]:
    """KSG / Ross 2014 mixed continuous-discrete MI estimator.

    For continuous ``T ∈ R^{N×d}`` and discrete ``Y``:

        Î(T;Y) = ψ(N) + ⟨ψ(k_i)⟩ − ⟨ψ(N_{y_i})⟩ − ⟨ψ(m_i + 1)⟩    [nats]

    where ``k_i = min(k, N_{y_i} − 1)`` is the within-class neighbour count,
    the radius ``d_i`` is the Chebyshev distance to the ``k_i``-th
    same-class NN of point ``i``, and ``m_i`` is the count of *other* points
    (any class) strictly inside that radius. Singleton-label samples (no
    same-class neighbour) are dropped, mirroring sklearn's
    ``_compute_mi_cd``.
    """
    from scipy.special import digamma
    from sklearn.neighbors import KDTree

    t0 = time.perf_counter()
    T = np.asarray(T, dtype=np.float64)
    if T.ndim == 1:
        T = T.reshape(-1, 1)
    Y = np.asarray(Y).ravel()
    N = T.shape[0]

    radius = np.zeros(N)
    label_counts = np.zeros(N, dtype=np.int64)
    k_per_sample = np.zeros(N, dtype=np.int64)
    valid = np.zeros(N, dtype=bool)

    for c in np.unique(Y):
        mask = (Y == c)
        n_c = int(mask.sum())
        if n_c <= 1:
            continue
        k_eff = min(k, n_c - 1)
        tree_c = KDTree(T[mask], metric="chebyshev")
        d, _ = tree_c.query(T[mask], k=k_eff + 1)  # includes self at d=0
        # nudge to open ball (Ross's m_i is strict <)
        radius[mask] = np.nextafter(d[:, -1], 0.0)
        k_per_sample[mask] = k_eff
        label_counts[mask] = n_c
        valid[mask] = True

    if not valid.any():
        return {"bits": 0.0, "bits_signed": 0.0, "k": int(k), "n_used": 0,
                "wall": time.perf_counter() - t0}

    T_v = T[valid]
    radius_v = radius[valid]
    k_v = k_per_sample[valid]
    Ny_v = label_counts[valid]
    n_used = T_v.shape[0]

    global_tree = KDTree(T_v, metric="chebyshev")
    m_counts = global_tree.query_radius(T_v, radius_v, count_only=True)
    m = np.asarray(m_counts, dtype=np.float64) - 1.0  # exclude self

    mi_nats = (
        digamma(n_used)
        + np.mean(digamma(k_v))
        - np.mean(digamma(Ny_v))
        - np.mean(digamma(m + 1.0))
    )
    return {
        "bits": float(max(0.0, mi_nats / LOG2)),
        "bits_signed": float(mi_nats / LOG2),
        "k": int(k),
        "n_used": int(n_used),
        "wall": time.perf_counter() - t0,
    }
