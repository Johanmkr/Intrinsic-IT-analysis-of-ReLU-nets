"""
Phase A.1 MI baselines: alternative `Î(Y;T)` estimators we benchmark our
routing-information estimator against.

Each estimator returns a ``dict`` with at least ``"bits"`` (the MI estimate
in bits) and ``"wall"`` (wall-clock seconds). See
``planning/phase_a_baselines.md`` §A.1 for the protocol and validation
tolerances.

Estimators
----------
- :func:`binning_mi`        — Tishby/Saxe-style per-neuron uniform binning.
- :func:`kmeans_mi`         — k-means cluster IDs as the discrete summary.
- :func:`ksg_mi`            — KSG / Ross 2014 mixed continuous-discrete kNN.
- :class:`InfoNCEEstimator` — bilinear critic, lower bound (van den Oord 2018).
- :class:`MINEEstimator`    — Donsker-Varadhan critic (Belghazi 2018, MINE-f).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src_experiment.routing_estimator import routing_information

LOG2 = float(np.log(2.0))


# ---------------------------------------------------------------------------
# Plug-in baselines (binning, k-means)
# ---------------------------------------------------------------------------
def _quantize_per_layer(T: np.ndarray, n_bins: int) -> np.ndarray:
    """Uniform-bin each neuron over ``[-c_l, c_l]`` with ``c_l = max |T|``."""
    T = np.asarray(T, dtype=np.float32)
    c_l = float(np.max(np.abs(T)))
    if c_l == 0.0:
        return np.zeros(T.shape, dtype=np.uint8)
    edges = np.linspace(-c_l, c_l, n_bins + 1)[1:-1]
    binned = np.digitize(T, edges)
    return np.clip(binned, 0, n_bins - 1).astype(np.uint8)


def _hash_rows(M: np.ndarray) -> np.ndarray:
    """Per-row md5 digest of a ``(N, d)`` integer matrix."""
    M = np.ascontiguousarray(M)
    out = np.empty(M.shape[0], dtype=object)
    for i in range(M.shape[0]):
        out[i] = hashlib.md5(M[i].tobytes()).digest()
    return out


def binning_mi(
    T: np.ndarray,
    Y: np.ndarray,
    n_bins: int,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """Plug-in MI with Miller-Madow on uniformly-binned pre-activations."""
    t0 = time.perf_counter()
    binned = _quantize_per_layer(T, n_bins)
    digests = _hash_rows(binned)
    plug_in, mm, R, _ = routing_information(digests, np.asarray(Y), num_classes=num_classes)
    return {
        "bits": float(mm),
        "bits_plugin": float(plug_in),
        "n_bins": int(n_bins),
        "num_regions": int(R),
        "wall": time.perf_counter() - t0,
    }


def kmeans_mi(
    T: np.ndarray,
    Y: np.ndarray,
    K: int,
    seed: int = 0,
    num_classes: Optional[int] = None,
) -> Dict[str, float]:
    """K-means cluster T into K clusters; plug-in (Miller-Madow) MI on (cluster, Y)."""
    from sklearn.cluster import KMeans

    t0 = time.perf_counter()
    T = np.asarray(T, dtype=np.float32)
    K_eff = int(min(K, T.shape[0]))
    km = KMeans(n_clusters=K_eff, n_init=10, random_state=seed).fit(T)
    cluster = km.labels_.astype(np.int64)
    plug_in, mm, R, _ = routing_information(cluster, np.asarray(Y), num_classes=num_classes)
    return {
        "bits": float(mm),
        "bits_plugin": float(plug_in),
        "K": K_eff,
        "num_regions": int(R),
        "wall": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# KSG / Ross 2014 mixed continuous-discrete
# ---------------------------------------------------------------------------
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
        return {"bits": 0.0, "k": int(k), "n_used": 0,
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


# ---------------------------------------------------------------------------
# Neural critics: InfoNCE and MINE
# ---------------------------------------------------------------------------
def _onehot(y: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


@dataclass
class InfoNCEEstimator:
    """Bilinear-critic InfoNCE lower bound (van den Oord et al. 2018)."""

    lr: float = 1e-3
    n_iter: int = 1000
    batch: int = 256
    device: str = "cpu"

    def estimate(
        self,
        T: np.ndarray,
        Y: np.ndarray,
        num_classes: int,
        seed: int = 0,
    ) -> Dict[str, float]:
        import torch
        import torch.nn as nn

        t0 = time.perf_counter()
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device(self.device)
        T_t = torch.as_tensor(np.asarray(T, dtype=np.float32), device=device)
        Y_t = torch.as_tensor(np.asarray(Y).astype(np.int64), device=device)
        N, d_T = T_t.shape

        W = nn.Parameter(torch.empty(num_classes, d_T, device=device))
        nn.init.normal_(W, std=0.01)
        opt = torch.optim.Adam([W], lr=self.lr)
        history: list[float] = []
        ce = nn.CrossEntropyLoss()

        for _ in range(self.n_iter):
            B = min(self.batch, N)
            ix = torch.randint(0, N, (B,), device=device)
            t_b = T_t[ix]                        # (B, d_T)
            y_b = Y_t[ix]                        # (B,)
            W_y = W[y_b]                         # (B, d_T) — per-batch class slots
            # S[i, j] = score for pairing t_i with the class label of sample j.
            # Diagonal entries are the matched (positive) pairs; off-diagonal are
            # in-batch negatives drawn from the marginal of Y.
            S = t_b @ W_y.T                      # (B, B)
            target = torch.arange(B, device=device)
            loss = ce(S, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            mi_lb_nats = float(np.log(B)) - float(loss.item())
            history.append(mi_lb_nats / LOG2)

        bits = float(np.mean(history[-100:]))
        return {
            "bits": bits,
            "history": np.asarray(history, dtype=np.float32),
            "wall": time.perf_counter() - t0,
        }


@dataclass
class MINEEstimator:
    """Donsker-Varadhan MINE-f estimator (Belghazi et al. 2018)."""

    hidden: int = 256
    lr: float = 5e-4
    n_iter: int = 2000
    batch: int = 256
    ema_decay: float = 0.99
    device: str = "cpu"

    def estimate(
        self,
        T: np.ndarray,
        Y: np.ndarray,
        num_classes: int,
        seed: int = 0,
    ) -> Dict[str, float]:
        import torch
        import torch.nn as nn

        t0 = time.perf_counter()
        torch.manual_seed(seed)
        np.random.seed(seed)
        device = torch.device(self.device)
        T_t = torch.as_tensor(np.asarray(T, dtype=np.float32), device=device)
        Y_oh = torch.as_tensor(_onehot(np.asarray(Y).astype(np.int64), num_classes),
                               device=device)
        N, d_T = T_t.shape
        d_in = d_T + num_classes

        critic = nn.Sequential(
            nn.Linear(d_in, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, self.hidden), nn.ReLU(),
            nn.Linear(self.hidden, 1),
        ).to(device)
        opt = torch.optim.Adam(critic.parameters(), lr=self.lr, betas=(0.5, 0.999))

        ema_E_exp_T: Optional[torch.Tensor] = None
        history: list[float] = []

        for _ in range(self.n_iter):
            B = min(self.batch, N)
            ix = torch.randint(0, N, (B,), device=device)
            iy = torch.randint(0, N, (B,), device=device)
            joint_in = torch.cat([T_t[ix], Y_oh[ix]], dim=1)
            marg_in = torch.cat([T_t[ix], Y_oh[iy]], dim=1)
            T_j = critic(joint_in).squeeze(-1)
            T_m = critic(marg_in).squeeze(-1)

            E_T_j = T_j.mean()
            E_exp_T_m = torch.exp(T_m).mean()

            # MINE-f gradient-bias correction: substitute E[exp T] / EMA into
            # the gradient path so the stochastic gradient is unbiased
            # (Belghazi 2018 App C).
            if ema_E_exp_T is None:
                ema_E_exp_T = E_exp_T_m.detach()
            else:
                ema_E_exp_T = (
                    self.ema_decay * ema_E_exp_T
                    + (1.0 - self.ema_decay) * E_exp_T_m.detach()
                )
            loss = -(E_T_j - E_exp_T_m / ema_E_exp_T.detach())
            opt.zero_grad()
            loss.backward()
            opt.step()

            # Reported estimate is the un-corrected DV bound (in bits).
            mi_dv_nats = float((E_T_j - torch.log(E_exp_T_m)).detach().item())
            history.append(mi_dv_nats / LOG2)

        bits = float(np.mean(history[-200:]))
        return {
            "bits": bits,
            "history": np.asarray(history, dtype=np.float32),
            "wall": time.perf_counter() - t0,
        }
