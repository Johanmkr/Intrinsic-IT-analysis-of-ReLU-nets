"""Bias-corrected plug-in estimators of I(Y; Ω) from a region × class count table.

Every estimator here is a discrete entropy estimator Ĥ applied to the three
count vectors of the contingency table and combined as

    Î(Y; Ω) = Ĥ(Ω) + Ĥ(Y) − Ĥ(Ω, Y),

which is how ``infomeasure`` 0.6.2 (used for rebuttal Exp 10) builds its MI
estimates; ``tests/test_estimators.py`` checks the two against each other.
The exception is Miller–Madow, which uses the paper's MI-level correction
Î − (|Ω| − 1)(|Y| − 1) / (2N ln 2), as in
:func:`routing_estimator.routing_information`.

Entropies are computed in nats and MI is returned in bits.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.special import digamma

LN2 = np.log(2.0)
EULER_GAMMA = 0.5772156649015329

ESTIMATORS = ("plug_in", "miller_madow", "grassberger", "chao_shen", "chao_wang_jost", "ansb")


def _counts(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, dtype=np.int64).ravel()
    return c[c > 0]


def entropy_plug_in(counts: np.ndarray) -> float:
    n = _counts(counts)
    p = n / n.sum()
    return float(-(p * np.log(p)).sum())


def entropy_miller_madow(counts: np.ndarray) -> float:
    n = _counts(counts)
    return entropy_plug_in(n) + (len(n) - 1) / (2 * n.sum())


def entropy_grassberger(counts: np.ndarray) -> float:
    """Grassberger (2003): mean over samples of ln N − ψ(n_i) − (−1)^{n_i}/(n_i + 1)."""
    n = _counts(counts)
    N = n.sum()
    local = np.log(N) - digamma(n) - (-1.0) ** n / (n + 1)
    return float((n * local).sum() / N)


def entropy_chao_shen(counts: np.ndarray) -> float:
    """Chao & Shen (2003): coverage-adjusted Horvitz–Thompson estimator."""
    n = _counts(counts)
    N = n.sum()
    f1 = int((n == 1).sum())
    if f1 == N:
        f1 -= 1  # avoid zero coverage, as infomeasure does
    pa = (1 - f1 / N) * n / N
    la = 1 - (1 - pa) ** N
    return float(-(pa * np.log(pa) / la).sum())


def _cwj_tail(A: float, N: int) -> float:
    """(1 − A)^{1−N} · (−ln A − Σ_{r=1}^{N−1} (1 − A)^r / r).

    The bracket is the tail Σ_{r≥N} z^r / r of −ln A = Σ_{r≥1} z^r / r with
    z = 1 − A, so the product equals Σ_{k≥1} z^k / (k + N − 1), which stays
    finite where the literal formula overflows or cancels. The literal formula
    is used only for A·N < 1, where z^N ≈ e^{−AN} is not small and the
    subtraction loses at most a couple of digits.
    """
    z = 1.0 - A
    if A * N < 1.0:
        r = np.arange(1, N, dtype=np.float64)
        bracket = -np.log(A) - np.sum(z ** r / r)
        return float(np.exp((1 - N) * np.log1p(-A)) * bracket)
    k = np.arange(1, int(np.ceil(40.0 / A)) + 1, dtype=np.float64)
    return float(np.sum(z ** k / (k + N - 1)))


def entropy_chao_wang_jost(counts: np.ndarray) -> float:
    """Chao, Wang & Jost (2013)."""
    n = _counts(counts)
    N = int(n.sum())
    f1 = int((n == 1).sum())
    f2 = int((n == 2).sum())
    if f2 > 0:
        A = 2 * f2 / ((N - 1) * f1 + 2 * f2)
    elif f1 > 0:
        A = 2 / ((N - 1) * (f1 - 1) + 2)
    else:
        A = 1.0
    h = float((n * (digamma(N) - digamma(n))).sum() / N)
    if A != 1 and f1 > 0:
        h += f1 / N * _cwj_tail(A, N)
    return h


def entropy_ansb(counts: np.ndarray) -> float:
    """Asymptotic NSB (Nemenman 2011): (γ − ln 2) + 2 ln N − ψ(N − K).

    Undefined (NaN) without coincidences (K = N). Only valid when strongly
    undersampled (N/K ≪ 1), a regime this estimator's callers report but
    never reach, since K ≤ N always.
    """
    n = _counts(counts)
    N, K = int(n.sum()), len(n)
    if N == K:
        return float("nan")
    return float((EULER_GAMMA - np.log(2)) + 2 * np.log(N) - digamma(N - K))


_ENTROPY = {
    "plug_in": entropy_plug_in,
    "miller_madow": entropy_miller_madow,
    "grassberger": entropy_grassberger,
    "chao_shen": entropy_chao_shen,
    "chao_wang_jost": entropy_chao_wang_jost,
    "ansb": entropy_ansb,
}


def mutual_information_bits(table: np.ndarray, estimator: str) -> float:
    """Î(Y; Ω) in bits from an (R, C) table of region × class counts."""
    table = np.asarray(table)
    if estimator == "miller_madow":
        R, C = (table.sum(axis=1) > 0).sum(), (table.sum(axis=0) > 0).sum()
        return mutual_information_bits(table, "plug_in") - (R - 1) * (C - 1) / (2 * table.sum() * LN2)
    H = _ENTROPY[estimator]
    return (H(table.sum(axis=1)) + H(table.sum(axis=0)) - H(table)) / LN2


def all_mutual_information_bits(table: np.ndarray) -> Dict[str, float]:
    """Every estimator in :data:`ESTIMATORS`, keyed ``<name>_bits``."""
    return {f"{name}_bits": mutual_information_bits(table, name) for name in ESTIMATORS}


def contingency_table(omega_ids: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(R, C) count table of region IDs (any hashable) against labels."""
    _, r = np.unique(np.asarray(omega_ids), return_inverse=True)
    _, c = np.unique(np.asarray(y), return_inverse=True)
    table = np.zeros((r.max() + 1, c.max() + 1), dtype=np.int64)
    np.add.at(table, (r, c), 1)
    return table
