"""Tests for src_experiment/estimators.py.

The comparison with infomeasure 0.6.2 (the package rebuttal Exp 10 used) runs
only when it is installed: ``uv sync --group rebuttal``.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import digamma

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src_experiment.estimators import (  # noqa: E402
    ESTIMATORS,
    _cwj_tail,
    all_mutual_information_bits,
    contingency_table,
    entropy_ansb,
    entropy_grassberger,
    entropy_plug_in,
    mutual_information_bits,
)
from src_experiment.routing_estimator import routing_information  # noqa: E402


def sample(N: int, R: int, C: int, seed: int):
    """Region IDs and labels with some dependence between the two."""
    rng = np.random.default_rng(seed)
    omega = rng.integers(0, R, size=N)
    y = np.where(rng.random(N) < 0.6, omega % C, rng.integers(0, C, size=N))
    return omega, y


# (N, number of possible regions, classes): well sampled → near-singleton.
REGIMES = [(2000, 10, 7), (2000, 300, 7), (569, 400, 2), (1000, 5000, 10), (114, 100, 2)]


@pytest.mark.parametrize("N,R,C", REGIMES)
def test_plug_in_and_mm_match_routing_information(N, R, C):
    omega, y = sample(N, R, C, seed=0)
    ids = np.array([str(w).encode() for w in omega])
    plug_in, mm, n_regions, _ = routing_information(ids, y)
    table = contingency_table(ids, y)
    assert table.shape[0] == n_regions
    assert mutual_information_bits(table, "plug_in") == pytest.approx(plug_in, abs=1e-12)
    assert mutual_information_bits(table, "miller_madow") == pytest.approx(mm, abs=1e-12)


def test_hand_computed_values():
    # Each region holds one class, uniformly: I = H(Y) = log2 4 for the plug-in.
    table = np.diag([5, 5, 5, 5])
    assert mutual_information_bits(table, "plug_in") == pytest.approx(2.0)
    # Grassberger, two singletons: ln 2 − ψ(1) + 1/2 = ln 2 + γ + 1/2.
    assert entropy_grassberger([1, 1]) == pytest.approx(np.log(2) - digamma(1) + 0.5)
    assert entropy_plug_in([3, 1]) == pytest.approx(-(0.75 * np.log(0.75) + 0.25 * np.log(0.25)))
    # ANSB is undefined without coincidences.
    assert np.isnan(entropy_ansb([1, 1, 1]))


def test_independent_labels_give_near_zero_mi():
    rng = np.random.default_rng(1)
    omega, y = rng.integers(0, 5, 20000), rng.integers(0, 3, 20000)
    out = all_mutual_information_bits(contingency_table(omega, y))
    # ANSB assumes N/K << 1; on well-sampled data it is meaningless (≈ 14 bits here).
    out.pop("ansb_bits")
    for name, value in out.items():
        assert abs(value) < 0.01, name


@pytest.mark.parametrize("A,N", [(0.5, 50), (0.01, 500), (1e-4, 2000), (0.2, 300), (0.9, 30), (1e-6, 10000)])
def test_cwj_tail_matches_high_precision_literal_formula(A, N):
    # In double precision the literal formula cancels catastrophically once
    # (1 − A)^N is small (e.g. A = 0.2, N = 300 gives −2e13); 60 digits do not.
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    a = mp.mpf(A)
    tail = -mp.log(a) - mp.fsum((1 - a) ** r / r for r in range(1, N))
    reference = float((1 - a) ** (1 - N) * tail)
    assert _cwj_tail(A, N) == pytest.approx(reference, rel=1e-9)


def test_cwj_tail_finite_where_literal_overflows():
    # (1 − A)^{1 − N} = 2^{9999} overflows a double.
    value = _cwj_tail(0.5, 10000)
    assert np.isfinite(value) and 0 < value < 1e-3


def test_all_estimators_reported():
    out = all_mutual_information_bits(np.diag([3, 4]))
    assert set(out) == {f"{name}_bits" for name in ESTIMATORS}


@pytest.mark.parametrize("N,R,C", REGIMES)
@pytest.mark.parametrize("name", ["grassberger", "chao_shen", "chao_wang_jost", "ansb"])
def test_matches_infomeasure(N, R, C, name):
    im = pytest.importorskip("infomeasure")
    omega, y = sample(N, R, C, seed=2)
    ours = mutual_information_bits(contingency_table(omega, y), name)
    try:
        theirs = float(im.mutual_information(omega.tolist(), y.tolist(), approach=name, base=2))
    except OverflowError:
        pytest.skip("infomeasure overflows here (literal CWJ tail); see test_cwj_tail_*")
    if np.isnan(theirs):
        assert np.isnan(ours)
    else:
        assert ours == pytest.approx(theirs, rel=1e-9, abs=1e-12)
