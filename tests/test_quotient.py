"""Functional quotient (src_experiment/functional_quotient.py) on constructed networks."""
import numpy as np
import pytest

from src_experiment.functional_quotient import (
    FunctionalQuotientEstimator,
    _RegionActive,
    _relative_frob,
    cluster_functional,
    collect_unique_region_patterns,
    compute_active_subnetwork,
)
from src_experiment.routing_estimator import cumulative_pattern_hashes, forward_activation_patterns


def random_net(widths, seed=0):
    rng = np.random.default_rng(seed)
    W = [rng.normal(size=(o, i)).astype(np.float64) for i, o in zip(widths[:-1], widths[1:])]
    b = [rng.normal(size=o).astype(np.float64) for o in widths[1:]]
    return W, b


def test_active_subnetwork_is_the_local_affine_map():
    # Inside a region the network is affine: the active units at layer l equal
    # tilde_A x + tilde_c for every point of that region.
    W, b = random_net([3, 8, 8, 8])
    X = np.random.default_rng(1).uniform(-1, 1, size=(300, 3))
    patterns = forward_activation_patterns(W, b, X)
    for layer in (1, 2, 3):
        a = X
        for Wi, bi in zip(W[:layer], b[:layer]):
            a = np.maximum(a @ Wi.T + bi, 0.0)
        for i in range(0, 300, 37):
            A, c, S = compute_active_subnetwork(W, b, [p[i] for p in patterns], layer)
            np.testing.assert_allclose(A @ X[i] + c, a[i, S], rtol=1e-10, atol=1e-10)
            assert np.all(a[i, np.setdiff1d(np.arange(a.shape[1]), S)] == 0)


def test_relative_frobenius_is_bounded():
    rng = np.random.default_rng(2)
    for _ in range(200):
        A1, A2 = rng.normal(size=(4, 3)), rng.normal(size=(4, 3)) * rng.uniform(0, 5)
        assert 0.0 <= _relative_frob(A1, A2) <= 2.0 + 1e-12
    assert _relative_frob(A1, -A1) == pytest.approx(2.0)
    assert _relative_frob(A1, A1) == 0.0


def region(A, S=(0, 1)):
    return _RegionActive(np.asarray(A, dtype=float), np.zeros(len(S)), np.asarray(S))


def test_epsilon_zero_merges_only_identical_maps_and_never_across_active_sets():
    I = np.eye(2)
    data = {b"a": region(I), b"b": region(I), b"c": region(1.001 * I), b"d": region(I[:1], S=(0,))}
    qmap, n = cluster_functional(data, 0.0)
    assert qmap[b"a"] == qmap[b"b"] and qmap[b"c"] != qmap[b"a"] and n == 3
    qmap, n = cluster_functional(data, 2.0)  # everything with the same active set merges ...
    assert qmap[b"a"] == qmap[b"b"] == qmap[b"c"] and qmap[b"d"] != qmap[b"a"] and n == 2  # ... but not across sets


def test_first_encounter_order_decides_non_transitive_chains():
    # d(A, B) = d(B, C) ≈ 0.18 ≤ ε < d(A, C) ≈ 0.36: greedy leader clustering
    # depends on which region is visited first (the paper's first-encounter order).
    A, B, C = region(np.eye(2)), region(1.2 * np.eye(2)), region(1.44 * np.eye(2))
    eps = 0.2
    q1, n1 = cluster_functional({b"A": A, b"B": B, b"C": C}, eps)
    assert n1 == 2 and q1[b"A"] == q1[b"B"] != q1[b"C"]
    q2, n2 = cluster_functional({b"B": B, b"A": A, b"C": C}, eps)
    assert n2 == 1


def test_regions_are_visited_in_first_encounter_order():
    W, b = random_net([2, 6, 6])
    X = np.random.default_rng(3).uniform(-1, 1, size=(200, 2))
    patterns = forward_activation_patterns(W, b, X)
    omega = cumulative_pattern_hashes(patterns, 2)
    first_seen = list(dict.fromkeys(omega.tolist()))
    assert list(collect_unique_region_patterns(patterns, omega, 2)) == first_seen


def test_estimator_rows_on_a_trained_network(tiny_h5):
    est = FunctionalQuotientEstimator(tiny_h5)
    rows = est.evaluate_epoch(1, epsilons=(0.0, 0.3, 2.0))
    assert len(rows) == 3 * 3  # layers × ε
    for r in rows:
        assert 1 <= r["num_quotient"] <= r["num_regions"]
        assert r["plug_in_func_bits"] <= r["plug_in_bits"] + 1e-12  # merging never adds information
        assert r["plug_in_bits"] <= r["H_Y_bits"] + 1e-12
        assert 0.0 <= r["accuracy"] <= 1.0
    layer1 = [r for r in rows if r["layer"] == 1]
    assert all(r["rho_func"] == 1.0 for r in layer1)  # one layer: distinct patterns are distinct maps
    by_layer = {}
    for r in rows:
        by_layer.setdefault(r["layer"], []).append(r["plug_in_bits"])
    assert all(len(set(v)) == 1 for v in by_layer.values())  # raw estimate is ε-independent
