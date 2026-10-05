"""Validation of parx linear-region partitions against the repo's own
routing / quotient machinery.

Checks, per plan section:
  2a  exact parx partitions on handcrafted nets: region counts, verify
      suite, local_affine against a plain numpy ReLU forward.
  2b  parx sparse partitions agree with the repo's grid-based
      cumulative-pattern counting on trained checkpoints (same region code
      sets), and every probe point routes to a region.
  2c  parx local_affine matrices equal the repo's compute_active_subnetwork
      (tilde_A, tilde_c) per routed region.
  2d  exact parx partitions dominate repo grid counts: full partition
      size >= layer-3 baselines at the last epoch, and per-layer cumulative
      counts >= 400x400-grid counts for every composite epoch.
  2e  Julia-backed parx methods are skipped cleanly when Julia is absent.

Checkpoints: ``./run.sh test`` sets SMOKE=1 and reads the ``./run.sh smoke``
outputs under smoke/; ``./run.sh test --full`` reads the step-1 outputs/.

parx operates on hidden layers only (weights/biases lists exclude the output
layer l{L+1}); activation_path / regions_at_layer index hidden layers 1..L.
"""
import shutil
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

# parx/__init__ runs check_julia() at import time; the pure-Python methods we
# exercise do not need Julia. Stub the checker before any parx import.
_julia_stub = types.ModuleType("parx._check")
_julia_stub.check_julia = lambda: None  # type: ignore[attr-defined]
sys.modules["parx._check"] = _julia_stub

from parx.partition import Partition
from parx.verify import (
    check_covers_space,
    check_no_overlaps,
    check_regions_nonempty,
    check_routing_consistency,
)
from parx.methods.exact_python import find as exact_find
from parx.methods.sparse_python import find as sparse_find

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src_experiment.routing_estimator import (  # noqa: E402
    RoutingEstimator,
    forward_activation_patterns,
)
from src_experiment.functional_quotient import compute_active_subnetwork  # noqa: E402
from src_experiment.smoke import LAST_EPOCH, SMOKE  # noqa: E402

DATA_ROOT = REPO_ROOT / "smoke" if SMOKE else REPO_ROOT
COMPOSITE_H5 = DATA_ROOT / "outputs/composite_label_noise/n0.0_[5, 5, 5]/seed_101.h5"
MNIST_777_H5 = DATA_ROOT / "outputs/mnist_capacity/2_dim_[7, 7, 7]/seed_101.h5"
MNIST_151515_H5 = DATA_ROOT / "outputs/mnist_capacity/2_dim_[15, 15, 15]/seed_101.h5"
MNIST_10DIM_H5 = DATA_ROOT / "outputs/mnist_capacity/10_dim_[3, 3, 3]/seed_101.h5"
FIG1_WEIGHTS = DATA_ROOT / ".cache/figure1_pedagogy_weights.pt"

# Layer-3 region counts of the full-run checkpoints at epoch 150. The smoke
# checkpoints differ, so there the repo's own count at the last epoch is used.
FULL_BASELINES = {COMPOSITE_H5: 59, MNIST_777_H5: 235, MNIST_151515_H5: 643}

_MISSING = f"needs {DATA_ROOT / 'outputs'}: run ./run.sh {'smoke' if SMOKE else 'step1'} first"
needs_outputs = pytest.mark.skipif(not COMPOSITE_H5.exists(), reason=_MISSING)
needs_fig1 = pytest.mark.skipif(not FIG1_WEIGHTS.exists(), reason=_MISSING)


def load_epoch_state_dict(h5_path: Path, epoch: int, dtype=np.float64):
    """Hidden-layer weights/biases for one checkpoint, hidden layers only."""
    with h5py.File(h5_path, "r") as f:
        n_hidden = len(f["metadata"].attrs["architecture"])
        grp = f[f"epochs/epoch_{epoch}"]
        sd = {}
        for i in range(1, n_hidden + 1):
            sd[f"l{i}.weight"] = np.asarray(grp[f"l{i}.weight"][:], dtype=dtype)
            sd[f"l{i}.bias"] = np.asarray(grp[f"l{i}.bias"][:], dtype=dtype)
    return sd


def weights_and_biases(sd):
    n_hidden = sum(1 for k in sd if k.endswith(".weight"))
    return ([sd[f"l{i}.weight"] for i in range(1, n_hidden + 1)],
            [sd[f"l{i}.bias"] for i in range(1, n_hidden + 1)])


def load_points(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f["points"][:])


def epoch_keys(h5_path: Path) -> list[int]:
    with h5py.File(h5_path, "r") as f:
        return sorted(int(k.split("_")[1]) for k in f["epochs"])


def cumulative_prefix_counts(part: Partition, n_layers: int) -> dict[int, int]:
    """Distinct cumulative activation codes per hidden-layer prefix 1..n."""
    out = {}
    for l in range(1, n_layers + 1):
        codes = {
            np.packbits(np.concatenate(r.activation_path[:l])).tobytes()
            for r in part.regions
        }
        out[l] = len(codes)
    return out


def grid_cumulative_counts(Ws32, bs32, grid: np.ndarray, n_layers: int) -> dict[int, int]:
    """Repo-side unique cumulative pattern counts on a sample grid."""
    pats = forward_activation_patterns(Ws32, bs32, grid.astype(np.float32))
    out = {}
    for l in range(1, n_layers + 1):
        codes = np.packbits(np.concatenate(pats[:l], axis=1), axis=1)
        out[l] = len(np.unique(codes, axis=0))
    return out


def region_key(region) -> tuple:
    return tuple(q.tobytes() for q in region.activation_path)


def interior_points(part: Partition, X: np.ndarray, margin: float = 1e-6) -> np.ndarray:
    """Sample rows lying strictly inside some region (|z| > margin for every
    hidden pre-activation), i.e. off all region boundaries.

    check_no_overlaps / check_covers_space count halfspace membership, which
    double-counts points sitting exactly on a shared boundary face — those
    are expected to belong to both adjacent regions.
    """
    keep = np.ones(len(X), dtype=bool)
    A = np.asarray(X, dtype=np.float64)
    for W, b in zip(part.weights, part.biases):
        Z = A @ W.T + b
        keep &= np.all(np.abs(Z) > margin, axis=1)
        A = np.maximum(Z, 0.0)
    return X[keep]


class TestHandcraftedExact:
    def test_1d_net_three_regions(self):
        Ws = [np.array([[2.0], [-2.0]])]
        bs = [np.array([0.0, 0.3])]
        part = Partition.from_result(exact_find(Ws, bs, np.zeros((1, 1))), Ws, bs)
        assert len(part.regions) == 3

        xs = np.linspace(-2.0, 2.0, 4001).reshape(-1, 1)
        routes = part.route(xs)
        assert all(r is not None for r in routes)

        interior = interior_points(part, xs)
        assert len(interior) > 0
        ok_overlap, _ = check_no_overlaps(part, interior)
        ok_cover, _ = check_covers_space(part, interior)
        ok_nonempty, _, _ = check_regions_nonempty(part)
        ok_routing, _ = check_routing_consistency(part, xs)
        assert ok_overlap and ok_cover and ok_nonempty and ok_routing

        # local_affine must reproduce the plain ReLU forward per point.
        max_diff = 0.0
        for x, r in zip(xs, routes):
            A, b = part.local_affine(r)
            h1 = np.maximum(Ws[0] @ x + bs[0], 0.0)
            max_diff = max(max_diff, float(np.max(np.abs(A @ x + b - h1))))
        assert max_diff < 1e-9

    @needs_fig1
    def test_fig1_net(self):
        import torch

        state = torch.load(FIG1_WEIGHTS, weights_only=True)
        Ws = [state["l1.weight"].numpy().astype(np.float64),
              state["l2.weight"].numpy().astype(np.float64)]
        bs = [state["l1.bias"].numpy().astype(np.float64),
              state["l2.bias"].numpy().astype(np.float64)]
        W1, b1, W2, b2 = Ws[0], bs[0], Ws[1], bs[1]

        part = Partition.from_result(exact_find(Ws, bs, np.zeros((1, 2))), Ws, bs)
        assert len(part.regions) == 38

        g = np.linspace(-1.5, 1.5, 601)
        XX, YY = np.meshgrid(g, g)
        grid = np.stack([XX.ravel(), YY.ravel()], axis=1)
        routes = part.route(grid)
        assert all(r is not None for r in routes)

        interior = interior_points(part, grid)
        assert len(interior) > 0
        ok_overlap, _ = check_no_overlaps(part, interior)
        ok_cover, _ = check_covers_space(part, interior)
        ok_nonempty, _, _ = check_regions_nonempty(part)
        ok_routing, _ = check_routing_consistency(part, grid)
        assert ok_overlap and ok_cover and ok_nonempty and ok_routing

        max_diff = 0.0
        for x, r in zip(grid, routes):
            A, b = part.local_affine(r)
            h1 = np.maximum(W1 @ x + b1, 0.0)
            h2 = np.maximum(W2 @ h1 + b2, 0.0)
            max_diff = max(max_diff, float(np.max(np.abs(A @ x + b - h2))))
        assert max_diff < 1e-9


@needs_outputs
class TestSparseVsRepo:
    @pytest.mark.parametrize("h5_path", [COMPOSITE_H5, MNIST_10DIM_H5])
    def test_sparse_matches_repo_routing(self, h5_path):
        est = RoutingEstimator(str(h5_path))
        points = np.asarray(est.points, dtype=np.float64)
        n_layers = est.num_hidden_layers

        for epoch in est.epochs:
            sd = load_epoch_state_dict(h5_path, epoch, dtype=np.float64)
            Ws, bs = weights_and_biases(sd)
            part = Partition.from_result(sparse_find(Ws, bs, points), Ws, bs)

            routed = part.route(points)
            assert all(r is not None for r in routed), (
                f"unrouted probe points at epoch {epoch}"
            )

            # parx routing must reproduce the repo's per-point activation
            # patterns exactly (same weights, same strict z>0 boundary).
            with h5py.File(h5_path, "r") as f:
                Ws32 = [np.asarray(f[f"epochs/epoch_{epoch}/l{i}.weight"][:], dtype=np.float32)
                        for i in range(1, n_layers + 1)]
                bs32 = [np.asarray(f[f"epochs/epoch_{epoch}/l{i}.bias"][:], dtype=np.float32)
                        for i in range(1, n_layers + 1)]
            repo_pats = forward_activation_patterns(
                Ws32, bs32, np.asarray(est.points, dtype=np.float32)
            )
            for layer in range(1, n_layers + 1):
                parx_mat = np.array(
                    [np.concatenate(r.activation_path[:layer]) for r in routed],
                    dtype=bool,
                )
                repo_mat = np.concatenate(repo_pats[:layer], axis=1)
                assert np.array_equal(parx_mat, repo_mat), (
                    f"epoch {epoch} layer {layer}: pattern mismatch"
                )
            assert len(set(map(region_key, part.regions))) == len(part.regions)


@needs_outputs
class TestLocalAffineVsSubnetwork:
    def test_local_affine_equals_active_subnetwork(self):
        sd = load_epoch_state_dict(COMPOSITE_H5, LAST_EPOCH, dtype=np.float64)
        Ws, bs = weights_and_biases(sd)
        n_layers = len(Ws)
        points = load_points(COMPOSITE_H5).astype(np.float64)

        part = Partition.from_result(sparse_find(Ws, bs, points), Ws, bs)
        routed = part.route(points)
        assert all(r is not None for r in routed)

        # distinct routed regions must cover at least the repo's counted regions
        est = RoutingEstimator(str(COMPOSITE_H5))
        n_repo = est.evaluate_epoch(LAST_EPOCH)[-1].num_regions
        distinct = {region_key(r) for r in routed}
        assert len(distinct) >= n_repo

        for key in distinct:
            r = next(r for r in part.regions if region_key(r) == key)
            pats = [r.activation_path[l] for l in range(n_layers)]
            tilde_A, tilde_c, S_l = compute_active_subnetwork(Ws, bs, pats, n_layers)
            A, b = part.local_affine(r)

            assert A[S_l] == pytest.approx(tilde_A, rel=1e-9, abs=1e-12)
            assert b[S_l] == pytest.approx(tilde_c, rel=1e-9, abs=1e-12)
            inactive = np.ones(A.shape[0], dtype=bool)
            inactive[S_l] = False
            assert np.all(A[inactive] == 0.0)


@needs_outputs
class TestExactVsGrid:
    @pytest.mark.parametrize("h5_path", [COMPOSITE_H5, MNIST_777_H5, MNIST_151515_H5])
    def test_exact_dominates_grid_last_epoch(self, h5_path):
        if SMOKE:
            baseline = RoutingEstimator(str(h5_path)).evaluate_epoch(LAST_EPOCH)[-1].num_regions
        else:
            baseline = FULL_BASELINES[h5_path]
        sd = load_epoch_state_dict(h5_path, LAST_EPOCH, dtype=np.float64)
        Ws, bs = weights_and_biases(sd)
        points = load_points(h5_path).astype(np.float64)

        part = Partition.from_result(exact_find(Ws, bs, points[0:1]), Ws, bs)
        assert len(part.regions) >= baseline

        # route an 800x800 grid over the probe bbox (+10% pad): no misses
        lo, hi = points.min(0), points.max(0)
        pad = 0.1 * (hi - lo)
        axes = [np.linspace(lo[d] - pad[d], hi[d] + pad[d], 800) for d in range(points.shape[1])]
        meshes = np.meshgrid(*axes)
        grid = np.stack([m.ravel() for m in meshes], axis=1)
        assert all(r is not None for r in part.route(grid))

        ok_cover, _ = check_covers_space(part, grid)
        ok_overlap, _ = check_no_overlaps(part, grid)
        assert ok_cover and ok_overlap

    def test_exact_dominates_grid_all_epochs(self):
        h5_path = COMPOSITE_H5
        sd = load_epoch_state_dict(h5_path, LAST_EPOCH, dtype=np.float64)
        Ws, bs = weights_and_biases(sd)
        n_layers = len(Ws)
        points = load_points(h5_path).astype(np.float64)

        grid = np.linspace(-1.0, 1.0, 400)
        XX, YY = np.meshgrid(grid, grid)
        G400 = np.stack([XX.ravel(), YY.ravel()], axis=1).astype(np.float32)

        for epoch in epoch_keys(h5_path):
            sd_e = load_epoch_state_dict(h5_path, epoch, dtype=np.float64)
            Ws_e, bs_e = weights_and_biases(sd_e)
            part = Partition.from_result(exact_find(Ws_e, bs_e, points[0:1]), Ws_e, bs_e)
            exact_cum = cumulative_prefix_counts(part, n_layers)

            with h5py.File(h5_path, "r") as f:
                Ws32 = [np.asarray(f[f"epochs/epoch_{epoch}/l{i}.weight"][:], dtype=np.float32)
                        for i in range(1, n_layers + 1)]
                bs32 = [np.asarray(f[f"epochs/epoch_{epoch}/l{i}.bias"][:], dtype=np.float32)
                        for i in range(1, n_layers + 1)]
            grid_cum = grid_cumulative_counts(Ws32, bs32, G400, n_layers)

            for l in range(1, n_layers + 1):
                assert exact_cum[l] >= grid_cum[l], (
                    f"epoch {epoch} layer {l}: exact {exact_cum[l]} < grid {grid_cum[l]}"
                )


class TestJuliaMethods:
    @pytest.mark.skipif(shutil.which("julia") is None, reason="Julia not on PATH")
    def test_julia_exact_matches_python(self):
        pytest.importorskip("parx.methods.exact_julia")
