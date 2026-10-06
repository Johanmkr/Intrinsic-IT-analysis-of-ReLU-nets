"""MI baselines of step 3 (src_experiment/baselines, run_baselines.py)."""
import h5py
import numpy as np
import pytest
import torch

from run_baselines import KSG_KS, baselines_row
from src_experiment.baselines.activations import load_layer_activations
from src_experiment.baselines.mi_baselines import ksg_mi, quantize_per_layer
from src_experiment.estimators import ESTIMATORS
from src_experiment.utils import NeuralNet


def test_quantize_per_layer_uniform_symmetric_bins():
    T = np.array([[-2.0, 0.0], [-0.1, 0.1], [1.0, 2.0]])
    binned = quantize_per_layer(T, 4)  # edges -1, 0, 1 over [-2, 2]
    np.testing.assert_array_equal(binned, [[0, 2], [1, 2], [3, 3]])
    assert quantize_per_layer(np.zeros((5, 3)), 8).max() == 0  # constant layer → one cell
    assert quantize_per_layer(np.random.default_rng(0).normal(size=(500, 4)), 30).max() <= 29


@pytest.mark.parametrize("k", [3, 5])
def test_ksg_matches_sklearn_in_one_dimension(k):
    from sklearn.feature_selection._mutual_info import _compute_mi_cd

    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, size=800)
    T = (y + rng.normal(scale=0.8, size=800)).reshape(-1, 1)
    ours = ksg_mi(T, y, k=k)
    assert ours["bits_signed"] * np.log(2) == pytest.approx(_compute_mi_cd(T.ravel(), y, k), rel=1e-9)


def test_ksg_limits():
    rng = np.random.default_rng(2)
    T, y = rng.normal(size=(2000, 2)), rng.integers(0, 2, size=2000)
    assert abs(ksg_mi(T, y)["bits_signed"]) < 0.05  # independent
    y = rng.integers(0, 4, size=2000)
    T = (10 * y + rng.normal(scale=0.1, size=2000)).reshape(-1, 1)
    assert ksg_mi(T, y)["bits"] == pytest.approx(2.0, abs=0.02)  # separated classes → H(Y)
    out = ksg_mi(rng.normal(size=(3, 2)), np.array([0, 1, 2]))  # no class with two samples
    assert out["bits"] == out["bits_signed"] == 0.0


def test_activations_match_the_torch_forward_pass(tmp_path):
    model = NeuralNet(input_size=3, hidden_sizes=[5, 4], num_classes=2, seed=7)
    path = tmp_path / "net.h5"
    with h5py.File(path, "w") as f:
        for name, t in model.state_dict().items():
            f.create_dataset(f"epochs/epoch_0/{name}", data=t.numpy())
    X = np.random.default_rng(3).uniform(-1, 1, size=(50, 3)).astype(np.float32)
    with torch.no_grad():
        z1 = model.l1(torch.from_numpy(X))
        z2 = model.l2(torch.relu(z1))
        logits = model(torch.from_numpy(X))
    np.testing.assert_allclose(load_layer_activations(path, 0, 1, X, "pre"), z1.numpy(), atol=1e-6)
    np.testing.assert_allclose(load_layer_activations(path, 0, 2, X, "post"), torch.relu(z2).numpy(), atol=1e-6)
    np.testing.assert_allclose(load_layer_activations(path, 0, 3, X, "pre"), logits.numpy(), atol=1e-6)
    np.testing.assert_array_equal(load_layer_activations(path, 0, 0, X), X)


def test_baselines_row_grid():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, size=300)
    T = (y[:, None] + rng.normal(scale=0.5, size=(300, 3))).astype(np.float32)
    row = baselines_row(T, y, num_classes=2, seed=0)
    kmeans = ["KY", "2KY", "4KY", "16", "64", "256"]  # |Y| = 2: K = 2, 4, 8, 16, 64, 256
    for prefix in [f"binning{k}" for k in (2, 4, 8, 16, 30)] + [f"kmeans{k}" for k in kmeans]:
        assert all(f"{prefix}_{est}_bits" in row for est in ESTIMATORS)
        assert 1 <= row[f"{prefix}_num_cells"] <= 300
    assert row["kmeansKY_num_cells"] == 2
    for k in KSG_KS:
        assert row[f"ksg{k}_bits"] == max(0.0, row[f"ksg{k}_signed_bits"])
    # With |Y| = 8 the 2|Y| setting is 16, so no separate kmeans16 column.
    row8 = baselines_row(T, rng.integers(0, 8, size=300), num_classes=8, seed=0)
    assert "kmeans2KY_num_cells" in row8 and "kmeans16_num_cells" not in row8
