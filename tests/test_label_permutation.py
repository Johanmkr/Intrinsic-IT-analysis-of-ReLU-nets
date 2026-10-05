"""Step 2 must rebuild exactly the (X, y) a label-permutation network trained on."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_estimate import _probe  # noqa: E402
from src_experiment.dataset import get_new_data  # noqa: E402


def _train_set(permute: bool):
    kwargs = {"permute_labels": True} if permute else {}
    train_loader, test_loader = get_new_data("composite", noise=0.0, split_seed=42, **kwargs)
    X, y = train_loader.dataset.tensors
    return X.numpy(), y.numpy(), len(test_loader.dataset)


def test_train_protocol_rebuilds_training_data():
    X_perm, y_perm, n_test = _train_set(permute=True)
    X_true, y_true, _ = _train_set(permute=False)
    assert np.array_equal(X_perm, X_true)  # permutation touches labels only
    assert not np.array_equal(y_perm, y_true)
    assert np.array_equal(np.bincount(y_perm), np.bincount(y_true))  # same marginal

    X, y = _probe("composite", "train", "train_labels", 42, n_test)
    assert np.array_equal(X, X_perm) and np.array_equal(y, y_perm)
    X, y = _probe("composite", "train", "true_labels", 42, n_test)
    assert np.array_equal(X, X_true) and np.array_equal(y, y_true)
