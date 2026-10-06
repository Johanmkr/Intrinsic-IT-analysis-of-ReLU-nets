"""Shared fixtures: a tiny trained network in the step-1 HDF5 layout.

Lets the unit tests exercise steps 2–4 end to end in seconds, without
./run.sh smoke checkpoints or any dataset download.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src_experiment.utils import NeuralNet  # noqa: E402

ARCH = [6, 6, 6]
N_POINTS = 400
N_CLASSES = 3


def _toy_data(seed: int = 0):
    """Points in [-1, 1]^2, labelled by angular sector (3 classes)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(N_POINTS, 2)).astype(np.float32)
    y = (np.floor((np.arctan2(X[:, 1], X[:, 0]) + np.pi) / (2 * np.pi / N_CLASSES)) % N_CLASSES).astype(np.int64)
    return X, y


@pytest.fixture(scope="session")
def tiny_h5(tmp_path_factory) -> Path:
    """seed_101.h5 of a 2→6→6→6→3 ReLU net, saved at epochs 0 and 1 (step-1 layout)."""
    X, y = _toy_data()
    model = NeuralNet(input_size=2, hidden_sizes=ARCH, num_classes=N_CLASSES, seed=101)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y)

    path = tmp_path_factory.mktemp("outputs") / "toy" / "toy_[6, 6, 6]" / "seed_101.h5"
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("metadata")
        for k, v in {"experiment_name": "toy_[6, 6, 6]", "dataset": "composite", "architecture": np.array(ARCH),
                     "model_seed": 101, "global_seed": 42, "noise": 0.0}.items():
            meta.attrs[k] = v
        for epoch in (0, 1):
            for _ in range(200 if epoch else 20):
                opt.zero_grad()
                loss_fn(model(Xt), yt).backward()
                opt.step()
            grp = f.create_group(f"epochs/epoch_{epoch}")
            for name, tensor in model.state_dict().items():
                grp.create_dataset(name, data=tensor.numpy())
        f.create_dataset("points", data=X)
        f.create_dataset("labels", data=y)
    return path
