import os
from pathlib import Path

import pandas as pd
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from ucimlrepo import fetch_ucirepo
from typing import Tuple, Callable
from torchvision import datasets
from src_experiment.smoke import SMOKE

N_SAMPLES = 10000
DEFAULT_BATCH_SIZE = 32

# Downloaded datasets (MNIST via torchvision, UCI tables cached by _load_uci).
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# ------------------------------------------------------------------------------
#       1. Optimized Utility Functions
# ------------------------------------------------------------------------------

def inject_label_noise_vectorized(y: np.ndarray, noise_ratio: float, n_classes: int, seed: int) -> np.ndarray:
    """Vectorized version of label noise injection."""
    if noise_ratio <= 0.0:
        return y

    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_ratio * n_samples)
    
    noisy_indices = rng.choice(n_samples, size=n_noisy, replace=False)
    shifts = rng.integers(low=1, high=n_classes, size=n_noisy)
    y_noisy[noisy_indices] = (y_noisy[noisy_indices] + shifts) % n_classes
    
    return y_noisy


def permute_labels(y: np.ndarray, seed: int) -> np.ndarray:
    """Memorization control: randomly permute the training labels with ``seed``.

    The label marginal is kept but the labels become independent of the inputs.
    Deterministic, so step 2 can rebuild exactly the labels a network trained on.
    """
    return np.random.default_rng(seed).permutation(y)


def _pca_project(X_train: np.ndarray, X_test: np.ndarray, target_dim: int, seed: int):
    """PCA fit on the train split, applied to both splits.

    BLAS is limited to one thread: the rounding of the PCA depends on the
    thread count, so this makes the projected inputs (and every network trained
    on them) the same on every machine with the same CPU kernels, independent
    of its core count or of OMP_NUM_THREADS.
    """
    if target_dim > X_train.shape[1]:
        raise ValueError(f"target_dim ({target_dim}) cannot be larger than actual dataset dimension ({X_train.shape[1]}).")
    with threadpool_limits(limits=1, user_api="blas"):
        pca = PCA(n_components=target_dim, random_state=seed)
        return pca.fit_transform(X_train), pca.transform(X_test)


def process_and_split(X: np.ndarray, y: np.ndarray, noise_level: float, test_size=0.2, seed=42, target_dim: int = None, shuffle_labels: bool = False) -> Tuple[TensorDataset, TensorDataset]:
    """Unified pipeline for splitting, PCA scaling, and noise injection."""
    # 1. Encode labels
    unique_classes = np.sort(np.unique(y))
    class_map = {val: i for i, val in enumerate(unique_classes)}
    y_mapped = np.array([class_map[val] for val in y], dtype=np.int64)
    n_classes = len(unique_classes)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_mapped, test_size=test_size, random_state=seed, stratify=y_mapped
    )

    # 3. Inject Noise (Training labels only)
    y_train = inject_label_noise_vectorized(y_train, noise_level, n_classes, seed)
    if shuffle_labels:
        y_train = permute_labels(y_train, seed)

    # 4. Dimensionality Reduction (PCA) - Fit ONLY on Train Data
    if target_dim is not None:
        X_train, X_test = _pca_project(X_train, X_test, target_dim, seed)

    # 5. Scale to Unit Hypercube [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_test = np.clip(X_test, -1.0, 1.0)

    # 6. Tensorize
    return (
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64)),
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))
    )

# ------------------------------------------------------------------------------
#       2. Dataset Loaders (Lazy Loading)
# ------------------------------------------------------------------------------

def _uci_tables(id: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Features and targets of a UCI dataset, cached as CSV in data/uci_<id>/.

    The first call fetches from the UCI repository and writes the cache
    (atomically, so parallel workers never read a partial file); later calls,
    including inside the Docker image, need no internet.
    """
    cache = DATA_DIR / f"uci_{id}"
    paths = {name: cache / f"{name}.csv" for name in ("features", "targets")}
    if all(p.exists() for p in paths.values()):
        return pd.read_csv(paths["features"]), pd.read_csv(paths["targets"])
    print(f"Fetching UCI dataset ID={id}...")
    dataset = fetch_ucirepo(id=id)
    tables = {"features": dataset.data.features, "targets": dataset.data.targets}
    cache.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        tmp = paths[name].with_name(f"{paths[name].name}.{os.getpid()}.tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(paths[name])
    return tables["features"], tables["targets"]


def _load_uci(id: int, target_col: str = None, target_val: str = None, map_func: Callable = None):
    """Generic helper to load UCI datasets only when requested."""
    X, y = _uci_tables(id)
    
    if target_col:
        y = y[target_col]
    if target_val:
        y = (y == target_val).astype(int)
    if map_func:
        X, y = map_func(X, y)
        
    return X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.int64)

def _make_composite_data(n_samples: int, seed: int, noise=None) -> Tuple[np.ndarray, np.ndarray]:
    """Generates the custom moons/circles/blobs composite dataset."""
    # Proportional splits based on the requested n_samples
    n_moons = int(n_samples * 0.4) 
    n_circles = int(n_samples * 0.4) 
    n_blobs = n_samples - n_moons - n_circles 
    
    # 1. Moons
    data_moons, labels_moons = make_moons(n_samples=n_moons, shuffle=True, noise=0.10, random_state=seed)
    data_moons = data_moons + 0.5

    # 2. Circles
    data_circles, labels_circles = make_circles(n_samples=n_circles, shuffle=True, noise=0.05, factor=0.75, random_state=seed)
    data_circles = data_circles * 3.0 + 1
    labels_circles += 2

    # 3. Blobs
    centers = [[-2, 4], [3, -3], [5, 4]]
    data_blobs, labels_blobs = make_blobs(n_samples=n_blobs, cluster_std=0.3, centers=centers, random_state=seed)
    labels_blobs += 4

    # 4. Combine
    X = np.concatenate([data_moons, data_circles, data_blobs], axis=0)
    y = np.concatenate([labels_moons, labels_circles, labels_blobs])
    
    return X, y
 

# ------------------------------------------------------------------------------
#       3. The Registry (Factory Pattern)
# ------------------------------------------------------------------------------

def get_new_data(dataset_name: str, noise: float = 0.0, batch_size: int = DEFAULT_BATCH_SIZE, split_seed=42, target_dim: int = None, **kwargs):
    """
    Central entry point. Handles logic dispatch cleanly.
    target_dim: Desired number of features after applying PCA.
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "composite":
        X, y = _make_composite_data(n_samples=N_SAMPLES, seed=split_seed)
        # We pass noise_level=0.0 to the pipeline since the feature noise is already baked into the shapes
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed, target_dim=target_dim,
                                              shuffle_labels=kwargs.get("permute_labels", False))

    # --- Standard Vision Datasets (Modified for Eager PCA Support) ---
    elif dataset_name == "mnist":
        print(f"Fetching {dataset_name} (torchvision)...")
        train_data = datasets.MNIST(root='./data', train=True, download=True)
        test_data = datasets.MNIST(root='./data', train=False, download=True)
        
        # Extract tensors natively to [0, 1] bounds
        X_train = train_data.data.float() / 255.0
        X_test = test_data.data.float() / 255.0
        y_train = train_data.targets.numpy()
        y_test = test_data.targets.numpy()
        
        # The paper uses the full 60k/10k MNIST split (App. C). Only ./run.sh smoke
        # subsamples both splits to 10% to keep the smoke run short.
        if SMOKE:
            rng_subsample = np.random.default_rng(split_seed)
            train_idx = rng_subsample.choice(len(X_train), size=int(len(X_train) * 0.10), replace=False)
            X_train = X_train[train_idx]
            y_train = y_train[train_idx]
            test_idx = rng_subsample.choice(len(X_test), size=int(len(X_test) * 0.10), replace=False)
            X_test = X_test[test_idx]
            y_test = y_test[test_idx]

        # Flatten to 2D numpy arrays for sklearn compatibility
        X_train = X_train.view(X_train.size(0), -1).numpy()
        X_test = X_test.view(X_test.size(0), -1).numpy()
        
        # MNIST keeps its standard 60k/10k split; PCA and scaling are fit on the
        # training split only, mirroring process_and_split.
        if noise > 0.0:
            y_train = inject_label_noise_vectorized(y_train, noise, 10, split_seed)
            
        if target_dim is not None:
            X_train, X_test = _pca_project(X_train, X_test, target_dim, split_seed)
        
        # Scale back to bounded interval
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_test = np.clip(X_test, -1.0, 1.0)
        
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))

    # --- UCI Datasets ---
    elif dataset_name == "wbc":
        X, y = _load_uci(id=17, target_col="Diagnosis", target_val="M")
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed, target_dim=target_dim,
                                              shuffle_labels=kwargs.get("permute_labels", False))
        
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )