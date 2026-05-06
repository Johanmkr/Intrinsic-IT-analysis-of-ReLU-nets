import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from typing import Tuple, Dict, Callable
from torchvision import datasets, transforms

N_SAMPLES = 10000
DEFAULT_BATCH_SIZE = 32

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


def process_and_split(X: np.ndarray, y: np.ndarray, noise_level: float, test_size=0.2, seed=42, target_dim: int = None) -> Tuple[TensorDataset, TensorDataset]:
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

    # 4. Dimensionality Reduction (PCA) - Fit ONLY on Train Data
    if target_dim is not None:
        if target_dim > X_train.shape[1]:
            raise ValueError(f"target_dim ({target_dim}) cannot be larger than actual dataset dimension ({X_train.shape[1]}).")
        
        pca = PCA(n_components=target_dim, random_state=seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

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

def _load_uci(id: int, target_col: str = None, target_val: str = None, map_func: Callable = None):
    """Generic helper to load UCI datasets only when requested."""
    print(f"Fetching UCI dataset ID={id}...")
    dataset = fetch_ucirepo(id=id)
    X = dataset.data.features
    y = dataset.data.targets
    
    if target_col:
        y = y[target_col]
    if target_val:
        y = (y == target_val).astype(int)
    if map_func:
        X, y = map_func(X, y)
        
    return X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.int64)

def _map_car_data(X_raw, y_raw):
    mapping = {
        "buying":   {"low": 0, "med": 1, "high": 2, "vhigh": 3},
        "maint":    {"low": 0, "med": 1, "high": 2, "vhigh": 3},
        "doors":    {"2": 0, "3": 1, "4": 2, "5more": 3},
        "persons":  {"2": 0, "4": 1, "more": 2},
        "lug_boot": {"small": 0, "med": 1, "big": 2},
        "safety":   {"low": 0, "med": 1, "high": 2}
    }
    X = X_raw.copy()
    for col, counts in mapping.items():
        X[col] = X[col].map(counts)
    target_mapping = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
    y = y_raw.iloc[:, 0].map(target_mapping)
    return X, y

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
    
    # --- Synthetic Datasets ---
    if dataset_name == "moons":
        X, y = make_moons(n_samples=N_SAMPLES, noise=noise, random_state=split_seed)
        train_ds, test_ds = process_and_split(X, y, noise_level=0.0, seed=split_seed, target_dim=target_dim) 
        
    elif dataset_name == "circles":
        X, y = make_circles(n_samples=N_SAMPLES, noise=noise, random_state=split_seed)
        train_ds, test_ds = process_and_split(X, y, noise_level=0.0, seed=split_seed, target_dim=target_dim)

    elif dataset_name == "blobs":
        centers = kwargs.get("centers", 3)
        n_features = kwargs.get("n_features", 2)
        X, y = make_blobs(n_samples=N_SAMPLES, centers=centers, n_features=n_features, random_state=split_seed)
        train_ds, test_ds = process_and_split(X, y, noise_level=0.0, seed=split_seed, target_dim=target_dim)
        
    elif dataset_name == "composite":
        X, y = _make_composite_data(n_samples=N_SAMPLES, seed=split_seed)
        # We pass noise_level=0.0 to the pipeline since the feature noise is already baked into the shapes
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed, target_dim=target_dim)

    # --- Standard Vision Datasets (Modified for Eager PCA Support) ---
    elif dataset_name in ["mnist", "mnist_minimal", "mnist_minimal_random"]:
        print(f"Fetching {dataset_name} (torchvision)...")
        train_data = datasets.MNIST(root='./data', train=True, download=True)
        test_data = datasets.MNIST(root='./data', train=False, download=True)
        
        # Extract tensors natively to [0, 1] bounds
        X_train = train_data.data.float() / 255.0
        X_test = test_data.data.float() / 255.0
        y_train = train_data.targets.numpy()
        y_test = test_data.targets.numpy()
        
        # --- NEW: DOWNSAMPLE TO 1/4 SIZE ---
        print("📉 Downsampling dataset to 25% of its original size...")
        rng_subsample = np.random.default_rng(split_seed) # Ensures the exact same subset is chosen every run
        
        # Subsample Train (60k -> 15k)
        train_idx = rng_subsample.choice(len(X_train), size=int(len(X_train) * 0.10), replace=False)
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]
        
        # Subsample Test (10k -> 2.5k)
        test_idx = rng_subsample.choice(len(X_test), size=int(len(X_test) * 0.10), replace=False)
        X_test = X_test[test_idx]
        y_test = y_test[test_idx]
        # -----------------------------------
        
        # Apply the minimal downsampling (28x28 -> 7x7)
        if dataset_name in ["mnist_minimal", "mnist_minimal_random"]:
            X_train = torch.nn.functional.avg_pool2d(X_train.unsqueeze(1), kernel_size=4).squeeze(1)
            X_test = torch.nn.functional.avg_pool2d(X_test.unsqueeze(1), kernel_size=4).squeeze(1)
            
        # Flatten to 2D numpy arrays for sklearn compatibility
        X_train = X_train.view(X_train.size(0), -1).numpy()
        X_test = X_test.view(X_test.size(0), -1).numpy()
        
        # OVERWRITE WITH RANDOM LABELS
        if dataset_name == "mnist_minimal_random":
            print("⚠️ Overwriting targets with completely random labels (0-9)...")
            rng_labels = np.random.default_rng(split_seed)
            y_train = rng_labels.integers(0, 10, size=y_train.shape)
            y_test = rng_labels.integers(0, 10, size=y_test.shape)
            noise = 0.0 

        # Call to your custom split function
        train_ds, test_ds = process_and_split(X_train, y_train, noise_level=noise, test_size=0.2, seed=split_seed, target_dim=target_dim)
        
        # Process test set manually to mirror the train set operations
        if noise > 0.0 and dataset_name != "mnist_minimal_random":
            y_train = inject_label_noise_vectorized(y_train, noise, 10, split_seed)
            
        if target_dim is not None:
             if target_dim > X_train.shape[1]:
                 raise ValueError(f"target_dim ({target_dim}) cannot be larger than actual dataset dimension ({X_train.shape[1]}).")
             pca = PCA(n_components=target_dim, random_state=split_seed)
             X_train = pca.fit_transform(X_train)
             X_test = pca.transform(X_test)
        
        # Scale back to bounded interval
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_test = np.clip(X_test, -1.0, 1.0)
        
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))

    # --- Full MNIST (Phase C: CNN sweep) ---
    elif dataset_name == "mnist_full":
        # Full 60k/10k MNIST as (N, 1, 28, 28), scaled to [-1, 1].
        # Optional label noise via the same `inject_label_noise_vectorized`
        # used elsewhere; no PCA, no subsample, no flatten.
        print("Fetching mnist_full (torchvision)...")
        train_data = datasets.MNIST(root="./data", train=True, download=True)
        test_data = datasets.MNIST(root="./data", train=False, download=True)
        X_train = (train_data.data.float() / 255.0 - 0.5) * 2.0  # -> [-1, 1]
        X_test = (test_data.data.float() / 255.0 - 0.5) * 2.0
        X_train = X_train.unsqueeze(1)  # (N, 1, 28, 28)
        X_test = X_test.unsqueeze(1)
        y_train = train_data.targets.numpy().astype(np.int64)
        y_test = test_data.targets.numpy().astype(np.int64)

        if noise > 0.0:
            y_train = inject_label_noise_vectorized(y_train, noise, 10, split_seed)

        train_ds = TensorDataset(X_train,
                                 torch.tensor(y_train, dtype=torch.int64))
        test_ds = TensorDataset(X_test,
                                torch.tensor(y_test, dtype=torch.int64))

    # --- UCI Datasets ---
    elif dataset_name == "wbc":
        X, y = _load_uci(id=17, target_col="Diagnosis", target_val="M")
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed, target_dim=target_dim)
        
    elif dataset_name == "wine":
        X, y = _load_uci(id=109)
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed, target_dim=target_dim)
        
    elif dataset_name == "hd":
        X, y = _load_uci(id=45)
        X[np.isnan(X)] = 0
        y[np.isnan(y)] = 0
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed, target_dim=target_dim)

    elif dataset_name == "car":
        X, y = _load_uci(id=19, map_func=_map_car_data)
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed, target_dim=target_dim)
        
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )