"""
Shared activation extractor for Phase A baselines.

Loads model weights from a training-output HDF5, runs a forward pass on a
probe matrix, and returns the pre- or post-activation at the chosen layer.
Both halves of Phase A (MI baselines, generalization-gap predictors) call
this. The forward-pass convention matches
``src_experiment.routing_estimator.forward_activation_patterns`` — strict
``z > 0`` ReLU gating, PyTorch weight layout.

For LeNet-5 HDF5s (``arch_type=lenet5``) the deepest hidden FC layer's
pre-activation is exposed via :func:`load_lenet_layer_activations`.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Tuple, Union

import h5py
import numpy as np

PathLike = Union[str, Path]
Kind = Literal["pre", "post"]


def _load_all_weights(
    h5_path: Path, epoch: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Load every stored weight/bias at ``epoch`` (hidden + output).

    Weights use PyTorch layout: ``W[i].shape == (n_{i+1}, n_i)``.
    """
    with h5py.File(h5_path, "r") as f:
        grp = f[f"epochs/epoch_{epoch}"]
        idx = 1
        Ws: List[np.ndarray] = []
        bs: List[np.ndarray] = []
        while f"l{idx}.weight" in grp:
            Ws.append(np.asarray(grp[f"l{idx}.weight"][:], dtype=np.float32))
            bs.append(np.asarray(grp[f"l{idx}.bias"][:], dtype=np.float32))
            idx += 1
    if not Ws:
        raise ValueError(f"no weights found at epoch {epoch} in {h5_path}")
    return Ws, bs


def load_layer_activations(
    h5_path: PathLike,
    epoch: int,
    layer: int,
    X: np.ndarray,
    kind: Kind = "pre",
) -> np.ndarray:
    """Forward pass on ``X`` and return one layer's activation matrix.

    Parameters
    ----------
    h5_path
        Training-output HDF5 in this project's standard layout.
    epoch
        Saved epoch index (must match a key under ``epochs/`` in the file).
    layer
        1-indexed layer. ``0`` returns the input ``X``. Hidden layers run
        ``1..L``; ``layer=L+1`` returns the output (logit) layer.
    X
        Probe set, shape ``(N, n_0)``, already in the model's training-time
        feature space (see ``src_experiment.probe_loader``).
    kind
        ``"pre"`` → ``z = a_{l-1} W_l^T + b_l`` (signed pre-activation).
        ``"post"`` → ``ReLU(z)`` on hidden layers; on the output layer post
        equals pre (no nonlinearity applied during training). At
        ``layer=0`` both kinds return ``X`` unchanged.

    Returns
    -------
    ``(N, d_layer)`` float32 array.
    """
    if layer < 0:
        raise ValueError("layer must be >= 0")
    if kind not in ("pre", "post"):
        raise ValueError(f"kind must be 'pre' or 'post', got {kind!r}")

    X = np.asarray(X, dtype=np.float32)
    if layer == 0:
        return X.copy()

    Ws, bs = _load_all_weights(Path(h5_path), epoch)
    n_layers = len(Ws)
    if layer > n_layers:
        raise ValueError(
            f"layer={layer} exceeds number of stored layers ({n_layers})"
        )

    a = X
    for idx, (W, b) in enumerate(zip(Ws, bs), start=1):
        z = a @ W.T + b
        if idx == layer:
            if kind == "pre":
                return np.asarray(z, dtype=np.float32)
            # post: ReLU on hidden layers; output layer is linear so post == pre.
            if idx < n_layers:
                return np.asarray(np.where(z > 0, z, 0.0), dtype=np.float32)
            return np.asarray(z, dtype=np.float32)
        a = np.where(z > 0, z, 0.0)

    raise RuntimeError("forward pass terminated without returning")  # unreachable


# ---------------------------------------------------------------------------
# LeNet-5 activation tap
# ---------------------------------------------------------------------------
def _is_lenet_h5(h5_path: Path) -> bool:
    with h5py.File(h5_path, "r") as f:
        return str(f["metadata"].attrs.get("arch_type", "")) == "lenet5"


def load_lenet_layer_activations(
    h5_path: PathLike,
    epoch: int,
    layer: int,
    X: np.ndarray,
    kind: Kind = "pre",
) -> np.ndarray:
    """Return the pre/post activation of one ReLU step in a LeNet-5 HDF5.

    Layer convention (matches :class:`src_experiment.cnn_estimator.LeNetSpec`):

    * ``layer = 1 .. n_conv`` — conv-ReLU output (pre-pool), flattened to
      ``(N, C * H * W)``. ``"pre"`` returns the conv output before ReLU
      (signed); ``"post"`` returns it after ReLU.
    * ``layer = n_conv + 1 .. n_conv + n_fc_hidden`` — FC hidden layer.
      ``"pre"`` returns the linear output before ReLU; ``"post"`` returns
      it after ReLU.
    * ``layer = n_conv + n_fc_hidden + 1`` — output logits (linear, no
      ReLU). Both ``"pre"`` and ``"post"`` return the same tensor.

    Probe ``X`` may be flat ``(N, n_0)`` or already shaped ``(N, C, H, W)``.
    """
    import torch

    from src_experiment.utils import LeNet5  # late import to avoid hard dep at import time

    if kind not in ("pre", "post"):
        raise ValueError(f"kind must be 'pre' or 'post', got {kind!r}")

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f["metadata"].attrs)
        conv_channels = tuple(int(x) for x in attrs.get("conv_channels", []))
        fc_widths = tuple(int(x) for x in attrs.get("fc_widths", []))
        kernel_size = int(attrs.get("kernel_size", 5))
        pool_size = int(attrs.get("pool_size", 2))
        input_shape = tuple(int(x) for x in attrs.get("inferred_input_shape", (1, 28, 28)))
        num_classes = int(attrs.get("inferred_num_classes", 10))
        grp = f[f"epochs/epoch_{epoch}"]
        state = {}
        idx = 1
        while f"conv{idx}.weight" in grp:
            state[f"conv{idx}.weight"] = torch.from_numpy(grp[f"conv{idx}.weight"][:])
            state[f"conv{idx}.bias"] = torch.from_numpy(grp[f"conv{idx}.bias"][:])
            idx += 1
        idx = 1
        while f"fc{idx}.weight" in grp:
            state[f"fc{idx}.weight"] = torch.from_numpy(grp[f"fc{idx}.weight"][:])
            state[f"fc{idx}.bias"] = torch.from_numpy(grp[f"fc{idx}.bias"][:])
            idx += 1

    n_conv = len(conv_channels)
    n_fc_hidden = len(fc_widths)
    n_relu = n_conv + n_fc_hidden
    if not 1 <= layer <= n_relu + 1:
        raise ValueError(
            f"layer={layer} out of range [1, {n_relu + 1}] for this LeNet HDF5"
        )

    model = LeNet5(
        conv_channels=conv_channels,
        fc_widths=fc_widths,
        num_classes=num_classes,
        input_shape=input_shape,
        kernel_size=kernel_size,
        pool_size=pool_size,
    )
    model.load_state_dict(state)
    model.eval()

    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        x = torch.from_numpy(X).view(X.shape[0], *input_shape)
    else:
        x = torch.from_numpy(X)

    captured: dict[str, torch.Tensor] = {}

    if layer <= n_conv:
        target = getattr(model, f"conv{layer}")

        def hook(_module, _inp, out):
            captured["v"] = out.detach()

        h = target.register_forward_hook(hook)
    elif layer <= n_relu:
        fc_idx = layer - n_conv  # 1-indexed FC hidden
        target = getattr(model, f"fc{fc_idx}")

        def hook(_module, _inp, out):
            captured["v"] = out.detach()

        h = target.register_forward_hook(hook)
    else:
        # Output logits — capture forward output.
        def hook(_module, _inp, out):
            captured["v"] = out.detach()

        h = model.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(x)
    h.remove()

    z = captured["v"]
    if z.dim() == 4:
        # Conv layer pre-/post-activation: flatten spatial.
        if kind == "post":
            z = torch.relu(z)
        return z.reshape(z.size(0), -1).cpu().numpy().astype(np.float32, copy=False)
    if kind == "post" and layer <= n_relu:
        z = torch.relu(z)
    return z.cpu().numpy().astype(np.float32, copy=False)


def load_activations_dispatch(
    h5_path: PathLike,
    epoch: int,
    layer: int,
    X: np.ndarray,
    kind: Kind = "pre",
) -> np.ndarray:
    """Dispatch by HDF5 ``arch_type`` to the MLP or LeNet activation loader."""
    h5_path = Path(h5_path)
    if _is_lenet_h5(h5_path):
        return load_lenet_layer_activations(h5_path, epoch, layer, X, kind=kind)
    return load_layer_activations(h5_path, epoch, layer, X, kind=kind)
