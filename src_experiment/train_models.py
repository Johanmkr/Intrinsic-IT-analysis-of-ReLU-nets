import copy
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange


# ================================================================
# Training Loop
# ================================================================

def _evaluate(model: nn.Module, data: DataLoader, loss_fn) -> Tuple[float, float]:
    """Mean batch loss and accuracy of ``model`` on ``data`` (no gradient)."""
    running_loss, num_correct, total = 0.0, 0, 0
    with torch.no_grad():
        for x, y in data:
            y_hat = model(x.float())
            y = y.long()
            running_loss += loss_fn(y_hat, y).item()
            num_correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return running_loss / len(data), num_correct / total


def train_model_multiclass(
    model: nn.Module,
    train_data: DataLoader,
    test_data: DataLoader,
    epochs: int,
    on_save_callback: Callable[[int, Dict, Dict], None],
    save_everyth_epoch: Optional[int] = None,
    save_for_epochs: Optional[list] = None,
    sgd_lr: float = 0.01,
    sgd_mom: float = 0.9,
    disable_progress: bool = False,
) -> pd.DataFrame:
    """Train with SGD + cross-entropy; return per-epoch loss/accuracy curves.

    At every epoch in ``save_for_epochs``, every ``save_everyth_epoch``-th epoch
    and the last epoch, ``on_save_callback(epoch, state_dict, metrics)`` is called.
    """
    curves = {k: np.zeros(epochs) for k in (
        "train_loss", "train_accuracy", "test_loss", "test_accuracy",
        "eval_train_loss", "eval_train_accuracy")}

    optimizer = torch.optim.SGD(model.parameters(), lr=sgd_lr, momentum=sgd_mom)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in trange(epochs, desc="Training", leave=False, disable=disable_progress):
        # --- Training phase ---
        model.train()
        running_loss, num_correct, total = 0.0, 0, 0
        for x, y in train_data:
            optimizer.zero_grad()
            x = x.float()
            y = y.long()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.size(0)
        curves["train_loss"][epoch] = running_loss / len(train_data)
        curves["train_accuracy"][epoch] = num_correct / total

        # --- Evaluation on train and test data ---
        model.eval()
        curves["eval_train_loss"][epoch], curves["eval_train_accuracy"][epoch] = _evaluate(model, train_data, loss_fn)
        curves["test_loss"][epoch], curves["test_accuracy"][epoch] = _evaluate(model, test_data, loss_fn)

        # --- Checkpoint ---
        should_save = save_for_epochs is not None and epoch in save_for_epochs
        if save_everyth_epoch is not None and (epoch % save_everyth_epoch == 0 or epoch == epochs - 1):
            should_save = True
        if should_save:
            metrics = {k: v[epoch] for k, v in curves.items()}
            on_save_callback(epoch, copy.deepcopy(model.state_dict()), metrics)

    return pd.DataFrame(curves)
