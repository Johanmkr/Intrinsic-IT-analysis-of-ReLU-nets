import copy
import pathlib as pl
from typing import Dict, Tuple, List, Optional, Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange
from .utils import createfolders

# ================================================================
# Geometric Penalty Module
# ================================================================

class GeometricPenalty(nn.Module):
    def __init__(self, alpha=0.1, beta=0.1, margin=10.0, sigmoid_steepness=10.0):
        super(GeometricPenalty, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.margin = margin
        self.steepness = sigmoid_steepness

    def forward(self, pre_activations, labels):
        # Flatten spatial dimensions for Conv2d layers (B, C, H, W) -> (B, C*H*W)
        if pre_activations.dim() > 2:
            pre_activations = pre_activations.view(pre_activations.size(0), -1)
            
        soft_activations = torch.sigmoid(pre_activations * self.steepness)
        distances = torch.cdist(soft_activations, soft_activations, p=1)
        
        labels_equal = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels_not_equal = 1.0 - labels_equal
        
        eye = torch.eye(labels.size(0), device=labels.device)
        labels_equal = labels_equal - eye
        
        num_intra = labels_equal.sum()
        intra_loss = (distances * labels_equal).sum() / (num_intra + 1e-8)
        
        num_inter = labels_not_equal.sum()
        inter_loss = (F.relu(self.margin - distances) * labels_not_equal).sum() / (num_inter + 1e-8)
        
        return (self.alpha * intra_loss) + (self.beta * inter_loss)


# ================================================================
# Training Loop
# ================================================================

def train_model_multiclass(
    model: nn.Module,
    train_data: DataLoader,
    test_data: DataLoader,
    epochs: int,
    num_classes: int = None,
    savepath: Optional[pl.Path] = None,
    SAVE_STATES: bool = False,
    save_everyth_epoch: int = None,
    save_for_epochs: list = None,
    RETURN_STATES: bool = False,
    sgd_lr=0.01,
    sgd_mom=0.9,
    on_save_callback: Optional[Callable[[int, Dict, Dict], None]] = None,
    disable_progress: bool = False,
    use_geo_penalty: bool = False,          # NEW: Toggle Geometric Penalty
    geo_alpha: float = 0.1,                 # NEW: Geometric Penalty hyperparams
    geo_beta: float = 0.1,
    geo_margin: float = 1.0,
    geo_steepness: float = 10.0
) -> Any:
    
    train_loss = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)
    eval_train_loss = np.zeros(epochs)
    eval_train_accuracy = np.zeros(epochs)
    
    if num_classes is None:
        num_classes = model.num_classes
    
    saved_states: Dict[int, Dict[str, torch.Tensor]] = {}

    optimizer = torch.optim.SGD(model.parameters(), lr=sgd_lr, momentum=sgd_mom)
    loss_fn = nn.CrossEntropyLoss()

    # ----------------------
    # Setup Geometric Penalty (Optional)
    # ----------------------
    captured_activations = {}
    hook_handles = []
    
    if use_geo_penalty:
        # Initialize penalty and move to the same device as the model
        device = next(model.parameters()).device
        geo_penalty = GeometricPenalty(
            alpha=geo_alpha, 
            beta=geo_beta, 
            margin=geo_margin, 
            sigmoid_steepness=geo_steepness
        ).to(device)
        
        def get_activation(name):
            def hook(model, input, output):
                captured_activations[name] = output
            return hook

        # Attach hooks to ALL linear and convolutional layers dynamically
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handle = module.register_forward_hook(get_activation(name))
                hook_handles.append(handle)

    # ----------------------
    # Training Loop
    # ----------------------
    for epoch in trange(epochs, desc="Training", leave=False, disable=disable_progress):
        
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        num_correct = 0
        total_samples = 0

        for x, y in train_data:
            optimizer.zero_grad()
            if use_geo_penalty:
                captured_activations.clear()

            x = x.float()
            y = y.long()

            y_hat = model(x)  
            loss = loss_fn(y_hat, y)
            
            # Apply geometric penalty to hidden layers if toggled
            if use_geo_penalty and len(captured_activations) > 0:
                # Dynamically identify and remove the final layer's output
                last_layer_key = list(captured_activations.keys())[-1]
                captured_activations.pop(last_layer_key)
                
                # Apply penalty to all remaining hidden layers
                for layer_name, pre_acts in captured_activations.items():
                    loss += geo_penalty(pre_acts, y)
                    
            loss.backward()
            optimizer.step()

            preds = y_hat.argmax(dim=1)
            running_loss += loss.item()
            num_correct += (preds == y).sum().item()
            total_samples += y.size(0)

        train_loss[epoch] = running_loss / len(train_data)
        train_accuracy[epoch] = num_correct / total_samples

        # --- Validation and evaluation Phase ---
        model.eval()
        
        # Eval on train data
        train_running_loss = 0.0
        train_num_correct = 0
        train_total_samples = 0
        with torch.no_grad():
            for x, y in train_data:
                if use_geo_penalty:
                    captured_activations.clear()
                    
                x = x.float()
                y = y.long()
                
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                
                if use_geo_penalty and len(captured_activations) > 0:
                    last_layer_key = list(captured_activations.keys())[-1]
                    captured_activations.pop(last_layer_key)
                    for layer_name, pre_acts in captured_activations.items():
                        loss += geo_penalty(pre_acts, y)
                
                preds = y_hat.argmax(dim=1)
                train_running_loss += loss.item()
                train_num_correct += (preds == y).sum().item()
                train_total_samples += y.size(0)
                
        eval_train_loss[epoch] = train_running_loss / len(train_data)
        eval_train_accuracy[epoch] = train_num_correct / train_total_samples
            
        # Eval on test data
        test_running_loss = 0.0
        test_num_correct = 0
        test_total_samples = 0
        with torch.no_grad():
            for x, y in test_data:
                if use_geo_penalty:
                    captured_activations.clear()
                    
                x = x.float()
                y = y.long()

                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                
                if use_geo_penalty and len(captured_activations) > 0:
                    last_layer_key = list(captured_activations.keys())[-1]
                    captured_activations.pop(last_layer_key)
                    for layer_name, pre_acts in captured_activations.items():
                        loss += geo_penalty(pre_acts, y)
                        
                preds = y_hat.argmax(dim=1)
                test_running_loss += loss.item()
                test_num_correct += (preds == y).sum().item()
                test_total_samples += y.size(0)

        test_loss[epoch] = test_running_loss / len(test_data)
        test_accuracy[epoch] = test_num_correct / test_total_samples

        # ----------------------
        # Save state Logic
        # ----------------------
        should_save = False
        if save_for_epochs is not None and epoch in save_for_epochs:
            should_save = True
        if save_everyth_epoch is not None:
            if (epoch % save_everyth_epoch == 0) or (epoch == epochs - 1):
                should_save = True

        if should_save:
            state = copy.deepcopy(model.state_dict())
            if on_save_callback:
                metrics = {
                    "train_loss": train_loss[epoch],
                    "train_accuracy": train_accuracy[epoch],
                    "test_loss": test_loss[epoch],
                    "test_accuracy": test_accuracy[epoch],
                    "eval_train_loss": eval_train_loss[epoch],
                    "eval_train_accuracy": eval_train_accuracy[epoch],
                }
                on_save_callback(epoch, state, metrics)
            elif RETURN_STATES or SAVE_STATES:
                saved_states[epoch] = state

    # ----------------------
    # Cleanup Hooks (Optional)
    # ----------------------
    if use_geo_penalty:
        for handle in hook_handles:
            handle.remove()

    # ----------------------
    # Compile results
    # ----------------------
    run_results = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "eval_train_loss": eval_train_loss,
        "eval_train_accuracy": eval_train_accuracy,
    }
    results = pd.DataFrame.from_dict(run_results)

    if SAVE_STATES and savepath is not None and not on_save_callback:
        createfolders(savepath / "state_dicts")
        for epoch, state in saved_states.items():
            torch.save(state, savepath / "state_dicts" / f"epoch{epoch}.pth")
        (savepath / "run_summary.csv").write_text(results.to_csv(index=False))
    elif not on_save_callback and not RETURN_STATES:
        print("Training not saved.")

    return (results,) if not RETURN_STATES else (results, saved_states)