import argparse
import yaml
import h5py
import torch
import numpy as np
import sys
from pathlib import Path
import random
import os

# Fix imports to work as a module
if __name__ == "__main__" and __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    __package__ = "src_experiment"

from .dataset import get_new_data
from .utils import NeuralNet, LeNet5
from .train_models import train_model_multiclass

def set_global_seed(seed: int):
    """
    Sets the seed for all sources of randomness to ensure reproducibility.
    """
    if seed is None:
        return

    # 1. Python built-in random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # 2. Numpy
    np.random.seed(seed)

    # 3. PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.

    # 4. CuDNN (GPU Determinism)
    # This ensures that CUDA selects deterministic algorithms, 
    # possibly at the cost of a small performance hit.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"✅ Global seed set to: {seed}")

def infer_dataset_properties(dataloader):
    """
    Dynamically determines input shape and num_classes from a DataLoader.

    Returns `(input_size, num_classes, input_shape)` where `input_size` is
    the flattened feature dim (preserves the existing 2D-input contract)
    and `input_shape` is the per-sample shape tuple (e.g. `(1, 28, 28)`
    for full-image MNIST).
    """
    dummy_x, _ = next(iter(dataloader))
    input_shape = tuple(dummy_x.shape[1:])
    input_size = int(np.prod(input_shape))

    if hasattr(dataloader.dataset, 'tensors'):
        all_labels = dataloader.dataset.tensors[1]
        num_classes = len(torch.unique(all_labels))
    else:
        print("Inferring num_classes by iterating full dataset...")
        all_labels = []
        for _, y in dataloader:
            all_labels.append(y)
        all_labels = torch.cat(all_labels)
        num_classes = len(torch.unique(all_labels))

    return input_size, num_classes, input_shape

def run(config_path, overwrite=False):
    # 1. Load Config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create output/experiment_name/experiment_name.h5
    base_out_dir = Path(config.get("output_dir", "./"))
    exp_name = config.get("experiment_name", "experiment")
    model_seed = config.get("model_seed", 0)
    
    # New specific directory for this experiment
    exp_dir = base_out_dir / exp_name
    h5_path = exp_dir / f"seed_{model_seed}.h5"
    
    if h5_path.exists() and not overwrite:
        print(f"Error: HDF5 file {h5_path} already exists. Use --overwrite to force re-computation.")
        return

    
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    
    
    # --- SEEDING LOGIC ---
    # Global seed for data splitting and shuffling
    global_seed = config.get("global_seed", 42)
    set_global_seed(global_seed) 
    
    # Model seed for weight initialization
    # model_seed = config.get("model_seed", 123)

    # 2. Prepare Data
    dataset_name = config.get("dataset", "moons")
    noise = config.get("noise", 0.0)
    batch_size = config.get("batch_size", 32)
    
    dataset_kwargs = {}
    if "centers" in config:
        dataset_kwargs["centers"] = config["centers"]
    if "n_features" in config:
        dataset_kwargs["n_features"] = config["n_features"]
    if "target_dim" in config:
        dataset_kwargs["target_dim"] = config["target_dim"]
    
    print(f"Loading {dataset_name}...")
    train_loader, test_loader = get_new_data(
        dataset_name, 
        noise=noise, 
        batch_size=batch_size,
        split_seed = global_seed,
        **dataset_kwargs
    )
    
    # --- DYNAMIC INFERENCE ---
    input_size, num_classes, input_shape = infer_dataset_properties(train_loader)

    # Config overrides
    if "input_size" in config: input_size = config["input_size"]
    if "num_classes" in config: num_classes = config["num_classes"]

    # 3. Initialize Model — dispatch on arch_type (default MLP for back-compat).
    arch_type = config.get("arch_type", "mlp").lower()
    if arch_type == "mlp":
        model = NeuralNet(
            input_size=input_size,
            hidden_sizes=config.get("architecture", [10, 10]),
            num_classes=num_classes,
            dropout=config.get("dropout", 0.0),
            seed=model_seed,
        )
    elif arch_type == "lenet5":
        model = LeNet5(
            conv_channels=tuple(config.get("conv_channels", (6, 16))),
            fc_widths=tuple(config.get("fc_widths", (120, 84))),
            num_classes=num_classes,
            input_shape=tuple(config.get("input_shape", input_shape)),
            kernel_size=config.get("kernel_size", 5),
            pool_size=config.get("pool_size", 2),
            seed=model_seed,
        )
    else:
        raise ValueError(f"Unknown arch_type: {arch_type!r}")

    # 4. Setup HDF5 Incremental Writer
    print(f"Initializing HDF5 file at {h5_path}...")
    
    with h5py.File(h5_path, 'w') as f:
        # A. Save Config Metadata
        meta_grp = f.create_group('metadata')
        config['inferred_input_size'] = input_size
        config['inferred_num_classes'] = num_classes
        config['inferred_input_shape'] = list(input_shape)
        config['inferred_arch_type'] = arch_type
        
        for k, v in config.items():
            try:
                if isinstance(v, list):
                    meta_grp.attrs[k] = np.array(v) if len(v) > 0 else "[]"
                else:
                    meta_grp.attrs[k] = v if v is not None else "None"
            except TypeError:
                meta_grp.attrs[k] = str(v)

        # Create groups to be filled during training
        epochs_grp = f.create_group('epochs')
        
        # B. Define Callback
        def on_save_callback(epoch, state_dict, metrics):
            print(f"Saving epoch {epoch} to HDF5...")
            ep_grp = epochs_grp.create_group(f'epoch_{epoch}')
            
            for m_key, m_val in metrics.items():
                ep_grp.attrs[m_key] = m_val

            for name, tensor in state_dict.items():
                ep_grp.create_dataset(name, data=tensor.cpu().numpy(), compression="gzip")
            
            f.flush()

        # 5. Train
        print(f"Starting training on {dataset_name} (Input: {input_size}, Classes: {num_classes})...")
        
        # --- FIXED CALL HERE ---
        print(f"Starting training on {dataset_name}...")
        train_results = train_model_multiclass(
            model=model,
            train_data=train_loader,
            test_data=test_loader,
            epochs=config.get("epochs", 100),
            num_classes=num_classes,
            sgd_lr=config.get("learning_rate", 0.01),
            sgd_mom=config.get("momentum", 0.9),
            save_everyth_epoch=config.get("save_interval", None),
            save_for_epochs=config.get("save_epochs", None),
            on_save_callback=on_save_callback,
            savepath=None,
            SAVE_STATES=False,
            RETURN_STATES=False,
            disable_progress=True 
        )
        
        # Safely extract results (works if return is (df,) or (df, dict))
        results_df = train_results[0]

        # 6. Save Final Training Curve
        res_grp = f.create_group('training_results')
        for col in results_df.columns:
            res_grp.create_dataset(col, data=results_df[col].values)
        
        all_points = []
        all_labels = []
        with torch.no_grad():
            for x, y in test_loader:
                # Flatten the inputs (e.g., for images, or just keep 2D as 2D)
                x = x.view(x.size(0), -1)
                
                # Append both to our lists
                all_points.append(x.cpu().numpy())
                all_labels.append(y.cpu().numpy())
                
        # Concatenate and save as separate datasets
        f.create_dataset("points", data=np.concatenate(all_points, axis=0))
        f.create_dataset("labels", data=np.concatenate(all_labels, axis=0))
        
        # Save test dataloader to the file 
    print("Experiment complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to config.yaml file")
    args = parser.parse_args()
    run(args.config)