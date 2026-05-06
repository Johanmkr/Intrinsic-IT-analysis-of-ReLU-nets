from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt

class NeuralNet(nn.Module):
    """
    Fully-connected feedforward neural network with ReLU activations
    and Gaussian Dropout regularization.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.num_classes = num_classes
        self.input_size = input_size

        # ------------------------------------------------------------------
        # Build Layers
        # ------------------------------------------------------------------
        # Layers are built first (with default PyTorch init)
        self._build_layers()

        # ------------------------------------------------------------------
        # Custom Initialization with Safety Logic
        # ------------------------------------------------------------------
        if seed is not None:
            # 1. Determine safe devices for fork_rng.
            # 'devices' expects a list of GPU IDs (integers). 
            # CPU is ALWAYS included in the fork by default.
            gpu_devices = []
            if torch.cuda.is_available():
                gpu_devices = [torch.cuda.current_device()]
                
            # 2. Fork the RNG state. 
            # Passing gpu_devices=[] ensures we don't trigger CUDA errors on CPU-only machines.
            with torch.random.fork_rng(devices=gpu_devices):
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                    
                # Apply the custom initialization within this seeded scope
                self._init_weights()
        else:
            # No seed provided: initialize using the current global state
            self._init_weights()

    def _build_layers(self):
        """Constructs the architecture and registers modules."""
        # Hidden Layers
        for i, hidden_dim in enumerate(self.hidden_sizes):
            layer_name = f"l{i + 1}"
            relu_name = f"relu{i + 1}"
            dropout_name = f"dropout{i + 1}"

            in_features = self.input_size if i == 0 else self.hidden_sizes[i - 1]

            setattr(self, layer_name, nn.Linear(in_features, hidden_dim))
            setattr(self, relu_name, nn.ReLU())
            setattr(self, dropout_name, GaussianDropout(p=self.dropout))

        # Output Layer
        # Handle case where hidden_sizes might be empty (Linear model)
        last_dim = self.hidden_sizes[-1] if self.hidden_sizes else self.input_size
        
        output_layer_name = f"l{len(self.hidden_sizes) + 1}"
        setattr(self, output_layer_name, nn.Linear(last_dim, self.num_classes))

    def _init_weights(self) -> None:
        self.apply(self._initialization_logic)

    def _initialization_logic(self, m):
        if isinstance(m, nn.Linear):
            # Check if it's the output layer by comparing output features
            if m.out_features == self.num_classes:
                init.xavier_uniform_(m.weight)
            else:
                init.kaiming_normal_(m.weight, nonlinearity="relu")
            
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        # Forward pass through hidden layers
        for i in range(len(self.hidden_sizes)):
            out = getattr(self, f"l{i + 1}")(out)
            out = getattr(self, f"relu{i + 1}")(out)
            out = getattr(self, f"dropout{i + 1}")(out)

        # Output layer
        out = getattr(self, f"l{len(self.hidden_sizes) + 1}")(out)
        return out


class LeNet5(nn.Module):
    """Parametric LeNet-5 (Conv-Pool-Conv-Pool-FC-FC-FC) for Phase C.

    Default `(conv_channels=(6, 16), fc_widths=(120, 84))` matches the
    LeNet-S reference; sweep widths via the constructor.

    Layer naming mirrors `NeuralNet` so the HDF5 saver picks up
    `state_dict()` keys in a predictable order. Hidden conv/linear use
    Kaiming-normal init; the output linear uses Xavier-uniform; biases
    zero. Seed scope follows the same `fork_rng` pattern as
    `NeuralNet`.
    """

    def __init__(
        self,
        conv_channels: Sequence[int] = (6, 16),
        fc_widths: Sequence[int] = (120, 84),
        num_classes: int = 10,
        input_shape: Tuple[int, int, int] = (1, 28, 28),
        kernel_size: int = 5,
        pool_size: int = 2,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.conv_channels = tuple(conv_channels)
        self.fc_widths = tuple(fc_widths)
        self.num_classes = num_classes
        self.input_shape = tuple(input_shape)
        self.kernel_size = kernel_size
        self.pool_size = pool_size

        self._build_layers()

        if seed is not None:
            gpu_devices = []
            if torch.cuda.is_available():
                gpu_devices = [torch.cuda.current_device()]
            with torch.random.fork_rng(devices=gpu_devices):
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                self._init_weights()
        else:
            self._init_weights()

    def _build_layers(self):
        in_c, in_h, in_w = self.input_shape
        prev_c, prev_h, prev_w = in_c, in_h, in_w
        for i, out_c in enumerate(self.conv_channels):
            setattr(self, f"conv{i + 1}",
                    nn.Conv2d(prev_c, out_c, kernel_size=self.kernel_size))
            setattr(self, f"relu_c{i + 1}", nn.ReLU())
            setattr(self, f"pool{i + 1}",
                    nn.MaxPool2d(kernel_size=self.pool_size))
            # Track output spatial size after conv (no padding) + pool.
            prev_h = (prev_h - self.kernel_size + 1) // self.pool_size
            prev_w = (prev_w - self.kernel_size + 1) // self.pool_size
            prev_c = out_c
        self._flat_dim = prev_c * prev_h * prev_w
        if self._flat_dim <= 0:
            raise ValueError(
                f"Conv stack produced non-positive flat dim "
                f"({prev_c}×{prev_h}×{prev_w}) for input {self.input_shape}."
            )

        prev_dim = self._flat_dim
        for j, w in enumerate(self.fc_widths):
            setattr(self, f"fc{j + 1}", nn.Linear(prev_dim, w))
            setattr(self, f"relu_f{j + 1}", nn.ReLU())
            prev_dim = w
        out_idx = len(self.fc_widths) + 1
        setattr(self, f"fc{out_idx}", nn.Linear(prev_dim, self.num_classes))

    def _init_weights(self) -> None:
        self.apply(self._initialization_logic)

    def _initialization_logic(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            is_output = (
                isinstance(m, nn.Linear)
                and m.out_features == self.num_classes
            )
            if is_output:
                init.xavier_uniform_(m.weight)
            else:
                init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.view(x.size(0), *self.input_shape)
        out = x
        for i in range(len(self.conv_channels)):
            out = getattr(self, f"conv{i + 1}")(out)
            out = getattr(self, f"relu_c{i + 1}")(out)
            out = getattr(self, f"pool{i + 1}")(out)
        out = out.flatten(1)
        for j in range(len(self.fc_widths)):
            out = getattr(self, f"fc{j + 1}")(out)
            out = getattr(self, f"relu_f{j + 1}")(out)
        out = getattr(self, f"fc{len(self.fc_widths) + 1}")(out)
        return out


class GaussianDropout(nn.Module):
    def __init__(self, p=0.5):
        super(GaussianDropout, self).__init__()
        if not (0 <= p < 1):
            raise ValueError("p value should be in the range [0, 1)")
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0:
            stddev = (self.p / (1.0 - self.p))**0.5
            noise = torch.randn_like(x) * stddev + 1.0
            return x * noise
        else:
            return x


# ----------------------------------------------------------------------
# Filesystem Utilities
# ----------------------------------------------------------------------
def createfolders(*dirs: Path) -> None:
    """Create folders for storing data."""
    for dir in dirs:
        dir.mkdir(parents=True, exist_ok=True)
        

def savefig(fig, path):
    # Convert string to a Path object for easy manipulation
    filepath = Path(path)
    
    # Strip the extension if one was accidentally provided
    base_path = filepath.with_suffix('') 
    
    # Save high-res PNG and vector PDF, ensuring labels aren't cut off
    fig.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{base_path}.pdf", bbox_inches="tight")
    
    print(f"Saved: {base_path}.png and .pdf")
  


# ----------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Verifying Reproducibility & Safety ---")

    # 1. Establish a Global State
    torch.manual_seed(999)
    initial_rand = torch.rand(1).item()
    print(f"Global random check before models: {initial_rand:.4f}")

    # 2. Initialize Model A with a specific seed (e.g., 42)
    # Note: Architecture must handle inputs. Using logic compatible with your config.
    model_a = NeuralNet(input_size=10, hidden_sizes=[20], num_classes=2, seed=42)
    print("Initialized Model A (Seed 42)")

    # 3. Check Global State - IT SHOULD BE DIFFERENT (randomness consumed normally)
    # BUT IT SHOULD NOT BE RESET to 42.
    mid_rand = torch.rand(1).item()
    print(f"Global random check between models: {mid_rand:.4f}")
    
    # 4. Initialize Model B with the SAME seed (42)
    model_b = NeuralNet(input_size=10, hidden_sizes=[20], num_classes=2, seed=42)
    print("Initialized Model B (Seed 42)")

    # 5. Initialize Model C with DIFFERENT seed (123)
    model_c = NeuralNet(input_size=10, hidden_sizes=[20], num_classes=2, seed=123)
    print("Initialized Model C (Seed 123)")

    # --- TESTS ---
    
    # Check if A and B are identical
    w_a = model_a.l1.weight.data
    w_b = model_b.l1.weight.data
    w_c = model_c.l1.weight.data

    if torch.equal(w_a, w_b):
        print("✅ SUCCESS: Model A and Model B weights are IDENTICAL.")
    else:
        print("❌ FAILURE: Model A and B differ (Reproducibility failed).")

    if not torch.equal(w_a, w_c):
        print("✅ SUCCESS: Model A and Model C weights are DIFFERENT.")
    else:
        print("❌ FAILURE: Model A and C are identical (Seeding failed).")

    # Final Global Check
    final_rand = torch.rand(1).item()
    print(f"Global random check after models:  {final_rand:.4f}")
    
    if mid_rand != final_rand and mid_rand != initial_rand:
         print("✅ SUCCESS: Global random state progressed naturally without being reset.")