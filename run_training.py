import argparse
import sys
from pathlib import Path

# Ensure src_experiment is importable
sys.path.append(str(Path(__file__).resolve().parent))

from src_experiment.run_experiment import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment pipeline.")
    
    # Positional argument
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    
    # Optional flag
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing results if they exist."
    )
    
    args = parser.parse_args()
    
    # Execute the experiment passing both the config path and the overwrite flag
    run(args.config, overwrite=args.overwrite)