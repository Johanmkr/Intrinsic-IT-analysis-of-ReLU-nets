"""Loading the stored results of steps 2 and 3 (``results/routing_*.csv``, ``results/baselines_*.csv``)."""

from pathlib import Path

import pandas as pd

# Protocol of the submitted paper; see run_estimate.py for the alternatives.
DEFAULT_PROTOCOL = "heldout"


def load_routing(results_dir: Path, sweep: str, protocol: str = DEFAULT_PROTOCOL,
                 labels: str = "true_labels") -> pd.DataFrame:
    """Routing-MI rows of one sweep for one estimation protocol and label set."""
    df = pd.read_csv(Path(results_dir) / f"routing_{sweep}.csv")
    keep = (df["protocol"] == protocol) & (df["labels"] == labels)
    if not keep.any():
        raise ValueError(f"routing_{sweep}.csv has no protocol={protocol!r}, labels={labels!r} rows")
    return df[keep].reset_index(drop=True)


def load_baselines(results_dir: Path, sweep: str, protocol: str = DEFAULT_PROTOCOL) -> pd.DataFrame:
    """Step-3 baseline rows (``results/baselines_<sweep>.csv``) for one protocol."""
    df = pd.read_csv(Path(results_dir) / f"baselines_{sweep}.csv")
    keep = df["protocol"] == protocol
    if not keep.any():
        raise ValueError(f"baselines_{sweep}.csv has no protocol={protocol!r} rows")
    return df[keep].reset_index(drop=True)
