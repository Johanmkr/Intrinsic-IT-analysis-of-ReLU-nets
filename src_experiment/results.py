"""Loading the stored step-2 results (``results/routing_<sweep>.csv``)."""

from pathlib import Path

import pandas as pd

# Protocol of the submitted paper; see run_estimate.py for the alternatives.
DEFAULT_PROTOCOL = "heldout"


def load_routing(results_dir: Path, sweep: str, protocol: str = DEFAULT_PROTOCOL) -> pd.DataFrame:
    """Routing-MI rows of one sweep, restricted to one estimation protocol."""
    df = pd.read_csv(Path(results_dir) / f"routing_{sweep}.csv")
    if protocol not in set(df["protocol"]):
        raise ValueError(f"routing_{sweep}.csv has no {protocol!r} rows")
    return df[df["protocol"] == protocol].reset_index(drop=True)
