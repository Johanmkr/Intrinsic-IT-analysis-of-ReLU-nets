"""Writing and loading the aggregated results of steps 2–4 (``results/*.csv.gz``).

The aggregated tables are gzipped CSVs so they can be kept in git. The gzip
timestamp is fixed, so rerunning a step on unchanged results rewrites
byte-identical files.
"""

from pathlib import Path

import pandas as pd

# Protocol of the submitted paper; see run_estimate.py for the alternatives.
DEFAULT_PROTOCOL = "heldout"

SUFFIX = ".csv.gz"
_GZIP = {"method": "gzip", "mtime": 0}


def write_table(df: pd.DataFrame, path: Path) -> None:
    """Write an aggregated table as gzipped CSV (``path`` ends in ``.csv.gz``)."""
    df.to_csv(path, index=False, compression=_GZIP)


def load_routing(results_dir: Path, sweep: str, protocol: str = DEFAULT_PROTOCOL,
                 labels: str = "true_labels") -> pd.DataFrame:
    """Routing-MI rows of one sweep for one estimation protocol and label set."""
    df = pd.read_csv(Path(results_dir) / f"routing_{sweep}{SUFFIX}")
    keep = (df["protocol"] == protocol) & (df["labels"] == labels)
    if not keep.any():
        raise ValueError(f"routing_{sweep}{SUFFIX} has no protocol={protocol!r}, labels={labels!r} rows")
    return df[keep].reset_index(drop=True)


def load_baselines(results_dir: Path, sweep: str, protocol: str = DEFAULT_PROTOCOL) -> pd.DataFrame:
    """Step-3 baseline rows (``results/baselines_<sweep>.csv.gz``) for one protocol."""
    df = pd.read_csv(Path(results_dir) / f"baselines_{sweep}{SUFFIX}")
    keep = df["protocol"] == protocol
    if not keep.any():
        raise ValueError(f"baselines_{sweep}{SUFFIX} has no protocol={protocol!r} rows")
    return df[keep].reset_index(drop=True)
