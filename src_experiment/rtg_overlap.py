"""Experiment 3 helper: routing-loss proxy on the data-supported RTG.

Sec 3.2 of the paper defines the data-supported region transition graph
(RTG) on layer-l cumulative activation patterns: vertices are occupied
regions, edges are Hamming-1 (single-bit-flip) adjacencies. Prop 4.2
says the routing-loss term I(Y;Π|T) is positive iff there exist
adjacent regions in the RTG whose dominant classes disagree.

`routing_loss_proxy` returns the empirical fraction of edges with
disagreeing dominant-class endpoints — a tractable surrogate for the
condition Prop 4.2 requires. It is ε-independent (only the bare RTG
plus per-region label majorities are used).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List

import numpy as np


def region_dominant_class(omega_ids: np.ndarray, y: np.ndarray) -> Dict[bytes, int]:
    """Map each region id to the majority class of its routed samples.

    Ties broken toward the lowest class index (Counter.most_common returns
    insertion-order on ties; we sort to make it deterministic).
    """
    by_rid: Dict[bytes, List[int]] = {}
    for w, label in zip(omega_ids, y):
        by_rid.setdefault(w, []).append(int(label))

    out: Dict[bytes, int] = {}
    for rid, labels in by_rid.items():
        c = Counter(labels)
        # most_common is order-stable; sort items to guarantee determinism on ties.
        ordered = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
        out[rid] = ordered[0][0]
    return out


def routing_loss_proxy(
    adjacency: Dict[bytes, List[bytes]],
    dominant: Dict[bytes, int],
) -> float:
    """Fraction of unique Hamming-1 RTG edges whose dominant classes disagree.

    Returns NaN if the RTG has no edges (e.g. one isolated region per
    sample, or only one occupied region).
    """
    seen: set = set()
    disagree = 0
    total = 0
    for rid, nbrs in adjacency.items():
        for nbr in nbrs:
            edge = (rid, nbr) if rid < nbr else (nbr, rid)
            if edge in seen:
                continue
            seen.add(edge)
            total += 1
            if dominant.get(rid) != dominant.get(nbr):
                disagree += 1
    if total == 0:
        return float("nan")
    return disagree / total
