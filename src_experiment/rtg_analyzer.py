"""
Recipe 4 from `claude_new_estimator_instructions.md`: data-supported region
transition graph (RTG).

Edges connect regions whose cumulative activation patterns differ in exactly
one bit. Implements the spec's "faster variant": for each region of length
$\\sum_i n_i$ bits, enumerate the single-bit flips and probe membership,
yielding $O(|\\Omega_{\\mathcal D}| \\cdot \\sum_i n_i)$ work instead of
$O(|\\Omega_{\\mathcal D}|^2)$.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Pattern lookup helpers
# ---------------------------------------------------------------------------
def cumulative_patterns_per_region(
    per_layer_patterns: Sequence[np.ndarray],
    omega_ids: np.ndarray,
    layer: int,
) -> Dict[bytes, np.ndarray]:
    """Map each unique region ID to its cumulative bool pattern up to ``layer``.

    The region ID equals ``md5(np.packbits(cumulative).tobytes())`` by construction
    (see :func:`routing_estimator.cumulative_pattern_hashes`), so we can probe
    bit-flipped neighbours by recomputing the digest.
    """
    first_idx: Dict[bytes, int] = {}
    for i, w in enumerate(omega_ids):
        if w not in first_idx:
            first_idx[w] = i

    cumulative = np.concatenate(
        [p.astype(bool, copy=False) for p in per_layer_patterns[:layer]],
        axis=1,
    )
    return {rid: cumulative[i].copy() for rid, i in first_idx.items()}


def _hash_pattern(pattern: np.ndarray) -> bytes:
    return hashlib.md5(np.packbits(pattern).tobytes()).digest()


# ---------------------------------------------------------------------------
# Adjacency (single-bit-flip enumeration)
# ---------------------------------------------------------------------------
def hamming1_adjacency(
    region_cum_patterns: Dict[bytes, np.ndarray],
) -> Dict[bytes, List[bytes]]:
    """Build the Hamming-1 adjacency list of $\\Omega_{\\mathcal D}^{\\le l}$.

    Each edge $(\\omega, \\omega')$ appears in both endpoint lists.
    """
    if not region_cum_patterns:
        return {}

    n_bits = next(iter(region_cum_patterns.values())).shape[0]
    adjacency: Dict[bytes, List[bytes]] = {rid: [] for rid in region_cum_patterns}

    for rid, pat in region_cum_patterns.items():
        flipped = pat.copy()
        for j in range(n_bits):
            flipped[j] = ~flipped[j]
            nbr_rid = _hash_pattern(flipped)
            if nbr_rid in region_cum_patterns and nbr_rid != rid:
                adjacency[rid].append(nbr_rid)
            flipped[j] = ~flipped[j]
    return adjacency


# ---------------------------------------------------------------------------
# Connected components (union-find)
# ---------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self.parent[rx] = ry


def connected_components(
    adjacency: Dict[bytes, List[bytes]],
) -> Dict[bytes, List[bytes]]:
    """Return ``{root_rid: [member_rids]}`` for each component."""
    uf = _UnionFind(adjacency.keys())
    for rid, neighbours in adjacency.items():
        for nbr in neighbours:
            uf.union(rid, nbr)

    comps: Dict[bytes, List[bytes]] = defaultdict(list)
    for rid in adjacency:
        comps[uf.find(rid)].append(rid)
    return dict(comps)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
@dataclass
class RTGDiagnostics:
    num_regions: int
    num_components: int
    component_sizes: np.ndarray  # sorted descending
    largest_component_frac: float
    isolated_frac: float  # fraction of regions in size-1 components


def rtg_diagnostics(
    adjacency: Dict[bytes, List[bytes]],
) -> RTGDiagnostics:
    R = len(adjacency)
    if R == 0:
        return RTGDiagnostics(
            num_regions=0,
            num_components=0,
            component_sizes=np.array([], dtype=np.int64),
            largest_component_frac=float("nan"),
            isolated_frac=float("nan"),
        )

    comps = connected_components(adjacency)
    sizes = np.array(sorted((len(v) for v in comps.values()), reverse=True), dtype=np.int64)
    isolated = int((sizes == 1).sum())  # equals number of size-1 regions

    return RTGDiagnostics(
        num_regions=R,
        num_components=len(comps),
        component_sizes=sizes,
        largest_component_frac=float(sizes[0] / R),
        isolated_frac=float(isolated / R),
    )


# ---------------------------------------------------------------------------
# CLI: dump diagnostics for one (epoch, layer)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "usage: python -m src_experiment.rtg_analyzer <path/to/file.h5> <epoch> [layer]"
        )
        sys.exit(1)

    from src_experiment.routing_estimator import (
        RoutingEstimator,
        cumulative_pattern_hashes,
        forward_activation_patterns,
    )

    h5 = sys.argv[1]
    epoch = int(sys.argv[2])
    layer = int(sys.argv[3]) if len(sys.argv) > 3 else None

    est = RoutingEstimator(h5)
    W, b = est._load_weights(epoch)
    patterns = forward_activation_patterns(W, b, est.points)
    layers = [layer] if layer is not None else list(range(1, est.num_hidden_layers + 1))

    for l in layers:
        omega = cumulative_pattern_hashes(patterns, l)
        rmap = cumulative_patterns_per_region(patterns, omega, l)
        adj = hamming1_adjacency(rmap)
        d = rtg_diagnostics(adj)
        print(
            f"layer={l} R={d.num_regions} "
            f"components={d.num_components} "
            f"largest_frac={d.largest_component_frac:.3f} "
            f"isolated_frac={d.isolated_frac:.3f} "
            f"top_sizes={list(d.component_sizes[:5])}"
        )
