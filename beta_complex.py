#!/usr/bin/env python3
"""β-complex edge construction from point clouds."""

import math
from typing import Set, Tuple

import numpy as np
from scipy.spatial import Delaunay

__all__ = ["build_beta_complex_edges"]


def build_beta_complex_edges(pts: np.ndarray, beta: float) -> Set[Tuple[int, int]]:
    """Return set of edge index pairs belonging to the β-complex."""
    edges: Set[Tuple[int, int]] = set()
    if pts.shape[0] < 3:
        return edges

    tri = Delaunay(pts)
    for simplex in tri.simplices:
        pa, pb, pc = pts[simplex]
        a = np.linalg.norm(pb - pc)
        b = np.linalg.norm(pa - pc)
        c = np.linalg.norm(pa - pb)
        s = (a + b + c) / 2
        area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        if area < 1e-8:
            continue
        R = (a * b * c) / (4 * area)
        if R > beta:
            continue
        for i, j in ((0, 1), (1, 2), (2, 0)):
            u, v = simplex[i], simplex[j]
            if np.linalg.norm(pts[u] - pts[v]) <= beta:
                edges.add((u, v))
    return edges
