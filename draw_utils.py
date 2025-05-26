#!/usr/bin/env python3
"""Shared drawing helpers for planners."""

from typing import Iterable

import numpy as np
from matplotlib.patches import Circle
from shapely.geometry import Polygon, MultiPolygon
from scipy.spatial import Delaunay

__all__ = [
    "circumradius",
    "draw_beta_triangles",
    "draw_beta_shape",
    "draw_tree",
    "draw_path",
]


def circumradius(pts: np.ndarray) -> float:
    a = np.linalg.norm(pts[1] - pts[0])
    b = np.linalg.norm(pts[2] - pts[1])
    c = np.linalg.norm(pts[0] - pts[2])
    s = (a + b + c) / 2
    area = max(s * (s - a) * (s - b) * (s - c), 0.0) ** 0.5
    return (a * b * c) / (4 * area) if area > 1e-6 else float("inf")


def draw_beta_triangles(ax, points: np.ndarray, beta: float) -> None:
    if len(points) < 3:
        return
    tri = Delaunay(points)
    drawn = False
    for simplex in tri.simplices:
        verts = points[simplex]
        if circumradius(verts) <= beta:
            loop = np.vstack([verts, verts[0]])
            lbl = "β-triangle" if not drawn else ""
            ax.plot(loop[:, 0], loop[:, 1], color="green", linewidth=1.0, alpha=0.6, label=lbl)
            drawn = True


def draw_beta_shape(ax, shape: Polygon) -> None:
    geoms = shape.geoms if isinstance(shape, MultiPolygon) else [shape]
    for poly in geoms:
        x, y = poly.exterior.xy
        ax.plot(x, y, color="blue", linewidth=1.0, alpha=0.5, label="β-shape")


def draw_tree(ax, tree: Iterable) -> None:
    for node in tree:
        parent = getattr(node, "parent", None)
        if parent is not None:
            xs = [node.pos[0], parent.pos[0]]
            ys = [node.pos[1], parent.pos[1]]
            ax.plot(xs, ys, color="gray", linewidth=0.5)


def draw_path(ax, path: np.ndarray, radius: float, *, color: str = "red", label: str = "final path") -> None:
    ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0, label=label)
    ax.add_patch(Circle(tuple(path[0]), radius, alpha=0.3, label="start"))
