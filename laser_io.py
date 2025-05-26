#!/usr/bin/env python3
"""Utilities for loading and filtering LIDAR scan data."""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = ["load_points", "load_and_filter"]


def _load_json(js_path: Path) -> dict:
    """Return JSON data from a .js file ignoring // comments."""
    with js_path.open(encoding="utf-8") as f:
        text = "".join(
            line for line in f
            if not line.lstrip().startswith("//") and line.strip()
        )
    return json.loads(text)


def load_points(js_path: Path, max_range: float) -> np.ndarray:
    """Return Nx2 array of points from a laser scan JSON file."""
    if not js_path.is_file():
        raise FileNotFoundError(f"Laser file not found: {js_path}")
    data = _load_json(js_path)
    pts = []
    for p in data.get("laser", []):
        r, theta = p.get("distance", 0.0), p.get("angle", 0.0)
        if 0 < r < max_range:
            pts.append((r * math.cos(theta), r * math.sin(theta)))
    return np.array(pts)


def load_and_filter(
    js_path: Path,
    max_range: float,
    *,
    start: Optional[np.ndarray] = None,
    exclude_radius: float = 0.0,
) -> np.ndarray:
    """Load scan points and drop any within ``exclude_radius`` of ``start``."""
    pts = load_points(js_path, max_range)
    if start is not None and exclude_radius > 0.0 and pts.size:
        mask = np.linalg.norm(pts - start, axis=1) > exclude_radius
        pts = pts[mask]
    return pts
