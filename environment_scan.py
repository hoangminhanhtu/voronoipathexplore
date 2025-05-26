#!/usr/bin/env python3
import json
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrow
from scipy.spatial import Delaunay

# === PARAMETERS ===
MAX_RANGE = 10.0     # max laser range in meters
BETA      = 0.6      # β‐complex threshold = robot diameter in meters
JS_DIR    = Path("list_file_laser")
JS_FILE   = JS_DIR / "FileLaserPoint6.js"  # continuously updated scan file


def load_and_clean(js_path: Path) -> dict:
    """Load JSON from a .js file (strip out // comments)."""
    with js_path.open(encoding="utf-8") as f:
        lines = [
            line for line in f
            if not line.lstrip().startswith("//") and line.strip()
        ]
    return json.loads("".join(lines))


def compute_scan_points(data: dict, max_range: float=MAX_RANGE):
    """
    Extracts valid scan points and heading from loaded data.
    Returns pts (Nx2 array) and heading (float, radians).
    """
    laser_pts = data.get("laser", [])
    heading   = data.get("heading", 0.0)
    pts = []
    for p in laser_pts:
        r = p.get("distance", 0.0)
        if 0 < r < max_range:
            θ = p.get("angle", 0.0)
            pts.append((r * math.cos(θ), r * math.sin(θ)))
    return np.array(pts), heading


"""
    Computes the β‐complex edges from scan points based on Delaunay triangulation.
    Returns a set of index pairs representing edges.
    """

def build_beta_complex_edges(pts: np.ndarray, beta: float=BETA):
    edges = set()
    if pts.shape[0] < 3:
        return edges

    tri = Delaunay(pts)
    for simplex in tri.simplices:
        pa, pb, pc = pts[simplex]
        # compute triangle circumradius R …
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

        # now only add those edges which individually obey length ≤ beta
        for (i, j) in ((0,1), (1,2), (2,0)):
            u, v = simplex[i], simplex[j]
            if np.linalg.norm(pts[u] - pts[v]) <= beta:
                # OR if you really want the half-radius test:
                # if (np.linalg.norm(pts[u] - pts[v]) / 2) <= beta:
                edges.add((u, v))

    return edges


def draw(js_path: Path, ax, max_range: float=MAX_RANGE, beta: float=BETA):
    data = load_and_clean(js_path)
    pts, heading = compute_scan_points(data, max_range)

    ax.clear()
    ax.set_aspect('equal')
    ax.set_title(f"{js_path.name} — range≤{max_range:.0f} m, β={beta:.2f} m")

    # Raw scan points
    if pts.size:
        ax.scatter(pts[:,0], pts[:,1], s=2, color='black')

    # Robot footprint
    ax.add_patch(Circle((0, 0), beta/2, edgecolor='red', facecolor='none', lw=1.5))

    # Heading arrow
    arrow_len = (beta/2) * 1.5
    dx = arrow_len * math.cos(heading)
    dy = arrow_len * math.sin(heading)
    ax.add_patch(FancyArrow(
        0, 0, dx, dy,
        width=0.02, length_includes_head=True, color='green'
    ))

    # β‐complex edges
    edges = build_beta_complex_edges(pts, beta)
    for i, j in edges:
        x0, y0 = pts[i]
        x1, y1 = pts[j]
        ax.plot([x0, x1], [y0, y1], '-', lw=1.0, color='blue')

    # Viewport limits
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)


def main():
    if not JS_DIR.is_dir():
        raise FileNotFoundError(f"{JS_DIR!r} not found")
    if not JS_FILE.is_file():
        raise FileNotFoundError(f"{JS_FILE!r} not found")

    fig, ax = plt.subplots(figsize=(12,12))

    def update(frame_idx):
        draw(JS_FILE, ax)
        ax.text(
            0.95, 0.05,
            f"Frame {frame_idx+1}",
            transform=ax.transAxes,
            ha='right', va='bottom',
            color='blue', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.6)
        )

    anim = FuncAnimation(
        fig,
        update,
        frames=10,
        interval=1000,
        repeat=False,
        blit=False
    )

    plt.show()

if __name__ == "__main__":
    main()
