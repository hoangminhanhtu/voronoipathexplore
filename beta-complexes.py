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
MAX_RANGE = 10.0     # max laser range in meters (now 10 m)
BETA      = 0.6      # beta‐complex threshold = robot diameter in meters

def load_and_clean(js_path):
    """Load JSON from a .js file (strip out // comments)."""
    with js_path.open(encoding="utf-8") as f:
        lines = [
            l for l in f
            if not l.lstrip().startswith("//") and l.strip()
        ]
    return json.loads("".join(lines))

def draw(js_path, ax, max_range=MAX_RANGE, beta=BETA):
    data = load_and_clean(js_path)
    laser_pts = data.get("laser", [])
    heading   = data.get("heading", 0.0)  # in radians

    # --- build point list in meters (no conversion) ---
    pts = []
    for p in laser_pts:
        r_m = p.get("distance", 0)            # assumed in meters
        if 0 < r_m < max_range:
            θ = p.get("angle", 0.0)
            pts.append((r_m * math.cos(θ), r_m * math.sin(θ)))
    pts = np.array(pts)

    ax.clear()
    ax.set_aspect('equal')
    ax.set_title(f"{js_path.name}  —  range≤{max_range:.0f} m, β={beta:.2f} m")

    # --- raw scan points ---
    if pts.size:
        ax.scatter(pts[:,0], pts[:,1], s=2, color='black')

    # --- robot footprint (0.6 m diameter) ---
    footprint = Circle((0, 0), 0.3, edgecolor='red', facecolor='none', lw=1.5)
    ax.add_patch(footprint)

    # --- heading arrow (green) ---
    arrow_len = 0.3 * 1.5
    dx = arrow_len * math.cos(heading)
    dy = arrow_len * math.sin(heading)
    ax.add_patch(FancyArrow(
        0, 0, dx, dy,
        width=0.02, length_includes_head=True, color='green'
    ))

    # --- β-complex via filtered Delaunay ---
    if pts.shape[0] >= 3:
        tri = Delaunay(pts)
        edges = set()
        for simplex in tri.simplices:
            pa, pb, pc = pts[simplex]
            a = np.linalg.norm(pb - pc)
            b = np.linalg.norm(pa - pc)
            c = np.linalg.norm(pa - pb)
            s = (a + b + c) / 2
            area = math.sqrt(max(s*(s-a)*(s-b)*(s-c), 0.0))
            if area > 1e-8:
                R = (a * b * c) / (4 * area)
                if R <= beta:
                    edges.update([
                        (simplex[0], simplex[1]),
                        (simplex[1], simplex[2]),
                        (simplex[2], simplex[0]),
                    ])
        for i, j in edges:
            x0, y0 = pts[i]
            x1, y1 = pts[j]
            ax.plot([x0, x1], [y0, y1], '-', lw=1.0, color='blue')

    # --- viewport ---
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

def main():
    laser_dir = Path("list_file_laser")
    if not laser_dir.is_dir():
        raise FileNotFoundError(f"{laser_dir} not found")
    files = sorted(laser_dir.glob("*.js"))
    if not files:
        raise FileNotFoundError("No .js files in list_file_laser")

    fig, ax = plt.subplots(figsize=(6,6))
    anim = FuncAnimation(
        fig, lambda i: draw(files[i], ax),
        frames=len(files), interval=1000, repeat=True
    )
    plt.show()

if __name__ == "__main__":
    main()
