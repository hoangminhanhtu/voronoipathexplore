import json
import math
import matplotlib.pyplot as plt
from pathlib import Path

# ── 1) Point at your folder ───────────────────────────────────
laser_dir = Path("list_file_laser")
assert laser_dir.is_dir(), f"{laser_dir} not found"

# ── 2) Loop over every .js in there ──────────────────────────
for js_path in laser_dir.glob("*.js"):
    print(f"\n▶ Processing {js_path.name}")
    
    # 2A) Read + strip comments
    with js_path.open(encoding="utf-8") as f:
        lines = []
        for line in f:
            if line.lstrip().startswith("//") or not line.strip():
                continue
            lines.append(line)
    clean = "".join(lines)

    # 2B) Parse JSON
    data = json.loads(clean)
    laser_pts = data["laser"]
    print(f"  → loaded {len(laser_pts)} points")

    # ── 3) Quick plot of the built‐in x/y ──────────────────────
    xs = [p["x"] for p in laser_pts]
    ys = [p["y"] for p in laser_pts]
    plt.figure(figsize=(5,5))
    plt.scatter(xs, ys, s=1)
    plt.axis("equal")
    plt.title(f"{js_path.name} – raw x/y")
    plt.show()

    # ── 4) (Optional) rebuild from distance/angle ──────────────
    MAX_RANGE = 20_000  # mm
    xs2, ys2 = [], []
    for p in laser_pts:
        r = p["distance"]
        if 0 < r < MAX_RANGE:
            θ = p["angle"]
            xs2.append(r * math.cos(θ))
            ys2.append(r * math.sin(θ))
    plt.figure(figsize=(5,5))
    plt.scatter(xs2, ys2, s=1, c="crimson")
    plt.axis("equal")
    plt.title(f"{js_path.name} – ≤{MAX_RANGE/1000:.0f} m rebuild")
    plt.show()
