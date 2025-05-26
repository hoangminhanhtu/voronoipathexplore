#!/usr/bin/env python3
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrow

# === CONFIGURABLE PARAMETERS ===
LASER_FILE         = Path("list_file_laser/FileLaserPoint6.js")
MAX_RANGE          = 10.0             # max sensor range in meters
ROBOT_DIAMETER     = 0.6              # robot diameter in meters
ROBOT_RADIUS       = ROBOT_DIAMETER / 2
START              = np.array([0.0, 0.0])
GOAL               = np.array([-3.0, -5.0])
GOAL_BIAS          = 0.1              # probability of sampling GOAL in RRT
MAX_ITERS          = 5000             # max RRT iterations
ANIMATION_INTERVAL = 1000             # ms between frames

def load_obstacles(js_path: Path) -> np.ndarray:
    if not js_path.is_file():
        raise FileNotFoundError(f"Laser file not found: {js_path}")
    with js_path.open(encoding="utf-8") as f:
        data_str = "".join(
            line for line in f
            if not line.lstrip().startswith("//") and line.strip()
        )
    data = json.loads(data_str)
    pts = []
    for p in data.get("laser", []):
        r, θ = p.get("distance", 0.0), p.get("angle", 0.0)
        if 0 < r < MAX_RANGE:
            x = r * math.cos(θ)
            y = r * math.sin(θ)
            pts.append((x, y))
    return np.array(pts) if pts else np.empty((0,2))

class Node:
    __slots__ = ("pos", "parent")
    def __init__(self, pos, parent=None):
        self.pos = np.array(pos)
        self.parent = parent

def nearest_node(tree, pt):
    dists = [np.linalg.norm(n.pos - pt) for n in tree]
    return tree[int(np.argmin(dists))]

def steer(from_node, to_pt, step):
    vec = to_pt - from_node.pos
    dist = np.linalg.norm(vec)
    return to_pt if dist < step else from_node.pos + vec/dist * step

def collision(pt, obstacles, robot_rad):
    if obstacles.size == 0:
        return False
    return np.any(np.linalg.norm(obstacles - pt, axis=1) <= robot_rad)

def rrt_plan(start, goal, obstacles):
    t0 = time.perf_counter() 
    tree = [Node(start)]
    goal_node = None
    for _ in range(MAX_ITERS):
        sample = goal if random.random() < GOAL_BIAS else np.random.uniform(-MAX_RANGE, MAX_RANGE, 2)
        nearest = nearest_node(tree, sample)
        new_pt = steer(nearest, sample, step=ROBOT_DIAMETER)
        if not collision(new_pt, obstacles, robot_rad=ROBOT_RADIUS):
            node = Node(new_pt, nearest)
            tree.append(node)
            if np.linalg.norm(new_pt - goal) < ROBOT_DIAMETER:
                goal_node = Node(goal, node)
                tree.append(goal_node)
                break

    end = goal_node or nearest
    path = []
    while end:
        path.append(end.pos)
        end = end.parent

    elapsed = time.perf_counter() - t0    # ← end timing
    print(f"RRT planning took {elapsed:.6f} seconds")

    return path[::-1], tree

def animate_path(path_pts, tree, obstacles):
    n_frames = len(path_pts)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal')
    ax.set_xlim(-MAX_RANGE, MAX_RANGE)
    ax.set_ylim(-MAX_RANGE, MAX_RANGE)

    # static draw: obstacles, tree edges, planned path
    ax.scatter(obstacles[:,0], obstacles[:,1], s=4, c='black', label='obstacles')
    for node in tree:
        if node.parent:
            x0,y0 = node.pos; x1,y1 = node.parent.pos
            ax.plot([x0,x1],[y0,y1], '-', lw=0.5, c='gray')
    ax.plot(path_pts[:,0], path_pts[:,1], '--', lw=1.5, c='blue', label='RRT path')
    ax.legend(loc='upper left')

    def update(i):
        pos = path_pts[min(i, n_frames-1)]
        print(f"[Frame {i+1}/{n_frames}] Robot at {pos}")

        # remove previous robot and text
        for art in list(ax.patches):
            art.remove()
        for txt in list(ax.texts):
            txt.remove()

        # compute heading
        nxt = path_pts[i+1] if i+1 < n_frames else pos
        θ = math.atan2(nxt[1]-pos[1], nxt[0]-pos[0])

        # draw robot footprint
        c = Circle(pos, ROBOT_RADIUS, edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(c)

        # draw heading arrow
        L = ROBOT_RADIUS * 1.5
        dx, dy = L*math.cos(θ), L*math.sin(θ)
        arr = FancyArrow(pos[0], pos[1], dx, dy,
                         width=0.02, length_includes_head=True, color='green')
        ax.add_patch(arr)

        # frame label
        ax.text(0.95, 0.05, f"Frame {i+1}",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # keep a reference to anim so it isn't garbage-collected
    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=ANIMATION_INTERVAL, repeat=False)
    plt.show()
    return anim

def main():
    obstacles = load_obstacles(LASER_FILE)
    start_time = time.perf_counter()
    path, tree = rrt_plan(START, GOAL, obstacles)
    end_time = time.perf_counter()
    print(f"RRT planning computation time: {end_time - start_time:.4f} seconds")
    path_pts = np.array(path)
    # capture anim so it lives until plt.show() returns
    _anim = animate_path(path_pts, tree, obstacles)

if __name__ == "__main__":
    main()
