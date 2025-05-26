#!/usr/bin/env python3
import json
import math
import random
import time                   # ← add this
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, FancyArrow

# === CONFIGURABLE PARAMETERS ===
LASER_FILE          = Path("list_file_laser/FileLaserPoint6.js")
MAX_RANGE           = 10.0
ROBOT_DIAMETER      = 0.6
ROBOT_RADIUS        = ROBOT_DIAMETER/2
START               = np.array([0.0, 0.0])
GOAL                = np.array([-3.0, -5.0])
GOAL_BIAS           = 0.1
MAX_ITERS           = 5000
RRT_STAR_RADIUS     = 1.0
ANIMATION_INTERVAL  = 1000

def load_obstacles(js_path: Path) -> np.ndarray:
    if not js_path.is_file():
        raise FileNotFoundError(f"Laser file not found: {js_path}")
    with js_path.open(encoding="utf-8") as f:
        lines = [L for L in f if not L.lstrip().startswith("//") and L.strip()]
    data = json.loads("".join(lines))
    pts = []
    for p in data.get("laser", []):
        r, θ = p.get("distance", 0.0), p.get("angle", 0.0)
        if 0 < r < MAX_RANGE:
            pts.append((r*math.cos(θ), r*math.sin(θ)))
    return np.array(pts) if pts else np.empty((0,2))

class Node:
    __slots__ = ("pos", "parent", "cost")
    def __init__(self, pos, parent=None, cost=0.0):
        self.pos    = np.array(pos)
        self.parent = parent
        self.cost   = cost

def distance(a, b):
    return np.linalg.norm(a - b)

def collision(pt, obstacles, robot_rad=ROBOT_RADIUS):
    if obstacles.size == 0: return False
    return np.any(np.linalg.norm(obstacles - pt, axis=1) <= robot_rad)

def collision_line(p1, p2, obstacles, robot_rad=ROBOT_RADIUS):
    dist = distance(p1, p2)
    steps = max(int(dist/(robot_rad/2)), 1)
    for i in range(steps+1):
        t = i/steps
        pt = p1 + t*(p2 - p1)
        if collision(pt, obstacles, robot_rad):
            return True
    return False

def nearest_node(tree, pt):
    dists = [distance(n.pos, pt) for n in tree]
    return tree[int(np.argmin(dists))]

def steer(from_node, to_pt, step=ROBOT_DIAMETER):
    vec  = to_pt - from_node.pos
    dist = np.linalg.norm(vec)
    return to_pt if dist < step else from_node.pos + vec/dist*step

def build_path(goal_node):
    path = []
    n = goal_node
    while n:
        path.append(n.pos)
        n = n.parent
    return path[::-1]

def rrt_star_plan(start, goal, obstacles):
    tree = [Node(start, cost=0.0)]
    goal_node = None

    for _ in range(MAX_ITERS):
        sample = goal if random.random() < GOAL_BIAS \
                 else np.random.uniform(-MAX_RANGE, MAX_RANGE, 2)
        nearest = nearest_node(tree, sample)
        new_pt  = steer(nearest, sample)
        if collision(new_pt, obstacles): 
            continue

        neighbors = [n for n in tree if distance(n.pos, new_pt) <= RRT_STAR_RADIUS]

        if neighbors:
            costs = [n.cost + distance(n.pos, new_pt) for n in neighbors]
            idx   = int(np.argmin(costs))
            parent, new_cost = neighbors[idx], costs[idx]
        else:
            parent   = nearest
            new_cost = nearest.cost + distance(nearest.pos, new_pt)

        new_node = Node(new_pt, parent, new_cost)
        tree.append(new_node)

        for nbr in neighbors:
            pot = new_node.cost + distance(new_node.pos, nbr.pos)
            if pot < nbr.cost and not collision_line(new_node.pos, nbr.pos, obstacles):
                nbr.parent = new_node
                nbr.cost   = pot

        if distance(new_pt, goal) <= ROBOT_DIAMETER:
            gcost = new_node.cost + distance(new_pt, goal)
            goal_node = Node(goal, new_node, gcost)
            tree.append(goal_node)
            return build_path(goal_node), tree

    # fallback if goal never reached
    best = min(tree, key=lambda n: n.cost + distance(n.pos, goal))
    gcost = best.cost + distance(best.pos, goal)
    goal_node = Node(goal, best, gcost)
    tree.append(goal_node)
    return build_path(goal_node), tree

def animate_path(path_pts, tree, obstacles):
    n_frames = len(path_pts)
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set_aspect('equal')
    ax.set_xlim(-MAX_RANGE, MAX_RANGE)
    ax.set_ylim(-MAX_RANGE, MAX_RANGE)

    ax.scatter(obstacles[:,0], obstacles[:,1], s=4, c='black', label='obs')
    for node in tree:
        if node.parent:
            ax.plot([node.pos[0], node.parent.pos[0]],
                    [node.pos[1], node.parent.pos[1]],
                    '-', lw=0.5, c='gray')
    ax.plot(path_pts[:,0], path_pts[:,1], '--', lw=1.5, c='blue', label='RRT*')
    ax.legend(loc='upper left')

    def update(i):
        pos = path_pts[min(i, n_frames-1)]
        print(f"[Frame {i+1}/{n_frames}] Robot at {pos}")

        for art in list(ax.patches): art.remove()
        for txt in list(ax.texts):  txt.remove()

        nxt = path_pts[i+1] if i+1 < n_frames else pos
        θ   = math.atan2(nxt[1]-pos[1], nxt[0]-pos[0])

        c = Circle(pos, ROBOT_RADIUS, edgecolor='red', facecolor='none', lw=2)
        ax.add_patch(c)
        L  = ROBOT_RADIUS * 1.5
        dx, dy = L*math.cos(θ), L*math.sin(θ)
        arr = FancyArrow(pos[0], pos[1], dx, dy,
                         width=0.02, length_includes_head=True, color='green')
        ax.add_patch(arr)

        ax.text(0.95, 0.05, f"Frame {i+1}",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, bbox=dict(boxstyle='round,pad=0.2',
                                      fc='white', alpha=0.7))

    anim = FuncAnimation(fig, update, frames=n_frames,
                         interval=ANIMATION_INTERVAL, repeat=False)
    plt.show()
    return anim

def main():
    obstacles = load_obstacles(LASER_FILE)

    # ---- time the planner ----
    t0 = time.perf_counter()
    path, tree = rrt_star_plan(START, GOAL, obstacles)
    t1 = time.perf_counter()
    print(f"RRT* planning took {t1 - t0:.4f} seconds")

    path_pts = np.array(path)
    _anim = animate_path(path_pts, tree, obstacles)

if __name__ == "__main__":
    main()
