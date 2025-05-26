#!/usr/bin/env python3
import json
import math
import random
import heapq
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
NUM_SAMPLES        = 300              # number of PRM samples
K_NEIGHBORS        = 10               # k for k-NN in roadmap
ANIMATION_INTERVAL = 1000             # ms between frames

def load_obstacles(js_path: Path) -> np.ndarray:
    """Load and parse laser points from a JS file into an array of obstacle coordinates."""
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
        r, theta = p.get("distance", 0.0), p.get("angle", 0.0)
        if 0 < r < MAX_RANGE:
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            pts.append((x, y))
    return np.array(pts) if pts else np.empty((0, 2))

def collision(pt: np.ndarray, obstacles: np.ndarray, robot_rad: float) -> bool:
    """Check if point pt is within robot_rad of any obstacle."""
    if obstacles.size == 0:
        return False
    return np.any(np.linalg.norm(obstacles - pt, axis=1) <= robot_rad)

def edge_collision(a: np.ndarray, b: np.ndarray, obstacles: np.ndarray,
                   robot_rad: float, step: float = 0.1) -> bool:
    """Check collision along straight-line segment from a to b."""
    dist = np.linalg.norm(b - a)
    direction = (b - a) / dist
    n_steps = int(dist / step)
    for i in range(n_steps + 1):
        p = a + direction * (i * step)
        if collision(p, obstacles, robot_rad):
            return True
    return False

def build_prm(start: np.ndarray, goal: np.ndarray, obstacles: np.ndarray):
    """Sample free configurations and build an undirected roadmap graph."""
    samples = [start, goal]
    while len(samples) < NUM_SAMPLES + 2:
        pt = np.random.uniform(-MAX_RANGE, MAX_RANGE, 2)
        if not collision(pt, obstacles, ROBOT_RADIUS):
            samples.append(pt)
    samples = np.array(samples)

    # Precompute all pairwise distances
    dists = np.linalg.norm(samples[:, None] - samples[None, :], axis=2)
    graph = {i: [] for i in range(len(samples))}

    for i in range(len(samples)):
        neighbors = np.argsort(dists[i])[1:K_NEIGHBORS + 1]
        for j in neighbors:
            if not edge_collision(samples[i], samples[j], obstacles, ROBOT_RADIUS):
                cost = float(dists[i, j])
                graph[i].append((j, cost))
                graph[j].append((i, cost))
    return samples, graph

def dijkstra(graph: dict, start_idx: int, goal_idx: int):
    """Find shortest-path in graph from start_idx to goal_idx via Dijkstra’s algorithm."""
    queue = [(0.0, start_idx)]
    dist = {start_idx: 0.0}
    parent = {start_idx: None}
    visited = set()

    while queue:
        d, u = heapq.heappop(queue)
        if u in visited:
            continue
        visited.add(u)
        if u == goal_idx:
            break
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                parent[v] = u
                heapq.heappush(queue, (nd, v))

    # Reconstruct path
    path = []
    node = goal_idx
    while node is not None:
        path.append(node)
        node = parent.get(node)
    return path[::-1]

def prm_plan(start: np.ndarray, goal: np.ndarray, obstacles: np.ndarray):
    """Build PRM and return the sequence of waypoints from start to goal."""
    samples, graph = build_prm(start, goal, obstacles)
    path_idx = dijkstra(graph, 0, 1)  # indices 0=start, 1=goal    
    path = [samples[i] for i in path_idx]
    
    print(f"PRM: samples={len(samples)}")
    return np.array(path), samples, graph

def animate_path(path_pts: np.ndarray, samples: np.ndarray,
                 graph: dict, obstacles: np.ndarray):
    """Visualize the PRM, path, and animate the robot moving along it."""
    n_frames = len(path_pts)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-MAX_RANGE, MAX_RANGE)
    ax.set_ylim(-MAX_RANGE, MAX_RANGE)

    # Draw obstacles
    ax.scatter(obstacles[:, 0], obstacles[:, 1], s=4, c='black', label='obstacles')

    # Draw PRM edges
    for u, nbrs in graph.items():
        for v, _ in nbrs:
            x0, y0 = samples[u]
            x1, y1 = samples[v]
            ax.plot([x0, x1], [y0, y1], '-', lw=0.5, c='lightgray')

    # Draw PRM nodes
    ax.scatter(samples[:, 0], samples[:, 1], s=10, c='orange',
               alpha=0.6, label='PRM nodes')

    # Draw planned path
    ax.plot(path_pts[:, 0], path_pts[:, 1], '--', lw=1.5,
            c='blue', label='PRM path')
    ax.legend(loc='upper left')

    def update(i):
        pos = path_pts[min(i, n_frames - 1)]
        print(f"[Frame {i+1}/{n_frames}] Robot at {pos}")

        # Clear previous robot & text
        for art in list(ax.patches):
            art.remove()
        for txt in list(ax.texts):
            txt.remove()

        # Compute heading
        nxt = path_pts[i+1] if (i + 1) < n_frames else pos
        theta = math.atan2(nxt[1] - pos[1], nxt[0] - pos[0])

        # Draw robot footprint
        circle = Circle(pos, ROBOT_RADIUS, edgecolor='red',
                        facecolor='none', lw=2)
        ax.add_patch(circle)

        # Draw heading arrow
        L = ROBOT_RADIUS * 1.5
        dx, dy = L * math.cos(theta), L * math.sin(theta)
        arrow = FancyArrow(pos[0], pos[1], dx, dy,
                           width=0.02, length_includes_head=True)
        ax.add_patch(arrow)

        # Frame label
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

    # Measure PRM planning time
    start_time = time.perf_counter()
    path, samples, graph = prm_plan(START, GOAL, obstacles)
    end_time = time.perf_counter()
    print(f"PRM planning computation time: {end_time - start_time:.4f} seconds")

    _anim = animate_path(path, samples, graph, obstacles)

if __name__ == "__main__":
    main()
