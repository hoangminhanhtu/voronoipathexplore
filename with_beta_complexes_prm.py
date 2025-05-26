#!/usr/bin/env python3
import json
import math
import random
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from shapely.ops import unary_union
from shapely.geometry import Point, LineString, Polygon, MultiPolygon

from scipy.spatial import Delaunay, KDTree
import heapq

# === CONFIGURATION ===
LASER_FILE       = Path("list_file_laser/FileLaserPoint5.js")
MAX_RANGE        = 10.0       # max sensor range (m)
ROBOT_DIAMETER   = 0.6        # robot diameter (m)
ROBOT_RADIUS     = ROBOT_DIAMETER / 2
BETA             = 0.3        # obstacle inflation radius (m)
START            = np.array([0.0, 0.0])
GOAL             = np.array([-3.0, -5.0])
N_SAMPLES        = 200        # number of random PRM nodes
K_NEIGHBORS      = 10         # k in k‐NN for roadmap
ANIM_INTERVAL_MS = 200  

# === LOAD & FILTER LIDAR POINTS ===
def load_and_filter(js_path: Path) -> np.ndarray:
    """Load LIDAR points, drop any within (ROBOT_RADIUS + BETA) of START."""
    if not js_path.is_file():
        raise FileNotFoundError(f"Laser file not found: {js_path}")
    raw = json.loads(
        "".join(
            line for line in js_path.open(encoding="utf-8")
            if not line.lstrip().startswith("//") and line.strip()
        )
    ).get("laser", [])
    pts = []
    for p in raw:
        r, θ = p.get("distance", 0.0), p.get("angle", 0.0)
        if 0 < r < MAX_RANGE:
            x, y = r * math.cos(θ), r * math.sin(θ)
            # filter out LIDAR returns too close to start
            if math.hypot(x - START[0], y - START[1]) > (ROBOT_RADIUS + BETA):
                pts.append((x, y))
    return np.array(pts)

# === BUILD INFLATED OBSTACLE SHAPE ===
def build_obstacle_shape(points: np.ndarray, inflate: float) -> Polygon:
    """Union circles of radius=inflate around each point."""
    circles = [Point(x, y).buffer(inflate) for x, y in points]
    return unary_union(circles)

# === COLLISION CHECKS ===
def point_in_collision(pt: np.ndarray, obs_shape: Polygon) -> bool:
    return obs_shape.covers(Point(pt))

def edge_in_collision(a: np.ndarray, b: np.ndarray, obs_shape: Polygon) -> bool:
    return LineString([tuple(a), tuple(b)]).intersects(obs_shape)

# === START/GOAL VALIDATION ===
def check_point_collision(pt: np.ndarray, obs_shape: Polygon, name: str):
    if point_in_collision(pt, obs_shape):
        print(f"⚠️ {name} {pt.tolist()} is in collision with obstacles. Please choose a different point.")        

# === PRM CONSTRUCTION ===
def sample_prm_nodes(n: int, obs_shape: Polygon) -> np.ndarray:
    """Uniformly sample n free points in the square."""
    nodes = []
    while len(nodes) < n:
        pt = np.random.uniform(-MAX_RANGE, MAX_RANGE, 2)
        if not point_in_collision(pt, obs_shape):
            nodes.append(pt)
    return np.array(nodes)

def build_prm_graph(nodes: np.ndarray, k: int, obs_shape: Polygon) -> dict:
    """
    Build adjacency: for each node index i, edges[i] = list of (j, distance).
    Undirected: we add both i→j and j→i.
    """
    tree = KDTree(nodes)
    edges = {i: [] for i in range(len(nodes))}
    for i, pt in enumerate(nodes):
        dists, idxs = tree.query(pt, k=k+1)  # include self at index 0
        for dist, j in zip(dists[1:], idxs[1:]):
            if not edge_in_collision(pt, nodes[j], obs_shape):
                edges[i].append((j, dist))
                edges[j].append((i, dist))
    return edges

def shortest_path_prm(nodes: np.ndarray,
                      edges: dict,
                      start_idx: int,
                      goal_idx: int) -> np.ndarray:
    """
    Dijkstra from start_idx to goal_idx on edges.
    Returns array of waypoint coords, or empty if no path.
    """
    dist = {start_idx: 0.0}
    prev = {}
    pq = [(0.0, start_idx)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == goal_idx:
            break
        for v, w in edges[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if goal_idx not in prev and goal_idx != start_idx:
        return np.zeros((0, 2))
    path = []
    cur = goal_idx
    while True:
        path.append(nodes[cur])
        if cur == start_idx:
            break
        cur = prev[cur]
    return np.array(path[::-1])

# === DRAWING HELPERS ===
def circumradius(pts: np.ndarray) -> float:
    a = np.linalg.norm(pts[1] - pts[0])
    b = np.linalg.norm(pts[2] - pts[1])
    c = np.linalg.norm(pts[0] - pts[2])
    s = (a + b + c) / 2
    area = max(s * (s - a) * (s - b) * (s - c), 0.0)**0.5
    return (a * b * c) / (4 * area) if area > 1e-6 else float('inf')

def draw_beta_shape(ax, shape: Polygon):
    geoms = shape.geoms if isinstance(shape, MultiPolygon) else [shape]
    for poly in geoms:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='blue', alpha=0.5, linewidth=1, label='β-shape')

def draw_beta_triangles(ax, pts: np.ndarray, beta: float):
    if len(pts) < 3:
        return
    tri = Delaunay(pts)
    shown = False
    for s in tri.simplices:
        tri_pts = pts[s]
        if circumradius(tri_pts) <= beta:
            loop = np.vstack([tri_pts, tri_pts[0]])
            lbl = 'β-triangle' if not shown else ''
            ax.plot(loop[:,0], loop[:,1],
                    color='green', alpha=0.6, linewidth=1, label=lbl)
            shown = True

def draw_prm_edges(ax, nodes: np.ndarray, edges: dict):
    """Draw every PRM edge once in light grey."""
    for i, nbrs in edges.items():
        for j, _ in nbrs:
            if j > i:
                x0, y0 = nodes[i]
                x1, y1 = nodes[j]
                ax.plot([x0, x1], [y0, y1],
                        color='lightgrey', linewidth=0.5)

def draw_prm_nodes(ax, nodes: np.ndarray):
    ax.scatter(nodes[:,0], nodes[:,1],
               c='grey', s=10, label='PRM nodes')

def draw_path(ax, path: np.ndarray):
    ax.plot(path[:,0], path[:,1],
            color='red', linewidth=2, label='PRM path')
    ax.add_patch(Circle(tuple(path[0]), ROBOT_RADIUS,
                        alpha=0.3, label='start'))

# === ANIMATION ===
def animate_prm(path: np.ndarray,
                prm_nodes: np.ndarray,
                edges: dict,
                raw_pts: np.ndarray,
                obs_shape: Polygon):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal','box')
    ax.set_xlim(-MAX_RANGE, MAX_RANGE)
    ax.set_ylim(-MAX_RANGE, MAX_RANGE)

    draw_beta_shape(ax, obs_shape)
    draw_beta_triangles(ax, raw_pts, BETA)
    ax.scatter(raw_pts[:,0], raw_pts[:,1], s=5,
               c='black', label='lidar')

    draw_prm_edges(ax, prm_nodes, edges)
    draw_prm_nodes(ax, prm_nodes)
    draw_path(ax, path)

    robot = Circle((0,0), ROBOT_RADIUS, color='green', alpha=0.4)
    ax.add_patch(robot)
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    for i in range(len(path)):
        robot.center = tuple(path[i])
        txt.set_text(f"Step {i}")
        plt.pause(ANIM_INTERVAL_MS / 1000.0)

    ax.legend(loc='upper right')
    plt.show()

# === MAIN ===
def main():
    raw_pts   = load_and_filter(LASER_FILE)
    obs_shape = build_obstacle_shape(raw_pts, BETA + ROBOT_RADIUS)

    # Check START and GOAL for collisions
    check_point_collision(START, obs_shape, 'START')
    check_point_collision(GOAL, obs_shape, 'GOAL')

    # measure PRM computation time
    t0 = time.perf_counter()
    random_nodes = sample_prm_nodes(N_SAMPLES, obs_shape)
    prm_nodes = np.vstack([START, GOAL, random_nodes])
    edges = build_prm_graph(prm_nodes, K_NEIGHBORS, obs_shape)
    t1 = time.perf_counter()
    path = shortest_path_prm(prm_nodes, edges, start_idx=0, goal_idx=1)
    t2 = time.perf_counter()

    print(f"Sampling & graph build time: {t1 - t0:.3f} s")
    print(f"Pathfinding time:          {t2 - t1:.3f} s")
    print(f"Total PRM time:            {t2 - t0:.3f} s")

    if path.size == 0:
        print("⚠️ No PRM path found.")
        return

    steps  = len(path) - 1
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    print(f"✅ PRM path: {steps} steps, {length:.3f} m total length")

    animate_prm(path, prm_nodes, edges, raw_pts, obs_shape)

if __name__ == "__main__":
    main()
