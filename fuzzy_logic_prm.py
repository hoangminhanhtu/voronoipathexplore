#!/usr/bin/env python3
import random
import time
from pathlib import Path
from typing import Tuple
import config

from fuzzy_utils import (
    membership_battery,
    membership_distance,
    fuzzy_replan_decision,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from scipy.spatial import Delaunay, KDTree

from laser_io import load_and_filter
from draw_utils import (
    draw_beta_shape,
    draw_beta_triangles,
    draw_path,
)


import heapq

# === CONFIGURATION ===
LASER_FILE       = config.LASER_FILE
MAX_RANGE        = config.MAX_RANGE
ROBOT_DIAMETER   = config.ROBOT_DIAMETER
ROBOT_RADIUS     = config.ROBOT_RADIUS
BETA             = config.BETA
START            = config.START
GOAL             = config.GOAL
N_SAMPLES        = config.N_SAMPLES
K_NEIGHBORS      = config.K_NEIGHBORS
ANIM_INTERVAL_MS = config.ANIM_INTERVAL_MS
X_STEP           = config.X_STEP
STEP_COST        = config.STEP_COST
MAX_TRIES        = config.MAX_TRIES

# === PRM UTILITIES ===

def build_obstacle_shape(points: np.ndarray, inflate: float) -> Polygon:
    circles = [Point(x,y).buffer(inflate) for x,y in points]
    return unary_union(circles)

def point_in_collision(pt: np.ndarray, obs: Polygon) -> bool:
    return obs.covers(Point(pt))

def edge_in_collision(a: np.ndarray, b: np.ndarray, obs: Polygon) -> bool:
    return LineString([tuple(a), tuple(b)]).intersects(obs)

def sample_prm_nodes(n: int, obs: Polygon) -> np.ndarray:
    nodes = []
    while len(nodes) < n:
        pt = np.random.uniform(-MAX_RANGE, MAX_RANGE, 2)
        if not point_in_collision(pt, obs):
            nodes.append(pt)
    return np.array(nodes)

def build_prm_graph(nodes: np.ndarray, k: int, obs: Polygon) -> dict:
    tree = KDTree(nodes)
    edges = {i: [] for i in range(len(nodes))}
    for i, pt in enumerate(nodes):
        dists, idxs = tree.query(pt, k=k+1)
        for dist, j in zip(dists[1:], idxs[1:]):
            if not edge_in_collision(pt, nodes[j], obs):
                edges[i].append((j, dist))
                edges[j].append((i, dist))
    return edges



def shortest_path_prm(nodes: np.ndarray, edges: dict,
                      start_idx: int, goal_idx: int) -> np.ndarray:
    dist, prev = {start_idx:0.0}, {}
    pq = [(0.0, start_idx)]
    seen = set()
    while pq:
        d,u = heapq.heappop(pq)
        if u in seen: continue
        seen.add(u)
        if u == goal_idx: break
        for v,w in edges[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v], prev[v] = nd, u
                heapq.heappush(pq, (nd, v))
    if goal_idx not in prev and goal_idx != start_idx:
        return np.zeros((0,2))
    path, cur = [], goal_idx
    while True:
        path.append(nodes[cur])
        if cur == start_idx: break
        cur = prev[cur]
    return np.array(path[::-1])



# === DRAWING HELPERS ===
def draw_prm_edges(ax, nodes: np.ndarray, edges: dict):
    for i, nbrs in edges.items():
        for j,_ in nbrs:
            if j > i:
                x0,y0 = nodes[i]
                x1,y1 = nodes[j]
                ax.plot([x0,x1],[y0,y1],
                        color='lightgrey', linewidth=0.5)

def draw_prm_nodes(ax, nodes: np.ndarray):
    ax.scatter(nodes[:,0], nodes[:,1],
               c='grey', s=10)

# draw_path provided by draw_utils

# === ANIMATION WITH TIMING & BATTERY PRINTS ===
def animate_prm(path, prm_nodes, edges, raw_pts, obs_shape):
    init_dist   = np.linalg.norm(START - GOAL)
    battery     = 100.0
    step_count  = 0
    replanned   = False

    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal','box')
    ax.set_xlim(-MAX_RANGE, MAX_RANGE)
    ax.set_ylim(-MAX_RANGE, MAX_RANGE)

    draw_beta_shape(ax, obs_shape)
    draw_beta_triangles(ax, raw_pts, BETA)
    ax.scatter(raw_pts[:,0], raw_pts[:,1], s=5, c='black')
    draw_prm_edges(ax, prm_nodes, edges)
    draw_prm_nodes(ax, prm_nodes)
    draw_path(ax, path, ROBOT_RADIUS)

    robot    = Circle((0,0), ROBOT_RADIUS, color='green', alpha=0.4)
    txt_batt = ax.text(0.02, 0.92, "", transform=ax.transAxes)
    txt_info = ax.text(0.02, 0.88, "", transform=ax.transAxes)
    ax.add_patch(robot)

    current_path  = path.copy()
    current_nodes = prm_nodes
    current_edges = edges

    while True:
        for idx in range(1, len(current_path)):
            pos = current_path[idx]
            robot.center = tuple(pos)

            step_count += 1
            battery    -= STEP_COST
            dist_rem    = np.linalg.norm(pos - GOAL)

            # --- print battery each step ---
            print(f"Step {step_count:3d}: Battery = {battery:5.1f}%")

            txt_batt.set_text(f"Battery: {battery:.1f}%")
            txt_info.set_text(f"Step {step_count}, Remain: {dist_rem:.2f} m")

            plt.pause(ANIM_INTERVAL_MS / 1000.0)

            # --- fuzzy & timing for replanning every X steps ---
            if step_count % X_STEP == 0 and battery > 0:
                t0 = time.time()
                if fuzzy_replan_decision(battery, dist_rem, init_dist):
                    new_nodes = np.vstack([pos, GOAL,
                                           sample_prm_nodes(N_SAMPLES, obs_shape)])
                    new_edges = build_prm_graph(new_nodes, K_NEIGHBORS, obs_shape)
                    new_path  = shortest_path_prm(new_nodes, new_edges, 0, 1)
                    t1 = time.time()
                    print(f"  → Replanning took {t1-t0:.3f}s at battery {battery:.1f}%")
                    if new_path.size:
                        current_path  = new_path
                        current_nodes = new_nodes
                        current_edges = new_edges
                        replanned     = True
                        break
        if replanned:
            replanned = False
            continue
        break

    ax.legend(["β-shape","β-triangles","PRM edges","PRM nodes","Path"], loc='upper right')
    plt.show()

# === MAIN ===
def main():
    raw_pts = load_and_filter(
        LASER_FILE,
        MAX_RANGE,
        start=START,
        exclude_radius=ROBOT_RADIUS + BETA,
    )
    obs_shape = build_obstacle_shape(raw_pts, BETA + ROBOT_RADIUS)

    t_start = time.time()
    #nodes      = sample_prm_nodes(N_SAMPLES, obs_shape)
    #prm_nodes  = np.vstack([START, GOAL, nodes])
    #edges      = build_prm_graph(prm_nodes, K_NEIGHBORS, obs_shape)
    #path       = shortest_path_prm(prm_nodes, edges, 0, 1)

    for attempt in range(1, MAX_TRIES + 1):
        nodes = sample_prm_nodes(N_SAMPLES, obs_shape)
        prm_nodes = np.vstack([START, GOAL, nodes])
        edges     = build_prm_graph(prm_nodes, K_NEIGHBORS, obs_shape)
        path      = shortest_path_prm(prm_nodes, edges, 0, 1)
        if path.size:
            print(f"✅ Found path on attempt {attempt}")
            break        
        print(f"⚠️ Attempt {attempt}: no path—retrying...")

    t_end = time.time()
    print(f"Initial PRM build & path planning took {t_end-t_start:.3f} seconds")

    if path.size == 0: 
        print(f"⚠️ No PRM path found. Failed aftter {MAX_TRIES}")
        return

    steps  = len(path) - 1
    length = np.sum(np.linalg.norm(np.diff(path,axis=0),axis=1))
    print(f"✅ PRM path: {steps} steps, {length:.3f} m total length")

    animate_prm(path, prm_nodes, edges, raw_pts, obs_shape)

if __name__ == "__main__":
    main()
