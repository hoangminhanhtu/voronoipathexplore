#!/usr/bin/env python3
import random
import time
from pathlib import Path
from typing import List, Tuple
import config

from fuzzy_utils import fuzzify_battery, need_replan

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

from shapely.ops import unary_union
from shapely.geometry import Point, LineString, Polygon, MultiPolygon

from laser_io import load_and_filter
from draw_utils import (
    draw_beta_shape,
    draw_beta_triangles,
    draw_tree,
    draw_path,
)

# === CONFIGURATION ===
LASER_FILE       = config.LASER_FILE
MAX_RANGE        = config.MAX_RANGE
ROBOT_DIAMETER   = config.ROBOT_DIAMETER
ROBOT_RADIUS     = config.ROBOT_RADIUS
BETA             = config.BETA
START            = config.START
GOAL             = config.GOAL
GOAL_BIAS        = config.GOAL_BIAS
MAX_ITERS        = config.MAX_ITERS
STEP_SIZE        = config.STEP_SIZE
ANIM_INTERVAL_MS = config.ANIM_INTERVAL_MS

# === FUZZY BATTERY LOGIC CONFIGURATION ===
BATTERY_INIT          = config.BATTERY_LEVEL
BATTERY_COST_PER_STEP = config.BATTERY_CONSUMPTION_PER_STEP
X_STEP                = config.X_STEP

# === LOAD & FILTER POINTS ===
# Handled by ``laser_io.load_and_filter``

# === BUILD MINKOWSKI OBSTACLE SHAPE ===

def build_obstacle_shape(points: np.ndarray,
                         inflate: float) -> Polygon:
    circles = [Point(x, y).buffer(inflate) for x,y in points]
    return unary_union(circles)

# === COLLISION CHECKS ===

def point_in_collision(pt: np.ndarray,
                       obs_shape: Polygon) -> bool:
    return obs_shape.covers(Point(pt))

def edge_in_collision(a: np.ndarray,
                      b: np.ndarray,
                      obs_shape: Polygon) -> bool:
    return LineString([tuple(a), tuple(b)]).intersects(obs_shape)

# === RRT DATA STRUCTURES ===

class Node:
    __slots__ = ("pos","parent")
    def __init__(self, pos: np.ndarray, parent=None):
        self.pos = pos
        self.parent = parent

def nearest(tree: List[Node], pt: np.ndarray) -> Node:
    return min(tree, key=lambda n: np.linalg.norm(n.pos - pt))

def steer(from_n: Node, to_pt: np.ndarray, step: float) -> np.ndarray:
    vec = to_pt - from_n.pos
    dist = np.linalg.norm(vec)
    return to_pt if dist<=step else from_n.pos + vec/dist*step

# === PLAIN RRT ===

def rrt_plan(start: np.ndarray,
             goal: np.ndarray,
             obs_shape: Polygon) -> Tuple[np.ndarray,List[Node]]:

    if point_in_collision(start, obs_shape):
        raise RuntimeError(f"Start {start.tolist()} is inside an obstacle.")
    if point_in_collision(goal, obs_shape):
        raise RuntimeError(f"Goal {goal.tolist()} is inside an obstacle.")

    tree: List[Node] = [Node(start)]
    goal_node = None
    t0 = time.perf_counter()

    for it in range(MAX_ITERS):
        sample = goal if random.random()<GOAL_BIAS else \
                 np.random.uniform(-MAX_RANGE, MAX_RANGE, 2)
        nearest_n = nearest(tree, sample)
        new_pt    = steer(nearest_n, sample, STEP_SIZE)

        if point_in_collision(new_pt, obs_shape):
            continue
        if edge_in_collision(nearest_n.pos, new_pt, obs_shape):
            continue

        node = Node(new_pt, parent=nearest_n)
        tree.append(node)

        if np.linalg.norm(new_pt - goal) <= STEP_SIZE:
            if not edge_in_collision(new_pt, goal, obs_shape):
                goal_node = Node(goal, parent=node)
                tree.append(goal_node)
                break

    elapsed = time.perf_counter() - t0
    if goal_node:
        print(f"✅ Path found in {elapsed:.2f}s ({it} iterations)")
        end = goal_node
    else:
        print(f"⚠️ No goal reached after {it} iterations; using nearest.")
        end = nearest(tree, goal)

    path = []
    cur = end
    while cur:
        path.append(cur.pos)
        cur = cur.parent
    return np.array(path[::-1]), tree

# === DRAWING HELPERS ===

def animate(path, tree, raw_pts, obs_shape):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal','box')
    ax.set_xlim(-MAX_RANGE, MAX_RANGE)
    ax.set_ylim(-MAX_RANGE, MAX_RANGE)

    draw_beta_shape(ax, obs_shape)
    draw_beta_triangles(ax, raw_pts, BETA)
    ax.scatter(raw_pts[:,0], raw_pts[:,1], s=5, c='black', label='lidar')
    draw_tree(ax, tree)
    draw_path(ax, path, ROBOT_RADIUS)

    robot = Circle((0,0), ROBOT_RADIUS, color='green', alpha=0.4)
    ax.add_patch(robot)
    txt = ax.text(0.02,0.95,"", transform=ax.transAxes)

    def upd(i):
        robot.center = tuple(path[i])
        txt.set_text(f"Step {i}")
        return robot, txt

    anim = FuncAnimation(fig, upd, frames=len(path),
                         interval=ANIM_INTERVAL_MS, blit=True,
                         repeat=False)
    ax.legend(loc='upper right')
    plt.show()
    return anim

# === MAIN ===

def main():
    raw_pts = load_and_filter(
        LASER_FILE,
        MAX_RANGE,
        start=START,
        exclude_radius=ROBOT_RADIUS + BETA,
    )
    obs_shape = build_obstacle_shape(raw_pts, BETA + ROBOT_RADIUS)

    path, tree = rrt_plan(START, GOAL, obs_shape)

    # --- Dynamic replanning based on fuzzy battery logic ---
    battery = BATTERY_INIT
    for idx in range(1, len(path)):
        battery -= BATTERY_COST_PER_STEP
        # only reevaluate at multiples of X_STEP, and not at goal
        if idx % X_STEP == 0 and idx < len(path)-1:
            low, med, high = fuzzify_battery(battery)
            print(f"[Fuzzy] Step {idx}, Battery {battery:.1f}% → low={low:.2f}, med={med:.2f}, high={high:.2f}")
            if need_replan((low, med, high)):
                print(f"🔄 Battery low ({battery:.1f}%), replanning from step {idx}")
                current_pos = path[idx]
                new_path, new_tree = rrt_plan(current_pos, GOAL, obs_shape)
                # stitch paths together (avoid duplicate current_pos)
                path = np.vstack([path[:idx+1], new_path[1:]])
                tree += new_tree
                break

    # report & animate updated plan
    steps = len(path) - 1
    length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    print(f"Total steps: {steps}")
    print(f"Total length: {length:.3f} m")
    animate(path, tree, raw_pts, obs_shape)

if __name__=="__main__":
    main()
