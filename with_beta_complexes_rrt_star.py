#!/usr/bin/env python3
import json, math, random, time
from pathlib import Path
from typing import List, Tuple

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
LASER_FILE       = Path("list_file_laser/FileLaserPoint6.js")
MAX_RANGE        = 10.0       # max sensor range (m)
ROBOT_DIAMETER   = 0.6        # robot diameter (m)
ROBOT_RADIUS     = ROBOT_DIAMETER / 2
BETA             = 0.3        # obstacle inflation radius (m)
START            = np.array([0.0, 0.0])
GOAL             = np.array([-3.0, -5.0])
GOAL_BIAS        = 0.1        # 10% goal sampling
MAX_ITERS        = 5000
STEP_SIZE        = 0.6        # maximum extension step (m)
NEIGHBOR_RADIUS  = 1.0        # RRT* rewiring radius (m)
ANIM_INTERVAL_MS = 200        # ms per animation frame


# === LOAD & FILTER LIDAR POINTS ===

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

# === NODE FOR RRT* ===

class Node:
    __slots__ = ("pos", "parent", "cost")
    def __init__(self, pos: np.ndarray, parent=None, cost: float = 0.0):
        self.pos = pos
        self.parent = parent
        self.cost = cost

def nearest(tree: List[Node], pt: np.ndarray) -> Node:
    return min(tree, key=lambda n: np.linalg.norm(n.pos - pt))

def steer(from_n: Node, to_pt: np.ndarray, step: float) -> np.ndarray:
    vec = to_pt - from_n.pos
    dist = np.linalg.norm(vec)
    return to_pt if dist <= step else from_n.pos + (vec / dist) * step

# === RRT* ALGORITHM ===

def rrt_star_plan(start: np.ndarray,
                  goal: np.ndarray,
                  obs_shape: Polygon
                 ) -> Tuple[np.ndarray, List[Node]]:
    # ensure start/goal are collision-free
    if point_in_collision(start, obs_shape):
        raise RuntimeError(f"Start {start.tolist()} is in collision.")
    if point_in_collision(goal, obs_shape):
        raise RuntimeError(f"Goal {goal.tolist()} is in collision.")

    tree: List[Node] = [Node(start, None, 0.0)]
    goal_node = None
    t0 = time.perf_counter()

    for it in range(MAX_ITERS):
        # sample with goal bias
        sample = goal if random.random() < GOAL_BIAS else \
                 np.random.uniform(-MAX_RANGE, MAX_RANGE, 2)

        nearest_n = nearest(tree, sample)
        new_pt    = steer(nearest_n, sample, STEP_SIZE)

        # collision checks
        if point_in_collision(new_pt, obs_shape):
            continue
        if edge_in_collision(nearest_n.pos, new_pt, obs_shape):
            continue

        # find neighbors for rewiring
        neighbors = [n for n in tree
                     if np.linalg.norm(n.pos - new_pt) <= NEIGHBOR_RADIUS]
        # ensure at least the nearest is considered
        if not neighbors:
            neighbors = [nearest_n]

        # choose best parent
        best_parent, best_cost = None, float('inf')
        for nbr in neighbors:
            d = np.linalg.norm(nbr.pos - new_pt)
            c = nbr.cost + d
            if c < best_cost and not edge_in_collision(nbr.pos, new_pt, obs_shape):
                best_parent, best_cost = nbr, c

        if best_parent is None:
            continue

        # create and add node
        new_node = Node(new_pt, best_parent, best_cost)
        tree.append(new_node)

        # rewire neighbors
        for nbr in neighbors:
            d = np.linalg.norm(new_pt - nbr.pos)
            c_new = new_node.cost + d
            if c_new + 1e-6 < nbr.cost and not edge_in_collision(new_pt, nbr.pos, obs_shape):
                nbr.parent, nbr.cost = new_node, c_new

        # try connecting to goal
        if np.linalg.norm(new_pt - goal) <= STEP_SIZE and not edge_in_collision(new_pt, goal, obs_shape):
            goal_node = Node(goal, new_node, new_node.cost + np.linalg.norm(new_pt - goal))
            tree.append(goal_node)
            break

    elapsed = time.perf_counter() - t0
    if goal_node:
        print(f"✅ RRT* found path in {elapsed:.2f}s after {it} iter, cost={goal_node.cost:.3f}")
        end_node = goal_node
    else:
        print(f"⚠️ RRT* failed after {it} iter ({elapsed:.2f}s), returning nearest")
        end_node = nearest(tree, goal)

    # backtrack path
    path = []
    cur = end_node
    while cur:
        path.append(cur.pos)
        cur = cur.parent
    return np.array(path[::-1]), tree


# === DRAWING & ANIMATION ===

def animate(path: np.ndarray, tree: List[Node],
            raw_pts: np.ndarray, obs_shape: Polygon):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal','box')
    ax.set_xlim(-MAX_RANGE, MAX_RANGE)
    ax.set_ylim(-MAX_RANGE, MAX_RANGE)

    draw_beta_shape(ax, obs_shape)
    draw_beta_triangles(ax, raw_pts, BETA)
    ax.scatter(raw_pts[:,0], raw_pts[:,1], s=5,
               c='black', label='lidar')
    draw_tree(ax, tree)
    # pass the robot radius to show the start circle
    draw_path(ax, path, ROBOT_RADIUS)

    robot = Circle((0,0), ROBOT_RADIUS, color='green', alpha=0.4)
    ax.add_patch(robot)
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def update(i):
        robot.center = tuple(path[i])
        txt.set_text(f"Step {i}")
        return robot, txt

    anim = FuncAnimation(fig, update,
                         frames=len(path),
                         interval=ANIM_INTERVAL_MS,
                         blit=True, repeat=False)
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
    # inflate by BETA + ROBOT_RADIUS for Minkowski sum
    obs_shape = build_obstacle_shape(raw_pts, BETA + ROBOT_RADIUS)

    path, tree = rrt_star_plan(START, GOAL, obs_shape)

    if path.size:
        steps = len(path) - 1
        length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        print(f"Total steps: {steps}")
        print(f"Total path length: {length:.3f} m")
        animate(path, tree, raw_pts, obs_shape)
    else:
        print("No feasible path found.")

if __name__ == "__main__":
    main()
