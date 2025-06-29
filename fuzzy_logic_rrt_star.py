#!/usr/bin/env python3
import random
import time
from pathlib import Path
from typing import List, Tuple
import config

from fuzzy_utils import fuzzy_neighbor_radius

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
NEIGHBOR_RADIUS  = config.NEIGHBOR_RADIUS
ANIM_INTERVAL_MS = config.ANIM_INTERVAL_MS

# === BATTERY & REPLANNING ===
BATTERY_CONSUMPTION_PER_STEP = config.BATTERY_CONSUMPTION_PER_STEP
X_STEP = config.X_STEP

# === LOAD & FILTER LIDAR POINTS ===
# Provided by ``laser_io.load_and_filter``


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
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal", "box")
    ax.set_xlim(-MAX_RANGE, MAX_RANGE)
    ax.set_ylim(-MAX_RANGE, MAX_RANGE)

    draw_beta_shape(ax, obs_shape)
    draw_beta_triangles(ax, raw_pts, BETA)
    ax.scatter(raw_pts[:, 0], raw_pts[:, 1], s=5, c="black", label="lidar")
    draw_tree(ax, tree)
    draw_path(ax, path, ROBOT_RADIUS)

    robot = Circle((0, 0), ROBOT_RADIUS, color="green", alpha=0.4)
    ax.add_patch(robot)
    txt = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def update(i):
        robot.center = tuple(path[i])
        txt.set_text(f"Step {i}")
        return robot, txt

    anim = FuncAnimation(
        fig,
        update,
        frames=len(path),
        interval=ANIM_INTERVAL_MS,
        blit=True,
        repeat=False,
    )
    ax.legend(loc="upper right")
    plt.show()
    return anim


# === REPLANNING‐ENABLED EXECUTION LOOP ===

def plan_with_replanning(start: np.ndarray,
                         goal: np.ndarray,
                         obs_shape: Polygon
                        ) -> Tuple[np.ndarray, List[Node], float]:
    """
    Execute the path with battery‐aware fuzzy replanning every X_STEP moves.
    Returns the executed path, the final tree, and remaining battery%.
    """
    battery = 100.0
    position = start.copy()
    executed = [position]

    path, tree = rrt_star_plan(position, goal, obs_shape)
    step_idx = 0

    while step_idx < len(path) - 1 and battery > 0:
        # move one step
        next_pt = path[step_idx + 1]
        battery -= BATTERY_CONSUMPTION_PER_STEP
        position = next_pt
        executed.append(position)
        step_idx += 1

        # periodic replanning
        if step_idx % X_STEP == 0 and np.linalg.norm(position - goal) > STEP_SIZE:
            global NEIGHBOR_RADIUS
            NEIGHBOR_RADIUS = fuzzy_neighbor_radius(battery)
            print(f"🔄 Replanning at step {step_idx}, battery={battery:.1f}%, "
                  f"NEIGHBOR_RADIUS={NEIGHBOR_RADIUS:.2f}m")
            path, tree = rrt_star_plan(position, goal, obs_shape)
            step_idx = 0

    return np.array(executed), tree, battery


# === MAIN ===

def main():
    raw_pts = load_and_filter(
        LASER_FILE,
        MAX_RANGE,
        start=START,
        exclude_radius=ROBOT_RADIUS + BETA,
    )
    obs_shape = build_obstacle_shape(raw_pts, BETA + ROBOT_RADIUS)

    executed_path, tree, remaining_batt = plan_with_replanning(START, GOAL, obs_shape)

    steps = len(executed_path) - 1
    length = np.sum(np.linalg.norm(np.diff(executed_path, axis=0), axis=1))
    used = steps * BATTERY_CONSUMPTION_PER_STEP

    print(f"Total executed steps: {steps}")
    print(f"Total path length: {length:.3f} m")
    print(f"Battery used: {used:.1f}% → remaining: {remaining_batt:.1f}%")

    animate(executed_path, tree, raw_pts, obs_shape)


if __name__ == "__main__":
    main()
