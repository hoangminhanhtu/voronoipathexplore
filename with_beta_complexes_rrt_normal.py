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

from scipy.spatial import Delaunay

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
STEP_SIZE        = 0.6        # extension step (m)
ANIM_INTERVAL_MS = 200        # ms per frame

# === LOAD & FILTER POINTS ===

def load_and_filter(js_path: Path) -> np.ndarray:
    """Load LIDAR points and drop any that lie within the robot's start circle."""
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
            x, y = r*math.cos(θ), r*math.sin(θ)
            # filter out any point within the starting robot radius + β
            if math.hypot(x-START[0], y-START[1]) > (ROBOT_RADIUS + BETA):
                pts.append((x, y))
    return np.array(pts)

# === BUILD MINKOWSKI OBSTACLE SHAPE ===

def build_obstacle_shape(points: np.ndarray,
                         inflate: float) -> Polygon:
    """
    Build a single polygon by unioning circles of radius=inflate
    around each point—the robot center must avoid this entire shape.
    """
    circles = [Point(x, y).buffer(inflate) for x,y in points]
    return unary_union(circles)

# === COLLISION CHECKS ===

def point_in_collision(pt: np.ndarray,
                       obs_shape: Polygon) -> bool:
    """True if robot‐center at pt lies inside (or on) obs_shape."""
    return obs_shape.covers(Point(pt))

def edge_in_collision(a: np.ndarray,
                      b: np.ndarray,
                      obs_shape: Polygon) -> bool:
    """True if the straight segment AB intersects obs_shape."""
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

    # 1) ensure start/goal are collision‐free
    if point_in_collision(start, obs_shape):
        raise RuntimeError(f"Start {start.tolist()} is inside an obstacle.")
    if point_in_collision(goal, obs_shape):
        raise RuntimeError(f"Goal {goal.tolist()} is inside an obstacle.")

    tree: List[Node] = [Node(start)]
    goal_node = None
    t0 = time.perf_counter()

    for it in range(MAX_ITERS):
        # biased sampling
        sample = goal if random.random()<GOAL_BIAS else \
                 np.random.uniform(-MAX_RANGE, MAX_RANGE, 2)

        nearest_n = nearest(tree, sample)
        new_pt    = steer(nearest_n, sample, STEP_SIZE)

        # 2) point collision?
        if point_in_collision(new_pt, obs_shape):
            continue
        # 3) edge collision?
        if edge_in_collision(nearest_n.pos, new_pt, obs_shape):
            continue

        node = Node(new_pt, parent=nearest_n)
        tree.append(node)

        # 4) goal check
        if np.linalg.norm(new_pt - goal) <= STEP_SIZE:
            # final link
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

    # backtrack
    path = []
    cur = end
    while cur:
        path.append(cur.pos)
        cur = cur.parent
    return np.array(path[::-1]), tree

# === DRAWING HELPERS ===

def circumradius(pts: np.ndarray) -> float:
    a,b,c = (np.linalg.norm(pts[(i+1)%3] - pts[i]) for i in range(3))
    s = (a+b+c)/2
    area = max(s*(s-a)*(s-b)*(s-c),0)**0.5
    return (a*b*c)/(4*area) if area>1e-6 else float('inf')

def draw_beta_triangles(ax, pts: np.ndarray, beta: float):
    if len(pts)<3: return
    tri = Delaunay(pts)
    shown = False
    for s in tri.simplices:
        tri_pts = pts[s]
        if circumradius(tri_pts) <= beta:
            loop = np.vstack([tri_pts, tri_pts[0]])
            lbl = "β-triangle" if not shown else ""
            ax.plot(loop[:,0], loop[:,1],
                    color='green', alpha=0.6, linewidth=1, label=lbl)
            shown = True

def draw_shape(ax, shape: Polygon):
    geoms = shape.geoms if isinstance(shape, MultiPolygon) else [shape]
    for poly in geoms:
        x,y = poly.exterior.xy
        ax.plot(x, y, color='blue', alpha=0.5, linewidth=1, label="β-shape")

def draw_tree(ax, tree: List[Node]):
    for n in tree:
        if n.parent:
            xs = [n.pos[0], n.parent.pos[0]]
            ys = [n.pos[1], n.parent.pos[1]]
            ax.plot(xs, ys, color='gray', linewidth=0.5)

def draw_path(ax, path: np.ndarray):
    ax.plot(path[:,0], path[:,1], color='red', linewidth=2, label='path')
    ax.add_patch(Circle(tuple(path[0]), ROBOT_RADIUS, alpha=0.3, label='start'))

def animate(path, tree, raw_pts, obs_shape):
    fig, ax = plt.subplots(figsize=(10,10))
    ax.set_aspect('equal','box')
    ax.set_xlim(-MAX_RANGE, MAX_RANGE)
    ax.set_ylim(-MAX_RANGE, MAX_RANGE)

    draw_shape(ax, obs_shape)
    draw_beta_triangles(ax, raw_pts, BETA)
    ax.scatter(raw_pts[:,0], raw_pts[:,1], s=5, c='black', label='lidar')
    draw_tree(ax, tree)
    draw_path(ax, path)

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
    raw_pts     = load_and_filter(LASER_FILE)
    # inflate by β + robot_radius so point tests suffice
    obs_shape   = build_obstacle_shape(raw_pts, BETA + ROBOT_RADIUS)

    path, tree = rrt_plan(START, GOAL, obs_shape)

    # report
    if path.size:
        steps = len(path)-1
        length = np.sum(np.linalg.norm(np.diff(path,axis=0),axis=1))
        print(f"Total steps: {steps}")
        print(f"Total length: {length:.3f} m")
        animate(path, tree, raw_pts, obs_shape)
    else:
        print("No feasible path.")

if __name__=="__main__":
    main()
