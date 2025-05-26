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
STEP_SIZE        = 0.6        # maximum extension step (m)
NEIGHBOR_RADIUS  = 1.0        # RRT* rewiring radius (m)
ANIM_INTERVAL_MS = 200        # ms per animation frame

# === BATTERY & REPLANNING ===
BATTERY_CONSUMPTION_PER_STEP = 2.0   # percent battery per executed step
X_STEP = 2                           # steps between replanning

def fuzzy_neighbor_radius(batt: float) -> float:
    """
    Fuzzy‐logic to adjust NEIGHBOR_RADIUS based on remaining battery.
    Membership functions (triangular):
      - low:    peak at 0, declines to zero at 30%
      - medium: peak at 50%, declines to zero at 20% and 80%
      - high:   peak at 100%, declines to zero at 50%
    Rule outputs for radius (m):
      low    → 0.5
      medium → 1.0
      high   → 1.5
    """
    # membership degrees
    low = max(min((30.0 - batt) / 30.0, 1.0), 0.0)
    med = max(min((batt - 20.0) / 30.0, (80.0 - batt) / 30.0), 0.0)
    high = max(min((batt - 50.0) / 50.0, 1.0), 0.0)

    # defuzzify by weighted average
    weight_sum = low + med + high
    if weight_sum == 0:
        return NEIGHBOR_RADIUS
    return (low * 0.5 + med * 1.0 + high * 1.5) / weight_sum


# === LOAD & FILTER LIDAR POINTS ===

def load_and_filter(js_path: Path) -> np.ndarray:
    """Load LIDAR points, drop those within (ROBOT_RADIUS + BETA) of START."""
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

def circumradius(pts: np.ndarray) -> float:
    a = np.linalg.norm(pts[1] - pts[0])
    b = np.linalg.norm(pts[2] - pts[1])
    c = np.linalg.norm(pts[0] - pts[2])
    s = (a + b + c) / 2
    area = max(s*(s-a)*(s-b)*(s-c), 0.0)**0.5
    return (a*b*c)/(4*area) if area > 1e-6 else float('inf')

def draw_beta_triangles(ax, points: np.ndarray, beta: float):
    if len(points) < 3:
        return
    tri = Delaunay(points)
    drawn = False
    for s in tri.simplices:
        verts = points[s]
        if circumradius(verts) <= beta:
            loop = np.vstack([verts, verts[0]])
            lbl = "β-triangle" if not drawn else ""
            ax.plot(loop[:,0], loop[:,1], color='green',
                    linewidth=1.0, alpha=0.6, label=lbl)
            drawn = True

def draw_beta_shape(ax, shape: Polygon):
    geoms = shape.geoms if isinstance(shape, MultiPolygon) else [shape]
    for poly in geoms:
        x, y = poly.exterior.xy
        ax.plot(x, y, color='blue', linewidth=1.0,
                alpha=0.5, label='β-shape')

def draw_tree(ax, tree: List[Node]):
    for n in tree:
        if n.parent:
            xs = [n.pos[0], n.parent.pos[0]]
            ys = [n.pos[1], n.parent.pos[1]]
            ax.plot(xs, ys, color='gray', linewidth=0.5)

def draw_path(ax, path: np.ndarray):
    ax.plot(path[:,0], path[:,1], color='red',
            linewidth=2.0, label='final path')
    ax.add_patch(Circle(tuple(path[0]), ROBOT_RADIUS,
                        alpha=0.3, label='start'))

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
    draw_path(ax, path)

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
    raw_pts   = load_and_filter(LASER_FILE)
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
