#!/usr/bin/env python3
"""Run a planner on a random goal within ``EXPLORE_RADIUS``."""

import numpy as np

import config
from algorithm_selector import run_planner_for_scan
from laser_io import load_and_filter
from shapely.geometry import Point
from shapely.ops import unary_union


def random_goal(radius: float) -> np.ndarray:
    """Return a random goal uniformly sampled inside a circle."""
    angle = np.random.uniform(0.0, 2 * np.pi)
    r = radius * np.sqrt(np.random.random())

    return np.array([r * np.cos(angle), r * np.sin(angle)])


def build_obstacle_shape(points: np.ndarray, inflate: float):
    """Return a Minkowski sum of points inflated by ``inflate``."""
    circles = [Point(x, y).buffer(inflate) for x, y in points]
    return unary_union(circles)


def point_in_collision(pt: np.ndarray, obs_shape) -> bool:
    """Return True if ``pt`` lies inside ``obs_shape``."""
    return obs_shape.covers(Point(pt))


def auto_explore() -> None:
    """Generate a random collision-free goal then run the planner."""
    if config.EXPLORE_RADIUS >= config.MAX_RANGE:
        raise ValueError("EXPLORE_RADIUS must be less than MAX_RANGE")

    print("[auto_explore] Loading scan data for collision checking…")
    raw_pts = load_and_filter(
        config.LASER_FILE,
        config.MAX_RANGE,
        start=config.START,
        exclude_radius=config.ROBOT_RADIUS + config.BETA,
    )
    obs_shape = build_obstacle_shape(raw_pts, config.BETA + config.ROBOT_RADIUS)

    attempt = 0
    while True:
        attempt += 1
        new_goal = random_goal(config.EXPLORE_RADIUS)
        print(f"[auto_explore] Attempt {attempt}: testing goal {new_goal.tolist()}")
        if point_in_collision(new_goal, obs_shape):
            print("[auto_explore] Goal in collision, retrying…")
            continue

        print("[auto_explore] Goal accepted")
        config.GOAL[:] = new_goal
        break

    run_planner_for_scan(str(config.LASER_FILE))


if __name__ == "__main__":
    auto_explore()
