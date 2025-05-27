#!/usr/bin/env python3
"""Run a planner on a random goal within ``EXPLORE_RADIUS``."""

import numpy as np

import config
from algorithm_selector import run_planner_for_scan


def random_goal(radius: float) -> np.ndarray:
    """Return a random goal uniformly sampled inside a circle."""
    angle = np.random.uniform(0.0, 2 * np.pi)
    r = radius * np.sqrt(np.random.random())

    return np.array([r * np.cos(angle), r * np.sin(angle)])


def auto_explore() -> None:
    """Generate a random goal and run the configured planner."""
    if config.EXPLORE_RADIUS >= config.MAX_RANGE:
        raise ValueError("EXPLORE_RADIUS must be less than MAX_RANGE")
    new_goal = random_goal(config.EXPLORE_RADIUS)
    # update in-place so planner modules see the change
    config.GOAL[:] = new_goal
    run_planner_for_scan(str(config.LASER_FILE))


if __name__ == "__main__":
    auto_explore()
