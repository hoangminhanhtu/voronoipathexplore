#!/usr/bin/env python3
"""Convenience functions to run fuzzy logic path planning.

Each function returns a NumPy array of robot positions representing
the executed motion steps.
"""

from __future__ import annotations

import numpy as np

import config
from laser_io import load_and_filter

# import primitives from the individual planners
import fuzzy_logic_rrt_normal as rrt
import fuzzy_logic_rrt_star as rrt_star
import fuzzy_logic_prm as prm


def rrt_path() -> np.ndarray:
    """Return a path planned with fuzzy RRT logic."""
    raw_pts = load_and_filter(
        config.LASER_FILE,
        config.MAX_RANGE,
        start=config.START,
        exclude_radius=config.ROBOT_RADIUS + config.BETA,
    )
    obs_shape = rrt.build_obstacle_shape(raw_pts, config.BETA + config.ROBOT_RADIUS)

    path, _ = rrt.rrt_plan(config.START, config.GOAL, obs_shape)

    battery = rrt.BATTERY_INIT
    for idx in range(1, len(path)):
        battery -= rrt.BATTERY_COST_PER_STEP
        if idx % rrt.X_STEP == 0 and idx < len(path) - 1:
            low, med, high = rrt.fuzzify_battery(battery)
            if rrt.need_replan((low, med, high)):
                current = path[idx]
                new_path, _ = rrt.rrt_plan(current, config.GOAL, obs_shape)
                path = np.vstack([path[: idx + 1], new_path[1:]])
                break
    return path


def rrt_star_path() -> np.ndarray:
    """Return a path executed with fuzzy RRT* replanning."""
    raw_pts = load_and_filter(
        config.LASER_FILE,
        config.MAX_RANGE,
        start=config.START,
        exclude_radius=config.ROBOT_RADIUS + config.BETA,
    )
    obs_shape = rrt_star.build_obstacle_shape(raw_pts, config.BETA + config.ROBOT_RADIUS)

    executed, _tree, _batt = rrt_star.plan_with_replanning(config.START, config.GOAL, obs_shape)
    return executed


def prm_path() -> np.ndarray:
    """Return a path executed with fuzzy PRM replanning."""
    raw_pts = load_and_filter(
        config.LASER_FILE,
        config.MAX_RANGE,
        start=config.START,
        exclude_radius=config.ROBOT_RADIUS + config.BETA,
    )
    obs_shape = prm.build_obstacle_shape(raw_pts, config.BETA + config.ROBOT_RADIUS)

    # initial PRM path search
    for _ in range(1, config.MAX_TRIES + 1):
        nodes = prm.sample_prm_nodes(config.N_SAMPLES, obs_shape)
        prm_nodes = np.vstack([config.START, config.GOAL, nodes])
        edges = prm.build_prm_graph(prm_nodes, config.K_NEIGHBORS, obs_shape)
        path = prm.shortest_path_prm(prm_nodes, edges, 0, 1)
        if path.size:
            break
    else:
        return np.zeros((0, 2))

    init_dist = np.linalg.norm(config.START - config.GOAL)
    battery = 100.0
    step_count = 0
    executed = [path[0]]

    current_path = path
    while True:
        replanned = False
        for idx in range(1, len(current_path)):
            pos = current_path[idx]
            step_count += 1
            battery -= config.STEP_COST
            executed.append(pos)
            dist_rem = np.linalg.norm(pos - config.GOAL)

            if step_count % config.X_STEP == 0 and battery > 0:
                if prm.fuzzy_replan_decision(battery, dist_rem, init_dist):
                    new_nodes = np.vstack(
                        [pos, config.GOAL, prm.sample_prm_nodes(config.N_SAMPLES, obs_shape)]
                    )
                    new_edges = prm.build_prm_graph(new_nodes, config.K_NEIGHBORS, obs_shape)
                    new_path = prm.shortest_path_prm(new_nodes, new_edges, 0, 1)
                    if new_path.size:
                        current_path = new_path
                        replanned = True
                        break
        if not replanned:
            break
    return np.array(executed)


__all__ = ["rrt_path", "rrt_star_path", "prm_path"]
