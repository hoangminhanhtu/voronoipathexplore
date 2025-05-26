#!/usr/bin/env python3
"""Select and run a planner based on fuzzy logic."""

from pathlib import Path
import config
from fuzzy_utils import membership_battery

import fuzzy_logic_prm
import fuzzy_logic_rrt_normal
import fuzzy_logic_rrt_star


PLANNERS = {
    "prm": fuzzy_logic_prm,
    "rrt": fuzzy_logic_rrt_normal,
    "rrt_star": fuzzy_logic_rrt_star,
}


def choose_algorithm(battery: float) -> str:
    """Return planner name chosen from battery level using fuzzy logic."""
    low, med, high = membership_battery(battery)
    scores = {
        "rrt_star": 0.6 * high + 0.3 * med + 0.1 * low,
        "prm": 0.6 * med + 0.3 * high + 0.1 * low,
        "rrt": 0.6 * low + 0.3 * med + 0.1 * high,
    }
    return max(scores, key=scores.get)


def run_planner_for_scan(scan_file: str) -> None:
    """Load given scan file and run the selected planner."""
    config.LASER_FILE = Path(scan_file)

    algo = choose_algorithm(config.BATTERY_LEVEL)
    print(f"Selected planner via fuzzy logic: {algo}")

    planner = PLANNERS[algo]
    planner.LASER_FILE = config.LASER_FILE
    planner.main()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: algorithm_selector.py <scan_file.js>")

    run_planner_for_scan(sys.argv[1])
