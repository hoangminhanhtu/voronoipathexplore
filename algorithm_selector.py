#!/usr/bin/env python3

"""Select and run a planner based on fuzzy logic."""

from pathlib import Path
from typing import Optional
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


def run_planner_for_scan(scan_file: str, algorithm: str | None = None) -> None:
    """Run the chosen planner on ``scan_file``."""
    config.LASER_FILE = Path(scan_file)

    algo = algorithm or config.ALGORITHM
    if algo not in PLANNERS:
        names = ", ".join(sorted(PLANNERS))
        raise ValueError(f"algorithm must be one of: {names}")

    planner = PLANNERS[algo]
    planner.LASER_FILE = config.LASER_FILE
    planner.main()

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        names = ", ".join(sorted(PLANNERS))
        raise SystemExit(
            f"Usage: algorithm_selector.py <scan_file.js> [algorithm]\n"
            f"Where [algorithm] is one of: {names}"
        )

    scan = sys.argv[1]
    algo = sys.argv[2] if len(sys.argv) > 2 else None
    run_planner_for_scan(scan, algo)
