from pathlib import Path
import numpy as np

# General robot and environment settings
LASER_FILE = Path("list_file_laser/FileLaserPoint6.js")
MAX_RANGE = 10.0
ROBOT_DIAMETER = 0.6
ROBOT_RADIUS = ROBOT_DIAMETER / 2
BETA = 0.3
START = np.array([0.0, 0.0])
GOAL = np.array([-3.0, -5.0])

# Radius for autonomous exploration.  Must be < MAX_RANGE.
EXPLORE_RADIUS = 4.0

# PRM parameters
N_SAMPLES = 200
K_NEIGHBORS = 10
ANIM_INTERVAL_MS = 200
X_STEP = 2
STEP_COST = 2.0
MAX_TRIES = 5

# RRT parameters
GOAL_BIAS = 0.1
MAX_ITERS = 5000
STEP_SIZE = 0.6

# RRT* parameters
NEIGHBOR_RADIUS = 1.0
BATTERY_CONSUMPTION_PER_STEP = 2.0

# Fuzzy algorithm selection
BATTERY_LEVEL = 100.0
