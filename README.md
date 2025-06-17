# Voronoi Path Explore

This project is a small playground for trying out different robot
path-planning algorithms. It starts from a set of laser scan points,
turns them into inflated obstacles using a **β-complex**, and then tries
to plan a path through the free space. The code is meant to be easy to
run so you can learn how the planners work.

## Setup

1. Install **Python 3**.
2. Grab the required packages:

   ```bash
   pip install numpy matplotlib shapely paho-mqtt
   ```

   The last package is only needed if you want to use the MQTT planner.

## Trying the planners

To see a simple demo of the PRM planner, run:

```bash
python3 fuzzy_logic_prm.py
```

A window will open showing the β-complex obstacle shape, the sampled
nodes and the resulting path.

### Using the MQTT planner

`mqtt_planner.py` exposes the planners over MQTT. It listens on
`mqtt_config.PLAN_REQUEST_TOPIC` for messages like

```json
{"algorithm": "prm"}
```

The `algorithm` field may be `prm`, `rrt` or `rrt_star`. After planning,
a JSON payload of the form

```json
{"path": [[x1, y1], [x2, y2], ...]}
```

is published on `mqtt_config.PLAN_RESULT_TOPIC`.

Adjust the broker host, port and topic names in `mqtt_config.py`.
With Mosquitto installed you can try a quick round trip:

```bash
mosquitto_sub -t planner/path &
mosquitto_pub -t planner/request -m '{"algorithm": "rrt"}'
```

### Changing parameters

Most tunable values live in `config.py`. Here you can set the laser scan
file to load, planner options and robot size. The MQTT-specific settings
are kept in `mqtt_config.py`.

#### Understanding `config.py`

`config.py` is a small Python module that holds nearly all of the knobs
used by the planners. Open it in your editor and you will see the
settings grouped by purpose:

* **General robot settings** – the path to the laser scan (`LASER_FILE`),
  sensor range (`MAX_RANGE`), robot size (`ROBOT_DIAMETER`) and the
  smoothing factor for the β-complex (`BETA`). The start and goal
  positions are defined by `START` and `GOAL`.
* **Automatic exploration** – when using auto exploration the planner
  looks ahead by `EXPLORE_RADIUS` and defaults to the algorithm named in
  `ALGORITHM`.
* **PRM options** – `N_SAMPLES`, `K_NEIGHBORS`, `ANIM_INTERVAL_MS` and
  related variables tune how the roadmap is built.
* **RRT options** – values such as `GOAL_BIAS`, `MAX_ITERS` and
  `STEP_SIZE` control the random tree growth.
* **RRT\*** options – parameters like `NEIGHBOR_RADIUS` influence how
  the tree is rewired. `BATTERY_CONSUMPTION_PER_STEP` provides a simple
  energy cost estimate.
* **Fuzzy logic** – `BATTERY_LEVEL` represents the starting battery
  charge used by the fuzzy algorithm selector.

## Key files

- `laser_io.py` – load and optionally filter scan points from the exported `.js` laser files.
- `beta_complex.py` – compute β-complex edges used to inflate obstacles.
- `draw_utils.py` – helpers for drawing robots and paths.
- `fuzzy_utils.py` – fuzzy-logic functions used by the planners.
