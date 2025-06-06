# voronoipathexplore

This repository collects a handful of path-planning experiments built on
laser scanner data. The code focuses on constructing **β-complex** obstacle
shapes from raw LIDAR points and exploring several planning algorithms. Most
scripts visualise their progress using Matplotlib.

## Key modules

- **`laser_io.py`** – load and optionally filter scan points from the exported
  `.js` laser files.
- **`beta_complex.py`** – compute β-complex edges used to inflate obstacles.
- **`draw_utils.py`** – small drawing helpers for robots, paths and β-complexes.
- **`fuzzy_utils.py`** – fuzzy-logic utilities used to decide when to replan
  based on remaining battery or distance.

Existing scripts import functionality from these modules. The main planners
implemented in the repository are:

- **Probabilistic Road Map (PRM)** with optional fuzzy replanning
  (`fuzzy_logic_prm.py`).
- **Rapidly-Exploring Random Trees (RRT)**
  and **RRT*** with fuzzy logic (`fuzzy_logic_rrt_normal.py`,
  `fuzzy_logic_rrt_star.py`).
- An **algorithm selector** (`algorithm_selector.py`) which chooses among the
  planners using simple fuzzy rules.
- **`auto_explore.py`** which picks a random collision-free goal inside a
  radius and then runs the selected planner.

## MQTT interface

The `mqtt_planner.py` script exposes the planners over MQTT. It listens on
`mqtt_config.PLAN_REQUEST_TOPIC` for JSON messages such as
```
{"algorithm": "prm"}
```
and publishes the resulting path on `mqtt_config.PLAN_RESULT_TOPIC`.
See `mqtt_config.py` to adjust broker settings and topic names.

## Running an example

Most scripts rely on the paths and parameters defined in `config.py`. To run a
basic PRM planner with fuzzy logic enabled:

```bash
python3 fuzzy_logic_prm.py
```

To execute the MQTT interface instead:

```bash
python3 mqtt_planner.py
```

The output will open a Matplotlib window showing the β-complex obstacle shape,
the sampled nodes and the evolving path.

