# voronoipathexplore

Common functionality such as LIDAR loading, β‑complex construction and drawing
helpers has been moved into small modules:

- `laser_io.py` – load and optionally filter laser scan points
- `beta_complex.py` – compute β‑complex edges
- `draw_utils.py` – shared drawing helpers
- `fuzzy_utils.py` – fuzzy-logic helpers for replanning decisions

Existing scripts continue to work and now import from these modules.

The repository also includes an MQTT interface for the fuzzy planners:

- `mqtt_planner.py` – listen for planning requests on `mqtt_config.PLAN_REQUEST_TOPIC`
  and publish computed paths to `mqtt_config.PLAN_RESULT_TOPIC`
