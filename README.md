# voronoipathexplore

Common functionality such as LIDAR loading, β‑complex construction and drawing
helpers has been moved into small modules:

- `laser_io.py` – load and optionally filter laser scan points
- `beta_complex.py` – compute β‑complex edges
- `draw_utils.py` – shared drawing helpers
- `fuzzy_utils.py` – fuzzy-logic helpers for replanning decisions
- `algorithm_selector.py` – choose or run planners and provides `auto_explore`
  for testing with random goals

Existing scripts continue to work and now import from these modules.
