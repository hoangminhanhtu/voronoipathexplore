# voronoipathexplore

Common functionality such as LIDAR loading, β‑complex construction and drawing
helpers has been moved into small modules:

- `laser_io.py` – load and optionally filter laser scan points
- `beta_complex.py` – compute β‑complex edges
- `draw_utils.py` – shared drawing helper
- `fuzzy_utils.py` – battery-aware fuzzy logic for replanning

Existing scripts continue to work and now import from these modules.
