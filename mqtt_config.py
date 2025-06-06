"""MQTT configuration parameters."""
HOST = "localhost"
PORT = 1883
USERNAME = None
PASSWORD = None
LASER_TOPIC = "laser/points"
AUTH_TOPIC = "laser/auth"
INTERVAL = 1.0  # seconds between requests
MAX_RANGE = 10.0

# Topics for planning requests and results
PLAN_REQUEST_TOPIC = "planner/request"
PLAN_RESULT_TOPIC = "planner/path"
