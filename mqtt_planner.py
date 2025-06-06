#!/usr/bin/env python3
"""MQTT interface for fuzzy planners.

This script listens on ``mqtt_config.PLAN_REQUEST_TOPIC`` for JSON messages of
``{"algorithm": "rrt"|"rrt_star"|"prm"}``.  After computing the requested path it
publishes ``{"path": [[x, y], ...]}`` on ``mqtt_config.PLAN_RESULT_TOPIC``.
"""

from __future__ import annotations

import json

import paho.mqtt.client as mqtt

import mqtt_config as cfg
import fuzzy_planner_channel as planner


def on_connect(client: mqtt.Client, userdata, flags, rc) -> None:
    print(f"[mqtt] Connected with result code {rc}")
    client.subscribe(cfg.PLAN_REQUEST_TOPIC)


def on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage) -> None:
    payload = msg.payload.decode("utf-8")
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        print(f"[mqtt] {msg.topic}: {payload}")
        return

    algo = str(data.get("algorithm", "prm")).lower()
    if algo == "rrt":
        path = planner.rrt_path()
    elif algo == "rrt_star":
        path = planner.rrt_star_path()
    else:
        path = planner.prm_path()

    out_msg = json.dumps({"path": path.tolist()})
    client.publish(cfg.PLAN_RESULT_TOPIC, out_msg)
    print(f"[mqtt] Published path with {len(path)} points")


def main() -> None:
    client = mqtt.Client()
    if cfg.USERNAME:
        client.username_pw_set(cfg.USERNAME, cfg.PASSWORD)

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(cfg.HOST, cfg.PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    main()
