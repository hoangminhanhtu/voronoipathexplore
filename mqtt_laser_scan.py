#!/usr/bin/env python3
"""Subscribe to laser scan points over MQTT.

Configuration parameters such as broker address and topics are read from
``mqtt_config.py``.  The script requests laser scan data every
``INTERVAL`` seconds and converts any received messages using
``laser_io.points_from_dict``.
"""

from __future__ import annotations

import json
import time

import paho.mqtt.client as mqtt

import mqtt_config as cfg
from laser_io import points_from_dict


def main() -> None:
    client = mqtt.Client()
    if cfg.USERNAME:
        client.username_pw_set(cfg.USERNAME, cfg.PASSWORD)

    def on_connect(client: mqtt.Client, userdata, flags, rc):
        print(f"[mqtt] Connected with result code {rc}")
        if cfg.AUTH_TOPIC:
            req_msg = json.dumps({"request": "laser"})
            client.publish(cfg.AUTH_TOPIC, req_msg)
        client.subscribe(cfg.LASER_TOPIC)

    def on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
        payload = msg.payload.decode("utf-8")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            print(f"[mqtt] {msg.topic}: {payload}")
            return

        pts = points_from_dict(data, cfg.MAX_RANGE)
        print(f"[mqtt] {msg.topic}: received {len(pts)} points")

    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(cfg.HOST, cfg.PORT, keepalive=60)
    client.loop_start()
    try:
        while True:
            time.sleep(cfg.INTERVAL)
            if cfg.AUTH_TOPIC:
                req_msg = json.dumps({"request": "laser"})
                client.publish(cfg.AUTH_TOPIC, req_msg)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
