"""Heartbeat monitoring for the main node."""

import time
import threading
from runtime_protocol import build_heartbeat, send_message
from constants import MAIN_NODE_NAME

class HeartbeatMonitor:
    def __init__(self, registry, config, logger, should_stop):
        self.registry = registry
        self.config = config
        self.logger = logger
        self.should_stop = should_stop

    def start_loop(self):
        while not self.should_stop():
            time.sleep(self.config.heartbeat_interval)
            workers = self.registry.list_workers()
            for worker in workers:
                try:
                    # Minimal Sprint 1 logic: just send a ping
                    msg = build_heartbeat(MAIN_NODE_NAME)
                    send_message(worker.conn, msg)
                except Exception as exc:
                    self.logger.warning(f"Worker {worker.node_name} failed heartbeat: {exc}")
                    self.registry.remove_worker(worker.worker_id)
