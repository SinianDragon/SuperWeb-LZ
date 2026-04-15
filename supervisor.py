"""Minimal supervisor — retained for backward compatibility.

NOTE: As of the bootstrap refactor, the main startup flow is handled
directly in bootstrap.py.  This module is kept so that existing tests
or scripts that import Supervisor continue to work.
"""

from __future__ import annotations
import logging
from common.state import RuntimeState
from common.types import DiscoveryResult
from discovery.pairing import run_pairing
from main_node.runtime import MainNodeRuntime
from compute_node.runtime import ComputeNodeRuntime
from trace_utils import trace_function


class Supervisor:
    @trace_function
    def __init__(self, config, platform_info, firewall_status, logger, on_promotion_callback=None) -> None:
        self.config, self.logger = config, logger
        self.on_promotion_callback = on_promotion_callback
        self._shutdown_requested = False

    def _set_state(self, state):
        self.logger.info("Supervisor state -> %s", state.value)

    @trace_function
    def run(self):
        # 尝试发现
        result = run_pairing(self.config)
        if result.success:
            self._set_state(RuntimeState.COMPUTE_NODE)
            runtime = ComputeNodeRuntime(self.config, result.peer_address, result.peer_port, self.logger)
            return runtime.run()

        # 发现失败，晋升为 Main Node
        if self.on_promotion_callback:
            self.logger.info("Triggering pre-promotion data preparation.")
            self.on_promotion_callback()

        self._set_state(RuntimeState.MAIN_NODE)
        runtime = MainNodeRuntime(self.config, self.logger, lambda: self._shutdown_requested)
        return runtime.run()

    def shutdown(self):
        self._shutdown_requested = True
