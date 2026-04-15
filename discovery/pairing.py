"""Sprint 1 discovery flow — bidirectional peer discovery.

Every node simultaneously sends discover probes AND listens for probes from
other nodes.  The first node to receive an announce reply becomes the Compute
Node; the first node to receive a discover query (meaning someone else is
looking for a Main Node) becomes the Main Node.  If nobody responds within the
timeout, the node self-promotes to Main Node.
"""

from __future__ import annotations

import logging
import threading
import time

from common.types import DiscoveryResult
from config import AppConfig
from discovery import multicast
from trace_utils import trace_function

logger = logging.getLogger("superweb_cluster")

# How often to re-broadcast the discover probe during the discovery window.
_PROBE_INTERVAL_SECONDS = 2.0


@trace_function
def run_pairing(config: AppConfig) -> DiscoveryResult:
    """Bidirectional discovery: simultaneously send probes and listen.

    Returns a DiscoveryResult:
      - success=True, peer_address set  → we found a Main Node (become Compute)
      - success=False                   → nobody answered (become Main)
    """

    result_holder: list[DiscoveryResult | None] = [None]
    stop_event = threading.Event()

    # ─── Thread 1: Sender — periodically broadcast discover probes ────────
    def _sender_loop() -> None:
        try:
            endpoint = multicast.create_sender(config)
        except OSError as exc:
            logger.debug(f"Discovery sender socket failed: {exc}")
            return

        try:
            deadline = time.monotonic() + config.discovery_timeout
            while not stop_event.is_set() and time.monotonic() < deadline:
                try:
                    multicast.send_discover(endpoint, config, config.node_name)
                    logger.debug("Sent discover probe")
                except OSError:
                    pass
                stop_event.wait(_PROBE_INTERVAL_SECONDS)
        finally:
            multicast.close(endpoint)

    # ─── Thread 2: Receiver — listen for discover probes from others ──────
    # If we receive a discover probe from another node, that means they are
    # looking for a Main Node.  We reply with an announce so they can find us.
    # (This makes *us* the Main Node.)
    def _receiver_loop() -> None:
        try:
            endpoint = multicast.create_receiver(config)
        except OSError as exc:
            logger.debug(f"Discovery receiver socket failed: {exc}")
            return

        try:
            deadline = time.monotonic() + config.discovery_timeout
            while not stop_event.is_set() and time.monotonic() < deadline:
                remaining = max(0.5, deadline - time.monotonic())
                endpoint.sock.settimeout(min(remaining, 1.0))

                try:
                    packet = multicast.recv_packet(endpoint, config.buffer_size)
                except Exception:
                    continue

                if packet is None:
                    continue

                addr, data = packet
                if multicast.parse_discover_message(data):
                    # Someone is looking for a Main Node — reply with announce
                    # and let the sender thread pick up their announce too.
                    try:
                        local_host = multicast.send_announce(
                            endpoint, addr, config, config.node_name
                        )
                        logger.info(
                            f"Replied to discover probe from {addr[0]}:{addr[1]} "
                            f"(announced as {local_host}:{config.tcp_port})"
                        )
                    except OSError as exc:
                        logger.debug(f"Failed to send announce reply: {exc}")
        finally:
            multicast.close(endpoint)

    # ─── Thread 3: Listener — wait for announce replies to our probes ─────
    def _announce_listener() -> None:
        try:
            endpoint = multicast.create_sender(config)
        except OSError as exc:
            logger.debug(f"Announce listener socket failed: {exc}")
            return

        try:
            deadline = time.monotonic() + config.discovery_timeout
            while not stop_event.is_set() and time.monotonic() < deadline:
                remaining = max(0.5, deadline - time.monotonic())
                endpoint.sock.settimeout(min(remaining, 1.0))

                try:
                    packet = multicast.recv_packet(endpoint, config.buffer_size)
                except Exception:
                    continue

                if packet is None:
                    continue

                addr, data = packet
                from protocol import parse_announce_message
                payload = parse_announce_message(data)
                if payload is not None:
                    logger.info(
                        f"Found Main Node: {payload.node_name} at "
                        f"{payload.host}:{payload.port}"
                    )
                    result_holder[0] = DiscoveryResult(
                        success=True,
                        peer_address=payload.host,
                        peer_port=payload.port,
                        source="mdns",
                        message=f"Found main node {payload.node_name} at {payload.host}:{payload.port}",
                    )
                    stop_event.set()
                    return
        finally:
            multicast.close(endpoint)

    # ─── Launch all threads ───────────────────────────────────────────────
    sender_thread = threading.Thread(target=_sender_loop, daemon=True)
    receiver_thread = threading.Thread(target=_receiver_loop, daemon=True)
    listener_thread = threading.Thread(target=_announce_listener, daemon=True)

    sender_thread.start()
    receiver_thread.start()
    listener_thread.start()

    # Wait for timeout or early termination
    sender_thread.join(timeout=config.discovery_timeout + 2)
    receiver_thread.join(timeout=2)
    listener_thread.join(timeout=2)
    stop_event.set()

    if result_holder[0] is not None:
        return result_holder[0]

    return DiscoveryResult(
        success=False,
        message="Discovery timed out — no Main Node found on LAN.",
    )


# ─── Legacy API compatibility ────────────────────────────────────────────────

@trace_function
def discover_peer(config: AppConfig) -> DiscoveryResult:
    """Send discovery and wait for an announce reply."""
    return run_pairing(config)


@trace_function
def announce_peer(config: AppConfig) -> DiscoveryResult:
    """Wait for one main-node query and reply with main-node details."""

    try:
        endpoint = multicast.create_receiver(config)
    except OSError as exc:
        return DiscoveryResult(
            success=False,
            message=f"Unable to create discovery receiver socket: {exc}.",
        )

    try:
        discovered = multicast.recv_discover(endpoint, config.buffer_size)
        if discovered is None:
            return DiscoveryResult(success=False, message="No main-node query packet received.")

        target, _message = discovered
        local_host = multicast.send_announce(endpoint, target, config, config.node_name)
        return DiscoveryResult(
            success=True,
            peer_address=target[0],
            peer_port=target[1],
            source="mdns",
            message=f"Reported main-node availability from {local_host}:{config.tcp_port}.",
        )
    finally:
        multicast.close(endpoint)
