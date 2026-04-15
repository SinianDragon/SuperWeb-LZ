"""Simple TCP message protocol for cluster communication.

Message format:
  [4 bytes: payload_length (big-endian)] [1 byte: msg_type] [payload bytes]

JSON messages: payload is UTF-8 JSON
Binary messages: payload is raw bytes with JSON header prepended
"""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from typing import Any

# ─── Message Types ─────────────────────────────────────────────────────────────

MSG_REGISTER = 1        # compute→main: JSON {node_name, gflops, backend}
MSG_TASK_ASSIGN = 2     # main→compute: JSON {worker_id, start_oc, end_oc, spec...}
MSG_WEIGHT_DATA = 3     # main→compute: binary weight slice
MSG_START = 4           # main→compute: JSON {} (signal to begin computation)
MSG_OUTPUT_DATA = 5     # compute→main: binary output slice
MSG_TASK_DONE = 6       # compute→main: JSON {elapsed_seconds, checksum}
MSG_ALL_DONE = 7        # main→compute: JSON {} (shutdown signal)
MSG_ERROR = 8           # either→either: JSON {error: str}


# ─── Wire Helpers ──────────────────────────────────────────────────────────────

def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Receive exactly `n` bytes from socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Connection closed while expecting data")
        buf.extend(chunk)
    return bytes(buf)


def send_json(sock: socket.socket, msg_type: int, data: dict) -> None:
    """Send a JSON message."""
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    header = struct.pack(">IB", len(payload) + 1, msg_type)
    sock.sendall(header + payload)


# Maximum accepted message size (4 GiB). Prevents unbounded allocation from
# corrupt or malicious length headers.
MAX_MESSAGE_SIZE = 4 * 1024 * 1024 * 1024


def send_binary(sock: socket.socket, msg_type: int, data: bytes) -> None:
    """Send a binary message."""
    header = struct.pack(">IB", len(data) + 1, msg_type)
    sock.sendall(header)
    sock.sendall(data)


def recv_msg(sock: socket.socket) -> tuple[int, bytes]:
    """Receive one message. Returns (msg_type, payload_bytes)."""
    header = _recv_exact(sock, 5)
    total_len, msg_type = struct.unpack(">IB", header)
    if total_len > MAX_MESSAGE_SIZE:
        raise ValueError(f"Message too large: {total_len} bytes (limit {MAX_MESSAGE_SIZE})")
    payload = _recv_exact(sock, total_len - 1) if total_len > 1 else b""
    return msg_type, payload


def recv_json(sock: socket.socket) -> tuple[int, dict]:
    """Receive a JSON message. Returns (msg_type, dict)."""
    msg_type, payload = recv_msg(sock)
    return msg_type, json.loads(payload.decode("utf-8")) if payload else {}


def recv_binary(sock: socket.socket) -> tuple[int, bytes]:
    """Receive a binary message. Returns (msg_type, raw_bytes)."""
    return recv_msg(sock)
