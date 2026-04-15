"""Worker registry for Main Node — tracks connected compute nodes."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional
import socket


@dataclass
class WorkerInfo:
    """Information about one registered compute node."""
    worker_id: int
    node_name: str
    gflops: float
    backend: str
    conn: socket.socket
    addr: tuple

    # Assigned after planning
    start_oc: int = 0
    end_oc: int = 0


class ClusterRegistry:
    """Thread-safe registry of compute-node workers."""

    def __init__(self):
        self._lock = threading.Lock()
        self._workers: dict[int, WorkerInfo] = {}
        self._next_id = 1

    def register_worker(self, node_name: str, gflops: float, backend: str,
                        conn: socket.socket, addr: tuple) -> WorkerInfo:
        """Register a new worker and return its WorkerInfo."""
        with self._lock:
            wid = self._next_id
            self._next_id += 1
            worker = WorkerInfo(
                worker_id=wid, node_name=node_name,
                gflops=gflops, backend=backend,
                conn=conn, addr=addr,
            )
            self._workers[wid] = worker
            return worker

    def remove_worker(self, worker_id: int) -> Optional[WorkerInfo]:
        """Remove a worker by ID."""
        with self._lock:
            return self._workers.pop(worker_id, None)

    def list_workers(self) -> list[WorkerInfo]:
        """Return a snapshot of all registered workers."""
        with self._lock:
            return list(self._workers.values())

    def count_workers(self) -> int:
        with self._lock:
            return len(self._workers)

    def allocate_slices(self, total_cout: int, main_gflops: float = 0.0) -> dict:
        """Allocate Cout slices proportional to GFLOPS.

        Returns dict: {worker_id: (start_oc, end_oc), 'main': (start_oc, end_oc)}
        """
        with self._lock:
            workers = list(self._workers.values())

        # Build participants: workers + main node itself
        participants = []
        for w in workers:
            participants.append(("worker", w.worker_id, w.gflops))
        if main_gflops > 0:
            participants.append(("main", 0, main_gflops))

        if not participants:
            return {"main": (0, total_cout)}

        total_gflops = sum(g for _, _, g in participants)
        if total_gflops <= 0:
            # Equal split
            chunk = total_cout // len(participants)
            allocation = {}
            oc = 0
            for kind, wid, _ in participants:
                end = oc + chunk
                key = f"worker_{wid}" if kind == "worker" else "main"
                allocation[key] = (oc, end)
                oc = end
            # Give remainder to last
            last_key = list(allocation.keys())[-1]
            allocation[last_key] = (allocation[last_key][0], total_cout)
            return allocation

        # Proportional split
        allocation = {}
        oc = 0
        for i, (kind, wid, gflops) in enumerate(participants):
            if oc >= total_cout:
                break  # no more channels to assign
            if i == len(participants) - 1:
                end = total_cout  # last one gets remainder
            else:
                share = gflops / total_gflops
                end = oc + max(1, round(total_cout * share))
                end = min(end, total_cout)
            key = f"worker_{wid}" if kind == "worker" else "main"
            allocation[key] = (oc, end)
            oc = end

        # Update worker objects
        with self._lock:
            for w in self._workers.values():
                key = f"worker_{w.worker_id}"
                if key in allocation:
                    w.start_oc, w.end_oc = allocation[key]

        return allocation
