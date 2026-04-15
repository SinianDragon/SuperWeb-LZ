"""Compute Node Runtime — connects to Main Node, receives work, executes, returns output."""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from pathlib import Path

from common.cluster_protocol import (
    MSG_REGISTER, MSG_TASK_ASSIGN, MSG_WEIGHT_DATA,
    MSG_START, MSG_OUTPUT_DATA, MSG_TASK_DONE, MSG_ALL_DONE, MSG_ERROR,
    send_json, send_binary, recv_msg,
)
from compute_node.executor import run_computation, _get_best_backend, _get_best_gflops

logger = logging.getLogger("superweb_cluster")

DEFAULT_TCP_PORT = 9800


class ComputeNodeRuntime:
    """Full Compute-Node runtime: connect to Main, receive weight, compute, return output."""

    def __init__(self, main_addr: str, config=None):
        self.main_addr = main_addr  # IP or hostname of the main node
        self.config = config
        self.root = Path(__file__).resolve().parent.parent
        self.dataset_dir = self.root / "compute_node" / "dataset" / "generated"
        self.node_name = os.environ.get("COMPUTERNAME", socket.gethostname())

    def run(self) -> dict:
        """Main entry point. Connects to main node and executes assigned work."""
        tcp_port = DEFAULT_TCP_PORT
        best_backend = _get_best_backend()
        best_gflops = _get_best_gflops()

        logger.info(f"Connecting to Main Node at {self.main_addr}:{tcp_port}...")
        logger.info(f"  Local benchmark: {best_backend.upper()} @ {best_gflops:.1f} GFLOPS")

        # ─── Connect to Main Node ────────────────────────────────────────
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(30.0)

        try:
            sock.connect((self.main_addr, tcp_port))
        except Exception as exc:
            logger.error(f"Cannot connect to Main Node: {exc}")
            return {"error": str(exc)}

        try:
            return self._execute_workflow(sock, best_backend, best_gflops)
        except Exception as exc:
            logger.error(f"Workflow failed: {exc}")
            return {"error": str(exc)}
        finally:
            try:
                sock.close()
            except Exception:
                pass

    def _execute_workflow(self, sock: socket.socket, backend: str, gflops: float) -> dict:
        """Execute the full compute node workflow."""

        # ─── Step 1: Register ────────────────────────────────────────────
        send_json(sock, MSG_REGISTER, {
            "node_name": self.node_name,
            "gflops": gflops,
            "backend": backend,
        })
        logger.info("Registration sent, waiting for task assignment...")

        # ─── Step 2: Receive TASK_ASSIGN ─────────────────────────────────
        msg_type, payload = recv_msg(sock)
        if msg_type != MSG_TASK_ASSIGN:
            raise RuntimeError(f"Expected TASK_ASSIGN, got type={msg_type}")

        task = json.loads(payload.decode("utf-8"))
        worker_id = task["worker_id"]
        start_oc = task["start_oc"]
        end_oc = task["end_oc"]
        h, w = task["h"], task["w"]
        c_in, c_out = task["c_in"], task["c_out"]
        k, pad = task["k"], task["pad"]
        slice_cout = end_oc - start_oc

        logger.info(f"Assigned: worker #{worker_id}, oc=[{start_oc},{end_oc}), "
                     f"scale={h}x{w}, slice_cout={slice_cout}")

        # ─── Step 3: Receive WEIGHT_DATA ─────────────────────────────────
        msg_type, weight_data = recv_msg(sock)
        if msg_type != MSG_WEIGHT_DATA:
            raise RuntimeError(f"Expected WEIGHT_DATA, got type={msg_type}")

        weight_mb = len(weight_data) / 1048576
        logger.info(f"Received weight data: {weight_mb:.1f} MB")

        # Save weight slice to file
        weight_path = self.dataset_dir / f"runtime_weight_{worker_id}.bin"
        weight_path.write_bytes(weight_data)

        # ─── Step 4: Wait for START signal ───────────────────────────────
        msg_type, _ = recv_msg(sock)
        if msg_type != MSG_START:
            raise RuntimeError(f"Expected START, got type={msg_type}")

        logger.info("START signal received. Executing computation...")

        # ─── Step 5: Execute computation ─────────────────────────────────
        input_path = self.dataset_dir / "runtime_input.bin"
        output_path = self.dataset_dir / f"runtime_output_{worker_id}.bin"

        start_time = time.perf_counter()

        result = run_computation(
            backend_name=backend,
            h=h, w=w, c_in=c_in, c_out=c_out,
            k=k, pad=pad,
            input_path=input_path,
            weight_path=weight_path,
            output_path=output_path,
            start_oc=start_oc, end_oc=end_oc,
            logger=logger,
        )

        elapsed = time.perf_counter() - start_time
        logger.info(f"Computation done: {elapsed:.2f}s, "
                     f"{result.effective_gflops:.1f} GFLOPS, "
                     f"checksum={result.checksum}")

        # ─── Step 6: Send TASK_DONE ──────────────────────────────────────
        send_json(sock, MSG_TASK_DONE, {
            "worker_id": worker_id,
            "elapsed_seconds": elapsed,
            "effective_gflops": result.effective_gflops,
            "checksum": result.checksum,
        })

        # ─── Step 7: Send OUTPUT_DATA ────────────────────────────────────
        output_data = output_path.read_bytes()
        send_binary(sock, MSG_OUTPUT_DATA, output_data)
        logger.info(f"Sent output: {len(output_data)/1048576:.1f} MB")

        # Cleanup worker-specific temp files
        for p in [weight_path, output_path]:
            if p.exists():
                p.unlink()

        # ─── Step 8: Wait for ALL_DONE ───────────────────────────────────
        try:
            msg_type, _ = recv_msg(sock)
            if msg_type == MSG_ALL_DONE:
                logger.info("ALL_DONE received. Shutting down.")
        except Exception:
            logger.info("Main node disconnected.")

        return {
            "mode": "compute",
            "worker_id": worker_id,
            "start_oc": start_oc,
            "end_oc": end_oc,
            "elapsed_seconds": elapsed,
            "effective_gflops": result.effective_gflops,
            "checksum": result.checksum,
        }
