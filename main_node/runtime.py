"""Main Node Runtime — TCP server orchestrating distributed computation."""

from __future__ import annotations

import json
import logging
import os
import socket
import struct
import tempfile
import threading
import time
from pathlib import Path

from common.cluster_protocol import (
    MSG_REGISTER, MSG_TASK_ASSIGN, MSG_WEIGHT_DATA,
    MSG_START, MSG_OUTPUT_DATA, MSG_TASK_DONE, MSG_ALL_DONE, MSG_ERROR,
    send_json, send_binary, recv_msg,
)
from compute_node.executor import run_computation, _get_best_gflops
from compute_node.performance_metrics.workloads import get_runtime_spec
from constants import DEFAULT_TCP_PORT
from main_node.aggregator import merge_output_slices, save_output
from main_node.dispatcher import TaskDispatcher
from main_node.registry import ClusterRegistry

logger = logging.getLogger("superweb_cluster")

WORKER_ACCEPT_TIMEOUT = 30  # seconds to wait for workers after discovery


class MainNodeRuntime:
    """Full Main-Node runtime: accept workers, distribute work, aggregate output."""

    def __init__(self, config=None):
        self.config = config
        self.registry = ClusterRegistry()
        self.dispatcher = TaskDispatcher(self.registry, logger)
        self.spec = get_runtime_spec()
        self.root = Path(__file__).resolve().parent.parent
        self.dataset_dir = self.root / "compute_node" / "dataset" / "generated"
        self._stop_event = threading.Event()

    def run(self) -> dict:
        """Main entry point. Returns execution summary dict."""
        tcp_port = DEFAULT_TCP_PORT

        # ─── Phase 1: Accept worker connections ──────────────────────────
        logger.info(f"Starting TCP server on port {tcp_port}, "
                    f"waiting {WORKER_ACCEPT_TIMEOUT}s for workers...")

        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_sock.settimeout(1.0)

        try:
            server_sock.bind(("0.0.0.0", tcp_port))
            server_sock.listen(10)
        except OSError as exc:
            logger.warning(f"Cannot bind TCP port {tcp_port}: {exc}. Running locally.")
            return self.dispatcher.run_locally()

        deadline = time.monotonic() + WORKER_ACCEPT_TIMEOUT

        while time.monotonic() < deadline and not self._stop_event.is_set():
            try:
                conn, addr = server_sock.accept()
                conn.settimeout(30.0)
                self._handle_registration(conn, addr)
            except socket.timeout:
                continue
            except Exception as exc:
                logger.warning(f"Accept error: {exc}")

        server_sock.close()
        worker_count = self.registry.count_workers()

        # ─── Phase 2: Decide execution mode ──────────────────────────────
        if worker_count == 0:
            logger.info("No workers connected. Running locally.")
            return self.dispatcher.run_locally()

        logger.info(f"{worker_count} worker(s) connected. Running distributed computation.")
        return self._run_distributed()

    def _handle_registration(self, conn: socket.socket, addr: tuple):
        """Process a REGISTER message from a new worker."""
        try:
            msg_type, payload = recv_msg(conn)
            if msg_type != MSG_REGISTER:
                logger.warning(f"Expected REGISTER from {addr}, got type={msg_type}")
                conn.close()
                return

            info = json.loads(payload.decode("utf-8"))
            worker = self.registry.register_worker(
                node_name=info.get("node_name", f"worker-{addr[0]}"),
                gflops=float(info.get("gflops", 0)),
                backend=info.get("backend", "cpu"),
                conn=conn, addr=addr,
            )
            logger.info(f"Registered worker #{worker.worker_id}: "
                        f"{worker.node_name} @ {addr[0]} ({worker.gflops:.1f} GFLOPS)")

        except Exception as exc:
            logger.warning(f"Registration failed from {addr}: {exc}")
            try:
                conn.close()
            except Exception:
                pass

    def _run_distributed(self) -> dict:
        """Distribute computation across workers + local, then aggregate."""
        spec = self.spec
        workers = self.registry.list_workers()
        main_gflops = _get_best_gflops()

        # ─── Allocate slices ─────────────────────────────────────────────
        allocation = self.registry.allocate_slices(spec.c_out, main_gflops)
        logger.info(f"Slice allocation: {allocation}")

        # ─── Load full weight file ───────────────────────────────────────
        weight_path = self.dataset_dir / "runtime_weight.bin"
        weight_data = weight_path.read_bytes()
        weight_per_oc = spec.k * spec.k * spec.c_in * 4  # bytes per output channel

        # ─── Distribute to workers ───────────────────────────────────────
        worker_threads = []
        worker_results: dict[int, tuple[int, int, bytes | None]] = {}
        errors = []
        results_lock = threading.Lock()

        for worker in workers:
            key = f"worker_{worker.worker_id}"
            if key not in allocation:
                continue

            start_oc, end_oc = allocation[key]
            worker.start_oc = start_oc
            worker.end_oc = end_oc

            # Extract weight slice: [start_oc:end_oc] from [Cout, K, K, Cin]
            w_start = start_oc * weight_per_oc
            w_end = end_oc * weight_per_oc
            weight_slice = weight_data[w_start:w_end]

            t = threading.Thread(
                target=self._distribute_to_worker,
                args=(worker, start_oc, end_oc, weight_slice, worker_results, errors, results_lock),
                daemon=True,
            )
            worker_threads.append(t)
            t.start()

        # ─── Execute local slice ─────────────────────────────────────────
        local_result = None
        if "main" in allocation:
            local_start, local_end = allocation["main"]
            logger.info(f"Main node computing local slice oc=[{local_start},{local_end})")

            # Write local weight slice to temp file
            local_w_start = local_start * weight_per_oc
            local_w_end = local_end * weight_per_oc
            local_weight_slice = weight_data[local_w_start:local_w_end]

            local_weight_path = self.dataset_dir / "_tmp_local_weight.bin"
            local_weight_path.write_bytes(local_weight_slice)

            local_output_path = self.dataset_dir / "_tmp_local_output.bin"

            try:
                exec_result = run_computation(
                    h=spec.h, w=spec.w, c_in=spec.c_in, c_out=spec.c_out,
                    k=spec.k, pad=spec.pad,
                    input_path=self.dataset_dir / "runtime_input.bin",
                    weight_path=local_weight_path,
                    output_path=local_output_path,
                    start_oc=local_start, end_oc=local_end,
                    logger=logger,
                )
                local_output_data = local_output_path.read_bytes()
                local_result = (local_start, local_end, local_output_data)
                logger.info(f"Main node local slice done: {exec_result.effective_gflops:.1f} GFLOPS")
            except Exception as exc:
                logger.error(f"Main node local execution failed: {exc}")
                errors.append(str(exc))
            finally:
                for p in [local_weight_path, local_output_path]:
                    if p.exists():
                        p.unlink()

        # ─── Wait for all workers ────────────────────────────────────────
        for t in worker_threads:
            t.join(timeout=300)

        # ─── Aggregate output ────────────────────────────────────────────
        out_h = spec.h + 2 * spec.pad - spec.k + 1
        out_w = spec.w + 2 * spec.pad - spec.k + 1

        slices = []
        for wid, (start_oc, end_oc, data) in worker_results.items():
            if data is not None:
                slices.append((start_oc, end_oc, data))
        if local_result is not None:
            slices.append(local_result)

        if not slices:
            logger.error("No output slices collected! Distributed computation failed.")
            return {"mode": "distributed", "error": "all slices failed"}

        logger.info(f"Aggregating {len(slices)} output slices...")
        final_output = merge_output_slices(slices, spec.c_out, out_h, out_w)

        output_path = self.dataset_dir / "runtime_output.bin"
        save_output(final_output, output_path)
        output_mb = len(final_output) / 1048576

        # ─── Send ALL_DONE to workers ────────────────────────────────────
        for worker in workers:
            try:
                send_json(worker.conn, MSG_ALL_DONE, {})
                worker.conn.close()
            except Exception:
                pass

        logger.info(f"Distributed computation complete: "
                    f"{len(slices)} slices, output={output_mb:.1f} MB")

        return {
            "mode": "distributed",
            "total_workers": len(workers),
            "slices_received": len(slices),
            "errors": errors,
            "output_path": str(output_path),
            "output_size_mb": output_mb,
        }

    def _distribute_to_worker(self, worker, start_oc, end_oc, weight_slice,
                              results_dict, errors_list, lock):
        """Send task to one worker, wait for output. Runs in a thread."""
        spec = self.spec
        try:
            # 1. Send TASK_ASSIGN
            send_json(worker.conn, MSG_TASK_ASSIGN, {
                "worker_id": worker.worker_id,
                "start_oc": start_oc,
                "end_oc": end_oc,
                "h": spec.h, "w": spec.w,
                "c_in": spec.c_in, "c_out": spec.c_out,
                "k": spec.k, "pad": spec.pad,
            })

            # 2. Send WEIGHT_DATA
            send_binary(worker.conn, MSG_WEIGHT_DATA, weight_slice)
            logger.info(f"Sent {len(weight_slice)/1048576:.1f} MB weight to worker #{worker.worker_id}")

            # 3. Send START
            send_json(worker.conn, MSG_START, {})

            # 4. Wait for TASK_DONE
            msg_type, payload = recv_msg(worker.conn)
            if msg_type == MSG_TASK_DONE:
                info = json.loads(payload.decode("utf-8"))
                logger.info(f"Worker #{worker.worker_id} done: "
                            f"{info.get('elapsed_seconds', 0):.2f}s, "
                            f"checksum={info.get('checksum', '?')}")

            # 5. Receive OUTPUT_DATA
            msg_type, output_data = recv_msg(worker.conn)
            if msg_type == MSG_OUTPUT_DATA:
                with lock:
                    results_dict[worker.worker_id] = (start_oc, end_oc, output_data)
                logger.info(f"Received {len(output_data)/1048576:.1f} MB output from worker #{worker.worker_id}")
            else:
                with lock:
                    errors_list.append(f"Worker #{worker.worker_id}: unexpected msg type {msg_type}")
                    results_dict[worker.worker_id] = (start_oc, end_oc, None)

        except Exception as exc:
            logger.error(f"Worker #{worker.worker_id} communication failed: {exc}")
            with lock:
                errors_list.append(str(exc))
                results_dict[worker.worker_id] = (start_oc, end_oc, None)
