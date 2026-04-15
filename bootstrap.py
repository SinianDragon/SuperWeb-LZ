"""SuperWeb Cluster Bootstrap — entry point for both Main and Compute nodes.

Usage:
    python bootstrap.py                  # Auto-discover role via LAN scan
    python bootstrap.py --role main      # Force Main Node mode
    python bootstrap.py --role compute --main-addr 192.168.1.100  # Force Compute Node
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# ─── Path setup ─────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "compute_node" / "performance_metrics"))

from logging_setup import configure_logging
from adapters.platform import detect_os, is_admin
from config import AppConfig
from constants import MAIN_NODE_NAME


def parse_args():
    parser = argparse.ArgumentParser(description="SuperWeb Distributed Cluster")
    parser.add_argument("--role", choices=["main", "compute"], default=None,
                        help="Force role (skip LAN discovery)")
    parser.add_argument("--main-addr", default=None,
                        help="Main node IP (required when --role compute)")
    parser.add_argument("--name", default=None,
                        help="Node name (default: hostname)")
    parser.add_argument("--force-benchmark", action="store_true",
                        help="Force re-run benchmark even if result.json already exists")
    return parser.parse_args()


def run_discovery(config) -> tuple[str, str | None]:
    """Run 10s LAN discovery. Returns (role, main_addr_or_None)."""
    from discovery.pairing import run_pairing

    logger = logging.getLogger("superweb_cluster")
    logger.info("Starting 10s LAN discovery scan...")

    result = run_pairing(config)
    if result and result.peer_address:
        logger.info(f"Main Node found at {result.peer_address}. Role -> Compute Node")
        return "compute", result.peer_address
    else:
        reason = result.message if result and result.message else "Discovery timed out."
        logger.info(f"No Main Node found ({reason}). Role -> Main Node")
        return "main", None


def ensure_data_generated(role: str, logger):
    """Generate test + runtime data files if they don't already exist."""
    import subprocess

    gen_dir = ROOT_DIR / "compute_node" / "dataset" / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Check what already exists
    has_test = (gen_dir / "test_input.bin").exists()
    has_runtime_input = (gen_dir / "runtime_input.bin").exists()
    has_runtime_weight = (gen_dir / "runtime_weight.bin").exists()

    if role == "main":
        if has_test and has_runtime_input and has_runtime_weight:
            logger.info("All data files already exist, skipping generation.")
            return
    else:  # compute
        if has_test and has_runtime_input:
            logger.info("Compute data files already exist, skipping generation.")
            return

    logger.info(f"Generating data files for role: {role}")
    subprocess.run(
        [sys.executable, str(ROOT_DIR / "compute_node" / "dataset" / "generate.py"),
         "--output-dir", str(gen_dir),
         "--role", role],
        check=True, cwd=str(ROOT_DIR),
    )


def ensure_benchmark_ready(role: str, logger, force: bool = False):
    """Run benchmark if result.json doesn't exist."""
    import subprocess

    result_path = ROOT_DIR / "compute_node" / "performance_metrics" / "result.json"

    if not force and result_path.exists():
        logger.info("Benchmark result already exists, skipping. Use --force-benchmark to re-run.")
        return

    logger.info(f"Generating data files and running benchmark for role: {role}")
    subprocess.run(
        [sys.executable,
         str(ROOT_DIR / "compute_node" / "performance_metrics" / "benchmark.py"),
         "--role", role],
        check=True, cwd=str(ROOT_DIR),
    )


def print_summary(result: dict, logger):
    """Print a human-readable summary of the computation result."""
    logger.info("=" * 60)
    logger.info("  COMPUTATION RESULT SUMMARY")
    logger.info("=" * 60)

    mode = result.get("mode", "unknown")
    if mode == "local":
        logger.info(f"  Mode:       Local (single node)")
        logger.info(f"  Backend:    {result.get('backend', '?').upper()}")
        logger.info(f"  Time:       {result.get('elapsed_seconds', 0):.2f}s")
        logger.info(f"  GFLOPS:     {result.get('effective_gflops', 0):.1f}")
        logger.info(f"  Checksum:   {result.get('checksum', '?')}")
        logger.info(f"  Output:     {result.get('output_path', '?')}")
        logger.info(f"  Output Size:{result.get('output_size_mb', 0):.1f} MB")
    elif mode == "distributed":
        logger.info(f"  Mode:       Distributed")
        logger.info(f"  Workers:    {result.get('total_workers', 0)}")
        logger.info(f"  Slices OK:  {result.get('slices_received', 0)}")
        logger.info(f"  Output:     {result.get('output_path', '?')}")
        logger.info(f"  Output Size:{result.get('output_size_mb', 0):.1f} MB")
        if result.get("errors"):
            logger.warning(f"  Errors:     {result['errors']}")
    elif mode == "compute":
        logger.info(f"  Mode:       Compute Node")
        logger.info(f"  Worker ID:  {result.get('worker_id', '?')}")
        logger.info(f"  OC Range:   [{result.get('start_oc', '?')}, {result.get('end_oc', '?')})")
        logger.info(f"  Time:       {result.get('elapsed_seconds', 0):.2f}s")
        logger.info(f"  GFLOPS:     {result.get('effective_gflops', 0):.1f}")
        logger.info(f"  Checksum:   {result.get('checksum', '?')}")
    else:
        logger.info(f"  Result: {json.dumps(result, indent=2)}")

    if result.get("error"):
        logger.error(f"  ERROR: {result['error']}")

    logger.info("=" * 60)


def main():
    configure_logging()
    logger = logging.getLogger("superweb_cluster")
    args = parse_args()

    detect_os()
    is_admin()

    config = AppConfig()

    # ─── Step 1: Determine role ──────────────────────────────────────────
    if args.role:
        role = args.role
        main_addr = args.main_addr
        if role == "compute" and not main_addr:
            logger.error("--main-addr is required when --role compute")
            sys.exit(1)
        logger.info(f"Role forced: {role.upper()}")
    else:
        role, main_addr = run_discovery(config)

    # ─── Step 2: Generate data ───────────────────────────────────────────
    ensure_data_generated(role, logger)

    # ─── Step 3: Run benchmark ───────────────────────────────────────────
    try:
        ensure_benchmark_ready(role, logger, force=args.force_benchmark)
    except Exception as exc:
        logger.error(f"Failed to run benchmark for role {role}: {exc}")
        sys.exit(1)

    # ─── Step 4: Start runtime ───────────────────────────────────────────
    start_time = time.perf_counter()

    if role == "main":
        from main_node.runtime import MainNodeRuntime
        runtime = MainNodeRuntime()
        result = runtime.run()
    else:
        from compute_node.runtime import ComputeNodeRuntime
        runtime = ComputeNodeRuntime(main_addr=main_addr)
        result = runtime.run()

    total_time = time.perf_counter() - start_time
    result["total_wall_time"] = total_time

    # ─── Step 5: Print summary ───────────────────────────────────────────
    print_summary(result, logger)

    # Save result to file
    result_path = ROOT_DIR / "compute_node" / "dataset" / "generated" / "execution_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Result saved to {result_path}")


if __name__ == "__main__":
    main()
