"""Executor — run Conv2D computation via the compiled C++ / CUDA backend."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
PERF_DIR = ROOT_DIR / "compute_node" / "performance_metrics"


@dataclass
class ExecutionResult:
    """Result of a Conv2D computation."""
    backend: str
    elapsed_seconds: float
    effective_gflops: float
    checksum: str
    output_path: Path | None


def _find_executable(backend_name: str) -> Path | None:
    """Locate the compiled runner executable."""
    fmvm_dir = PERF_DIR / "conv2d_runners"

    if backend_name == "cpu":
        if sys.platform == "win32":
            exe = fmvm_dir / "cpu" / "windows" / "build" / "fmvm_cpu_windows.exe"
        elif sys.platform == "darwin":
            exe = fmvm_dir / "cpu" / "macos" / "build" / "fmvm_cpu_macos"
        else:
            return None
        return exe if exe.exists() else None

    if backend_name == "cuda":
        exe = fmvm_dir / "cuda" / "build" / (
            "fmvm_cuda_runner.exe" if os.name == "nt" else "fmvm_cuda_runner"
        )
        return exe if exe.exists() else None

    return None


def _get_best_backend() -> str:
    """Read benchmark result.json to find best backend."""
    result_path = PERF_DIR / "result.json"
    if not result_path.exists():
        return "cpu"
    try:
        data = json.loads(result_path.read_text())
        ranking = data.get("ranking", [])
        return ranking[0] if ranking else "cpu"
    except Exception:
        return "cpu"


def _get_best_gflops() -> float:
    """Read benchmark result.json to find best GFLOPS."""
    result_path = PERF_DIR / "result.json"
    if not result_path.exists():
        return 0.0
    try:
        data = json.loads(result_path.read_text())
        best_backend = data.get("best_backend", "cpu")
        results = data.get("backend_results", {})
        br = results.get(best_backend, {})
        trial = br.get("best_trial") or br.get("best_result") or br.get("autotune_trial") or {}
        return float(trial.get("effective_gflops", 0))
    except Exception:
        return 0.0


def run_computation(
    *,
    backend_name: str | None = None,
    h: int, w: int,
    c_in: int, c_out: int,
    k: int, pad: int,
    input_path: Path,
    weight_path: Path,
    output_path: Path,
    start_oc: int = 0,
    end_oc: int | None = None,
    timeout: float = 600.0,
    logger=None,
) -> ExecutionResult:
    """Execute Conv2D using the compiled C++ / CUDA runner.

    The runner writes the output to `output_path` and prints timing JSON to stdout.
    """
    if backend_name is None:
        backend_name = _get_best_backend()

    executable = _find_executable(backend_name)
    if executable is None:
        # Fallback to CPU if preferred backend not available
        backend_name = "cpu"
        executable = _find_executable("cpu")
        if executable is None:
            raise FileNotFoundError(f"No compiled runner found for any backend")

    if end_oc is None:
        end_oc = c_out

    slice_c_out = end_oc - start_oc
    workers_arg = str(os.cpu_count() or 4)

    cmd = [
        str(executable),
        "--input", str(input_path.resolve()),
        "--weight", str(weight_path.resolve()),
        "--output", str(output_path.resolve()),
        "--h", str(h), "--w", str(w),
        "--cin", str(c_in),
        "--k", str(k), "--pad", str(pad),
        "--autotune-repeats", "1",
        "--measurement-repeats", "1",
    ]

    if backend_name == "cpu":
        # CPU runner supports --start-oc/--end-oc and --workers
        cmd.extend(["--cout", str(c_out)])
        cmd.extend(["--start-oc", str(start_oc), "--end-oc", str(end_oc)])
        cmd.extend(["--workers", workers_arg])
    else:
        # CUDA runner uses --cout as slice channel count (no --start-oc support)
        # Weight file must already be sliced to the correct range
        cmd.extend(["--cout", str(slice_c_out)])

    if logger:
        logger.info(f"Executing {backend_name.upper()} runner: oc=[{start_oc},{end_oc})")

    try:
        completed = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            timeout=timeout, cwd=str(PERF_DIR),
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{backend_name} runner timed out after {timeout}s")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{backend_name} runner failed: {exc.stderr[:500]}")

    try:
        metrics = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"{backend_name} runner produced invalid JSON: {completed.stdout[:200]}")

    elapsed = float(metrics.get("measurement_wall_clock_latency_seconds", 0))
    gflops = float(metrics.get("measurement_effective_gflops", 0))
    checksum = str(metrics.get("measurement_checksum", ""))

    return ExecutionResult(
        backend=backend_name,
        elapsed_seconds=elapsed,
        effective_gflops=gflops,
        checksum=checksum,
        output_path=output_path,
    )
