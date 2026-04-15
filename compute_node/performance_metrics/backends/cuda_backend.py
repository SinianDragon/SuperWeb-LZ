"""CUDA backend for the Convolution benchmark."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from models import (
        DEFAULT_AUTOTUNE_REPEATS,
        DEFAULT_MEASUREMENT_REPEATS,
        BackendResult,
        BenchmarkSpec,
        DatasetLayout,
        TrialRecord,
    )
    from path_utils import sanitize_text, to_relative_cli_path, to_relative_string
    from scoring import linear_time_score
except ImportError:
    from compute_node.performance_metrics.models import (
        DEFAULT_AUTOTUNE_REPEATS,
        DEFAULT_MEASUREMENT_REPEATS,
        BackendResult,
        BenchmarkSpec,
        DatasetLayout,
        TrialRecord,
    )
    from compute_node.performance_metrics.path_utils import sanitize_text, to_relative_cli_path, to_relative_string
    from compute_node.performance_metrics.scoring import linear_time_score

ROOT_DIR = Path(__file__).resolve().parents[1]
CUDA_DIR = ROOT_DIR / "conv2d_runners" / "cuda"
CUDA_SOURCE_PATH = CUDA_DIR / "fmvm_cuda_runner.cu"
CUDA_BUILD_DIR = CUDA_DIR / "build"
CUDA_EXECUTABLE_PATH = CUDA_BUILD_DIR / ("fmvm_cuda_runner.exe" if os.name == "nt" else "fmvm_cuda_runner")
WINDOWS_PREBUILT_SMS = ("75", "80", "86", "89", "90")


def _relative_project_path(path: Path) -> str:
    return to_relative_string(path, start=ROOT_DIR)

def _sanitize_note(text: str) -> str:
    return sanitize_text(text, start=ROOT_DIR)

def _relative_cli_path(path: Path) -> str:
    return to_relative_cli_path(path, start=ROOT_DIR)

def _binary_is_stale(binary_path: Path, inputs: list[Path]) -> bool:
    if not binary_path.exists(): return True
    binary_mtime = binary_path.stat().st_mtime
    for input_path in inputs:
        if input_path.exists() and binary_mtime < input_path.stat().st_mtime: return True
    return False

def _candidate_block_sizes() -> list[int]:
    return [64, 128, 256, 512]

def _candidate_tile_sizes() -> list[int]:
    return [1, 2, 4, 8]

def _candidate_transpose_modes() -> list[int]:
    return [0, 1]

def _autotune_repeats(_spec: BenchmarkSpec | None = None) -> int:
    return DEFAULT_AUTOTUNE_REPEATS

def _measurement_repeats(_spec: BenchmarkSpec | None = None) -> int:
    return DEFAULT_MEASUREMENT_REPEATS

class CudaBackend:
    name = "cuda"

    def probe(self) -> tuple[bool, str]:
        if not CUDA_SOURCE_PATH.exists(): return False, "Missing CUDA source"
        return True, "CUDA Backend Ready"

    def run(self, spec: BenchmarkSpec, dataset: DatasetLayout, *, time_budget_seconds: float, force_rebuild: bool = False) -> BackendResult:
        available, message = self.probe()
        notes = [message]
        if not available: return BackendResult(self.name, False, None, None, None, [], notes)

        try:
            executable_path = CUDA_EXECUTABLE_PATH # Skipping dynamic build logic for brevity
        except Exception as exc:
            notes.append(_sanitize_note(f"failed to prepare CUDA runner: {exc}"))
            return BackendResult(self.name, False, None, None, None, [], notes)

        block_sizes = _candidate_block_sizes()
        tile_sizes = _candidate_tile_sizes()
        transpose_modes = _candidate_transpose_modes()

        # 核心修改：改为传递卷积特有的参数
        command = [
            str(executable_path),
            "--input", _relative_cli_path(dataset.input_path),
            "--weight", _relative_cli_path(dataset.weight_path),
            "--h", str(spec.h), "--w", str(spec.w),
            "--cin", str(spec.c_in), "--cout", str(spec.c_out),
            "--k", str(spec.k), "--pad", str(spec.pad),
            "--transpose-modes", ",".join(str(v) for v in transpose_modes),
            "--block-sizes", ",".join(str(v) for v in block_sizes),
            "--tile-sizes", ",".join(str(v) for v in tile_sizes),
            "--autotune-repeats", str(_autotune_repeats(spec)),
            "--measurement-repeats", str(_measurement_repeats(spec)),
        ]

        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=max(time_budget_seconds, 30.0), cwd=ROOT_DIR)
            metrics = json.loads(completed.stdout)
        except Exception as exc:
            notes.append("CUDA benchmark failed or timed out")
            return BackendResult(self.name, False, None, None, None, [], notes)

        autotune_score = linear_time_score(
            float(metrics["autotune_wall_clock_latency_seconds"]),
            ideal_seconds=spec.ideal_seconds,
            zero_score_seconds=spec.zero_score_seconds
        )
        measurement_score = linear_time_score(
            float(metrics["measurement_wall_clock_latency_seconds"]),
            ideal_seconds=spec.ideal_seconds,
            zero_score_seconds=spec.zero_score_seconds
        )

        config = {
            "transpose": bool(metrics["transpose"]), "block_size": int(metrics["block_size"]),
            "tile_size": int(metrics["tile_size"]), "autotune_repeats": int(metrics["autotune_repeats"]),
            "measurement_repeats": int(metrics["measurement_repeats"]), "trials_run": int(metrics["trials_run"]),
        }
        autotune_trial = TrialRecord(self.name, config, float(metrics["autotune_wall_clock_latency_seconds"]), float(metrics["autotune_effective_gflops"]), str(metrics["autotune_checksum"]), autotune_score, [])
        trial = TrialRecord(self.name, config, float(metrics["measurement_wall_clock_latency_seconds"]), float(metrics["measurement_effective_gflops"]), str(metrics["measurement_checksum"]), measurement_score, [])
        return BackendResult(self.name, True, dict(config), autotune_trial, trial, [autotune_trial, trial], notes)
