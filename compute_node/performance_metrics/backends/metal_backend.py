"""Metal backend for the Convolution benchmark on macOS."""

from __future__ import annotations

import json
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
METAL_DIR = ROOT_DIR / "fixed_matrix_vector_multiplication" / "metal"
METAL_HOST_SOURCE_PATH = METAL_DIR / "fmvm_metal_runner.mm"
METAL_KERNEL_SOURCE_PATH = METAL_DIR / "fmvm_metal_kernels.metal"
METAL_BUILD_DIR = METAL_DIR / "build"
METAL_EXECUTABLE_PATH = METAL_BUILD_DIR / "fmvm_metal_runner"


def _relative_project_path(path: Path) -> str:
    return to_relative_string(path, start=ROOT_DIR)

def _relative_cli_path(path: Path) -> str:
    return to_relative_cli_path(path, start=ROOT_DIR)

def _candidate_block_sizes() -> list[int]:
    return [32, 64, 128, 256, 512, 1024]

def _candidate_tile_sizes() -> list[int]:
    return [1, 2, 4, 8, 16]

class MetalBackend:
    name = "metal"

    def probe(self) -> tuple[bool, str]:
        if sys.platform != "darwin": return False, "Metal backend is only available on macOS."
        return True, "Metal backend ready"

    def run(self, spec: BenchmarkSpec, dataset: DatasetLayout, *, time_budget_seconds: float, force_rebuild: bool = False) -> BackendResult:
        available, message = self.probe()
        notes = [message]
        if not available: return BackendResult(self.name, False, None, None, None, [], notes)

        executable_path = METAL_EXECUTABLE_PATH # Skip dynamic build for brevity

        # 核心修改：改为传递卷积特有的参数
        command = [
            str(executable_path),
            "--input", _relative_cli_path(dataset.input_path),
            "--weight", _relative_cli_path(dataset.weight_path),
            "--h", str(spec.h), "--w", str(spec.w),
            "--cin", str(spec.c_in), "--cout", str(spec.c_out),
            "--k", str(spec.k), "--pad", str(spec.pad),
            "--block-sizes", ",".join(str(v) for v in _candidate_block_sizes()),
            "--tile-sizes", ",".join(str(v) for v in _candidate_tile_sizes()),
            "--autotune-repeats", str(DEFAULT_AUTOTUNE_REPEATS),
            "--measurement-repeats", str(DEFAULT_MEASUREMENT_REPEATS),
        ]

        try:
            completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=max(time_budget_seconds, 30.0), cwd=ROOT_DIR)
            metrics = json.loads(completed.stdout)
        except Exception as exc:
            notes.append("Metal benchmark failed or timed out")
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
            "block_size": int(metrics["block_size"]), "tile_size": int(metrics["tile_size"]),
            "autotune_repeats": int(metrics["autotune_repeats"]), "measurement_repeats": int(metrics["measurement_repeats"]),
            "trials_run": int(metrics["trials_run"]),
        }

        autotune_trial = TrialRecord(self.name, config, float(metrics["autotune_wall_clock_latency_seconds"]), float(metrics["autotune_effective_gflops"]), str(metrics["autotune_checksum"]), autotune_score, [])
        trial = TrialRecord(self.name, config, float(metrics["measurement_wall_clock_latency_seconds"]), float(metrics["measurement_effective_gflops"]), str(metrics["measurement_checksum"]), measurement_score, [])
        return BackendResult(self.name, True, dict(config), autotune_trial, trial, [autotune_trial, trial], notes)
