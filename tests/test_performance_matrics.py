"""Tests for the performance-metrics workspace (Conv2D architecture)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

PERF_DIR = Path(__file__).resolve().parents[1] / "compute_node" / "performance_metrics"
if str(PERF_DIR) not in sys.path:
    sys.path.insert(0, str(PERF_DIR))

import benchmark
from backends.cpu_backend import (
    CpuArtifacts,
    CpuBackend,
    _binary_tree_worker_candidates,
    _candidate_tile_sizes as cpu_candidate_tile_sizes,
    _cpu_artifacts_for_platform,
)
from backends.cuda_backend import (
    CudaBackend,
    _candidate_block_sizes as cuda_candidate_block_sizes,
    _candidate_tile_sizes as cuda_candidate_tile_sizes,
    _candidate_transpose_modes as cuda_candidate_transpose_modes,
)
from backends.metal_backend import (
    MetalBackend,
)
from fmvm_dataset import (
    build_dataset_layout,
    dataset_is_generated,
    generate_dataset,
    load_float32_file,
)
from models import DEFAULT_AUTOTUNE_REPEATS, DEFAULT_MEASUREMENT_REPEATS, BenchmarkSpec
from path_utils import to_relative_cli_path
from scoring import linear_time_score
from workloads import build_benchmark_spec, get_test_spec, get_runtime_spec


class ScoringTests(unittest.TestCase):
    def test_linear_score_midpoint_is_half(self) -> None:
        score = linear_time_score(0.6, ideal_seconds=0.2, zero_score_seconds=1.0, max_score=100.0)
        self.assertAlmostEqual(score, 50.0)

    def test_linear_score_below_ideal_is_max(self) -> None:
        score = linear_time_score(0.1, ideal_seconds=0.2, zero_score_seconds=1.0, max_score=100.0)
        self.assertEqual(score, 100.0)

    def test_linear_score_above_zero_score_is_zero(self) -> None:
        score = linear_time_score(2.0, ideal_seconds=0.2, zero_score_seconds=1.0, max_score=100.0)
        self.assertEqual(score, 0.0)


class WorkerCandidateTests(unittest.TestCase):
    def test_worker_binary_tree_sequence(self) -> None:
        self.assertEqual(_binary_tree_worker_candidates(16), [16, 8, 32, 4, 64])

    def test_worker_binary_tree_single_core(self) -> None:
        result = _binary_tree_worker_candidates(1)
        self.assertIn(1, result)
        self.assertTrue(all(x >= 1 for x in result))


class BenchmarkParserTests(unittest.TestCase):
    def test_benchmark_parser_accepts_backend_flag(self) -> None:
        args = benchmark.build_parser().parse_args(["--backend", "cpu"])
        self.assertEqual(args.backend, ["cpu"])

    def test_benchmark_parser_defaults(self) -> None:
        args = benchmark.build_parser().parse_args([])
        self.assertIsNone(args.backend)
        self.assertIsNone(args.h)
        self.assertEqual(args.role, "compute")


class CpuArtifactTests(unittest.TestCase):
    def test_cpu_artifacts_follow_platform(self) -> None:
        windows_artifacts = _cpu_artifacts_for_platform("win32")
        assert windows_artifacts is not None
        self.assertEqual(windows_artifacts.platform_key, "windows")
        self.assertEqual(windows_artifacts.executable_path.name, "fmvm_cpu_windows.exe")

        macos_artifacts = _cpu_artifacts_for_platform("darwin")
        assert macos_artifacts is not None
        self.assertEqual(macos_artifacts.platform_key, "macos")
        self.assertEqual(macos_artifacts.executable_path.name, "fmvm_cpu_macos")

        self.assertIsNone(_cpu_artifacts_for_platform("linux"))


class WorkloadSpecTests(unittest.TestCase):
    def test_test_spec_dimensions(self) -> None:
        spec = get_test_spec()
        self.assertEqual(spec.h, 256)
        self.assertEqual(spec.w, 256)
        self.assertEqual(spec.c_in, 32)
        self.assertEqual(spec.c_out, 64)
        self.assertEqual(spec.k, 3)
        self.assertEqual(spec.pad, 1)

    def test_runtime_spec_dimensions(self) -> None:
        spec = get_runtime_spec()
        self.assertEqual(spec.h, 2048)
        self.assertEqual(spec.w, 2048)
        self.assertEqual(spec.c_in, 128)
        self.assertEqual(spec.c_out, 256)

    def test_default_build_gives_test_spec(self) -> None:
        spec = build_benchmark_spec()
        test_spec = get_test_spec()
        self.assertEqual(spec.h, test_spec.h)
        self.assertEqual(spec.w, test_spec.w)
        self.assertEqual(spec.c_in, test_spec.c_in)
        self.assertEqual(spec.c_out, test_spec.c_out)

    def test_custom_build_overrides(self) -> None:
        spec = build_benchmark_spec(h=64, w=64, c_in=8, c_out=16, k=3, pad=1)
        self.assertEqual(spec.h, 64)
        self.assertEqual(spec.c_in, 8)
        self.assertEqual(spec.c_out, 16)

    def test_flops_per_run_calculation(self) -> None:
        spec = BenchmarkSpec(
            name="test", h=4, w=4, c_in=2, c_out=3, k=3, pad=1,
            ideal_seconds=1.0, zero_score_seconds=10.0,
        )
        out_h = spec.h + 2 * spec.pad - spec.k + 1  # 4
        out_w = spec.w + 2 * spec.pad - spec.k + 1  # 4
        expected = 2 * out_h * out_w * spec.c_out * spec.c_in * spec.k * spec.k
        self.assertEqual(spec.flops_per_run, expected)

    def test_input_and_weight_bytes(self) -> None:
        spec = BenchmarkSpec(
            name="test", h=8, w=8, c_in=4, c_out=8, k=3, pad=1,
            ideal_seconds=1.0, zero_score_seconds=10.0,
        )
        self.assertEqual(spec.input_bytes, 8 * 8 * 4 * 4)
        self.assertEqual(spec.weight_bytes, 3 * 3 * 4 * 8 * 4)


class DatasetTests(unittest.TestCase):
    def test_generate_dataset_with_small_spec(self) -> None:
        spec = build_benchmark_spec(h=8, w=8, c_in=4, c_out=8, k=3, pad=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec)
            self.assertTrue(dataset_is_generated(layout, spec))
            self.assertEqual(layout.input_path.name, "input.bin")
            self.assertEqual(layout.weight_path.name, "weight.bin")
            self.assertEqual(layout.input_path.stat().st_size, spec.input_bytes)
            self.assertEqual(layout.weight_path.stat().st_size, spec.weight_bytes)

    def test_generate_dataset_with_prefix(self) -> None:
        spec = build_benchmark_spec(h=8, w=8, c_in=4, c_out=8, k=3, pad=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_dataset_layout(Path(temp_dir), prefix="test_")
            generate_dataset(layout, spec)
            self.assertEqual(layout.input_path.name, "test_input.bin")
            self.assertEqual(layout.weight_path.name, "test_weight.bin")

    def test_dataset_is_generated_rejects_mismatched_metadata(self) -> None:
        spec = build_benchmark_spec(h=8, w=8, c_in=4, c_out=8, k=3, pad=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec)
            metadata = json.loads(layout.meta_path.read_text(encoding="utf-8"))
            metadata["benchmark"]["h"] = 16  # tamper with dimensions
            layout.meta_path.write_text(json.dumps(metadata), encoding="utf-8")
            self.assertFalse(dataset_is_generated(layout, spec))

    def test_skip_weight_generation(self) -> None:
        spec = build_benchmark_spec(h=8, w=8, c_in=4, c_out=8, k=3, pad=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec, skip_weight=True)
            self.assertTrue(layout.input_path.exists())
            self.assertFalse(layout.weight_path.exists())
            self.assertTrue(dataset_is_generated(layout, spec, skip_weight=True))


class BackendProbeTests(unittest.TestCase):
    def test_cpu_backend_probe_returns_status(self) -> None:
        backend = CpuBackend()
        available, message = backend.probe()
        self.assertIsInstance(available, bool)
        self.assertTrue(message)

    def test_cuda_backend_probe_returns_status(self) -> None:
        backend = CudaBackend()
        available, message = backend.probe()
        self.assertIsInstance(available, bool)
        self.assertTrue(message)

    def test_metal_backend_probe_returns_status(self) -> None:
        backend = MetalBackend()
        available, message = backend.probe()
        self.assertIsInstance(available, bool)
        self.assertTrue(message)


class CpuBenchmarkTests(unittest.TestCase):
    def test_cpu_tile_sizes_always_return_values(self) -> None:
        tiles = cpu_candidate_tile_sizes(64)
        self.assertTrue(len(tiles) > 0)
        self.assertIn(64, tiles)

    def test_cpu_tile_sizes_small_cout(self) -> None:
        tiles = cpu_candidate_tile_sizes(4)
        self.assertIn(4, tiles)


class CudaBenchmarkTests(unittest.TestCase):
    def test_cuda_block_sizes(self) -> None:
        sizes = cuda_candidate_block_sizes()
        self.assertTrue(len(sizes) > 0)
        self.assertTrue(all(s > 0 for s in sizes))

    def test_cuda_tile_sizes(self) -> None:
        sizes = cuda_candidate_tile_sizes()
        self.assertTrue(len(sizes) > 0)

    def test_cuda_transpose_modes(self) -> None:
        modes = cuda_candidate_transpose_modes()
        self.assertIn(0, modes)
        self.assertIn(1, modes)


class PerformanceSummaryTests(unittest.TestCase):
    """Tests for the performance_summary module that reads result.json."""

    def test_load_performance_summary_with_valid_result(self) -> None:
        """Verify that load_performance_summary correctly reads best_trial fields."""
        from compute_node.performance_summary import load_performance_summary

        result_data = {
            "schema_version": 2,
            "ranking": ["cuda", "cpu"],
            "backend_results": {
                "cuda": {
                    "available": True,
                    "best_trial": {
                        "effective_gflops": 500.0,
                        "wall_clock_latency_seconds": 0.005,
                    },
                },
                "cpu": {
                    "available": True,
                    "best_trial": {
                        "effective_gflops": 100.0,
                        "wall_clock_latency_seconds": 0.02,
                    },
                },
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            result_path.write_text(json.dumps(result_data), encoding="utf-8")
            summary = load_performance_summary(result_path)

        self.assertEqual(summary.hardware_count, 2)
        self.assertEqual(len(summary.ranked_hardware), 2)
        # cuda should be rank 1 (first in ranking list)
        self.assertEqual(summary.ranked_hardware[0].hardware_type, "cuda")
        self.assertEqual(summary.ranked_hardware[0].rank, 1)
        self.assertAlmostEqual(summary.ranked_hardware[0].effective_gflops, 500.0)
        # cpu should be rank 2
        self.assertEqual(summary.ranked_hardware[1].hardware_type, "cpu")
        self.assertEqual(summary.ranked_hardware[1].rank, 2)

    def test_load_performance_summary_missing_file(self) -> None:
        from compute_node.performance_summary import load_performance_summary

        summary = load_performance_summary(Path("/nonexistent/result.json"))
        self.assertEqual(summary.hardware_count, 0)
        self.assertEqual(len(summary.ranked_hardware), 0)

    def test_load_performance_summary_unavailable_backend(self) -> None:
        from compute_node.performance_summary import load_performance_summary

        result_data = {
            "ranking": [],
            "backend_results": {
                "metal": {"available": False, "best_trial": None},
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result_path = Path(temp_dir) / "result.json"
            result_path.write_text(json.dumps(result_data), encoding="utf-8")
            summary = load_performance_summary(result_path)

        self.assertEqual(summary.hardware_count, 0)


class IntegrationTests(unittest.TestCase):
    """Integration tests that require compiled backends to be available."""

    def test_cpu_benchmark_smoke(self) -> None:
        """Run a minimal CPU benchmark if the backend is available."""
        backend = CpuBackend()
        available, _message = backend.probe()
        if not available:
            self.skipTest("CPU backend is unavailable in this environment.")

        spec = build_benchmark_spec(h=8, w=8, c_in=4, c_out=8, k=3, pad=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec)

            with mock.patch("backends.cpu_backend.os.cpu_count", return_value=1):
                result = backend.run(spec, layout, time_budget_seconds=60.0)

        self.assertTrue(result.available)
        if result.best_trial is not None:
            self.assertGreater(result.best_trial.effective_gflops, 0.0)

    def test_cuda_benchmark_smoke(self) -> None:
        """Run a minimal CUDA benchmark if the backend is available."""
        backend = CudaBackend()
        available, _message = backend.probe()
        if not available:
            self.skipTest("CUDA backend is unavailable in this environment.")

        spec = build_benchmark_spec(h=8, w=8, c_in=4, c_out=8, k=3, pad=1)
        with tempfile.TemporaryDirectory() as temp_dir:
            layout = build_dataset_layout(Path(temp_dir))
            generate_dataset(layout, spec)
            result = backend.run(spec, layout, time_budget_seconds=60.0)

        self.assertTrue(result.available)
        if result.best_trial is not None:
            self.assertGreater(result.best_trial.effective_gflops, 0.0)
            # Sanity check: GFLOPS should not be absurdly high
            self.assertLess(result.best_trial.effective_gflops, 100_000.0)


if __name__ == "__main__":
    unittest.main()
