"""Benchmark constants for the Convolution workload."""

from __future__ import annotations
import sys
from pathlib import Path

# 使用显式导入防止冲突
try:
    from .models import BenchmarkSpec
except (ImportError, ValueError):
    from models import BenchmarkSpec

# === 1. Test 规模 (跑分用) ===
TEST_H, TEST_W = 256, 256
TEST_CIN, TEST_COUT = 32, 64
TEST_K, TEST_PAD = 3, 1
TEST_IDEAL_SECONDS = 0.5

# === 2. Runtime 规模 (2048 大规模任务，适合笔记本设备) ===
# 2048x2048, 128→256: GPU ~8s, CPU ~17s, 显存 ~6GB
RUNTIME_H, RUNTIME_W = 2048, 2048
RUNTIME_CIN, RUNTIME_COUT = 128, 256
RUNTIME_K, RUNTIME_PAD = 3, 1
RUNTIME_IDEAL_SECONDS = 60.0

def get_test_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        name=f"test-conv2d-{TEST_H}x{TEST_W}",
        h=TEST_H, w=TEST_W, c_in=TEST_CIN, c_out=TEST_COUT, k=TEST_K, pad=TEST_PAD,
        ideal_seconds=TEST_IDEAL_SECONDS, zero_score_seconds=TEST_IDEAL_SECONDS * 10,
    )

def get_runtime_spec() -> BenchmarkSpec:
    return BenchmarkSpec(
        name=f"runtime-conv2d-{RUNTIME_H}x{RUNTIME_W}",
        h=RUNTIME_H, w=RUNTIME_W, c_in=RUNTIME_CIN, c_out=RUNTIME_COUT, k=RUNTIME_K, pad=RUNTIME_PAD,
        ideal_seconds=RUNTIME_IDEAL_SECONDS, zero_score_seconds=RUNTIME_IDEAL_SECONDS * 10,
    )

def build_benchmark_spec(*, h=None, w=None, c_in=None, c_out=None, k=None, pad=None, **kwargs) -> BenchmarkSpec:
    if all(v is None for v in [h, w, c_in, c_out, k, pad]):
        return get_test_spec()
    return BenchmarkSpec(
        name="custom-conv2d",
        h=h or TEST_H, w=w or TEST_W, c_in=c_in or TEST_CIN, c_out=c_out or TEST_COUT,
        k=k or TEST_K, pad=pad or TEST_PAD,
        ideal_seconds=TEST_IDEAL_SECONDS, zero_score_seconds=TEST_IDEAL_SECONDS * 10
    )
