"""Dataset helpers for the Convolution benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

try:
    from models import BenchmarkSpec, DatasetLayout
except ImportError:
    from compute_node.performance_metrics.models import BenchmarkSpec, DatasetLayout

MASK32 = 0xFFFFFFFF
DEFAULT_INPUT_SEED = 0x123456789ABCDEF0
DEFAULT_WEIGHT_SEED = 0x0FEDCBA987654321
DEFAULT_CHUNK_VALUES = 1_048_576
PROGRESS_STEP_BYTES = 32 * 1024 * 1024


def build_dataset_layout(root_dir: Path, prefix: str = "") -> DatasetLayout:
    """返回带前缀的数据集路径布局，用于区分 test 和 runtime"""
    return DatasetLayout(
        root_dir=root_dir,
        input_path=root_dir / f"{prefix}input.bin",
        weight_path=root_dir / f"{prefix}weight.bin",
        meta_path=root_dir / f"{prefix}dataset_meta.json",
    )


def dataset_is_generated(layout: DatasetLayout, spec: BenchmarkSpec, skip_weight: bool = False) -> bool:
    """检查所需的数据集是否已经在硬盘上生成完毕"""
    if not layout.input_path.exists() or not layout.meta_path.exists():
        return False
    if not skip_weight and not layout.weight_path.exists():
        return False

    try:
        metadata = json.loads(layout.meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    # 校验维度是否对齐
    bm = metadata.get("benchmark", {})
    if bm.get("h") != spec.h or bm.get("w") != spec.w:
        return False
    if bm.get("c_in") != spec.c_in or bm.get("c_out") != spec.c_out:
        return False
    if bm.get("k") != spec.k or bm.get("pad") != spec.pad:
        return False

    return True


def _xorshift32_next(state: int) -> int:
    state ^= (state << 13) & MASK32
    state ^= (state >> 17) & MASK32
    state ^= (state << 5) & MASK32
    return state & MASK32


def _float32_word_from_state(state: int) -> int:
    sign = (state & 0x1) << 31
    mantissa = (state >> 1) & 0x007FFFFF
    return sign | 0x3F000000 | mantissa


def _float32_chunk_from_prng(count: int, state: int) -> tuple[bytearray, int]:
    payload = bytearray(count * 4)
    words = memoryview(payload).cast("I")
    try:
        current_state = state & MASK32
        if current_state == 0:
            current_state = 0x6D2B79F5

        for index in range(count):
            current_state = _xorshift32_next(current_state)
            words[index] = _float32_word_from_state(current_state)
    finally:
        words.release()
    return payload, current_state


def _write_float32_file(
        path: Path, total_values: int, seed: int, chunk_values: int, *, label: str,
        progress: Callable[[str, int, int], None] | None = None,
) -> str:
    """基于伪随机数生成器在硬盘上写入大块浮点数文件"""
    sha256 = hashlib.sha256()
    state = seed & MASK32
    written_values = 0
    total_bytes = total_values * 4
    next_progress_bytes = PROGRESS_STEP_BYTES

    with path.open("wb") as handle:
        while written_values < total_values:
            current_chunk_values = min(chunk_values, total_values - written_values)
            chunk, state = _float32_chunk_from_prng(current_chunk_values, state)
            handle.write(chunk)
            sha256.update(chunk)
            written_values += current_chunk_values
            written_bytes = written_values * 4
            handle.flush()

            if progress and (written_bytes >= next_progress_bytes or written_values == total_values):
                progress(label, written_bytes, total_bytes)
                while next_progress_bytes <= written_bytes:
                    next_progress_bytes += PROGRESS_STEP_BYTES

    return sha256.hexdigest()


def generate_dataset(
        layout: DatasetLayout, spec: BenchmarkSpec, skip_weight: bool = False, *, progress: Callable[[str, int, int], None] | None = None,
) -> None:
    """生成卷积运算所需的 Input 和 Weights 数据集"""
    layout.root_dir.mkdir(parents=True, exist_ok=True)

    # Check if we can skip input generation
    input_sha256 = "existing"
    if not layout.input_path.exists():
        input_sha256 = _write_float32_file(
            layout.input_path,
            total_values=spec.h * spec.w * spec.c_in,
            seed=DEFAULT_INPUT_SEED, chunk_values=DEFAULT_CHUNK_VALUES,
            label=layout.input_path.name, progress=progress,
        )

    weight_sha256 = "skipped"
    if not skip_weight:
        if layout.weight_path.exists():
            weight_sha256 = "existing"
        else:
            weight_sha256 = _write_float32_file(
                layout.weight_path,
                total_values=spec.k * spec.k * spec.c_in * spec.c_out,
                seed=DEFAULT_WEIGHT_SEED, chunk_values=DEFAULT_CHUNK_VALUES,
                label=layout.weight_path.name, progress=progress,
            )

    metadata = {
        "benchmark": {
            "name": spec.name,
            "h": spec.h, "w": spec.w,
            "c_in": spec.c_in, "c_out": spec.c_out,
            "k": spec.k, "pad": spec.pad,
            "dtype": "float32", "endianness": "little",
            "operation": "Conv2D",
        },
        "files": {
            "input_feature_map": {
                "path": layout.input_path.name, "bytes": spec.input_bytes,
                "sha256": input_sha256, "shape": [spec.h, spec.w, spec.c_in]
            },
            "weights": {
                "path": layout.weight_path.name,
                "bytes": spec.weight_bytes if not skip_weight else 0,
                "sha256": weight_sha256,
                "skipped": skip_weight,
                # 【架构升级】：适配网络切片分发的 [C_out, K, K, C_in] 连续内存排布
                "shape": [spec.c_out, spec.k, spec.k, spec.c_in]
            },
        },
    }
    layout.meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_float32_file(path: Path) -> list[float]:
    """小工具：从硬盘加载二进制文件并转为 python 的 list 结构"""
    with path.open("rb") as handle:
        raw = handle.read()
    return list(memoryview(raw).cast("f"))
