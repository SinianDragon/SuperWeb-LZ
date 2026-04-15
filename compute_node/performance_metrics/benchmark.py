"""Top-level benchmark entry point with hardware ranking."""

from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# FIX: 注入路径解决 Conda 环境下 models 库冲突 (ModuleNotFoundError: No module named 'project')
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from workloads import build_benchmark_spec, get_test_spec
from backends import build_backends
from fmvm_dataset import build_dataset_layout, dataset_is_generated
from path_utils import to_relative_cli_path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark backends.")
    parser.add_argument("--backend", action="append", default=None)
    parser.add_argument("--dataset-dir", type=Path, default=ROOT_DIR.parent / "dataset" / "generated")
    parser.add_argument("--output", type=Path, default=ROOT_DIR / "result.json")
    parser.add_argument("--role", choices=("compute", "main"), default="compute")
    parser.add_argument("--h", type=int)
    parser.add_argument("--w", type=int)
    parser.add_argument("--cin", type=int)
    parser.add_argument("--cout", type=int)
    parser.add_argument("--k", type=int)
    parser.add_argument("--pad", type=int)
    return parser

def _generate_if_needed(dataset_dir: Path, spec, role: str, has_custom_h: bool = False) -> None:
    """仅在文件缺失时调用生成脚本，避免重复生成大文件时的冗余日志。"""
    test_layout = build_dataset_layout(dataset_dir, prefix="test_")
    runtime_layout = build_dataset_layout(dataset_dir, prefix="runtime_")

    has_test = test_layout.input_path.exists() and test_layout.weight_path.exists()
    has_runtime_input = runtime_layout.input_path.exists()
    has_runtime_weight = runtime_layout.weight_path.exists()

    needs_gen = False
    if not (has_test and has_runtime_input):
        needs_gen = True
    elif role == "main" and not has_runtime_weight:
        needs_gen = True

    if needs_gen:
        script = ROOT_DIR.parent / "dataset" / "generate.py"
        cmd = [sys.executable, str(script), "--output-dir", str(dataset_dir), "--role", role]
        if has_custom_h:
            cmd.extend(["--h", str(spec.h), "--w", str(spec.w)])
        subprocess.run(cmd, check=True)

def run_benchmark(args: argparse.Namespace) -> dict:
    spec = build_benchmark_spec(h=args.h, w=args.w, c_in=args.cin, c_out=args.cout, k=args.k, pad=args.pad)
    dataset_dir = Path(args.dataset_dir)

    # 1. 确保数据存在 (1025 规模逻辑在 generate.py 中由 get_runtime_spec 锁定)
    _generate_if_needed(dataset_dir, spec, args.role, args.h is not None)

    # 2. 探测并运行硬件测试
    backends = build_backends(args.backend)
    backend_results = {}

    print(f"\n=== Benchmarking Backends for {args.role.upper()} NODE ===")
    test_layout = build_dataset_layout(dataset_dir, prefix="test_")

    for b in backends:
        print(f" -> Probing {b.name}...")
        # 调用后端运行 test 规模的 benchmark
        result = b.run(spec, test_layout, time_budget_seconds=spec.zero_score_seconds)
        res = result.to_dict() if hasattr(result, 'to_dict') else result
        backend_results[b.name] = res

        if res.get("available"):
            best = res.get("best_trial") or res.get("best_result") or {}
            gflops = best.get("effective_gflops", 0)
            print(f"    Available. Performance: {gflops:.2f} GFLOPS")
        else:
            notes = res.get("notes", ["No details"])
            print(f"    Not available: {notes[0] if notes else 'No details'}")

    # 3. 生成排名
    ranking = sorted(
        [k for k, v in backend_results.items() if v.get("available")],
        key=lambda x: (backend_results[x].get("best_trial") or backend_results[x].get("best_result") or {}).get("effective_gflops", 0),
        reverse=True
    )

    return {
        "schema_version": 2,
        "method": "Conv2D",
        "generated_at_unix": time.time(),
        "role": args.role,
        "best_backend": ranking[0] if ranking else None,
        "ranking": ranking,
        "backend_results": backend_results
    }

def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark(args)

    # 写入并打印结果
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nBenchmark Report Summary:")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
