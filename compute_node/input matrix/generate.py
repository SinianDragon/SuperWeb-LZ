"""Generate deterministic datasets for Test and Runtime."""

import argparse
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR.parent / "performance_metrics"))

from fmvm_dataset import build_dataset_layout, generate_dataset
from workloads import get_test_spec, get_runtime_spec, build_benchmark_spec

def main() -> int:
    parser = argparse.ArgumentParser(description="Generate datasets.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--role", choices=["compute", "main"], default="compute")
    parser.add_argument("--h", type=int)
    parser.add_argument("--w", type=int)
    args = parser.parse_args()

    def _progress(label: str, w_bytes: int, t_bytes: int) -> None:
        print(f"\r  {label}: {w_bytes / 1048576:.1f} / {t_bytes / 1048576:.1f} MB", end="", flush=True)
        if w_bytes == t_bytes: print()

    test_spec = build_benchmark_spec(h=args.h, w=args.w) if args.h else get_test_spec()
    runtime_spec = get_runtime_spec()

    # 1. TEST dataset
    test_layout = build_dataset_layout(args.output_dir, prefix="test_")
    if not (test_layout.input_path.exists() and test_layout.weight_path.exists()):
        print(f"\n[Data Gen] Creating TEST dataset ({test_spec.h}x{test_spec.w})...")
        generate_dataset(test_layout, test_spec, skip_weight=False, progress=_progress)

    # 2. RUNTIME dataset
    runtime_layout = build_dataset_layout(args.output_dir, prefix="runtime_")
    skip_rt_weight = (args.role == "compute")

    needs_input = not runtime_layout.input_path.exists()
    needs_weight = not skip_rt_weight and not runtime_layout.weight_path.exists()

    if needs_input or needs_weight:
        label = "RUNTIME INPUT" if needs_input and skip_rt_weight else "FULL RUNTIME"
        print(f"\n[Data Gen] Creating {label} ({runtime_spec.h}x{runtime_spec.w})...")
        generate_dataset(runtime_layout, runtime_spec, skip_weight=skip_rt_weight, progress=_progress)

    return 0

if __name__ == "__main__":
    sys.exit(main())
