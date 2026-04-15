"""Task dispatcher — orchestrates local and distributed computation."""

from __future__ import annotations

import json
import time
from pathlib import Path

from compute_node.executor import run_computation, _get_best_backend, _get_best_gflops
from compute_node.performance_metrics.workloads import get_runtime_spec


class TaskDispatcher:
    def __init__(self, registry, logger):
        self.registry = registry
        self.logger = logger
        self.runtime_spec = get_runtime_spec()
        self.root = Path(__file__).resolve().parent.parent

    def _get_best_backend_name(self) -> str:
        return _get_best_backend()

    def _get_best_gflops(self) -> float:
        return _get_best_gflops()

    def run_locally(self) -> dict:
        """Execute the full computation locally using the best backend."""
        spec = self.runtime_spec
        dataset_dir = self.root / "compute_node" / "dataset" / "generated"
        input_path = dataset_dir / "runtime_input.bin"
        weight_path = dataset_dir / "runtime_weight.bin"
        output_path = dataset_dir / "runtime_output.bin"

        backend_name = self._get_best_backend_name()
        self.logger.info(f"Local execution: backend={backend_name.upper()}, "
                         f"scale={spec.h}x{spec.w}, cin={spec.c_in}, cout={spec.c_out}")

        start = time.perf_counter()

        result = run_computation(
            backend_name=backend_name,
            h=spec.h, w=spec.w, c_in=spec.c_in, c_out=spec.c_out,
            k=spec.k, pad=spec.pad,
            input_path=input_path,
            weight_path=weight_path,
            output_path=output_path,
            start_oc=0, end_oc=spec.c_out,
            logger=self.logger,
        )

        total_elapsed = time.perf_counter() - start
        output_mb = output_path.stat().st_size / 1048576 if output_path.exists() else 0

        self.logger.info(
            f"Local execution complete: {total_elapsed:.2f}s total, "
            f"{result.effective_gflops:.1f} GFLOPS, "
            f"output={output_mb:.1f} MB, checksum={result.checksum}"
        )

        return {
            "mode": "local",
            "backend": result.backend,
            "elapsed_seconds": total_elapsed,
            "effective_gflops": result.effective_gflops,
            "checksum": result.checksum,
            "output_path": str(output_path),
            "output_size_mb": output_mb,
        }
