"""Helpers for turning benchmark `result.json` into a small runtime summary."""

from __future__ import annotations

import json
from pathlib import Path
from common.types import ComputeHardwarePerformance, ComputePerformanceSummary

DEFAULT_RESULT_PATH = Path(__file__).resolve().parent / "performance_metrics" / "result.json"

def load_performance_summary(result_path: Path | None = None) -> ComputePerformanceSummary:
    """Load the ranked backend summary for runtime registration."""
    resolved_path = DEFAULT_RESULT_PATH if result_path is None else Path(result_path)
    if not resolved_path.exists():
        return ComputePerformanceSummary(hardware_count=0, ranked_hardware=[])

    payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    backend_results = payload.get("backend_results", {})

    ranked_hardware: list[ComputeHardwarePerformance] = []
    for name, data in backend_results.items():
        if data.get("available") and data.get("best_result"):
            ranked_hardware.append(
                ComputeHardwarePerformance(
                    hardware_type=name,
                    effective_gflops=float(data["best_result"]["effective_gflops"]),
                    rank=int(data.get("rank") or 99),
                )
            )

    ranked_hardware.sort(key=lambda item: item.rank)
    return ComputePerformanceSummary(
        hardware_count=len(ranked_hardware),
        ranked_hardware=ranked_hardware,
    )
