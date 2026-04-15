"""Output slice aggregator — merges partial Cout slices into final output."""

from __future__ import annotations

import array
from pathlib import Path


def merge_output_slices(
    slices: list[tuple[int, int, bytes]],
    total_cout: int,
    out_h: int,
    out_w: int,
) -> bytes:
    """Merge output slices from multiple workers into a single output buffer.

    Args:
        slices: List of (start_oc, end_oc, raw_float32_bytes)
                Each slice has shape [out_h * out_w, slice_cout] in row-major float32.
        total_cout: Total output channels.
        out_h, out_w: Output spatial dimensions.

    Returns:
        Complete output as bytes with shape [out_h, out_w, total_cout] in float32.
    """
    spatial_size = out_h * out_w
    output = array.array("f", [0.0] * (spatial_size * total_cout))

    for start_oc, end_oc, data in slices:
        slice_cout = end_oc - start_oc
        slice_arr = array.array("f")
        slice_arr.frombytes(data)

        # Scatter: each pixel's slice_cout channels → correct position in output
        for pixel in range(spatial_size):
            src_base = pixel * slice_cout
            dst_base = pixel * total_cout + start_oc
            output[dst_base:dst_base + slice_cout] = slice_arr[src_base:src_base + slice_cout]

    return output.tobytes()


def save_output(data: bytes, path: Path) -> None:
    """Write raw output bytes to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
