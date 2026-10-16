"""Analytical FLOP / byte accounting for prefill and decode."""
from __future__ import annotations

from dataclasses import dataclass, asdict


def active_attention_pairs(nq: int, nk: int, causal: bool) -> int:
    if not causal:
        return nq * nk
    total = 0
    offset = nk - nq
    for qi in range(nq):
        total += max(0, min(nk, qi + offset + 1))
    return total


def attention_flops(b: int, h: int, nq: int, nk: int, d: int, causal: bool = False) -> int:
    # QK^T: 2*d FLOPs/pair; P@V: 2*d FLOPs/pair.
    return 4 * b * h * active_attention_pairs(nq, nk, causal) * d


def ideal_fused_bytes(b: int, h: int, nq: int, nk: int, d: int, dtype_bytes: int = 2) -> int:
    # Algorithmic lower bound: read Q/K/V once, write O once.
    elements = b * h * (nq * d + nk * d + nk * d + nq * d)
    return elements * dtype_bytes


def naive_materialized_bytes(b: int, h: int, nq: int, nk: int, d: int, dtype_bytes: int = 4) -> int:
    base = ideal_fused_bytes(b, h, nq, nk, d, dtype_bytes)
    scores = b * h * nq * nk
    # score write; score read by softmax; probability write; probability read by PV.
    return base + 4 * scores * dtype_bytes


@dataclass(frozen=True)
class RooflineMetrics:
    flops: int
    bytes: int
    arithmetic_intensity: float
    compute_ceiling_tflops: float
    bandwidth_ceiling_tflops: float
    roofline_tflops: float
    estimated_ms_at_roofline: float

    def as_dict(self):
        return asdict(self)


def roofline_metrics(*, flops: int, bytes_moved: int, compute_tflops: float, bandwidth_gbs: float) -> RooflineMetrics:
    ai = flops / bytes_moved
    bw_ceiling = bandwidth_gbs * ai / 1000.0
    roof = min(compute_tflops, bw_ceiling)
    ms = (flops / (roof * 1e12)) * 1e3 if roof > 0 else float("inf")
    return RooflineMetrics(
        flops=flops,
        bytes=bytes_moved,
        arithmetic_intensity=ai,
        compute_ceiling_tflops=compute_tflops,
        bandwidth_ceiling_tflops=bw_ceiling,
        roofline_tflops=roof,
        estimated_ms_at_roofline=ms,
    )
