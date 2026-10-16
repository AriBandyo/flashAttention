#!/usr/bin/env python3
"""Benchmark PyTorch SDPA or the flash-attn package on the same shape sweep.

PyTorch tensors are [B,H,N,D]. flash-attn tensors are [B,N,H,D].
For unequal causal lengths this uses lower-right causal semantics, matching
FlashAttention >= 2.1.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from a fresh checkout without pip install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
import statistics

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.bias import causal_lower_right

from flashforge.roofline import attention_flops, ideal_fused_bytes


def _backend(name: str):
    table = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "math": SDPBackend.MATH,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    return table[name]


def timed_warmup(fn, target_ms=1000.0):
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    elapsed = 0.0
    start.record()
    while elapsed < target_ms:
        for _ in range(20):
            fn()
        stop.record(); stop.synchronize()
        elapsed = start.elapsed_time(stop)


def make_runner(fn, use_graph: bool):
    if not use_graph:
        return fn
    # Static-shape decode microbenchmarks benefit from removing Python launch overhead.
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return g.replay


def median_ms(fn, flush_tensor, iters=100, use_graph=False):
    timed_warmup(fn)
    run = make_runner(fn, use_graph)
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(iters):
        flush_tensor.zero_()
        start.record(); run(); stop.record(); stop.synchronize()
        times.append(start.elapsed_time(stop))
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["torch", "flash-attn"], default="torch")
    ap.add_argument("--backend", choices=["flash", "math", "efficient", "cudnn"], default="flash")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--nq", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--nk", type=int, default=None, help="override kv length; default nk=nq")
    ap.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--l2-flush-mb", type=int, default=96)
    ap.add_argument("--cuda-graph", action="store_true")
    ap.add_argument("--output", default="results/pytorch.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA PyTorch build + NVIDIA GPU required")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    flush = torch.empty(args.l2_flush_mb * 1024 * 1024, device="cuda", dtype=torch.uint8)

    if args.provider == "flash-attn":
        from flash_attn import flash_attn_func

    rows = []
    for d in args.head_dims:
        for nq in args.nq:
            nk = args.nk or nq
            q = torch.randn(args.batch, args.heads, nq, d, device="cuda", dtype=dtype)
            k = torch.randn(args.batch, args.heads, nk, d, device="cuda", dtype=dtype)
            v = torch.randn(args.batch, args.heads, nk, d, device="cuda", dtype=dtype)

            if args.provider == "torch":
                if args.causal and nq != nk:
                    bias = causal_lower_right(nq, nk)
                    def op():
                        with sdpa_kernel(_backend(args.backend)):
                            return F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
                else:
                    def op():
                        with sdpa_kernel(_backend(args.backend)):
                            return F.scaled_dot_product_attention(q, k, v, is_causal=args.causal)
            else:
                qf, kf, vf = (x.transpose(1, 2).contiguous() for x in (q, k, v))
                def op():
                    return flash_attn_func(qf, kf, vf, dropout_p=0.0, causal=args.causal)

            ms = median_ms(op, flush, iters=args.iters, use_graph=args.cuda_graph)
            flops = attention_flops(args.batch, args.heads, nq, nk, d, args.causal)
            tflops = flops / (ms / 1e3) / 1e12
            ideal_bytes = ideal_fused_bytes(args.batch, args.heads, nq, nk, d, dtype.itemsize)
            ideal_gbs = ideal_bytes / (ms / 1e3) / 1e9
            row = dict(provider=args.provider, backend=args.backend, dtype=args.dtype,
                       B=args.batch, H=args.heads, NQ=nq, NK=nk, D=d, causal=int(args.causal),
                       cuda_graph=int(args.cuda_graph), ms=ms, tflops=tflops, ideal_gbs=ideal_gbs)
            rows.append(row)
            print(row)

    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
