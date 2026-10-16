#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running directly from a fresh checkout without pip install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import csv
from flashforge.roofline import attention_flops, ideal_fused_bytes, naive_materialized_bytes, roofline_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compute-tflops", type=float, required=True)
    ap.add_argument("--bandwidth-gbs", type=float, required=True)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--sizes", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    ap.add_argument("--head-dims", type=int, nargs="+", default=[64, 128])
    ap.add_argument("--output", default="results/roofline.csv")
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for causal in (False, True):
        for d in args.head_dims:
            for n in args.sizes:
                flops = attention_flops(args.batch, args.heads, n, n, d, causal)
                for model in ("ideal_fused", "naive_materialized"):
                    if model == "ideal_fused":
                        moved = ideal_fused_bytes(args.batch, args.heads, n, n, d, args.dtype_bytes)
                    else:
                        moved = naive_materialized_bytes(args.batch, args.heads, n, n, d, args.dtype_bytes)
                    m = roofline_metrics(flops=flops, bytes_moved=moved, compute_tflops=args.compute_tflops, bandwidth_gbs=args.bandwidth_gbs)
                    rows.append({
                        "model": model,
                        "causal": int(causal),
                        "N": n,
                        "D": d,
                        **m.as_dict(),
                    })
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
