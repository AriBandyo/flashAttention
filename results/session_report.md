# FlashForge validation report — 2026-10-07

## Executed in this session

- Python adversarial/reference suite: **11/11 passed**.
- Python syntax compilation: passed for package, tests, and benchmark scripts.
- Roofline sweep generated for the Coalesce RTX 4060 Laptop FP32 configuration: 14.6 TFLOP/s compute, 272 GB/s DRAM.
- Nominal A100 40 GB FP16 roofline sweep generated: 312 TFLOP/s tensor compute, 1555 GB/s HBM.

## GPU execution status

GPU CUDA kernels were **not executed in this environment**. The runtime has:

- no `nvcc`
- no `nvidia-smi`
- PyTorch `2.10.0+cpu`
- `torch.cuda.is_available() == False`

CMake was intentionally probed and fails at CUDA compiler discovery. Therefore this report contains no invented custom-kernel milliseconds, TFLOP/s, occupancy, register count, or Nsight counters.

## Selected analytical ceilings

These are roofline ceilings, not measured kernel performance.

| hardware / regime | shape | traffic model | arithmetic intensity | roofline ceiling | roofline time |
|---|---:|---|---:|---:|---:|
| RTX 4060 FP32 prefill | N=512, D=64 | ideal fused | 128 FLOP/B | 14.6 TFLOP/s | 0.00460 ms |
| RTX 4060 FP32 prefill | N=512, D=64 | naive materialized | 14.22 FLOP/B | 3.87 TFLOP/s | 0.01735 ms |
| RTX 4060 FP32 prefill | N=4096, D=64 | ideal fused | 1024 FLOP/B | 14.6 TFLOP/s | 0.29418 ms |
| RTX 4060 FP32 prefill | N=4096, D=64 | naive materialized | 15.75 FLOP/B | 4.29 TFLOP/s | 1.00232 ms |
| RTX 4060 FP32 decode | NKV=16384, D=128 | ideal fused | ~0.500 FLOP/B | 0.136 TFLOP/s | 0.06168 ms |
| A100 40GB FP16 prefill | N=512, D=128 | ideal fused | 256 FLOP/B | 312 TFLOP/s | 0.000430 ms |
| A100 40GB FP16 prefill | N=512, D=128 | naive materialized | 51.2 FLOP/B | 79.6 TFLOP/s | 0.001686 ms |
| A100 40GB FP16 prefill | N=4096, D=128 | ideal fused | 2048 FLOP/B | 312 TFLOP/s | 0.02753 ms |
| A100 40GB FP16 prefill | N=4096, D=128 | naive materialized | 62.06 FLOP/B | 96.5 TFLOP/s | 0.08901 ms |
| A100 40GB FP16 decode | NKV=16384, D=128 | ideal fused | ~1.000 FLOP/B | 1.555 TFLOP/s | 0.005395 ms |

The fused numbers use the algorithmic lower-bound traffic model (read Q/K/V once, write O once). Actual kernels will move more data and run below these ceilings. The naive model additionally accounts for score/probability materialization but is still an analytical model rather than a hardware counter.
