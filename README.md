# FlashForge — FlashAttention from First Principles

A from-scratch attention project built around two regimes that behave very differently:

- **Prefill:** increase arithmetic intensity by never materializing the N×N attention matrix and then move toward tensor-core FlashAttention-2 style kernels.
- **Decode:** q_len=1 is primarily a KV-cache bandwidth and grid-parallelism problem, so split KV across CTAs and stably reduce partial `(m, l, O)` states.

The project is deliberately **correctness-first and measurement-first**. Every CUDA version is meant to pass the same adversarial suite before its performance is reported.

## Current status

| version | idea | status |
|---|---|---|
| v0 | three-stage materialized QKᵀ → softmax → PV | implemented |
| v1 | fused FP32 online softmax, shared K/V tiles | implemented |
| v2 | FA2-style Q-row/warp partition, no cross-warp reduction | implemented |
| decode | split-KV partial states + stable reduction | implemented |
| v3 | FP16/BF16 tensor cores (`mma.sync`, `ldmatrix`) | staged |
| v4 | `cp.async` double buffering + shared-memory swizzle | staged |
| v5 | tensor-core causal tile specialization | staged |

The CUDA sources in this first cut are written but **not claimed as GPU-validated in this repository snapshot** unless `flashforge_test` has been run on a CUDA machine. The Python algorithmic reference and adversarial tests are runnable on CPU.

## What carries over from Coalesce

The benchmark discipline intentionally follows the Coalesce harness: sustained timed warm-up, a buffer at least 2× L2 for eviction, one CUDA-event measurement per iteration, and median reporting. The code is copied/reworked into this repo rather than imported across projects.

## Correctness traps covered

- large-magnitude logits (online-softmax overflow resistance)
- sequence lengths not divisible by a tile size
- fully masked rows (must become zero, not NaN)
- lower-right causal attention with `NQ != NK`
- split-KV decode with uneven split boundaries

FlashAttention 2.1+ uses lower-right alignment for unequal causal sequence lengths; the CPU reference and CUDA kernels use that convention.

## Python tests

```bash
python -m unittest discover -s tests -v
```

## Build CUDA kernels

RTX 4060 Laptop (Ada):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build -j
./build/flashforge_test
```

A100:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j
./build/flashforge_test
```

## Custom CUDA benchmark

```bash
./build/flashforge_bench --version v0 --nq 1024 --nk 1024 --d 64
./build/flashforge_bench --version v1 --nq 4096 --nk 4096 --d 128
./build/flashforge_bench --version v2 --nq 4096 --nk 4096 --d 128 --causal
./build/flashforge_bench --version decode --nk 16384 --d 128 --splits 16
```

Output fields include median milliseconds, attention TFLOP/s, and **ideal** input/output GB/s. `ideal_gbs` is explicitly not an Nsight HBM counter.

## PyTorch SDPA and flash-attn baselines

The PyTorch path pins the backend with `torch.nn.attention.sdpa_kernel`. For unequal causal lengths it uses `causal_lower_right`, so the semantics agree with FlashAttention rather than PyTorch's default upper-left `is_causal` convention.

```bash
python bench/bench_pytorch.py --provider torch --backend flash --dtype fp16 --heads 8 --nq 512 1024 2048 4096 --head-dims 64 128
python bench/bench_pytorch.py --provider torch --backend flash --dtype fp16 --heads 8 --nq 1 --nk 16384 --head-dims 128 --cuda-graph
python bench/bench_pytorch.py --provider flash-attn --dtype fp16 --heads 8 --nq 512 1024 2048 4096 --head-dims 64 128
```

For decode, use CUDA graphs when you want to remove Python launch overhead from microsecond-scale kernels.

## Roofline sweep

The FLOP count is `4 * B * H * active_pairs * D`, where `active_pairs=NQ*NK` for non-causal attention and the exact lower-right triangular pair count for causal attention.

Example using the FP32 numbers recorded in Coalesce for the RTX 4060 Laptop:

```bash
python bench/roofline_sweep.py \
  --compute-tflops 14.6 \
  --bandwidth-gbs 272 \
  --dtype-bytes 4 \
  --output results/rtx4060_fp32_roofline.csv
```

For a final A100 FP16 study, replace the nominal numbers with the exact rented SKU and clock state you actually benchmark.

## Repository layout

```text
flashforge/                  CPU reference + analytical roofline model
tests/                       adversarial CPU tests
cuda/include/                public launch API + CUDA helpers
cuda/src/v0_naive.cu         materialized attention baseline
cuda/src/v1_online.cu        fused online softmax
cuda/src/v2_fa2_partition.cu FA2-style warp/Q-row partition
cuda/src/decode_splitkv.cu   FlashDecoding-style split-KV
cuda/src/test_cuda.cu        GPU correctness suite vs CPU reference
cuda/src/bench_cuda.cu       Coalesce-style median/L2-flush benchmark
bench/bench_pytorch.py       PyTorch SDPA + flash-attn baselines
bench/roofline_sweep.py      ceiling-first analytical sweep
configs/                     machine notes
```

## Measurement rules

1. Correctness before timing.
2. Timed warm-up until clocks are under sustained load.
3. Evict L2 between independent timed iterations.
4. Time each iteration separately and report the median.
5. Keep theoretical/ideal byte rates labeled as such; use Nsight Compute on native Linux/A100 for actual DRAM counters.
6. Never report v3/v4/v5 speedups until those kernels compile, pass adversarial tests, and are profiled.
