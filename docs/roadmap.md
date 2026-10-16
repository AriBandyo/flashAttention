# Optimization ladder

## Implemented in the correctness-first cut

- **v0 naive:** QK^T -> materialized score/probability matrix -> PV.
- **v1 fused online softmax:** one query row per block, K/V streamed through shared memory, running `(m, l, O)` state, no NQxNK materialization.
- **v2 FA2-style partition:** four warps own four Q rows, no cross-warp reduction, K/V tile shared across those rows, normalization deferred to the end.
- **decode split-KV:** q_len=1, KV sequence split across CTAs, stable `(m,l,O)` partial reduction.
- Lower-right causal semantics and fully-masked-row zeroing are implemented throughout the references and prefill CUDA kernels.

## Deliberately not marked complete yet

- **v3 tensor cores:** FP16/BF16, `mma.sync`, `ldmatrix`, FP32 accumulation, register-layout reuse between QK and PV.
- **v4 pipelining:** `cp.async` double buffering + swizzled shared memory.
- **v5 optimized causal:** tile scheduler that removes fully masked work at CTA granularity and specializes diagonal tiles. v2 already skips fully masked K tiles per Q-row warp, but this is not yet the final tensor-core causal kernel.

These phases must be compiled, correctness-tested, and profiled before the repository claims them.
