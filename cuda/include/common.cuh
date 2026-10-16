#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <math_constants.h>

#define CUDA_CHECK(call) do { \
    cudaError_t e__ = (call); \
    if (e__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e__)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

namespace flashforge {

__device__ __forceinline__ bool causal_keep(int qi, int ki, int nq, int nk) {
    return ki <= qi + nk - nq;
}

__device__ __forceinline__ float warp_sum(float x) {
    for (int off = 16; off > 0; off >>= 1) x += __shfl_down_sync(0xffffffffu, x, off);
    return x;
}

__device__ __forceinline__ float warp_max(float x) {
    for (int off = 16; off > 0; off >>= 1) x = fmaxf(x, __shfl_down_sync(0xffffffffu, x, off));
    return x;
}

inline int next_pow2_threads(int d) {
    int t = 32;
    while (t < d) t <<= 1;
    return t > 1024 ? 1024 : t;
}

} // namespace flashforge
