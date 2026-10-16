#pragma once
#include <cuda_runtime.h>

namespace flashforge {

void launch_v0_naive(const float* q, const float* k, const float* v, float* out,
                     float* scores, int B, int H, int NQ, int NK, int D,
                     bool causal, cudaStream_t stream = 0);

void launch_v1_online(const float* q, const float* k, const float* v, float* out,
                      int B, int H, int NQ, int NK, int D,
                      bool causal, cudaStream_t stream = 0);

void launch_v2_fa2_partition(const float* q, const float* k, const float* v, float* out,
                             int B, int H, int NQ, int NK, int D,
                             bool causal, cudaStream_t stream = 0);

void launch_decode_splitkv(const float* q, const float* k, const float* v, float* out,
                           float* partial_m, float* partial_l, float* partial_o,
                           int B, int H, int NK, int D, int splits,
                           cudaStream_t stream = 0);

} // namespace flashforge
