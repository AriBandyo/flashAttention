#include "flashforge.cuh"
#include "common.cuh"
#include <cfloat>

namespace flashforge {

__global__ void qk_kernel(const float* q, const float* k, float* s,
                          int B, int H, int NQ, int NK, int D, bool causal) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * H * NQ * NK;
    if (idx >= total) return;
    int ki = idx % NK; long long t = idx / NK;
    int qi = t % NQ; t /= NQ;
    int h = t % H; int b = t / H;
    if (causal && !causal_keep(qi, ki, NQ, NK)) {
        s[idx] = -CUDART_INF_F;
        return;
    }
    const float* qrow = q + (((long long)b * H + h) * NQ + qi) * D;
    const float* krow = k + (((long long)b * H + h) * NK + ki) * D;
    float acc = 0.0f;
    for (int d = 0; d < D; ++d) acc = fmaf(qrow[d], krow[d], acc);
    s[idx] = acc * rsqrtf((float)D);
}

__global__ void softmax_rows_kernel(float* s, int rows, int NK) {
    int row = blockIdx.x;
    if (row >= rows) return;
    extern __shared__ float sm[];
    float local_max = -CUDART_INF_F;
    for (int j = threadIdx.x; j < NK; j += blockDim.x) local_max = fmaxf(local_max, s[(long long)row * NK + j]);
    sm[threadIdx.x] = local_max; __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) sm[threadIdx.x] = fmaxf(sm[threadIdx.x], sm[threadIdx.x + stride]);
        __syncthreads();
    }
    float m = sm[0];
    if (isinf(m) && m < 0) {
        for (int j = threadIdx.x; j < NK; j += blockDim.x) s[(long long)row * NK + j] = 0.0f;
        return;
    }
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < NK; j += blockDim.x) {
        float x = s[(long long)row * NK + j];
        float e = isinf(x) && x < 0 ? 0.0f : expf(x - m);
        s[(long long)row * NK + j] = e;
        local_sum += e;
    }
    sm[threadIdx.x] = local_sum; __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) sm[threadIdx.x] += sm[threadIdx.x + stride];
        __syncthreads();
    }
    float inv = sm[0] > 0.0f ? 1.0f / sm[0] : 0.0f;
    for (int j = threadIdx.x; j < NK; j += blockDim.x) s[(long long)row * NK + j] *= inv;
}

__global__ void pv_kernel(const float* p, const float* v, float* o,
                          int B, int H, int NQ, int NK, int D) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)B * H * NQ * D;
    if (idx >= total) return;
    int d = idx % D; long long t = idx / D;
    int qi = t % NQ; t /= NQ;
    int h = t % H; int b = t / H;
    const float* prow = p + (((long long)b * H + h) * NQ + qi) * NK;
    const float* vbase = v + ((long long)b * H + h) * NK * D;
    float acc = 0.0f;
    for (int ki = 0; ki < NK; ++ki) acc = fmaf(prow[ki], vbase[(long long)ki * D + d], acc);
    o[idx] = acc;
}

void launch_v0_naive(const float* q, const float* k, const float* v, float* out,
                     float* scores, int B, int H, int NQ, int NK, int D,
                     bool causal, cudaStream_t stream) {
    long long score_elems = (long long)B * H * NQ * NK;
    qk_kernel<<<(score_elems + 255) / 256, 256, 0, stream>>>(q, k, scores, B, H, NQ, NK, D, causal);
    int rows = B * H * NQ;
    softmax_rows_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(scores, rows, NK);
    long long out_elems = (long long)B * H * NQ * D;
    pv_kernel<<<(out_elems + 255) / 256, 256, 0, stream>>>(scores, v, out, B, H, NQ, NK, D);
}

} // namespace flashforge
