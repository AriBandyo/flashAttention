#include "flashforge.cuh"
#include "common.cuh"
#include <cfloat>

namespace flashforge {

// Correctness-first fused online softmax. One block owns one query row.
// K/V are streamed in 32-row tiles through shared memory. This removes the NQxNK
// materialization but still pays a block-wide reduction for every QK dot product.
__global__ void v1_online_kernel(const float* q, const float* k, const float* v, float* o,
                                 int B, int H, int NQ, int NK, int D, bool causal) {
    constexpr int TK = 32;
    int row = blockIdx.x;
    int qi = row % NQ; int t = row / NQ;
    int h = t % H; int b = t / H;
    int tid = threadIdx.x;

    extern __shared__ float sh[];
    float* sk = sh;
    float* sv = sk + TK * D;
    float* red = sv + TK * D;
    float* scalars = red + blockDim.x;

    const float* qrow = q + (((long long)b * H + h) * NQ + qi) * D;
    const float* kbase = k + ((long long)b * H + h) * NK * D;
    const float* vbase = v + ((long long)b * H + h) * NK * D;
    float acc = (tid < D) ? 0.0f : 0.0f;
    float m = -CUDART_INF_F, l = 0.0f;
    float scale = rsqrtf((float)D);

    for (int ks = 0; ks < NK; ks += TK) {
        int tile = min(TK, NK - ks);
        for (int idx = tid; idx < tile * D; idx += blockDim.x) {
            int r = idx / D, d = idx % D;
            sk[idx] = kbase[(long long)(ks + r) * D + d];
            sv[idx] = vbase[(long long)(ks + r) * D + d];
        }
        __syncthreads();
        for (int j = 0; j < tile; ++j) {
            int ki = ks + j;
            bool keep = !causal || causal_keep(qi, ki, NQ, NK);
            float part = 0.0f;
            for (int d = tid; d < D; d += blockDim.x) part = fmaf(qrow[d], sk[j * D + d], part);
            red[tid] = part; __syncthreads();
            for (int stride = blockDim.x / 2; stride; stride >>= 1) {
                if (tid < stride) red[tid] += red[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                float score = keep ? red[0] * scale : -CUDART_INF_F;
                if (keep) {
                    float m_new = fmaxf(m, score);
                    float alpha = isinf(m) && m < 0 ? 0.0f : expf(m - m_new);
                    float p = expf(score - m_new);
                    scalars[0] = alpha; scalars[1] = p;
                    m = m_new; l = l * alpha + p;
                } else { scalars[0] = 1.0f; scalars[1] = 0.0f; }
                scalars[2] = m; scalars[3] = l;
            }
            __syncthreads();
            if (tid < D) acc = acc * scalars[0] + scalars[1] * sv[j * D + tid];
            m = scalars[2]; l = scalars[3];
            __syncthreads();
        }
        __syncthreads();
    }
    if (tid < D) o[(((long long)b * H + h) * NQ + qi) * D + tid] = l > 0.0f ? acc / l : 0.0f;
}

void launch_v1_online(const float* q, const float* k, const float* v, float* out,
                      int B, int H, int NQ, int NK, int D,
                      bool causal, cudaStream_t stream) {
    if (D > 128) { std::fprintf(stderr, "v1 supports D <= 128\n"); std::exit(EXIT_FAILURE); }
    int threads = 128;
    size_t shmem = (size_t)(2 * 32 * D + threads + 4) * sizeof(float);
    v1_online_kernel<<<B * H * NQ, threads, shmem, stream>>>(q, k, v, out, B, H, NQ, NK, D, causal);
}

} // namespace flashforge
