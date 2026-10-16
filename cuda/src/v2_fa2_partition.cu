#include "flashforge.cuh"
#include "common.cuh"

namespace flashforge {

// FA2-style work partitioning: four warps own four independent Q rows.
// No reduction crosses warp boundaries. K/V tiles are shared by all four rows.
__global__ void v2_fa2_kernel(const float* q, const float* k, const float* v, float* o,
                              int B, int H, int NQ, int NK, int D, bool causal) {
    constexpr int WARPS = 4;
    constexpr int TK = 32;
    int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    int bh = blockIdx.y;
    int b = bh / H, h = bh % H;
    int qi = blockIdx.x * WARPS + warp;

    extern __shared__ float sh[];
    float* sk = sh;
    float* sv = sk + TK * D;

    float qreg[4] = {0,0,0,0};
    float acc[4] = {0,0,0,0};
    if (qi < NQ) {
        const float* qrow = q + (((long long)b * H + h) * NQ + qi) * D;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int d = lane + 32 * r;
            if (d < D) qreg[r] = qrow[d];
        }
    }
    float m = -CUDART_INF_F, l = 0.0f, scale = rsqrtf((float)D);
    const float* kbase = k + ((long long)b * H + h) * NK * D;
    const float* vbase = v + ((long long)b * H + h) * NK * D;

    for (int ks = 0; ks < NK; ks += TK) {
        int tile = min(TK, NK - ks);
        for (int idx = tid; idx < tile * D; idx += blockDim.x) {
            int r = idx / D, d = idx % D;
            sk[idx] = kbase[(long long)(ks + r) * D + d];
            sv[idx] = vbase[(long long)(ks + r) * D + d];
        }
        __syncthreads();
        if (qi < NQ) {
            // A full masked tile can be skipped outright for causal attention.
            int limit = qi + NK - NQ;
            bool tile_fully_masked = causal && ks > limit;
            if (!tile_fully_masked) {
                for (int j = 0; j < tile; ++j) {
                    int ki = ks + j;
                    bool keep = !causal || ki <= limit;
                    float dot = 0.0f;
                    #pragma unroll
                    for (int r = 0; r < 4; ++r) {
                        int d = lane + 32 * r;
                        if (d < D) dot = fmaf(qreg[r], sk[j * D + d], dot);
                    }
                    dot = warp_sum(dot);
                    float score = __shfl_sync(0xffffffffu, dot, 0) * scale;
                    if (keep) {
                        float m_new = fmaxf(m, score);
                        float alpha = isinf(m) && m < 0 ? 0.0f : expf(m - m_new);
                        float p = expf(score - m_new);
                        #pragma unroll
                        for (int r = 0; r < 4; ++r) {
                            int d = lane + 32 * r;
                            if (d < D) acc[r] = acc[r] * alpha + p * sv[j * D + d];
                        }
                        l = l * alpha + p;
                        m = m_new;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (qi < NQ) {
        float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float* orow = o + (((long long)b * H + h) * NQ + qi) * D;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int d = lane + 32 * r;
            if (d < D) orow[d] = acc[r] * inv;
        }
    }
}

void launch_v2_fa2_partition(const float* q, const float* k, const float* v, float* out,
                             int B, int H, int NQ, int NK, int D,
                             bool causal, cudaStream_t stream) {
    if (D > 128) { std::fprintf(stderr, "v2 supports D <= 128\n"); std::exit(EXIT_FAILURE); }
    dim3 grid((NQ + 3) / 4, B * H);
    size_t shmem = (size_t)(2 * 32 * D) * sizeof(float);
    v2_fa2_kernel<<<grid, 128, shmem, stream>>>(q, k, v, out, B, H, NQ, NK, D, causal);
}

} // namespace flashforge
