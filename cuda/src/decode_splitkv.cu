#include "flashforge.cuh"
#include "common.cuh"

namespace flashforge {

__global__ void splitkv_partial_kernel(const float* q, const float* k, const float* v,
                                       float* pm, float* pl, float* po,
                                       int B, int H, int NK, int D, int splits) {
    int split = blockIdx.x;
    int bh = blockIdx.y;
    int b = bh / H, h = bh % H;
    int tid = threadIdx.x;
    int ks = (split * NK) / splits;
    int ke = ((split + 1) * NK) / splits;

    extern __shared__ float red[];
    const float* qrow = q + ((long long)b * H + h) * D; // q_len = 1
    const float* kbase = k + ((long long)b * H + h) * NK * D;
    const float* vbase = v + ((long long)b * H + h) * NK * D;
    float acc = 0.0f, m = -CUDART_INF_F, l = 0.0f, scale = rsqrtf((float)D);

    for (int ki = ks; ki < ke; ++ki) {
        float part = 0.0f;
        for (int d = tid; d < D; d += blockDim.x) part = fmaf(qrow[d], kbase[(long long)ki * D + d], part);
        red[tid] = part; __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (tid < stride) red[tid] += red[tid + stride];
            __syncthreads();
        }
        float score = red[0] * scale;
        float m_new = fmaxf(m, score);
        float alpha = isinf(m) && m < 0 ? 0.0f : expf(m - m_new);
        float p = expf(score - m_new);
        if (tid < D) acc = acc * alpha + p * vbase[(long long)ki * D + tid];
        l = l * alpha + p; m = m_new;
        __syncthreads();
    }
    long long state = (long long)bh * splits + split;
    if (tid == 0) { pm[state] = m; pl[state] = l; }
    if (tid < D) po[state * D + tid] = acc;
}

__global__ void splitkv_reduce_kernel(const float* pm, const float* pl, const float* po,
                                      float* o, int BH, int D, int splits) {
    int bh = blockIdx.x, tid = threadIdx.x;
    extern __shared__ float sh[];
    float* sms = sh;
    if (tid < splits) sms[tid] = pm[(long long)bh * splits + tid];
    __syncthreads();
    float gm = -CUDART_INF_F;
    for (int s = 0; s < splits; ++s) gm = fmaxf(gm, sms[s]);
    float denom = 0.0f, numer = 0.0f;
    for (int s = 0; s < splits; ++s) {
        float m = pm[(long long)bh * splits + s];
        float l = pl[(long long)bh * splits + s];
        float w = (l > 0.0f) ? expf(m - gm) : 0.0f;
        denom += l * w;
        if (tid < D) numer += po[((long long)bh * splits + s) * D + tid] * w;
    }
    if (tid < D) o[(long long)bh * D + tid] = denom > 0.0f ? numer / denom : 0.0f;
}

void launch_decode_splitkv(const float* q, const float* k, const float* v, float* out,
                           float* partial_m, float* partial_l, float* partial_o,
                           int B, int H, int NK, int D, int splits,
                           cudaStream_t stream) {
    if (D > 128) { std::fprintf(stderr, "decode supports D <= 128\n"); std::exit(EXIT_FAILURE); }
    if (splits <= 0 || splits > 128) { std::fprintf(stderr, "decode supports 1 <= splits <= 128\n"); std::exit(EXIT_FAILURE); }
    int threads = 128;
    dim3 grid(splits, B * H);
    splitkv_partial_kernel<<<grid, threads, threads * sizeof(float), stream>>>(q, k, v, partial_m, partial_l, partial_o, B, H, NK, D, splits);
    int reduce_threads = 128;
    splitkv_reduce_kernel<<<B * H, reduce_threads, splits * sizeof(float), stream>>>(partial_m, partial_l, partial_o, out, B * H, D, splits);
}

} // namespace flashforge
