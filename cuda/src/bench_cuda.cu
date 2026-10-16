#include "flashforge.cuh"
#include "common.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

using namespace flashforge;

struct L2Flusher {
    void* buf = nullptr; size_t bytes = 0;
    L2Flusher() {
        int l2 = 0; CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
        bytes = std::max<size_t>((size_t)l2 * 2, 1);
        CUDA_CHECK(cudaMalloc(&buf, bytes));
        std::printf("# L2 %.1f MB; flush %.1f MB\n", l2 / 1048576.0, bytes / 1048576.0);
    }
    ~L2Flusher(){ if (buf) cudaFree(buf); }
    void flush(cudaStream_t s=0){ CUDA_CHECK(cudaMemsetAsync(buf, 0, bytes, s)); }
};

void warmup(const std::function<void()>& fn, float target_ms=1000.0f) {
    cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    float ms=0; int n=0; CUDA_CHECK(cudaEventRecord(a));
    do { fn(); ++n; if (n%20==0) { CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b)); CUDA_CHECK(cudaEventElapsedTime(&ms,a,b)); } } while(ms<target_ms);
    CUDA_CHECK(cudaDeviceSynchronize()); cudaEventDestroy(a); cudaEventDestroy(b);
}

float median_ms(const std::function<void()>& fn, L2Flusher& l2, int iters) {
    warmup(fn); std::vector<float> ts(iters); cudaEvent_t a,b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    for(int i=0;i<iters;++i){ l2.flush(); CUDA_CHECK(cudaEventRecord(a)); fn(); CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b)); CUDA_CHECK(cudaEventElapsedTime(&ts[i],a,b)); }
    std::sort(ts.begin(),ts.end()); cudaEventDestroy(a); cudaEventDestroy(b); return ts[iters/2];
}

long long active_pairs(int nq,int nk,bool causal){ if(!causal) return (long long)nq*nk; long long x=0; int off=nk-nq; for(int q=0;q<nq;++q) x += std::max(0,std::min(nk,q+off+1)); return x; }

int main(int argc, char** argv){
    std::string version="v2"; int B=1,H=1,NQ=1024,NK=1024,D=64,iters=100,splits=8; bool causal=false;
    for(int i=1;i<argc;++i){
        auto need=[&](const char* n){ if(i+1>=argc){std::fprintf(stderr,"missing %s\n",n);std::exit(2);} return argv[++i];};
        if(!std::strcmp(argv[i],"--version")) version=need("version"); else if(!std::strcmp(argv[i],"--b")) B=std::atoi(need("b"));
        else if(!std::strcmp(argv[i],"--h")) H=std::atoi(need("h")); else if(!std::strcmp(argv[i],"--nq")) NQ=std::atoi(need("nq"));
        else if(!std::strcmp(argv[i],"--nk")) NK=std::atoi(need("nk")); else if(!std::strcmp(argv[i],"--d")) D=std::atoi(need("d"));
        else if(!std::strcmp(argv[i],"--iters")) iters=std::atoi(need("iters")); else if(!std::strcmp(argv[i],"--splits")) splits=std::atoi(need("splits"));
        else if(!std::strcmp(argv[i],"--causal")) causal=true;
    }
    if(version=="decode") NQ=1;
    size_t qn=(size_t)B*H*NQ*D, kvn=(size_t)B*H*NK*D, on=qn;
    std::vector<float> hq(qn),hk(kvn),hv(kvn); for(size_t i=0;i<qn;++i) hq[i]=std::sin((double)i*0.013); for(size_t i=0;i<kvn;++i){hk[i]=std::cos((double)i*0.017);hv[i]=std::sin((double)i*0.019);}
    float *q,*k,*v,*o,*s=nullptr,*pm=nullptr,*pl=nullptr,*po=nullptr;
    CUDA_CHECK(cudaMalloc(&q,qn*4)); CUDA_CHECK(cudaMalloc(&k,kvn*4)); CUDA_CHECK(cudaMalloc(&v,kvn*4)); CUDA_CHECK(cudaMalloc(&o,on*4));
    CUDA_CHECK(cudaMemcpy(q,hq.data(),qn*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(k,hk.data(),kvn*4,cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(v,hv.data(),kvn*4,cudaMemcpyHostToDevice));
    if(version=="v0") CUDA_CHECK(cudaMalloc(&s,(size_t)B*H*NQ*NK*4));
    if(version=="decode"){ CUDA_CHECK(cudaMalloc(&pm,(size_t)B*H*splits*4)); CUDA_CHECK(cudaMalloc(&pl,(size_t)B*H*splits*4)); CUDA_CHECK(cudaMalloc(&po,(size_t)B*H*splits*D*4)); }
    std::function<void()> fn;
    if(version=="v0") fn=[&]{launch_v0_naive(q,k,v,o,s,B,H,NQ,NK,D,causal);};
    else if(version=="v1") fn=[&]{launch_v1_online(q,k,v,o,B,H,NQ,NK,D,causal);};
    else if(version=="v2") fn=[&]{launch_v2_fa2_partition(q,k,v,o,B,H,NQ,NK,D,causal);};
    else if(version=="decode") fn=[&]{launch_decode_splitkv(q,k,v,o,pm,pl,po,B,H,NK,D,splits);};
    else { std::fprintf(stderr,"unknown version\n"); return 2; }
    fn(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    L2Flusher l2; float ms=median_ms(fn,l2,iters); CUDA_CHECK(cudaGetLastError());
    long long pairs=active_pairs(NQ,NK,causal); double flops=4.0*B*H*pairs*D; double tflops=flops/(ms/1000.0)/1e12;
    double ideal_bytes=(double)B*H*(NQ*D + 2LL*NK*D + NQ*D)*4.0; double ideal_gbs=ideal_bytes/(ms/1000.0)/1e9;
    std::printf("version,B,H,NQ,NK,D,causal,splits,ms,tflops,ideal_gbs\n");
    std::printf("%s,%d,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.3f\n",version.c_str(),B,H,NQ,NK,D,(int)causal,splits,ms,tflops,ideal_gbs);
    cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(o); if(s)cudaFree(s); if(pm)cudaFree(pm); if(pl)cudaFree(pl); if(po)cudaFree(po);
}
