#include "flashforge.cuh"
#include "common.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
using namespace flashforge;

static bool keep(int qi,int ki,int nq,int nk,bool causal){return !causal || ki <= qi+nk-nq;}

static void cpu_ref(const std::vector<float>&q,const std::vector<float>&k,const std::vector<float>&v,std::vector<float>&o,int B,int H,int NQ,int NK,int D,bool causal){
  float sc=1/std::sqrt((float)D); std::fill(o.begin(),o.end(),0);
  for(int b=0;b<B;++b)for(int h=0;h<H;++h)for(int qi=0;qi<NQ;++qi){
    std::vector<double>s(NK,-INFINITY); double m=-INFINITY;
    for(int ki=0;ki<NK;++ki)if(keep(qi,ki,NQ,NK,causal)){double x=0;for(int d=0;d<D;++d)x+=(double)q[(((b*H+h)*NQ+qi)*D+d)]*k[(((b*H+h)*NK+ki)*D+d)];s[ki]=x*sc;m=std::max(m,s[ki]);}
    if(!std::isfinite(m))continue; double l=0;for(int ki=0;ki<NK;++ki)if(std::isfinite(s[ki]))l+=std::exp(s[ki]-m);
    for(int d=0;d<D;++d){double x=0;for(int ki=0;ki<NK;++ki)if(std::isfinite(s[ki]))x+=std::exp(s[ki]-m)/l*v[(((b*H+h)*NK+ki)*D+d)];o[(((b*H+h)*NQ+qi)*D+d)]=(float)x;}
  }
}

static double maxdiff(const std::vector<float>&a,const std::vector<float>&b){double m=0;for(size_t i=0;i<a.size();++i)m=std::max(m,std::abs((double)a[i]-b[i]));return m;}

int main(){
  struct C{int nq,nk,d;bool causal;}; std::vector<C> cases={{7,13,64,false},{9,73,64,false},{5,2,64,true},{2,5,64,true},{11,37,128,true}};
  for(auto c:cases){int B=1,H=1;size_t qn=(size_t)B*H*c.nq*c.d,kvn=(size_t)B*H*c.nk*c.d;std::vector<float>hq(qn),hk(kvn),hv(kvn),ref(qn),out(qn);
    for(size_t i=0;i<qn;++i){int z=(int)((i*7)%23)-11;hq[i]=z*0.17f;}for(size_t i=0;i<kvn;++i){int a=(int)((i*5)%19)-9;int b=(int)((i*3)%17)-8;hk[i]=a*0.13f;hv[i]=b*0.11f;}cpu_ref(hq,hk,hv,ref,B,H,c.nq,c.nk,c.d,c.causal);
    float *q,*k,*v,*o,*s;CUDA_CHECK(cudaMalloc(&q,qn*4));CUDA_CHECK(cudaMalloc(&k,kvn*4));CUDA_CHECK(cudaMalloc(&v,kvn*4));CUDA_CHECK(cudaMalloc(&o,qn*4));CUDA_CHECK(cudaMalloc(&s,(size_t)c.nq*c.nk*4));
    CUDA_CHECK(cudaMemcpy(q,hq.data(),qn*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(k,hk.data(),kvn*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(v,hv.data(),kvn*4,cudaMemcpyHostToDevice));
    auto check=[&](const char*name){CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(out.data(),o,qn*4,cudaMemcpyDeviceToHost));double d=maxdiff(ref,out);std::printf("%s nq=%d nk=%d d=%d causal=%d maxdiff=%.3e\n",name,c.nq,c.nk,c.d,(int)c.causal,d);if(!(d<2e-4))std::exit(1);};
    launch_v0_naive(q,k,v,o,s,B,H,c.nq,c.nk,c.d,c.causal);check("v0");
    launch_v1_online(q,k,v,o,B,H,c.nq,c.nk,c.d,c.causal);check("v1");
    launch_v2_fa2_partition(q,k,v,o,B,H,c.nq,c.nk,c.d,c.causal);check("v2");
    cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(o);cudaFree(s);
  }
  {int B=1,H=2,NK=257,D=128,S=7;size_t qn=(size_t)B*H*D,kvn=(size_t)B*H*NK*D;std::vector<float>hq(qn),hk(kvn),hv(kvn),ref(qn),out(qn);for(size_t i=0;i<qn;++i)hq[i]=std::sin(i*.1);for(size_t i=0;i<kvn;++i){hk[i]=std::cos(i*.07);hv[i]=std::sin(i*.03);}cpu_ref(hq,hk,hv,ref,B,H,1,NK,D,false);float*q,*k,*v,*o,*pm,*pl,*po;CUDA_CHECK(cudaMalloc(&q,qn*4));CUDA_CHECK(cudaMalloc(&k,kvn*4));CUDA_CHECK(cudaMalloc(&v,kvn*4));CUDA_CHECK(cudaMalloc(&o,qn*4));CUDA_CHECK(cudaMalloc(&pm,B*H*S*4));CUDA_CHECK(cudaMalloc(&pl,B*H*S*4));CUDA_CHECK(cudaMalloc(&po,(size_t)B*H*S*D*4));CUDA_CHECK(cudaMemcpy(q,hq.data(),qn*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(k,hk.data(),kvn*4,cudaMemcpyHostToDevice));CUDA_CHECK(cudaMemcpy(v,hv.data(),kvn*4,cudaMemcpyHostToDevice));launch_decode_splitkv(q,k,v,o,pm,pl,po,B,H,NK,D,S);CUDA_CHECK(cudaDeviceSynchronize());CUDA_CHECK(cudaMemcpy(out.data(),o,qn*4,cudaMemcpyDeviceToHost));double d=maxdiff(ref,out);std::printf("decode nk=%d d=%d splits=%d maxdiff=%.3e\n",NK,D,S,d);if(!(d<2e-4))return 1;cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(o);cudaFree(pm);cudaFree(pl);cudaFree(po);}
  std::puts("CUDA correctness suite passed");return 0;
}
