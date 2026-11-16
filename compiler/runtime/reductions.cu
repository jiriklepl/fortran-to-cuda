#include "reductions.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>

namespace {
constexpr unsigned threads=256;
template<class T> struct Result { T maximum; int32_t any_nan; };
template<class T> struct Work { size_t elements; unsigned blocks; T initial; Result<T> *scratch; Result<T> result; };
template<class T> __device__ Result<T> merge(Result<T> a, Result<T> b) {
    return {fmax(a.maximum,b.maximum),a.any_nan|b.any_nan};
}
template<class T> __device__ void reduce_block(Result<T> value,Result<T> *output) {
    __shared__ Result<T> partial[threads];
    partial[threadIdx.x]=value;
    __syncthreads();
    for(unsigned stride=threads/2;stride;stride/=2) {
        if(threadIdx.x<stride) partial[threadIdx.x]=merge(partial[threadIdx.x],partial[threadIdx.x+stride]);
        __syncthreads();
    }
    if(!threadIdx.x) *output=partial[0];
}
template<class T> __global__ void partial_max(const T *values,size_t elements,T initial,Result<T> *output) {
    Result<T> value{initial,0};
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<elements;i+=size_t(gridDim.x)*blockDim.x) {
        const T x=values[i];
        value.any_nan|=isnan(x);
        value.maximum=fmax(value.maximum,x);
    }
    reduce_block(value,output+blockIdx.x);
}
template<class T> __global__ void final_max(const Result<T> *partial,unsigned count,T initial,Result<T> *output) {
    Result<T> value{initial,0};
    for(unsigned i=threadIdx.x;i<count;i+=blockDim.x) value=merge(value,partial[i]);
    reduce_block(value,output);
}
template<class T> struct ReductionTraits;
template<> struct ReductionTraits<float> {
    static constexpr int scope_type=FORT_SCOPE_REAL32;
    static constexpr const char *trace_name="max_real32";
};
template<> struct ReductionTraits<double> {
    static constexpr int scope_type=FORT_SCOPE_REAL64;
    static constexpr const char *trace_name="max_real64";
};
template<class T>
int worker(void *user,const fort_scope_external_view_v1 *views,size_t count,void *stream,int *started) noexcept {
    *started=0;
    if(count!=1 || !user || !views) return FORT_SCOPE_ARGUMENT;
    auto &work=*static_cast<Work<T> *>(user);const auto &v=views[0];
    if(!v.device || !v.extents || !v.strides || v.type!=ReductionTraits<T>::scope_type || v.element_bytes!=sizeof(T))
        return FORT_SCOPE_ARGUMENT;
    size_t bytes=sizeof(T);
    for(unsigned axis=0;axis<v.rank;++axis) {
        if(v.strides[axis]!=bytes || !v.extents[axis] || bytes>SIZE_MAX/v.extents[axis])
            return FORT_SCOPE_ARGUMENT;
        bytes*=v.extents[axis];
    }
    if(bytes!=v.bytes || bytes/sizeof(T)!=work.elements) return FORT_SCOPE_ARGUMENT;
    const auto cuda_stream=static_cast<cudaStream_t>(stream);
    *started=1;
    partial_max<T><<<work.blocks,threads,0,cuda_stream>>>(static_cast<const T *>(v.device),work.elements,
                                                          work.initial,work.scratch);
    if(cudaGetLastError()!=cudaSuccess) return FORT_SCOPE_EXECUTION;
    final_max<T><<<1,threads,0,cuda_stream>>>(work.scratch,work.blocks,work.initial,work.scratch+work.blocks);
    if(cudaGetLastError()!=cudaSuccess) return FORT_SCOPE_EXECUTION;
    if(cudaMemcpyAsync(&work.result,work.scratch+work.blocks,sizeof(Result<T>),cudaMemcpyDeviceToHost,cuda_stream)!=cudaSuccess)
        return FORT_SCOPE_EXECUTION;
    return cudaStreamSynchronize(cuda_stream)==cudaSuccess?FORT_SCOPE_OK:FORT_SCOPE_EXECUTION;
}
template<class T>
int scope_max_impl(fort_scope_t context,fort_buffer_t input,uint32_t rank,
    size_t elements,T initial,T *maximum,int32_t *any_nan,fort_scope_external_report_v1 *report) {
    if(report) *report={1,FORT_SCOPE_ARGUMENT,0,0};
    if(!report || !maximum || !any_nan || !input || !rank || rank>15 || !elements ||
       elements>SIZE_MAX/sizeof(T) || std::isnan(initial)) return FORT_SCOPE_ARGUMENT;
    fort_scope_scratch_lease_v1 lease{};
    try {
        const unsigned blocks=unsigned(elements/threads+(elements%threads!=0)>1024?1024:
                                       elements/threads+(elements%threads!=0));
        const size_t bytes=(blocks+1)*sizeof(Result<T>);
        int status=fort_scope_scratch_acquire_v1(context,bytes,&lease);
        if(status) { report->status=status;return status; }
        Work<T> work{elements,blocks,initial,static_cast<Result<T> *>(lease.device),{}};
        fort_scope_access access{FORT_SCOPE_READ_ALL,0,nullptr,0,nullptr,0,nullptr};
        fort_scope_external_binding_v1 binding{input,ReductionTraits<T>::scope_type,rank,sizeof(T),access};
        // Scratch is already charged by scratch_acquire. External workspace
        // claims refer to domain workspace reservations, not scratch tokens.
        fort_scope_external_call_v1 call{1,0,&binding,1,0,0};
        status=fort_scope_external_execute_v1(context,&call,worker<T>,&work,report);
        const int released=fort_scope_scratch_release_v1(context,lease.token);lease.token=0;
        if(!status) status=released;
        if(!status) {
            *maximum=work.result.maximum;*any_nan=work.result.any_nan;
            const char *trace=std::getenv("FORT_RUNTIME_TRACE");
            if(trace && trace[0] && trace[0]!='0')
                std::fprintf(stderr,"FORT_REDUCTION %s context=%llu elements=%zu scalar_download_bytes=%zu kernels=2 any_nan=%d\n",
                    ReductionTraits<T>::trace_name,static_cast<unsigned long long>(context),elements,sizeof(Result<T>),work.result.any_nan);
        }
        report->status=status;return status;
    } catch(...) {
        if(lease.token) fort_scope_scratch_release_v1(context,lease.token);
        report->status=FORT_SCOPE_EXECUTION;return FORT_SCOPE_EXECUTION;
    }
}
}

extern "C" int fort_scope_max_real32_v1(fort_scope_t context,fort_buffer_t input,uint32_t rank,
    size_t elements,float initial,float *maximum,int32_t *any_nan,fort_scope_external_report_v1 *report) {
    return scope_max_impl<float>(context,input,rank,elements,initial,maximum,any_nan,report);
}

extern "C" int fort_scope_max_real64_v1(fort_scope_t context,fort_buffer_t input,uint32_t rank,
    size_t elements,double initial,double *maximum,int32_t *any_nan,fort_scope_external_report_v1 *report) {
    return scope_max_impl<double>(context,input,rank,elements,initial,maximum,any_nan,report);
}
