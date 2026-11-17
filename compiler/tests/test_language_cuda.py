"""Exercise CUDA host-branch coherence with synchronous simulated device work."""




CUDA_RUNTIME = r"""
#pragma once
#define __CUDACC__
#define __global__
#define __host__
#define __device__
#include <cstdlib>
#include <cstring>
using cudaError_t = int;
using cudaEvent_t = float*;
constexpr int cudaSuccess = 0;
constexpr int cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2;
constexpr int cudaHostRegisterPortable = 1;
inline const char* cudaGetErrorString(int) { return "simulated CUDA error"; }
inline int cudaMalloc(void** p, std::size_t n) { *p = std::malloc(n); return *p ? 0 : 1; }
inline int cudaFree(void* p) { std::free(p); return 0; }
inline int cudaMemcpy(void* to, const void* from, std::size_t n, int) { std::memcpy(to, from, n); return 0; }
inline int cudaDeviceSynchronize() { return 0; }
inline int cudaGetLastError() { return 0; }
inline int cudaHostRegister(void*, std::size_t, unsigned) { return 0; }
inline int cudaHostUnregister(void*) { return 0; }
inline int cudaEventCreate(cudaEvent_t* event) { *event = new float(0); return 0; }
inline int cudaEventDestroy(cudaEvent_t event) { delete event; return 0; }
inline int cudaEventRecord(cudaEvent_t, int) { return 0; }
inline int cudaEventSynchronize(cudaEvent_t) { return 0; }
inline int cudaEventElapsedTime(float* value, cudaEvent_t, cudaEvent_t) { *value = 0; return 0; }
struct Dimension { unsigned x = 0; };
inline Dimension blockIdx, threadIdx, blockDim, gridDim;
template <typename Function, typename... Args>
void fort_test_launch(Function function, unsigned blocks, unsigned threads, Args... args) {
    blockDim.x = threads;
    gridDim.x = blocks;
    for (blockIdx.x = 0; blockIdx.x < blocks; ++blockIdx.x)
        for (threadIdx.x = 0; threadIdx.x < threads; ++threadIdx.x)
            function(args...);
}
"""
