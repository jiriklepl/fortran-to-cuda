"""Exercise CUDA host-branch coherence with synchronous simulated device work."""

import re
from dataclasses import replace

import pytest

from compiler.analysis import build_execution_plan
from compiler.emission import generate_sources
from compiler.tests import test_language as language

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


@pytest.mark.native
def test_cuda_host_predicates_branch_joins_and_partial_writes_preserve_values(tmp_path, monkeypatch):
    function = language.lower(
        tmp_path,
        "do i=1,n\na(i)=a(i)+1\nenddo\n"
        "if(a(1)>0)then\nif(enabled)then\ndo i=1,n\na(i)=a(i)*2\nenddo\n"
        "if(a(2)>1) a(3)=-a(3)\nelse\na(2)=-7\nendif\n"
        "else\ndo i=1,n\na(i)=a(i)-3\nenddo\nendif\n"
        "a(n)=a(n)+10\ndo i=1,n\na(i)=abs(a(i))+2\nenddo",
        "logical,intent(in)::enabled",
        "a,n,enabled",
    )
    driver = """program main
use language_case
real(knd)::a(4)
a=[-4.d0,1.d0,2.d0,3.d0]
call entry(a,4,.true.)
print *,a
a=[1.d0,2.d0,3.d0,4.d0]
call entry(a,4,.true.)
print *,a
a=[1.d0,2.d0,3.d0,4.d0]
call entry(a,4,.false.)
print *,a
end program
"""
    (tmp_path / "cuda_runtime.h").write_text(CUDA_RUNTIME)

    def simulated_sources(function, plan):
        sources = generate_sources(function, plan)
        source = sources.cuda.replace("#include <cuda_runtime.h>", '#include "cuda_runtime.h"')
        source, launches = re.subn(r"(kernel_region_\d+_device)<<<([^>]+)>>>\(", r"fort_test_launch(\1, \2, ", source)
        assert launches == len(plan.regions)
        return replace(sources, cpp=source)

    monkeypatch.setattr(language, "generate_sources", simulated_sources)
    language.run_reference_and_cpp(tmp_path, function, build_execution_plan(function), driver)
