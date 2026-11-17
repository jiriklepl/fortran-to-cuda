"""Exact pitched copies and bounded scratch ownership, independent of selection."""

import os
import shutil
import subprocess

import pytest

from compiler.emission.common.resources import read_common_header


def checked(command, path):
    result = subprocess.run(command, cwd=path, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.native
@pytest.mark.cuda
def test_pitched_copies_match_independent_coordinates_and_cost_counts(tmp_path):
    nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
    if not nvcc:
        pytest.skip("CUDA compiler required")
    (tmp_path / "common.hpp").write_text(read_common_header())
    (tmp_path / "copies.cu").write_text(r'''
#include <cuda_runtime.h>
#include <cassert>
#include <numeric>
#include <random>
#include <vector>
int copies=0;
cudaError_t counted1(void *d,const void *s,size_t n,cudaMemcpyKind k) {
    ++copies; return cudaMemcpy(d,s,n,k);
}
cudaError_t counted2(void *d,size_t dp,const void *s,size_t sp,size_t w,size_t h,cudaMemcpyKind k) {
    ++copies; return cudaMemcpy2D(d,dp,s,sp,w,h,k);
}
cudaError_t counted3(const cudaMemcpy3DParms *p) { ++copies; return cudaMemcpy3D(p); }
#define cudaMemcpy counted1
#define cudaMemcpy2D counted2
#define cudaMemcpy3D counted3
#include "common.hpp"
#undef cudaMemcpy
#undef cudaMemcpy2D
#undef cudaMemcpy3D
int main() {
    using namespace generated_kernels::offload;
    std::mt19937 random(42);
    for (unsigned rank=1;rank<=4;++rank) {
        std::vector<size_t> dimensions{7,6,5,4}; dimensions.resize(rank);
        size_t n=1; for(auto d:dimensions) n*=d;
        std::vector<double> host(n),expected(n),actual(n),sentinel(n,-123);
        void *device=nullptr;
        CUCH(cudaMalloc(&device,n*sizeof(double)));
        Array a{host.data(),sizeof(double),n*sizeof(double),dimensions};
        for(unsigned sample=0;sample<80;++sample) {
            Box box;
            for(auto d:dimensions) {
                const auto lo=sample==0 ? 0 : random()%d;
                const auto hi=sample==0 ? d-1 : lo+random()%(d-lo);
                box.lower.push_back(lo); box.upper.push_back(hi);
            }
            // Explicit opposite faces, singleton middle axes, and a 3D interior.
            if(sample<=2*rank && sample) {
                box.lower.assign(rank,0); box.upper=dimensions;
                for(auto &v:box.upper) --v;
                const auto axis=(sample-1)/2;
                box.lower[axis]=box.upper[axis]=(sample%2) ? 0 : dimensions[axis]-1;
            }
            if(sample==2*rank+1) {
                box.lower.assign(rank,1); box.upper=dimensions;
                for(auto &v:box.upper) v-=2;
            }
            for(bool upload:{true,false}) {
                std::iota(host.begin(),host.end(),10.0+sample);
                expected=sentinel;
                // Element-by-element oracle uses logical coordinates, not CUDA pitches.
                size_t selected=0;
                for(size_t i=0;i<n;++i) {
                    auto rest=i; bool inside=true;
                    for(unsigned axis=0;axis<rank;++axis) {
                        const auto at=rest%dimensions[axis]; rest/=dimensions[axis];
                        inside &= at>=box.lower[axis] && at<=box.upper[axis];
                    }
                    if(inside) { expected[i]=host[i]; ++selected; }
                }
                CUCH(cudaMemcpy(device,upload ? sentinel.data() : host.data(),a.bytes,cudaMemcpyHostToDevice));
                if(!upload) host=sentinel;
                copies=0;
                copy_box(a,box,device,upload);
                bool valid=true;
                assert(copy_operations(a,box,valid)==static_cast<size_t>(copies) && valid);
                assert(box_bytes(a,box,valid)==selected*sizeof(double));
                if(upload) CUCH(cudaMemcpy(actual.data(),device,a.bytes,cudaMemcpyDeviceToHost));
                else actual=host;
                assert(actual==expected);
            }
        }
        CUCH(cudaFree(device));
    }
}
''')
    command = [nvcc, "-std=c++17", "-O2", "-DFORT_OFFLOAD_ENABLED", "-Xcompiler=-fopenmp"]
    host = shutil.which(os.environ.get("CUDAHOSTCXX", "g++-14"))
    if host:
        command += ["-ccbin", host]
    checked([*command, "copies.cu", "-o", "copies"], tmp_path)
    checked([str(tmp_path / "copies")], tmp_path)
