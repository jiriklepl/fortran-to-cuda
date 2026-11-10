"""Exact power-of-two division keeps scalar types and exceptional bit patterns."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission import read_common_header
from compiler.emission.common.c_family import render_expression
from compiler.ir import Binary, Literal, Reference, ScalarType, Symbol
from compiler.numerical_contract import cuda_compile_options, require_explicit_cuda_environment


@pytest.mark.parametrize(("left", "right"), [
    (ScalarType.REAL, ScalarType.REAL), (ScalarType.REAL32, ScalarType.REAL32),
    (ScalarType.INTEGER, ScalarType.REAL), (ScalarType.REAL, ScalarType.INTEGER),
    (ScalarType.REAL32, ScalarType.REAL), (ScalarType.REAL, ScalarType.REAL32),
])
def test_dynamic_floating_division_preserves_operand_types(left, right):
    a, b = Symbol(0, "a", left), Symbol(1, "b", right)
    assert render_expression(Binary("/", Reference(a), Reference(b))) == (
        "::generated_kernels::numeric::divide(fort_v0_a, fort_v1_b)")


def test_integer_division_and_constant_floating_divisors_retain_original_lowering():
    a, b = Symbol(0, "a", ScalarType.INTEGER), Symbol(1, "b", ScalarType.INTEGER)
    assert render_expression(Binary("/", Reference(a), Reference(b))) == "(fort_v0_a / fort_v1_b)"
    x = Symbol(2, "x", ScalarType.REAL)
    assert render_expression(Binary("/", Reference(x), Literal("3.", ScalarType.REAL))) == "(fort_v2_x / 3.)"


@pytest.mark.cuda
def test_cuda_exact_division_matches_original_bits_across_exponents_and_special_values(tmp_path):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA and C++ compiler required")
    require_explicit_cuda_environment()
    (tmp_path / "runtime.hpp").write_text(read_common_header())
    source = tmp_path / "division.cu"
    source.write_text(r'''
#include "runtime.hpp"
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstdio>
template<class T> __global__ void evaluate(const T* a,const T* b,T* reference,T* candidate,int n) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) {
        reference[i]=a[i]/b[i];
        candidate[i]=generated_kernels::numeric::divide(a[i],b[i]);
        reference[n+i]=(i-41)/b[i];
        candidate[n+i]=generated_kernels::numeric::divide(i-41,b[i]);
    }
}
template<class T,class U,int M,int E> void check() {
    constexpr U sign=U(1)<<(sizeof(U)*8-1), exponent=U((1<<E)-1)<<M;
    const U special[]={0,sign,1,sign|1,(U(1)<<M)-1,U(1)<<M,exponent-1,
        sign|(exponent-1),exponent,sign|exponent,exponent|1,exponent|(U(1)<<(M-1)),
        exponent|12345,sign|exponent|54321, U((1<<(E-1))-1)<<M,
        sign|(U((1<<(E-1))-1)<<M)};
    const int power_count=(1<<E)*32,n=power_count+8192;
    std::vector<T> a(n),b(n),reference(n*2),candidate(n*2);
    uint64_t state=0x123456789abcdef0ULL;
    auto random=[&](){state^=state<<13;state^=state>>7;state^=state<<17;return U(state);};
    for(int i=0;i<n;++i) {
        U av=i%32<16?special[i%16]:random();
        U bv=i<power_count?(U(i/32)<<M)|((i&1)?sign:0):i%32<16?special[i%16]:random();
        std::memcpy(&a[i],&av,sizeof(T));std::memcpy(&b[i],&bv,sizeof(T));
    }
    T *da,*db,*dr,*dc;
    CUCH(cudaMalloc(&da,n*sizeof(T)));CUCH(cudaMalloc(&db,n*sizeof(T)));
    CUCH(cudaMalloc(&dr,n*2*sizeof(T)));CUCH(cudaMalloc(&dc,n*2*sizeof(T)));
    CUCH(cudaMemcpy(da,a.data(),n*sizeof(T),cudaMemcpyHostToDevice));
    CUCH(cudaMemcpy(db,b.data(),n*sizeof(T),cudaMemcpyHostToDevice));
    evaluate<<<(n+255)/256,256>>>(da,db,dr,dc,n);CUCH(cudaGetLastError());CUCH(cudaDeviceSynchronize());
    CUCH(cudaMemcpy(reference.data(),dr,n*2*sizeof(T),cudaMemcpyDeviceToHost));
    CUCH(cudaMemcpy(candidate.data(),dc,n*2*sizeof(T),cudaMemcpyDeviceToHost));
    for(int i=0;i<2*n;++i)if(std::memcmp(&reference[i],&candidate[i],sizeof(T))) {
        std::fprintf(stderr,"division mismatch: width %zu, sample %d\n",sizeof(T),i);std::exit(2);
    }
    CUCH(cudaFree(da));CUCH(cudaFree(db));CUCH(cudaFree(dr));CUCH(cudaFree(dc));
    std::printf("PASS %zu %d\n",sizeof(T),2*n);
}
int main(){check<float,uint32_t,23,8>();check<double,uint64_t,52,11>();}
''')
    command = [nvcc, "-ccbin", host, "-std=c++17", "-O3", "-Xcompiler=-fopenmp",
               "-arch=" + os.environ.get("FORT_TEST_CUDA_ARCH", "native"),
               *cuda_compile_options(), str(source), "-o", str(tmp_path / "division")]
    build = subprocess.run(command, text=True, capture_output=True, timeout=120)
    (tmp_path / "compile.log").write_text(build.stdout + build.stderr)
    assert build.returncode == 0, build.stdout + build.stderr
    result = subprocess.run([str(tmp_path / "division")], text=True, capture_output=True, timeout=60)
    (tmp_path / "result.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == ["PASS 4 32768", "PASS 8 147456"]
