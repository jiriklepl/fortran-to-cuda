"""Independent native-reference checks for packed, disjoint CUDA/CPU windows."""

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from compiler.analysis import build_execution_plan
from compiler.driver.options import CompilerOptions
from compiler.emission.common.abi import abi_arguments
from compiler.emission.common.resources import read_common_header
from compiler.emission.cuda.generator import generate_cuda
from compiler.emission.cuda.hybrid import generate_hybrid
from compiler.emission.fortran.generator import generate_fortran
from compiler.frontend import lower_file
from compiler.offload import analyze_offload

SOURCE = """module hybrid_case
contains
subroutine advance(a,b,n,m,scale)
real(8),intent(inout)::a(:,:)
real(8),intent(in)::b(:,:)
integer,intent(in)::n,m
real(8),intent(in)::scale
integer::i,j
real(8)::required_scale
do j=2,m-1
 do i=2,n-1
  required_scale=scale
  if(b(i,j)>0.0_8) a(i,j)=b(i-1,j-1)+b(i+1,j+1)+real(size(a,2),8)+required_scale
 end do
end do
do j=2,m-1
 do i=2,n-1
  a(i,j)=a(i,j)*2.0_8
 end do
end do
end subroutine
end module
"""


def prepare(tmp_path, source=SOURCE):
    path = tmp_path / "original.f90"
    path.write_text(source)
    function = lower_file(path, "advance")
    plan = build_execution_plan(function, options=CompilerOptions(opt_level=0))
    return function, plan, analyze_offload(function, plan)


def test_hybrid_preserves_logical_extents_and_namespaces_kernels(tmp_path):
    function, plan, analysis = prepare(tmp_path)
    output = generate_hybrid(function, plan, analysis, policy="chunked")
    assert output.available, output.reason
    assert "hybrid::View<const double, 2>" in output.helpers
    assert "slot.stream" in output.helpers
    assert "timing::measure_kernel" not in output.helpers
    assert "cudaDeviceSynchronize" not in output.helpers
    cpu = output.helpers.split("__global__", 1)[0]
    assert "#pragma omp" not in cpu
    renamed = generate_hybrid(replace(function, name="different"), plan, analysis, policy="chunked")
    assert output.entry_name != renamed.entry_name
    assert output.entry_name + "_select" in output.decision_body[0]
    assert "static_cast<int>" in output.helpers  # SIZE still uses the logical extent ABI.


def test_cross_slab_written_dependence_has_no_hybrid_dispatch(tmp_path):
    source = SOURCE.replace("b(i-1,j-1)+b(i+1,j+1)+real(size(a,2),8)+required_scale", "b(i,j)")
    source = source.replace("a(i,j)=a(i,j)*2.0_8", "a(i,j)=a(i,j-1)+a(i-1,j)")
    # Each second loop is unsafe itself, so use two pointwise units with
    # shifted destinations instead of relaxing the compiler dependence proof.
    source = source.replace("a(i,j)=a(i,j-1)+a(i-1,j)", "a(i+1,j+1)=b(i,j)")
    function, plan, analysis = prepare(tmp_path, source)
    output = generate_hybrid(function, plan, analysis)
    assert not output.available
    assert output.decision_body[-1] == "return 0;"
    assert '"native"' in output.decision_body[0]
    assert "hybrid::execute" not in output.helpers
    assert output.cpu_entry_name in output.body[0]


def test_uncertain_conditional_work_stays_native_and_query_scalars_are_references(tmp_path):
    function, plan, analysis = prepare(tmp_path)
    assert any(unit.work_is_upper_bound for unit in analysis.units)
    output = generate_hybrid(function, plan, analysis, policy="hybrid", collective=True)
    selector = output.helpers.split(f"static hybrid::Choice {output.entry_name}_select(", 1)[1]
    assert "const int&" in selector.split(") {", 1)[0]
    assert selector.index("Uncertain work") < selector.index("hybrid::select(")
    assert '"slabs"' in "\n".join(output.decision_body)
    entry = output.helpers.split(f"static void {output.entry_name}(", 1)[1]
    assert entry.index("#pragma omp barrier") < entry.index("#pragma omp single copyprivate")


@pytest.mark.native
def test_allocation_failure_and_actual_team_mismatch_fall_back_before_any_work(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("A C++ compiler with OpenMP is required")
    runtime = Path(__file__).parents[1] / "runtime/hybrid.hpp"
    source = tmp_path / "allocation_failure.cpp"
    source.write_text(
        r"""
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <vector>
#include <omp.h>
#define __CUDACC__
#define __device__
using cudaError_t = int;
using cudaStream_t = int*;
using cudaEvent_t = int*;
constexpr int cudaSuccess=0, cudaStreamNonBlocking=1, cudaEventDisableTiming=2;
constexpr int cudaErrorMemoryAllocation=1, cudaErrorNotReady=2;
constexpr int cudaMemcpyHostToDevice=1, cudaMemcpyDeviceToHost=2;
int streams=0, events=0, host_live=0, device_live=0, host_calls=0;
bool fail_second=true;
int cudaGetLastError() { return 0; }
int cudaGetDevice(int* device) { *device=0; return 0; }
int cudaSetDevice(int) { return 0; }
int cudaStreamCreateWithFlags(cudaStream_t* p,unsigned) { *p=new int; ++streams; return 0; }
int cudaEventCreateWithFlags(cudaEvent_t* p,unsigned) { *p=new int; ++events; return 0; }
int cudaStreamDestroy(cudaStream_t p) { delete p; --streams; return 0; }
int cudaEventDestroy(cudaEvent_t p) { delete p; --events; return 0; }
int cudaEventSynchronize(cudaEvent_t) { return 0; }
int cudaStreamSynchronize(cudaStream_t) { return 0; }
int cudaEventRecord(cudaEvent_t,cudaStream_t) { return 0; }
int cudaMallocHost(void** p,std::size_t bytes) {
    if(++host_calls==2 && fail_second) return 1;
    *p=std::malloc(bytes); ++host_live; return 0;
}
int cudaMalloc(void** p,std::size_t bytes) { *p=std::malloc(bytes); ++device_live; return 0; }
int cudaFreeHost(void* p) { std::free(p); --host_live; return 0; }
int cudaFree(void* p) { std::free(p); --device_live; return 0; }
int cudaMemcpyAsync(void* a,const void* b,std::size_t bytes,int,cudaStream_t) {
    std::memcpy(a,b,bytes); return 0;
}
#define CUCH(call) do { if ((call) != cudaSuccess) std::abort(); } while(0)
namespace generated_kernels::storage {
void trace(const char*,std::size_t=0) {}
[[noreturn]] void fail(const char*) { std::abort(); }
}
namespace generated_kernels::offload {
int native_decisions=0;
void decision_trace(const char*,const char* mode,std::size_t gpu,std::size_t cpu,const char*) {
    if(std::strcmp(mode,"native") || gpu || cpu!=32) std::abort();
    ++native_decisions;
}
}
"""
        + f'#include "{runtime}"\n'
        + r"""
int main() {
    using namespace generated_kernels;
    double values[32] = {};
    std::vector<hybrid::Array> arrays{{values,sizeof(double),{32},0,1,0,0,true}};
    auto cpu=[&](std::size_t begin,std::size_t count) {
        for(std::size_t i=begin;i<begin+count;++i) values[i]+=1;
    };
    auto gpu=[](hybrid::Slot&,std::size_t,std::size_t) { std::abort(); };
    hybrid::Choice choice{16,8,64,0};
    choice.threads=4;
    hybrid::execute(arrays,32,1,1,choice,4,false,cpu,gpu,"failure","hybrid");
    if(host_calls!=2 || host_live || device_live || streams || events) return 1;
    for(double value:values) if(value!=1) return 2;
    if(hybrid::budget_state().reserved) return 3;
    // A partial split in an actual single-thread collective must do all work
    // once on the CPU, without creating a nested team or attempting staging.
    choice.threads=1;
    #pragma omp parallel num_threads(1)
    { hybrid::execute(arrays,32,1,1,choice,1,true,cpu,gpu,"one","hybrid"); }
    for(double value:values) if(value!=2) return 4;
    // A runtime team smaller than the calibrated team invalidates its split.
    choice.threads=4;
    hybrid::execute(arrays,32,1,1,choice,2,false,cpu,gpu,"team","hybrid");
    for(double value:values) if(value!=3) return 5;
    // An ordinary serial call from an unrecognized existing team's SINGLE
    // computes exactly once on that caller; it cannot recruit a nested team.
    #pragma omp parallel num_threads(4)
    {
        #pragma omp single
        { hybrid::execute(arrays,32,1,1,choice,4,false,cpu,gpu,"single","hybrid"); }
    }
    for(double value:values) if(value!=4) return 6;
    if(host_calls!=2 || offload::native_decisions!=4) return 7;
    // A small packed footprint cannot justify overflowing full logical strides.
    std::vector<hybrid::Array> overflow{{values,8,{static_cast<std::size_t>(-1)/8,3},0,1,0,0,false}};
    std::vector<hybrid::Layout> layout;
    std::size_t bytes=0;
    if(hybrid::layouts(overflow,0,1,1,1,layout,bytes)) return 8;
    std::cout << "HYBRID_RESOURCE_FALLBACK_PASS\n";
}
"""
    )
    binary = tmp_path / "failure"
    result = subprocess.run(
        [compiler, "-std=c++17", "-O2", "-fopenmp", str(source), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [str(binary)],
        env=dict(os.environ, OMP_DYNAMIC="FALSE"),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HYBRID_RESOURCE_FALLBACK_PASS" in result.stdout


@pytest.mark.native
@pytest.mark.cuda
@pytest.mark.parametrize(("collective", "reverse_permuted"), [(False, False), (True, False), (True, True)])
def test_packed_two_slot_windows_match_fortran_with_halos_and_slot_reuse(tmp_path, collective, reverse_permuted):
    nvcc = shutil.which(os.environ.get("NVCC", "nvcc"))
    fortran = shutil.which(os.environ.get("FC", "gfortran-15")) or shutil.which("gfortran")
    if not nvcc or not fortran:
        pytest.skip("CUDA and GNU Fortran compilers are required")
    source = SOURCE
    if reverse_permuted:
        source = source.replace("do j=2,m-1", "do j=m-1,2,-1")
        source = source.replace("b(i,j)", "b(j,i)").replace("b(i-1,j-1)", "b(j-1,i-1)")
        source = source.replace("b(i+1,j+1)", "b(j+1,i+1)")
    function, plan, analysis = prepare(tmp_path, source)
    output = generate_hybrid(function, plan, analysis, policy="chunked", collective=collective)
    assert output.available, output.reason
    (tmp_path / "common.hpp").write_text(read_common_header())
    (tmp_path / "interface.f90").write_text(generate_fortran(function, abi_arguments(function.parameters)))
    (tmp_path / "reference.f90").write_text(source.replace("module hybrid_case", "module reference_case"))
    args = ", ".join(arg.name for arg in abi_arguments(function.parameters))
    from compiler.emission.c.declarations import cpp_declaration

    signature = ", ".join(cpp_declaration(arg) for arg in abi_arguments(function.parameters))
    # A second call exercises the same engine with a deterministic half split.
    # This internal control tests races/packing independently of cost calibration.
    legacy = generate_cuda(function, plan, abi_arguments(function.parameters), "common.hpp")
    legacy = legacy.replace('extern "C" void cpp_advance(', 'extern "C" void cpp_advance_legacy(')
    (tmp_path / "generated.cu").write_text(
        legacy + "\nnamespace generated_kernels {\n"
        "using namespace indexing;\n" + output.helpers + f'extern "C" void cpp_advance({signature}) {{\n'
        f"auto choice={output.entry_name}_select({args});\n"
        'const char *half=std::getenv("FORT_TEST_HALF");\n'
        "if(half && choice.gpu_iterations>1) choice.gpu_iterations/=2;\n"
        f"{output.entry_name}_dispatch({args},choice);\n}}\n}}\n"
    )
    parallel_start = "!$omp parallel num_threads(4)" if collective else ""
    parallel_end = "!$omp end parallel" if collective else ""
    b_shape = "ny,nx" if reverse_permuted else "nx,ny"
    b_index = "j,i" if reverse_permuted else "i,j"
    (tmp_path / "driver.f90").write_text(f"""program compare
use hybrid_case,only:candidate=>advance,advance_trim_cache
use reference_case,only:reference=>advance
implicit none
integer,parameter::nx=513
real(8),allocatable::a(:,:),b(:,:),expected(:,:)
integer::i,j,rep,ny
do rep=1,3
 ny=1026
 if(rep==3) ny=513
 allocate(a(nx,ny),b({b_shape}),expected(nx,ny))
 do j=1,ny
  do i=1,nx
   b({b_index})=real(i+100*j+rep,8)
   if(modulo(i+j,2)==0) b({b_index})=-b({b_index})
  end do
 end do
 a=-17.0_8-rep
 expected=a
 call reference(expected,b,nx,ny,0.25_8*rep)
 {parallel_start}
 call candidate(a,b,nx,ny,0.25_8*rep)
 {parallel_end}
 if(any(a/=expected)) error stop 'hybrid mismatch or halo corruption'
 deallocate(a,b,expected)
end do
call advance_trim_cache()
print *, 'HYBRID_NATIVE_PASS'
end program
""")
    environment = dict(os.environ, OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE")
    host = shutil.which(os.environ.get("CUDAHOSTCXX", "g++-14"))
    command = [nvcc, "-std=c++17", "-O2", "-DFORT_OFFLOAD_ENABLED", "-Xcompiler=-fopenmp", "-arch=sm_86"]
    if host:
        command += ["-ccbin", host]
    commands = [
        [*command, "-c", "generated.cu", "-o", "generated.o"],
        [
            fortran,
            "-O2",
            "-fopenmp",
            "-ffree-line-length-none",
            "-fcheck=all",
            "-c",
            "interface.f90",
            "reference.f90",
            "driver.f90",
        ],
        [
            fortran,
            "-fopenmp",
            "interface.o",
            "reference.o",
            "driver.o",
            "generated.o",
            "-lstdc++",
            "-L" + str(Path(nvcc).resolve().parents[1] / "lib64"),
            "-lcudart",
            "-o",
            "run",
        ],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
    for half in (False, True):
        env = dict(environment, FORT_RUNTIME_TRACE="1", FORT_OFFLOAD_TRACE="1")
        if half:
            env["FORT_TEST_HALF"] = "1"
        result = subprocess.run(
            [str(tmp_path / "run")], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90
        )
        log = result.stdout + result.stderr
        (tmp_path / ("half.log" if half else "chunked.log")).write_text(log)
        if "no CUDA-capable device" in log or "CUDA driver version is insufficient" in log:
            pytest.skip(log)
        assert result.returncode == 0, log
        assert "HYBRID_NATIVE_PASS" in log
        assert log.count("FORT_RUNTIME kernel") > 4  # Both slots are reused.
        assert log.count("FORT_RUNTIME alloc bytes=") == 2  # Retained across reallocation and shape changes.
        assert log.count("FORT_RUNTIME scratch_reuse bytes=") == 2
        assert log.count("FORT_RUNTIME free") == 2  # Explicit public cache teardown.
