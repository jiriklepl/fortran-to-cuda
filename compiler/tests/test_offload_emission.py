"""Runtime partition and section-copy checks against independent Fortran."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission.common.resources import read_common_header
from compiler.emission.driver import generate_sources
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig
from compiler.offload.profile import load_profile


def run(command, path, environment=None):
    result = subprocess.run(command, cwd=path, env=environment, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def test_partition_costs_preserve_faces_and_choose_cpu_gpu_gpu_cpu(tmp_path):
    cxx = shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler required")
    header = Path(__file__).resolve().parents[1] / "runtime" / "offload.hpp"
    (tmp_path / "check.cpp").write_text('''
#include <cstdlib>
#include <cassert>
#include "''' + str(header) + '''"
using namespace generated_kernels::offload;
int main() {
    Profile p; p.valid=true; p.launch_seconds=10;
    p.cpu_flops=1; p.gpu_flops=100;
    Data d; d.units.resize(4); d.arrays.push_back({nullptr,8,8,{1}});
    for(auto &u:d.units) { u.iterations=1; u.arrays.resize(1); }
    d.units[1].arrays[0].upload=d.units[2].arrays[0].upload={{{0},{0}}};
    p.h2d_latency=1;
    d.units[0].flops=d.units[3].flops=1;
    d.units[1].flops=d.units[2].flops=1000;
    auto choice=select(d,p,true);
    assert(choice.choices.size()==3);
    assert(!choice.choices[0].gpu && choice.choices[0].end==1);
    assert(choice.choices[1].gpu && choice.choices[1].begin==1 && choice.choices[1].end==3);
    assert(!choice.choices[2].gpu && choice.choices[2].begin==3);
    p.valid=false; assert(!select(d,p,true).has_gpu());
    d.valid=false; assert(!select(d,p,false).has_gpu());
    std::vector<Box> faces;
    append_box(faces,{{0,0,0},{0,19,29}});
    append_box(faces,{{9,0,0},{9,19,29}});
    assert(faces.size()==2);
    Array a{nullptr,8,10*20*30*8,{10,20,30}};
    bool valid=true;
    assert(box_bytes(a,faces[0],valid)+box_bytes(a,faces[1],valid)==2*20*30*8);
    assert(copy_operations(a,{{1,1,1},{8,18,28}},valid)==1);
    std::size_t result=0;
    assert(!mul(std::numeric_limits<std::size_t>::max(),2,result));
    index(1.0e50L,valid); assert(!valid);
}
''')
    run([cxx, "-std=c++17", "-Wall", "-Werror", "check.cpp", "-o", "check"], tmp_path)
    run([str(tmp_path / "check")], tmp_path)


SOURCE = """module section_case
contains
subroutine advance(a,b,n,m,l)
real(8),intent(inout)::a(:,:,:)
real(8),intent(in)::b(:,:,:)
integer,intent(in)::n,m,l
integer::i,j,k
do k=1,l
 do j=2,m-1
  do i=-3,n-4
   if(b(i+4,j,k)>0.0_8) a(i+4,j,k)=b(i+4,j,k)+real(i+100*j+10000*k,8)
  end do
 end do
end do
do k=1,l
 do j=1,m
  a(1,j,k)=b(1,j,k)*2.0_8
  a(n,j,k)=b(n,j,k)*3.0_8
 end do
end do
end subroutine
end module
"""
BOUNDARY = """module section_case
contains
subroutine advance(a,b,n,m,l)
real(8),intent(inout)::a(:,:,:)
real(8),intent(in)::b(:,:,:)
integer,intent(in)::n,m,l
integer::j,k
do k=1,l
 do j=1,m
  a(1,j,k)=b(1,j,k)*2.0_8
  a(n,j,k)=b(n,j,k)*3.0_8
 end do
end do
end subroutine
end module
"""
MIXED = """module section_case
contains
subroutine advance(a,b,n,m,l)
real(8),intent(inout)::a(:,:,:)
real(8),intent(in)::b(:,:,:)
integer,intent(in)::n,m,l
integer::i,j,k
do k=1,l
 do j=1,m
  a(1,j,k)=b(1,j,k)*2.0_8
 end do
end do
do k=1,l
 do j=1,m
  do i=1,n
   a(i,j,k)=b(i,j,k)*2.0_8
  end do
 end do
end do
do k=1,l
 do j=1,m
  do i=1,n
   a(i,j,k)=a(i,j,k)*3.0_8
  end do
 end do
end do
do k=1,l
 do j=1,m
  a(n,j,k)=b(n,j,k)*3.0_8
 end do
end do
end subroutine
end module
"""


@pytest.mark.native
@pytest.mark.cuda
@pytest.mark.parametrize(("policy", "collective", "boundary"), [
    ("sections", False, False), ("sections", True, False),
    ("sections", True, True), ("auto", True, True), ("auto_mixed", True, False),
])
def test_sections_and_native_decisions_match_fortran(tmp_path, policy, collective, boundary):
    nvcc = shutil.which(os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"))
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not fc:
        pytest.skip("CUDA and Fortran compilers required")
    mixed = policy == "auto_mixed"
    if mixed:
        policy = "auto"
    profile_path = Path(__file__).resolve().parents[2] / "benchmarks/results/gpu-strategies-20261006/calibration.json"
    profile = load_profile(profile_path) if policy == "auto" and profile_path.exists() else None
    source = BOUNDARY if boundary else SOURCE
    if mixed:
        if profile is None:
            pytest.skip("A compatible local calibration identity is required")
        # Deterministic correctness control, never used for performance evidence.
        rates = profile["rates"]
        rates.update(cpu_flops_per_second=1e6, gpu_flops_per_second=1e9,
                     cpu_memory_bytes_per_second=1e12, gpu_memory_bytes_per_second=1e12,
                     launch_latency_seconds=.002)
        for direction in ("h2d_pageable", "d2h_pageable"):
            rates[direction] = {"latency_seconds": .0001, "bandwidth_bytes_per_second": 1e12}
        source = MIXED
    (tmp_path / "original.f90").write_text(source)
    function, plan = prepare_function(lower_file(tmp_path / "original.f90", "advance"),
                                     options=CompilerOptions(gpu_policy=policy))
    generated = generate_sources(function, plan, common_header="common.hpp",
                                 offload_config=OffloadConfig(policy, profile, 4, collective))
    assert generated.offload["analysis"]["available"]
    query = generated.offload["native_fallback_query"]
    (tmp_path / "common.hpp").write_text(read_common_header())
    (tmp_path / "generated.cu").write_text(generated.cuda)
    (tmp_path / "interface.f90").write_text(generated.fortran)
    (tmp_path / "reference.f90").write_text(source.replace("module section_case", "module reference_case"))
    start = "!$omp parallel num_threads(4)" if collective else ""
    stop = "!$omp end parallel" if collective else ""
    native = "!$omp single\n call reference(a,b,n,m,l)\n!$omp end single" if collective else "call reference(a,b,n,m,l)"
    (tmp_path / "driver.f90").write_text(f"""program compare
use section_case,only:candidate=>advance, decide=>{query}
use reference_case,only:reference=>advance
implicit none
real(8),allocatable::a(:,:,:),b(:,:,:),expected(:,:,:)
integer::n,m,l,i,j,k,rep
do rep=1,4
 n=32+rep
 m=17+rep
 l=7+rep
 if(rep==4) l=0
 allocate(a(n,m,l),b(n,m,l),expected(n,m,l))
 do k=1,l
  do j=1,m
   do i=1,n
    b(i,j,k)=real(i+10*j+100*k+rep,8)
    if(modulo(i+j,2)==0) b(i,j,k)=-b(i,j,k)
   end do
  end do
 end do
 a=-17.0_8-rep
 expected=a
 call reference(expected,b,n,m,l)
 {start}
 if(decide(a,b,n,m,l)) then
  call candidate(a,b,n,m,l)
 else
  {native}
 end if
 {stop}
 if(any(a/=expected)) error stop 'section mismatch or halo corruption'
 deallocate(a,b,expected)
end do
print *, 'SECTIONS_NATIVE_PASS'
end program
""")
    environment = dict(os.environ, OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE", FORT_OFFLOAD_TRACE="1", FORT_RUNTIME_TRACE="1")
    host = shutil.which(os.environ.get("CUDAHOSTCXX", "g++-14"))
    command = [nvcc, "-std=c++17", "-O2", "-Xcompiler=-fopenmp", "-arch=sm_86"]
    if host:
        command += ["-ccbin", host]
    run([*command, "-c", "generated.cu", "-o", "generated.o"], tmp_path, environment)
    run([fc, "-O2", "-fopenmp", "-ffree-line-length-none", "-fcheck=all", "-c",
         "interface.f90", "reference.f90", "driver.f90"], tmp_path, environment)
    run([fc, "-fopenmp", "interface.o", "reference.o", "driver.o", "generated.o", "-lstdc++",
         "-L" + str(Path(nvcc).resolve().parents[1] / "lib64"), "-lcudart", "-o", "run"], tmp_path, environment)
    result = subprocess.run([str(tmp_path / "run")], cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=90)
    log = result.stdout + result.stderr
    if "no CUDA-capable device" in log or "CUDA driver version is insufficient" in log:
        pytest.skip(log)
    assert result.returncode == 0, log
    assert "SECTIONS_NATIVE_PASS" in log
    assert log.count("FORT_OFFLOAD entry=") == 4, log
    decisions = "\n".join(line for line in log.splitlines() if line.startswith("FORT_OFFLOAD entry="))
    if mixed:
        assert log.count("mode=mixed gpu_units=2 cpu_units=2") == 3, log
        assert log.count("FORT_RUNTIME kernel") == 6, log
    elif policy == "auto":
        assert decisions.count("mode=native") == 4, log
        assert "FORT_RUNTIME upload" not in log
        assert "FORT_RUNTIME kernel" not in log
    else:
        assert decisions.count("mode=gpu") == 3, log
        if boundary:
            import re
            uploaded = sum(map(int, re.findall(r"FORT_RUNTIME upload bytes=(\d+)", log)))
            downloaded = sum(map(int, re.findall(r"FORT_RUNTIME download bytes=(\d+)", log)))
            # Two separate faces: read input and fully overwritten output.
            expected_faces = sum(2 * (17 + rep) * (7 + rep) * 8 for rep in (1, 2, 3))
            assert uploaded == expected_faces, log
            assert downloaded == expected_faces, log
