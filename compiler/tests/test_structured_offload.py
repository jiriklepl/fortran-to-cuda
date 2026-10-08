"""Generic host preparation, guarded arguments, and shared GPU intervals."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.emission import generate_sources, read_common_header
from compiler.emission.cuda.offload import generate_offload
from compiler.frontend import lower_file
from compiler.offload.config import OffloadConfig
from compiler.offload.profile import load_profile

SOURCE = """module prepared_case
contains
subroutine advance(a,b,configuration,n,scale,unused)
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:)
integer,intent(in)::configuration(:),n,unused
real(8),intent(in)::scale
integer::i,limit
real(8)::coefficient
limit=configuration(1)
coefficient=scale/2.0_8
if(limit>0) then
 do i=1,n
  a(i+1)=b(i+1)*coefficient+real(i,8)
 end do
end if
limit=configuration(2)
coefficient=scale/4.0_8
if(limit>0) then
 do i=1,n
  a(i+1)=a(i+1)+b(i+1)*coefficient
 end do
end if
end subroutine
end module
"""


def generate(tmp_path, source=SOURCE, policy="sections", collective=False, profile=None):
    path = tmp_path / "source.f90"
    path.write_text(source)
    function, plan = prepare_function(lower_file(path, "advance"), options=CompilerOptions(gpu_policy=policy))
    config = OffloadConfig(policy, profile, 4, collective)
    return function, generate_offload(function, plan, config), generate_sources(function, plan, offload_config=config)


def run(argv, directory, environment=None):
    result = subprocess.run(argv, cwd=directory, env=environment, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout + result.stderr


def test_generic_preparation_is_available_without_enabling_chunked_hybrid(tmp_path):
    _, emitted, _ = generate(tmp_path)
    report = emitted.report
    assert report["analysis"]["available"]
    assert report["supported_strategies"] == ["native", "sections", "auto"]
    assert report["analysis"]["chunk_axis"] is None
    assert [s.name for s in emitted.unused_scalars] == ["unused"]
    for policy in ("chunked", "hybrid"):
        _, rejected, _ = generate(tmp_path, policy=policy)
        assert not rejected.report["analysis"]["available"]


def test_immutable_array_bounds_are_public_and_queryable(tmp_path):
    _, emitted, _ = generate(tmp_path, SOURCE.replace("do i=1,n", "do i=1,min(n,configuration(2))"))
    assert emitted.report["analysis"]["available"]
    assert "configuration(2)" in json.dumps(emitted.report)


def test_conditional_logical_query_input_remains_native(tmp_path):
    source = SOURCE.replace("scale,unused)", "scale,unused,outer,inner)")
    source = source.replace("integer::i,limit", "logical,intent(in)::outer,inner\ninteger::i,limit")
    source = source.replace("limit=configuration(1)", "if(outer) then\nif(inner) then\nlimit=configuration(1)")
    source = source.replace("end subroutine", "end if\nend if\nend subroutine")
    _, emitted, _ = generate(tmp_path, source)
    assert not emitted.report["analysis"]["available"]
    assert "conditional LOGICAL" in emitted.report["analysis"]["reason"]


@pytest.mark.parametrize(
    ("replacement", "reason"),
    [
        (("limit=configuration(1)", "a(1)=1.0_8\nlimit=configuration(1)"), "write only local"),
        (("limit=configuration(1)", "limit=int(a(1))"), "storage written"),
        (("if(limit>0)", "if(coefficient>0.0_8)"), "INTEGER/LOGICAL"),
        (("limit=configuration(1)", "limit=int(scale)"), "numerical scalar setup"),
    ],
)
def test_unsupported_preparation_retains_native(tmp_path, replacement, reason):
    _, emitted, _ = generate(tmp_path, SOURCE.replace(*replacement))
    assert not emitted.report["analysis"]["available"]
    assert reason in emitted.report["analysis"]["reason"]


@pytest.mark.native
def test_preflight_does_not_execute_numerical_setup_and_checks_indices_and_overflow(tmp_path):
    cxx = shutil.which("g++")
    if not cxx:
        pytest.skip("requires C++")
    source = SOURCE.replace("limit=configuration(1)", "limit=configuration(n-n+1)")
    _, emitted, _ = generate(tmp_path, source)
    # Metadata is target-independent. Exercise it without creating a CUDA context.
    metadata = emitted.helpers.split("static void fort_structured_", 1)[0]
    name = emitted.helpers.split("static offload::Data ")[-1].split("_data(", 1)[0] + "_data"
    runtime = Path(__file__).resolve().parents[1] / "runtime/offload.hpp"
    (tmp_path / "check.cpp").write_text(f"""
#include <cassert>
#include <cfenv>
#include <climits>
#include <sys/mman.h>
#include "{runtime}"
using namespace generated_kernels;
{metadata}
int main() {{
 double a[8]={{}},b[8]={{}};
 int configuration[2]={{1,1}};
 void* page=mmap(nullptr,4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 assert(page!=MAP_FAILED);
 const auto& unused=*static_cast<int*>(page);
 const auto& numerical=*static_cast<double*>(page);
 std::feclearexcept(FE_ALL_EXCEPT);
 auto d={name}(a,8,b,8,configuration,2,4,numerical,unused);
 assert(d.valid && d.units.size()==2 && d.guarded_inputs_checked);
 assert(std::fetestexcept(FE_ALL_EXCEPT)==0);
 assert(offload::select(d,offload::Profile{{}},false).has_gpu());
 configuration[0]=configuration[1]=0;
 const auto& protected_bound=*static_cast<int*>(page);
 // This source's n-n index genuinely needs n, so a protected bound is tested
 // separately below. Invalid configuration length must never read its pointer.
 d={name}(a,8,b,8,configuration,0,4,numerical,unused);
 assert(!d.valid);
 bool valid=true;
 assert(offload::query_integer(static_cast<long long>(INT_MAX)+1,valid)==0 && !valid);
 valid=true; assert(offload::query_divide(INT_MIN,-1,valid)==0 && !valid);
 valid=true; assert(offload::query_divide(1,0,valid)==0 && !valid);
 (void)protected_bound;
 munmap(page,4096);
}}
""")
    run([cxx, "-std=c++17", "-O2", "-fopenmp", "check.cpp", "-o", "check"], tmp_path)
    run([str(tmp_path / "check")], tmp_path)


@pytest.mark.native
def test_guarded_path_does_not_read_inactive_bounds_or_unused_arguments(tmp_path):
    cxx = shutil.which("g++")
    if not cxx:
        pytest.skip("requires C++")
    _, emitted, _ = generate(tmp_path)
    metadata = emitted.helpers.split("static void fort_structured_", 1)[0]
    name = emitted.helpers.split("static offload::Data ")[-1].split("_data(", 1)[0] + "_data"
    runtime = Path(__file__).resolve().parents[1] / "runtime/offload.hpp"
    (tmp_path / "guard.cpp").write_text(f"""
#include <cassert>
#include <climits>
#include <sys/mman.h>
#include "{runtime}"
using namespace generated_kernels;
{metadata}
int main() {{
 double a[8]={{}},b[8]={{}}; int configuration[2]={{0,0}};
 void* page=mmap(nullptr,4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0);
 assert(page!=MAP_FAILED);
 const auto& protected_bound=*static_cast<int*>(page);
 auto d={name}(a,8,b,8,configuration,2,protected_bound,2.0,protected_bound);
 assert(!d.valid && d.units.empty());
 configuration[0]=1;
 d={name}(a,8,b,8,configuration,2,4,2.0,protected_bound);
 assert(d.valid && d.units.size()==1 && d.units[0].source_region==0);
 configuration[0]=0; configuration[1]=1;
 d={name}(a,8,b,8,configuration,2,4,2.0,protected_bound);
 assert(d.valid && d.units.size()==1 && d.units[0].source_region==1);
 munmap(page,4096);
}}
""")
    run([cxx, "-std=c++17", "-O2", "-fopenmp", "guard.cpp", "-o", "guard"], tmp_path)
    run([str(tmp_path / "guard")], tmp_path)


@pytest.mark.native
@pytest.mark.cuda
@pytest.mark.parametrize("collective", [False, True])
@pytest.mark.parametrize("policy", ["sections", "auto", "auto_mixed"])
def test_structured_workers_match_native_with_shared_transfers(tmp_path, collective, policy):
    nvcc = shutil.which(os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"))
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not fc:
        pytest.skip("requires CUDA and Fortran")
    mixed = policy == "auto_mixed"
    if mixed:
        policy = "auto"
    source = SOURCE
    if mixed:
        source = source.replace("limit=configuration(1)", "do i=1,1\na(i)=b(i)*scale\nend do\nlimit=configuration(1)")
        source = source.replace("end subroutine", "do i=n+2,n+2\na(i)=b(i)*scale\nend do\nend subroutine")
    profile = None
    if policy == "auto":
        profile = load_profile(
            Path(__file__).resolve().parents[2] / "benchmarks/results/gpu-strategies-20261006/calibration.json"
        )
        # A deterministic correctness control, excluded from performance evidence.
        profile["rates"].update(
            cpu_flops_per_second=1,
            gpu_flops_per_second=1e12,
            cpu_memory_bytes_per_second=1,
            gpu_memory_bytes_per_second=1e12,
        )
        if mixed:
            profile["rates"].update(
                cpu_flops_per_second=1e6, cpu_memory_bytes_per_second=1e12, launch_latency_seconds=0.002
            )
            for direction in ("h2d_pageable", "d2h_pageable"):
                profile["rates"][direction] = {"latency_seconds": 0.0001, "bandwidth_bytes_per_second": 1e12}
    function, emitted, generated = generate(tmp_path, source, policy=policy, collective=collective, profile=profile)
    assert emitted.report["analysis"]["available"]
    query = emitted.query_name
    (tmp_path / "common_functions.cuh").write_text(read_common_header())
    (tmp_path / "generated.cu").write_text(generated.cuda)
    (tmp_path / "interface.f90").write_text(generated.fortran)
    (tmp_path / "reference.f90").write_text(source.replace("module prepared_case", "module reference_case"))
    (tmp_path / "guard.cpp").write_text("""#include <sys/mman.h>
extern "C" void* protected_argument() { return mmap(nullptr,4096,PROT_NONE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0); }
""")
    start = "!$omp parallel num_threads(4)" if collective else ""
    stop = "!$omp end parallel" if collective else ""
    fallback = (
        "!$omp single\ncall reference(a,b,configuration,n,scale,0)\n!$omp end single"
        if collective
        else "call reference(a,b,configuration,n,scale,0)"
    )
    (tmp_path / "driver.f90").write_text(f"""program verify
use prepared_case,only:candidate=>advance,decide=>{query}
use reference_case,only:reference=>advance
use iso_c_binding
implicit none
interface
 function protected_argument() bind(C) result(address)
 import c_ptr
 type(c_ptr)::address
 end function
end interface
real(8),allocatable::a(:),b(:),expected(:)
integer,pointer::unused
integer::configuration(2),n,i,rep
real(8)::scale
call c_f_pointer(protected_argument(),unused)
do rep=1,5
 n={4096 if mixed else 7}+rep
 allocate(a(-3:n),b(-3:n),expected(-3:n))
 do i=-3,n
  b(i)=real(i*i+rep,8)
 end do
 a=-91.0_8-rep
 expected=a
 scale=real(rep+2,8)
 configuration=1
 if(rep==3) configuration(1)=0
 if(rep==4) configuration(2)=0
 if(rep==5) configuration=0
 call reference(expected,b,configuration,n,scale,0)
 {start}
 if(decide(a,b,configuration,n,scale,unused)) then
  call candidate(a,b,configuration,n,scale,unused)
 else
  {fallback}
 end if
 {stop}
 if(any(a/=expected)) error stop 'structured output or halo mismatch'
 deallocate(a,b,expected)
end do
print *, 'STRUCTURED_GPU_PASS'
end program
""")
    env = dict(os.environ, OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE", FORT_RUNTIME_TRACE="1", FORT_OFFLOAD_TRACE="1")
    host = shutil.which(os.environ.get("CUDAHOSTCXX", "g++-14"))
    flags = [nvcc, "-O2", "-std=c++17", "-Xcompiler=-fopenmp", "-arch=sm_86"]
    if host:
        flags += ["-ccbin", host]
    run([*flags, "-c", "generated.cu", "-o", "generated.o"], tmp_path, env)
    run(["g++", "-c", "guard.cpp", "-o", "guard.o"], tmp_path)
    run(
        [
            fc,
            "-O2",
            "-fopenmp",
            "-fcheck=all",
            "-ffree-line-length-none",
            "-c",
            "interface.f90",
            "reference.f90",
            "driver.f90",
        ],
        tmp_path,
        env,
    )
    run(
        [
            fc,
            "-fopenmp",
            "interface.o",
            "reference.o",
            "driver.o",
            "generated.o",
            "guard.o",
            "-lstdc++",
            "-L" + str(Path(nvcc).resolve().parents[1] / "lib64"),
            "-lcudart",
            "-o",
            "verify",
        ],
        tmp_path,
        env,
    )
    log = run([str(tmp_path / "verify")], tmp_path, env)
    (tmp_path / "run.log").write_text(log)
    (tmp_path / "compiler-report.json").write_text(json.dumps(emitted.report, indent=2) + "\n")
    assert "STRUCTURED_GPU_PASS" in log
    assert log.count("FORT_RUNTIME kernel\n") == 6, log
    assert log.count("FORT_RUNTIME download") == 4, log
    assert sum(map(int, re.findall(r"FORT_RUNTIME upload bytes=(\d+)", log))) == (229504 if mixed else 520), log
    assert sum(map(int, re.findall(r"FORT_RUNTIME download bytes=(\d+)", log))) == (131152 if mixed else 304), log
    if mixed:
        assert log.count("mode=mixed") == 4, log
    assert "mode=native" in log
