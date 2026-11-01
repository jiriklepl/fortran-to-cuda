"""Actual original-Fortran equivalence and runtime-leased indexed tile scratch."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission.cuda.scoped_tile_private import support_code
from compiler.emission.cuda.tile_private import (
    generate_cooperative_kernel,
    generate_cooperative_launch,
    generate_kernel,
    generate_launch,
)
from compiler.frontend.lowering import lower_tile_private
from compiler.frontend.tile_private import prove_cooperative_tile_private, prove_tile_private
from compiler.tests.test_tile_private import fixture
from compiler.tests.test_tile_private_emission import original_arguments, original_source

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


def checked(command, directory, name, timeout=120):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    (directory / (name + ".stdout")).write_text(result.stdout)
    (directory / (name + ".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def tile_runtime_object(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fc = shutil.which("gfortran")
    if not nvcc or not host or not fc:
        pytest.skip("CUDA toolkit and gfortran required")
    directory = tmp_path_factory.mktemp("tile-private-runtime")
    target = directory / "runtime.o"
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-I", str(RUNTIME),
             "-c", str(RUNTIME / "scoped_runtime.cu"), "-o", str(target)], directory, "runtime")
    return nvcc, host, fc, target


@pytest.fixture(scope="module", params=[(kind, mapping) for kind in (4, 8) for mapping in ("thread", "cooperative")])
def tile_cuda_executable(request, tmp_path_factory, tile_runtime_object):
    nvcc, host, fc, runtime = tile_runtime_object
    kind, mapping = request.param
    directory = tmp_path_factory.mktemp(f"tile-private-cuda-real{kind}-{mapping}")
    _, analysis, routine, node = fixture(directory, original_source(kind, branch=True))
    proof = prove_tile_private(analysis, routine, node)
    lowered = lower_tile_private(proof)
    element = "float" if kind == 4 else "double"
    arguments = original_arguments(lowered)
    arguments = [value.replace("a.data()", f"static_cast<{element}*>(access.device(ha))")
                 .replace("b.data()", f"static_cast<{element}*>(access.device(hb))")
                 .replace("a.size()", "extent").replace("b.size()", "extent") for value in arguments]
    arguments += ["1", "t", "tiles", "static_cast<unsigned char*>(scratch)", "stride", "0", "-1", "t+4"]
    if mapping == "cooperative":
        cooperative = prove_cooperative_tile_private(proof)
        launch_lines = generate_cooperative_launch(
            lowered.region, cooperative, arguments, context="context",
            stream="static_cast<cudaStream_t>(stream)", first="first", count="count")
        kernel_lines = generate_cooperative_kernel(lowered.region, cooperative, addressing=lowered.addressing)
    else:
        launch_lines = generate_launch(lowered.region, arguments, context="context",
                                       stream="static_cast<cudaStream_t>(stream)", first="first", count="count")
        kernel_lines = generate_kernel(lowered.region, addressing=lowered.addressing)
    launch = "\n".join(launch_lines)
    kernels = "\n".join(kernel_lines)
    bridge = directory / "bridge.f90"
    bridge.write_text(f'''subroutine original_bridge(a,b,n,t)bind(C)
use iso_c_binding
use stencil
implicit none
integer(c_int),value::n,t
real(c_float{'' if kind == 4 else '64'}),intent(in)::a(-3:n+4)
real(c_float{'' if kind == 4 else '64'}),intent(inout)::b(-3:n+4)
call advance(a,b,n,t)
end subroutine
'''.replace("c_float64", "c_double"))
    checked([fc, "-O2", "-fopenmp", "-ffp-contract=off", "-J", str(directory), "-c",
             str(directory / "source.f90"), "-o", str(directory / "original.o")], directory, "original")
    checked([fc, "-O2", "-fopenmp", "-ffp-contract=off", "-I", str(directory), "-c", str(bridge),
             "-o", str(directory / "bridge.o")], directory, "bridge")
    code = directory / "probe.cu"
    code.write_text(f'''#include "scoped_entry.hpp"
#include "{Path('compiler/emission/common/templates/common_functions.cuh').resolve()}"
#include "numeric.hpp"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <vector>
using generated_kernels::indexing::F_IDX;
extern "C" void original_bridge(const {element}*,{element}*,int,int);
static void check(int s){{if(s){{std::fprintf(stderr,"status=%d %s\\n",s,fort_scope_error());std::abort();}}}}
{chr(10).join(support_code())}
{kernels}
int main(int argc,char**argv){{
 const bool budget=argc>1&&!std::strcmp(argv[1],"budget");
 const bool hidden=argc>1&&!std::strcmp(argv[1],"hidden");
 const bool fault=argc>1&&!std::strcmp(argv[1],"fault");
 for(int n:{{0,1,7,13,19}}){{
  const int t=3;const std::size_t extent=n+8,tiles=n?(n-1)/t+1:0;
  const std::size_t stride=(static_cast<std::size_t>(t+4)*sizeof({element})+7)&~std::size_t(7);
  std::vector<{element}>a(extent),b(extent);
  for(int i=-3;i<=n+4;++i){{a[i+3]=static_cast<{element}>(17*i-3)/static_cast<{element}>(7);b[i+3]=static_cast<{element}>(500-i)/static_cast<{element}>(32);}}
  const auto initial_a=a,initial_b=b;auto expected=b;
  original_bridge(a.data(),expected.data(),n,t);
  fort_scope_t context=0;fort_buffer_t ha=0,hb=0;
  if(n){{
   check(fort_scope_create(0,&context));
   check(fort_scope_set_device_budget(context,budget?1:2*extent*sizeof({element})+3*stride));
   const std::int64_t lower=-3;
   fort_scope_layout la{{1,{('FORT_SCOPE_REAL32' if kind == 4 else 'FORT_SCOPE_REAL64')},{kind},a.data(),&extent,&lower,1}};
   fort_scope_layout lb{{1,{('FORT_SCOPE_REAL32' if kind == 4 else 'FORT_SCOPE_REAL64')},{kind},b.data(),&extent,&lower,1}};
   check(fort_scope_register(context,1,1,&la,1,&ha));check(fort_scope_register(context,2,1,&lb,1,&hb));
  }}
  fort_scoped::AccessBatch<2> access(context);
  fort_scope_access read{{}},readwrite{{}};read.flags=FORT_SCOPE_READ_ALL;
  readwrite.flags=FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL;
  if(n){{check(access.add(ha,read));check(access.add(hb,readwrite));}}
  int descriptors=0,started=0,accepted=0,native_fallbacks=0;
  auto prepare=[&](std::size_t total,std::size_t& batch,std::size_t& bytes){{
   ++descriptors;assert(total==tiles);batch=total<3?total:3;bytes=stride;return FORT_SCOPE_OK;}};
  auto submit=[&](std::size_t first,std::size_t count,void* scratch,void* stream)->int{{
   auto launch=[&](){{{launch}}};
   const int status=launch();if(status)return status;++accepted;
   return fault?FORT_SCOPE_RESOURCE:FORT_SCOPE_OK;
  }};
  const int status=fort_scoped::tile_private_submit(context,access,tiles,false,prepare,submit,[&](){{++started;}});
  if(!n){{assert(status==0&&descriptors==0&&started==0&&accepted==0);}}
  else if(budget||hidden){{
   assert(status==FORT_SCOPE_RESOURCE&&started==0&&accepted==0);
   fort_scope_stats s{{}};check(fort_scope_stats_get(context,&s));
   assert(s.launches==0&&s.uploads==0&&s.downloads==0);
   original_bridge(a.data(),b.data(),n,t);++native_fallbacks;
   check(fort_scope_close(context));
  }}else if(fault){{
   assert(status==FORT_SCOPE_EXECUTION&&started==1&&accepted==1&&native_fallbacks==0);
   assert(a==initial_a&&b==initial_b);
   assert(fort_scope_close(context)==FORT_SCOPE_EXECUTION);
   check(fort_scope_abandon(context));continue;
  }}else{{
   check(status);assert(started==1&&accepted==static_cast<int>((tiles+2)/3));
   fort_scope_scratch_stats_v1 scratch{{}};check(fort_scope_scratch_stats_get_v1(context,&scratch));
   assert(scratch.active==0&&scratch.active_bytes==0&&scratch.acquisitions==1&&scratch.releases==1);
   assert(scratch.peak_scratch_bytes==(tiles<3?tiles:3)*stride);
   assert(scratch.peak_total_device_bytes<=2*extent*sizeof({element})+3*stride);
   check(fort_scope_host_begin(context,hb,&read));check(fort_scope_host_end(context,hb));
   fort_scope_stats s{{}};check(fort_scope_stats_get(context,&s));
   assert(s.launches==static_cast<std::uint64_t>(accepted));
   assert(s.uploads==2&&s.upload_bytes==2*extent*sizeof({element}));
   assert(s.downloads==1&&s.download_bytes==extent*sizeof({element}));
   check(fort_scope_close(context));
  }}
  assert(a==initial_a);assert(std::memcmp(b.data(),expected.data(),extent*sizeof({element}))==0);
 }}
 std::puts("ORIGINAL_FULL_FIELDS_AND_HALOS_OK");
}}
''')
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "--fmad=false", "--ftz=false",
             "--prec-div=true", "--prec-sqrt=true", "-I", str(RUNTIME), "-c", str(code),
             "-o", str(directory / "probe.o")], directory, "probe")
    executable = directory / "run"
    checked([nvcc, "-ccbin", host, str(directory / "probe.o"), str(runtime), str(directory / "original.o"),
             str(directory / "bridge.o"), "-lgfortran", "-lgomp", "-o", str(executable)], directory, "link")
    return executable


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["full", "budget", "hidden", "fault"])
def test_source_lowered_tiles_use_bounded_scratch_and_original_fallback(tile_cuda_executable, mode):
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
           "FORT_RUNTIME_TRACE": "1", "FORT_SCOPE_TRACE": "1"}
    if mode == "hidden":
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([str(tile_cuda_executable), mode], capture_output=True, text=True, env=env, timeout=30)
    directory = tile_cuda_executable.parent
    (directory / (mode + ".stdout")).write_text(result.stdout)
    (directory / (mode + ".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ORIGINAL_FULL_FIELDS_AND_HALOS_OK" in result.stdout
