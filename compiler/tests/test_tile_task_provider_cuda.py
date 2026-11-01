"""Real source-team transaction against the original Fortran full-array oracle."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission.cuda.tile_task_provider import BATCH_TARGET_BYTES, emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.tests.test_tile_private_cuda import RUNTIME, checked
from compiler.tests.test_tile_private_emission import original_source
from compiler.tests.test_tile_private_team import team_fixture


@pytest.fixture(scope="module")
def source_runtime_object(tmp_path_factory):
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++"), shutil.which("gfortran")
    if not nvcc or not host or not fc:
        pytest.skip("CUDA toolkit and original Fortran compiler are required")
    directory = tmp_path_factory.mktemp("tile-source-runtime")
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp", "-DFORT_SCOPE_TEST_FAULTS",
             "-I", str(RUNTIME), "-c", str(RUNTIME / "scoped_runtime.cu"), "-o", str(directory / "runtime.o")], directory, "runtime")
    return nvcc, host, fc, directory / "runtime.o"


@pytest.fixture(scope="module", params=[(kind, mapping, padding) for kind in (4, 8) for mapping in ("thread", "cooperative") for padding in (0, 196608, 524288)])
def provider_executable(request, tmp_path_factory, source_runtime_object):
    nvcc, host, fc, runtime = source_runtime_object
    kind, mapping, padding = request.param
    directory = tmp_path_factory.mktemp(f"tile-source-real{kind}-{mapping}-pad{padding}")
    source = original_source(kind, branch=True).replace("tmp(-1:t+2)", f"tmp(-1:t+2+{padding})")
    first = source.index("!$omp do ")
    last = source.index("!$omp end do", first) + len("!$omp end do\n")
    source = source[:last] + source[first:last] + source[last:]
    _, analysis, routine, nodes = team_fixture(directory, source)
    issued = source_device_task(analysis, routine, nodes)
    provider = emit_tile_task_provider(issued.team, mapping=mapping, readable_scalars=issued)
    assert len(issued.team.units) == 2
    element = "float" if kind == 4 else "double"
    iso = "c_float" if kind == 4 else "c_double"
    (directory / "provider.cu").write_text(provider.cuda_source)
    (directory / "provider.json").write_text(__import__("json").dumps(provider.public(), sort_keys=True, indent=2))
    (directory / "bridge.f90").write_text(f'''subroutine original_bridge(a,b,n,t)bind(C)
use iso_c_binding
use stencil
implicit none
integer(c_int),value::n,t
real({iso}),intent(in)::a(-3:n+4)
real({iso}),intent(inout)::b(-3:n+4)
call advance(a,b,n,t)
end subroutine
''')
    checked([fc, "-O2", "-fopenmp", "-ffp-contract=off", "-J", str(directory), "-c", str(directory / "source.f90"),
             "-o", str(directory / "original.o")], directory, "original")
    checked([fc, "-O2", "-fopenmp", "-ffp-contract=off", "-I", str(directory), "-c", str(directory / "bridge.f90"),
             "-o", str(directory / "bridge.o")], directory, "bridge")
    values = {("scalar", "n", None): "&n", ("scalar", "t", None): "&t",
              ("external_lower", "a", 1): "&lower", ("external_lower", "b", 1): "&lower",
              ("scratch_lower", "tmp", 1): "&scratch_lower", ("scratch_upper", "tmp", 1): "&scratch_upper"}
    parameters = ",".join(values[p.role, p.binding.name, p.axis] for p in provider.parameters)
    prototype = ",".join(["fort_scope_t", "fort_buffer_t", "fort_buffer_t", *["const int*" for _ in provider.parameters], "fort_scope_external_report_v1*"])
    (directory / "driver.cpp").write_text(f'''#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" int {provider.entry}({prototype});
extern "C" void original_bridge(const {element}*,{element}*,int,int);
static int fault_launch=0,attempted_launches=0;
extern "C" cudaError_t __real___cudaLaunchKernel(cudaKernel_t,dim3,dim3,void**,std::size_t,cudaStream_t);
extern "C" cudaError_t __wrap___cudaLaunchKernel(cudaKernel_t f,dim3 grid,dim3 block,void** args,std::size_t bytes,cudaStream_t stream){{
 ++attempted_launches;if(fault_launch&&attempted_launches==fault_launch)grid.x=0;
 return __real___cudaLaunchKernel(f,grid,block,args,bytes,stream);
}}
static void check(int status){{if(status){{std::fprintf(stderr,"status=%d %s\\n",status,fort_scope_error());std::abort();}}}}
int main(int argc,char**argv){{
 const bool budget=argc>1&&!std::strcmp(argv[1],"budget"),hidden=argc>1&&!std::strcmp(argv[1],"hidden");
 const bool bounds=argc>1&&!std::strcmp(argv[1],"bounds"),fault=argc>1&&!std::strcmp(argv[1],"fault");
 for(const int n:{{0,1,7,13,19}}){{
  const int t=3,lower=-3,scratch_lower=-1,scratch_upper=bounds?t+1:t+2+{padding};
  const std::size_t extent=n+8,tiles=n?(n-1)/t+1:0,stride=((t+4+{padding})*sizeof({element})+7)&~std::size_t(7);
  const std::size_t batch=std::min(tiles,std::max<std::size_t>(1,{BATCH_TARGET_BYTES}/stride));
  const std::size_t launches=tiles?2*(tiles/batch+(tiles%batch!=0)):0,scratch_bytes=stride*batch;
  std::vector<{element}>a(extent),b(extent);
  for(int i=-3;i<=n+4;++i){{a[i+3]=static_cast<{element}>(17*i-3)/static_cast<{element}>(7);b[i+3]=static_cast<{element}>(500-i)/static_cast<{element}>(32);}}
  const auto initial_a=a,initial_b=b;auto expected=b;
  original_bridge(a.data(),expected.data(),n,t);
  fort_scope_t context=0;fort_buffer_t ha=0,hb=0;
  if(n){{check(fort_scope_create(0,&context));check(fort_scope_set_device_budget(context,budget?1:2*extent*sizeof({element})+scratch_bytes));
   const std::int64_t original_lower=lower;
   fort_scope_layout la{{1,{('FORT_SCOPE_REAL32' if kind == 4 else 'FORT_SCOPE_REAL64')},{kind},a.data(),&extent,&original_lower,1}};
   fort_scope_layout lb{{1,{('FORT_SCOPE_REAL32' if kind == 4 else 'FORT_SCOPE_REAL64')},{kind},b.data(),&extent,&original_lower,1}};
   check(fort_scope_register(context,1,1,&la,1,&ha));check(fort_scope_register(context,2,1,&lb,1,&hb));
  }}
  fort_scope_external_report_v1 report{{}};
  fault_launch=fault?2:0;attempted_launches=0;
  const int status={provider.entry}(context,ha,hb,{parameters},&report);
  assert(report.version==1&&report.status==status);
  if(!n){{assert(status==0&&report.started==0&&report.committed==0);assert(a==initial_a&&b==initial_b);continue;}}
  if(fault){{assert(status==FORT_SCOPE_EXECUTION&&report.started==1&&report.committed==0);
   assert(attempted_launches==2);fort_scope_stats stats{{}};
   assert(fort_scope_stats_get(context,&stats)==FORT_SCOPE_EXECUTION);
   assert(a==initial_a&&b==initial_b);assert(fort_scope_close(context)==FORT_SCOPE_EXECUTION);check(fort_scope_abandon(context));continue;}}
  if(budget||hidden||bounds){{assert(status==FORT_SCOPE_RESOURCE&&report.started==0&&report.committed==0);
   fort_scope_stats stats{{}};check(fort_scope_stats_get(context,&stats));assert(stats.launches==0&&stats.uploads==0&&stats.downloads==0);
   original_bridge(a.data(),b.data(),n,t);check(fort_scope_close(context));
  }}else{{check(status);assert(report.started==1&&report.committed==1);
   fort_scope_scratch_stats_v1 scratch{{}};check(fort_scope_scratch_stats_get_v1(context,&scratch));
   assert(scratch.active==0&&scratch.active_bytes==0&&scratch.acquisitions==1&&scratch.releases==1);
   assert(scratch.peak_scratch_bytes==scratch_bytes&&scratch.peak_total_device_bytes<=2*extent*sizeof({element})+scratch_bytes);
   fort_scope_access read{{}};read.flags=FORT_SCOPE_READ_ALL;check(fort_scope_host_begin(context,hb,&read));check(fort_scope_host_end(context,hb));
   fort_scope_stats stats{{}};check(fort_scope_stats_get(context,&stats));assert(stats.launches==launches);
   assert(stats.uploads==2&&stats.upload_bytes==2*extent*sizeof({element}));assert(stats.downloads==1&&stats.download_bytes==extent*sizeof({element}));
   check(fort_scope_close(context));}}
  assert(a==initial_a);assert(std::memcmp(b.data(),expected.data(),extent*sizeof({element}))==0);
 }}
 std::puts("SOURCE_TEAM_FULL_FIELDS_HALOS_AND_TRANSACTION_OK");
}}
''')
    includes = ["-I", str(RUNTIME), "-I", str(Path("compiler/emission/common/templates").resolve())]
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "--fmad=false", "--ftz=false",
             "--prec-div=true", "--prec-sqrt=true", *includes, "-c", str(directory / "provider.cu"),
             "-o", str(directory / "provider.o")], directory, "provider")
    checked([nvcc, "-std=c++17", "-ccbin", host, *includes, str(directory / "driver.cpp"), str(directory / "provider.o"),
             str(runtime), str(directory / "original.o"), str(directory / "bridge.o"), "-lgfortran", "-lgomp",
             "-Xlinker=--wrap=__cudaLaunchKernel", "-o", str(directory / "run")], directory, "link")
    return directory / "run"


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["full", "budget", "hidden", "bounds", "fault"])
def test_issued_source_provider_preserves_original_fields_and_transaction_failure_rules(provider_executable, mode):
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
           "OMP_STACKSIZE": "64M", "FORT_RUNTIME_TRACE": "1", "FORT_SCOPE_TRACE": "1"}
    if mode == "hidden":
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([str(provider_executable), mode], capture_output=True, text=True, env=env, timeout=30)
    (provider_executable.parent / (mode + ".stdout")).write_text(result.stdout)
    (provider_executable.parent / (mode + ".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SOURCE_TEAM_FULL_FIELDS_HALOS_AND_TRANSACTION_OK" in result.stdout
    if mode == "fault":
        # Poisoned contexts reject ordinary API observations. Retain the raw
        # completed child trace and check its actual accepted prefix instead.
        contexts = {}
        for line in result.stderr.splitlines():
            words = line.split()
            if len(words) < 2 or words[0] != "FORT_SCOPED" or words[1] not in {
                "create", "upload", "download", "launch", "external_commit"
            }:
                continue
            fields = dict(word.split("=", 1) for word in words[2:])
            context = fields["context"]
            if words[1] == "create":
                assert context not in contexts
                contexts[context] = []
            contexts[context].append(words[1])
        assert len(contexts) == 4
        for events in contexts.values():
            assert events == ["create", "upload", "upload", "launch"]
