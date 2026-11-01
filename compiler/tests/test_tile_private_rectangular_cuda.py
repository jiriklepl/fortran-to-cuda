"""Full rectangular point distribution against the original Fortran oracle."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.tests.test_tile_private import SOURCE_3D
from compiler.tests.test_tile_private_cuda import RUNTIME, checked
from compiler.tests.test_tile_private_team import team_fixture


@pytest.fixture(scope="module")
def runtime_object(tmp_path_factory):
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not all((nvcc, host, fc)):
        pytest.skip("CUDA and original GNU Fortran toolchain required")
    directory = tmp_path_factory.mktemp("rectangular-runtime")
    target = directory / "runtime.o"
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp",
             "-I", str(RUNTIME), "-c", str(RUNTIME / "scoped_runtime.cu"), "-o", str(target)], directory, "runtime")
    return nvcc, host, fc, target


@pytest.fixture(scope="module", params=[(kind, boxes) for kind in (4, 8) for boxes in (False, True)])
def executable(request, tmp_path_factory, runtime_object):
    nvcc, host, fc, runtime = runtime_object
    kind, boxes = request.param
    directory = tmp_path_factory.mktemp(f"rectangular-real{kind}-boxes{int(boxes)}")
    source = SOURCE_3D.replace("real(8)", f"real({kind})").replace("3.0_8", f"3.0_{kind}")
    source = source.replace("a(:,:,:)", "a(-3:,-2:,-1:)").replace("b(:,:,:)", "b(-3:,-2:,-1:)")
    source = source.replace("b(i,j,k)=b(i,j,k)+(tmp(li+1,lj,lk)-tmp(li-1,lj,lk))", """if(mod(i+j+k,2)==0)then
      b(i,j,k)=b(i,j,k)+(tmp(li+1,lj,lk)-tmp(li-1,lj,lk))
      else
      b(i,j,k)=b(i,j,k)-(tmp(li+1,lj,lk)-tmp(li-1,lj,lk))
      endif""")
    _, analysis, routine, nodes = team_fixture(directory, source)
    issued = source_device_task(analysis, routine, nodes)
    provider = emit_tile_task_provider(issued.team, readable_scalars=issued, access_envelopes=boxes)
    assert [binding.name for binding, _ in issued.team.fields] == ["a", "b"]
    (directory / "provider.cu").write_text(provider.cuda_source)
    (directory / "provider.json").write_text(json.dumps(provider.public(), indent=2))
    iso, element = ("c_float", "float") if kind == 4 else ("c_double", "double")
    (directory / "bridge.f90").write_text(f"""subroutine original_bridge(a,b,nx,ny,nz,tx,ty,tz)bind(C)
use iso_c_binding
use stencil
implicit none
integer(c_int),value::nx,ny,nz,tx,ty,tz
real({iso}),intent(in)::a(-3:nx+4,-2:ny+3,-1:nz+2)
real({iso}),intent(inout)::b(-3:nx+4,-2:ny+3,-1:nz+2)
call advance(a,b,nx,ny,nz,tx,ty,tz)
end subroutine
""")
    for name in ("source", "bridge"):
        checked([fc, "-O2", "-fopenmp", "-ffp-contract=off", "-fcheck=bounds", "-J", str(directory),
                 "-I", str(directory), "-c", str(directory / (name + ".f90")),
                 "-o", str(directory / (name + ".o"))], directory, name)
    arguments = []
    for parameter in provider.parameters:
        if parameter.role == "scalar":
            arguments.append("&" + parameter.binding.name)
        elif parameter.role == "external_lower":
            arguments.append(f"&lower[{parameter.axis - 1}]")
        else:
            vector = "scratch_lower" if parameter.role == "scratch_lower" else "scratch_upper"
            arguments.append(f"&{vector}[{parameter.axis - 1}]")
    prototype = ",".join(["fort_scope_t", "fort_buffer_t", "fort_buffer_t",
                          *["const int*" for _ in provider.parameters], "fort_scope_external_report_v1*"])
    (directory / "driver.cpp").write_text(f"""#include "scoped_runtime.h"
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" int {provider.entry}({prototype});
extern "C" void original_bridge(const {element}*,{element}*,int,int,int,int,int,int);
static void check(int s){{if(s){{std::fprintf(stderr,"%d %s\\n",s,fort_scope_error());std::abort();}}}}
int main(int argc,char**){{
 const bool hidden=argc>1;
 const int shapes[][6]={{{{0,3,2,2,3,2}},{{3,0,2,2,3,2}},{{3,2,0,2,3,2}},{{1,1,1,2,3,4}},
   {{7,5,3,2,3,2}},{{33,7,5,9,4,3}},{{3,33,7,2,9,4}},{{7,3,33,4,2,9}}}};
 for(const auto& shape:shapes){{
  const int nx=shape[0],ny=shape[1],nz=shape[2],tx=shape[3],ty=shape[4],tz=shape[5];
  const int scratch_lower[3]={{-1,-1,-1}},scratch_upper[3]={{tx+2,ty+2,tz+2}};
  const std::int64_t lower64[3]={{-3,-2,-1}};
  const int lower[3]={{-3,-2,-1}};
  const std::size_t extents[3]={{std::size_t(nx+8),std::size_t(ny+6),std::size_t(nz+4)}};
  const std::size_t cells=extents[0]*extents[1]*extents[2],bytes=cells*sizeof({element});
  std::vector<{element}>a(cells),b(cells);
  for(std::size_t i=0;i<cells;++i){{a[i]={element}(int(i%101)-50)/{element}(7);b[i]={element}(i%313)/{element}(32);}}
  const auto initial_a=a;auto expected=b;
  original_bridge(a.data(),expected.data(),nx,ny,nz,tx,ty,tz);
  const bool reached=nx&&ny&&nz;
  fort_scope_t context=0;fort_buffer_t ha=0,hb=0;
  if(reached){{check(fort_scope_create(0,&context));
   fort_scope_layout la{{3,{('FORT_SCOPE_REAL32' if kind == 4 else 'FORT_SCOPE_REAL64')},{kind},a.data(),extents,lower64,1}};
   fort_scope_layout lb{{3,{('FORT_SCOPE_REAL32' if kind == 4 else 'FORT_SCOPE_REAL64')},{kind},b.data(),extents,lower64,1}};
   check(fort_scope_register(context,1,1,&la,1,&ha));check(fort_scope_register(context,2,1,&lb,1,&hb));}}
  fort_scope_external_report_v1 report{{}};
  const int status={provider.entry}(context,ha,hb,{','.join(arguments)},&report);
  const bool mixed={str(boxes).lower()}&&!hidden&&reached&&nx>=7;
  if(mixed){{
   check(status);
   fort_scope_access readonly{{}};readonly.flags=FORT_SCOPE_READ_ALL;
   check(fort_scope_host_begin(context,ha,&readonly));
   volatile double sum=0;for(const auto value:a)sum=sum+double(value);
   assert(sum==sum);check(fort_scope_host_end(context,ha));
   // Completely overwrite five interior cells and one halo cell on the CPU.
   // Neither operation needs to read the old device values.
   const std::size_t lo[3]={{4,3,2}},hi[3]={{9,4,3}},halo_lo[3]={{0,0,0}},halo_hi[3]={{1,1,1}};
   fort_scope_section interior{{lo,hi}},halo{{halo_lo,halo_hi}};
   fort_scope_access write{{}};write.write_count=write.overwrite_count=1;
   write.writes=write.overwrites=&interior;
   check(fort_scope_host_begin(context,hb,&write));
   for(std::size_t i=0;i<5;++i){{
    const auto index=4+i+extents[0]*(3+extents[1]*2);
    b[index]=expected[index]={element}(700+i);
   }}
   check(fort_scope_host_end(context,hb));
   write.writes=write.overwrites=&halo;
   check(fort_scope_host_begin(context,hb,&write));b[0]=expected[0]={element}(12345);
   check(fort_scope_host_end(context,hb));
   original_bridge(a.data(),expected.data(),nx,ny,nz,tx,ty,tz);
   check({provider.entry}(context,ha,hb,{','.join(arguments)},&report));
  }}
  if(hidden&&reached){{assert(status&&report.started==0&&report.committed==0);
   fort_scope_stats stats{{}};check(fort_scope_stats_get(context,&stats));
   assert(stats.launches==0&&stats.uploads==0&&stats.downloads==0);
   original_bridge(a.data(),b.data(),nx,ny,nz,tx,ty,tz);
  }}
  else{{check(status);assert(report.status==FORT_SCOPE_OK);
   if(reached){{assert(report.started==1&&report.committed==1);
    fort_scope_access read{{}};read.flags=FORT_SCOPE_READ_ALL;
    check(fort_scope_host_begin(context,hb,&read));check(fort_scope_host_end(context,hb));
    fort_scope_stats stats{{}};check(fort_scope_stats_get(context,&stats));
    const std::size_t input_bytes=std::size_t(nx+3)*ny*nz*sizeof({element});
    const std::size_t output_bytes=std::size_t(nx)*ny*nz*sizeof({element});
    assert(stats.launches==std::uint64_t(mixed?2:1)&&stats.uploads==std::uint64_t(mixed?3:2));
    assert(stats.upload_bytes==({str(boxes).lower()}?input_bytes+output_bytes+(mixed?5*sizeof({element}):0):2*bytes));
    assert(stats.downloads==1&&stats.download_bytes==({str(boxes).lower()}?output_bytes:bytes));
   }}else assert(report.started==0&&report.committed==0);
  }}
  if(context)check(fort_scope_close(context));
  assert(std::memcmp(a.data(),initial_a.data(),bytes)==0);
  assert(std::memcmp(b.data(),expected.data(),bytes)==0);
 }}
 std::puts("RECTANGULAR_ORIGINAL_FULL_FIELDS_HALOS_OK");
}}
""")
    includes = ["-I", str(RUNTIME), "-I", str(Path("compiler/emission/common/templates").resolve())]
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "--fmad=false", "--ftz=false",
             "--prec-div=true", "--prec-sqrt=true", *includes, "-c", str(directory / "provider.cu"),
             "-o", str(directory / "provider.o")], directory, "provider")
    checked([nvcc, "-std=c++17", "-ccbin", host, *includes, str(directory / "driver.cpp"),
             str(directory / "provider.o"), str(runtime), str(directory / "source.o"),
             str(directory / "bridge.o"), "-lgfortran", "-lgomp", "-o", str(directory / "run")], directory, "link")
    return directory / "run"


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True])
def test_rectangular_points_preserve_original_bitwise_fields_and_halos(executable, hidden):
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if hidden:
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run([str(executable), *(["hidden"] if hidden else [])],
                            capture_output=True, text=True, env=env, timeout=30)
    name = "hidden" if hidden else "enabled"
    (executable.parent / (name + ".stdout")).write_text(result.stdout)
    (executable.parent / (name + ".stderr")).write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "RECTANGULAR_ORIGINAL_FULL_FIELDS_HALOS_OK" in result.stdout
