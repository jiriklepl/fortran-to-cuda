"""Original Fortran versus direct CUDA for reached and unreached real reads."""
import os
import subprocess
from pathlib import Path

import pytest

from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.tests.test_byte_scalar_transport import byte_fixture, issue
from compiler.tests.test_tile_private_cuda import RUNTIME, checked
from compiler.tests.test_tile_task_provider_cuda import source_runtime_object  # noqa: F401


@pytest.mark.cuda
@pytest.mark.parametrize("kind", [4, 8])
@pytest.mark.parametrize("mapping", ["thread", "cooperative"])
@pytest.mark.parametrize("storage_transport", [False, True], ids=["strict-fields", "storage-fields"])
def test_conditional_scalar_bytes_against_fortran(tmp_path, source_runtime_object, kind, mapping, storage_transport):  # noqa: F811
    nvcc, host, fc, runtime = source_runtime_object
    builder, nodes = byte_fixture(tmp_path, kind)
    if storage_transport:
        from compiler.frontend.source_device_tasks import source_device_task

        proof = source_device_task(builder.analysis, builder.entry, nodes, builder=builder, storage_transport=True)
    else:
        proof = issue(builder, nodes)
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof, mapping=mapping, access_envelopes=True)
    (tmp_path / "provider.cu").write_text(provider.cuda_source)
    element, iso = ("float", "c_float") if kind == 4 else ("double", "c_double")
    (tmp_path / "bridge.f90").write_text(f'''subroutine original_bridge(a,b,n,t,scale)bind(C)
use iso_c_binding
use stencil
implicit none
integer(c_int),value::n,t
real({iso}),intent(in)::a(-3:n+4),scale
real({iso}),intent(inout)::b(-3:n+4)
call advance(a,b,n,t,scale)
end subroutine
''')
    for name in ("source", "bridge"):
        checked([fc, "-O2", "-fopenmp", "-ffp-contract=off", "-J", str(tmp_path), "-I", str(tmp_path),
                 "-c", str(tmp_path / (name + ".f90")), "-o", str(tmp_path / (name + ".o"))], tmp_path, name)
    values = {("scalar", "n", None): "&n", ("scalar", "t", None): "&t", ("scalar", "scale", None): "scale",
              ("external_lower", "a", 1): "&lower", ("external_lower", "b", 1): "&lower",
              ("scratch_lower", "tmp", 1): "&sl", ("scratch_upper", "tmp", 1): "&su"}
    arguments = ",".join(values[p.role, p.binding.name, p.axis] for p in provider.parameters)
    ctypes = {"c_int": "int", "c_float": "float", "c_double": "double"}
    prototype = ",".join(["fort_scope_t", "fort_buffer_t", "fort_buffer_t",
                          *["const " + ctypes[p.kind] + "*" for p in provider.parameters], "fort_scope_external_report_v1*"])
    snan = "0x7f800001U" if kind == 4 else "0x7ff0000000000001ULL"
    integer = "std::uint32_t" if kind == 4 else "std::uint64_t"
    (tmp_path / "driver.cpp").write_text(f'''#include "scoped_runtime.h"
#include <cassert>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>
extern "C" int {provider.entry}({prototype});
extern "C" void original_bridge(const {element}*,{element}*,int,int,const {element}*);
void check(int status){{if(status){{std::fprintf(stderr,"%d %s\\n",status,fort_scope_error());std::abort();}}}}
int main(){{
 for(const bool undefined:{{false,true}})for(const int n:{{0,1,7,19}}){{
  const int t=3,lower=-3,sl=-1,su=t+2;const std::size_t extent=n+8;
  std::vector<{element}>a(extent),b(extent);
  for(int i=-3;i<=n+4;++i){{a[i+3]=static_cast<{element}>(17*i-3)/static_cast<{element}>(7);b[i+3]=static_cast<{element}>(500-i)/static_cast<{element}>(32);}}
  const auto initial_a=a;auto expected=b;
  std::unique_ptr<{element}> storage(new {element});
  if(n>1){{const {element} value=-0.625;std::memcpy(storage.get(),&value,sizeof(value));}}
  else if(!undefined){{const {integer} bits={snan};std::memcpy(storage.get(),&bits,sizeof(bits));}}
  const {element}* scale=n?storage.get():reinterpret_cast<const {element}*>(std::uintptr_t(1));
  std::feclearexcept(FE_ALL_EXCEPT);
  if(n)original_bridge(a.data(),expected.data(),n,t,scale);
  assert(!std::fetestexcept(FE_INVALID));
  fort_scope_t context=0;fort_buffer_t ha=0,hb=0;
  if(n){{check(fort_scope_create(0,&context));const std::int64_t lb=lower;
   fort_scope_layout la{{1,{('FORT_SCOPE_REAL32' if kind == 4 else 'FORT_SCOPE_REAL64')},{kind},a.data(),&extent,&lb,1}};
   fort_scope_layout lbview{{1,{('FORT_SCOPE_REAL32' if kind == 4 else 'FORT_SCOPE_REAL64')},{kind},b.data(),&extent,&lb,1}};
   check(fort_scope_register(context,1,1,&la,{0 if storage_transport else 1},&ha));check(fort_scope_register(context,2,1,&lbview,{0 if storage_transport else 1},&hb));}}
  fort_scope_external_report_v1 report{{}};check({provider.entry}(context,ha,hb,{arguments},&report));
  assert(!std::fetestexcept(FE_INVALID));
  if(n){{assert(report.started==1&&report.committed==1);fort_scope_access read{{}};read.flags=FORT_SCOPE_READ_ALL;
   {('assert(fort_scope_host_begin(context,hb,&read)==FORT_SCOPE_UNINITIALIZED);check(fort_scope_observe_v1(context,&hb,1));' if storage_transport else 'check(fort_scope_host_begin(context,hb,&read));check(fort_scope_host_end(context,hb));')}
   fort_scope_stats stats{{}};check(fort_scope_stats_get(context,&stats));
   assert(stats.launches==1&&stats.uploads==2&&stats.downloads==1);
   assert(stats.upload_bytes==(2*n+3)*sizeof({element})&&stats.download_bytes==n*sizeof({element}));
   std::printf("n=%d undefined=%d upload=%llu download=%llu\\n",n,int(undefined),
    static_cast<unsigned long long>(stats.upload_bytes),static_cast<unsigned long long>(stats.download_bytes));
   check(fort_scope_close(context));
  }}else{{assert(report.started==0&&report.committed==0);}}
  assert(a==initial_a);assert(std::memcmp(b.data(),expected.data(),extent*sizeof({element}))==0);
 }}
 std::puts("BYTE_SCALAR_FORTRAN_FIELDS_HALOS_LEDGER_AND_UNREACHED_READS_OK");
}}
''')
    includes = ["-I", str(RUNTIME), "-I", str(Path("compiler/emission/common/templates").resolve())]
    checked([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "--fmad=false", "--ftz=false",
             "--prec-div=true", "--prec-sqrt=true", *includes, "-c", str(tmp_path / "provider.cu"),
             "-o", str(tmp_path / "provider.o")], tmp_path, "provider")
    executable = tmp_path / "probe"
    checked([nvcc, "-std=c++17", "-ccbin", host, *includes, str(tmp_path / "driver.cpp"), str(tmp_path / "provider.o"),
             str(tmp_path / "source.o"), str(tmp_path / "bridge.o"), str(runtime), "-lgfortran", "-lgomp", "-o", str(executable)],
            tmp_path, "link")
    result = subprocess.run([str(executable)], cwd=tmp_path, env={**os.environ, "OMP_NUM_THREADS": "4"},
                            capture_output=True, text=True, timeout=60)
    (tmp_path / "run.stdout").write_text(result.stdout)
    (tmp_path / "run.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "BYTE_SCALAR_FORTRAN_FIELDS_HALOS_LEDGER_AND_UNREACHED_READS_OK" in result.stdout
