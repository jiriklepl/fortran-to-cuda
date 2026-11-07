"""Batched physical copies preserve source-only values and undefined holes."""
import json
import os
import shutil
import subprocess

import pytest

from compiler.numerical_contract import cuda_compile_options
from compiler.tests.test_scoped_compact_cuda import CLIENT as COMPACT_CLIENT
from compiler.tests.test_scoped_runtime import RUNTIME

# Reuse the existing deliberately synthetic dispatch rates. They select the
# physical transport for correctness checks and are never hardware estimates.
CONFIGURE = COMPACT_CLIENT[COMPACT_CLIENT.index("void configure("):COMPACT_CLIENT.index("fort_scope_stats stats(")]
CLIENT = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
void check(int status){if(status){std::fprintf(stderr,"%d %s\n",status,fort_scope_error());std::abort();}}
''' + CONFIGURE + r'''
constexpr std::size_t nx=34,ny=9,nz=5,n=nx*ny*nz,payload=4*ny*nz*8;
__host__ __device__ bool touched(std::size_t i){
 const auto x=i%nx,y=i/nx%ny,z=i/(nx*ny);
 return (x==1||x==7||x==13||x==19)&&y%2==0&&z>0&&z+1<nz;
}
__global__ void original(double* data,int phase){
 const std::size_t i=blockIdx.x*blockDim.x+threadIdx.x;
 if(i<n&&touched(i)){if(phase==0)data[i]+=10;else data[i]*=2;}
}
int submit(void* phase,const fort_scope_external_view_v1* views,std::size_t count,void* stream,
           int* started,uint64_t* accepted){
 *started=0;*accepted=0;assert(count==1);*started=1;
 original<<<(n+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
  static_cast<double*>(views[0].device),*static_cast<int*>(phase));
 if(cudaGetLastError()!=cudaSuccess)return FORT_SCOPE_EXECUTION;
 *accepted=1;return FORT_SCOPE_OK;
}
fort_scope_stats stats(fort_scope_t ctx){fort_scope_stats s{};check(fort_scope_stats_get(ctx,&s));return s;}
uint64_t bits(double value){uint64_t b;std::memcpy(&b,&value,8);return b;}
int main(int argc,char** argv){
 assert(argc==4);const bool writer=!std::strcmp(argv[1],"writer"),borrowed=!std::strcmp(argv[2],"borrowed");
 const bool compact=!std::strcmp(argv[3],"compact"),limited=!std::strcmp(argv[3],"budget");
 std::vector<double> host(n);std::vector<uint64_t> expected(n);
 for(std::size_t i=0;i<n;++i){
  expected[i]=touched(i)?bits(double(i)):0x7ff0000000000001ULL+i;
  std::memcpy(&host[i],&expected[i],8);
 }
 fort_scope_domain_t domain=0;fort_scope_t ctx=0;
 if(borrowed){check(fort_scope_domain_create_v1(0,&domain));check(fort_scope_create_in_domain_v1(domain,&ctx));}
 else check(fort_scope_create(0,&ctx));
 if(compact||limited)configure(ctx,false);
 if(limited)check(fort_scope_set_device_budget(ctx,n*8));
 const std::size_t shape[]={nx,ny,nz};const int64_t lower[]={-17,2,-9};
 fort_scope_layout layout{3,FORT_SCOPE_REAL64,8,host.data(),shape,lower,1};fort_buffer_t buffer=0;
 check(fort_scope_register(ctx,31,7,&layout,0,&buffer));
 std::size_t lo[4][3],hi[4][3];fort_scope_section sections[4];
 for(std::size_t k=0;k<4;++k){lo[k][0]=1+6*k;lo[k][1]=lo[k][2]=0;
  hi[k][0]=lo[k][0]+1;hi[k][1]=ny;hi[k][2]=nz;sections[k]={lo[k],hi[k]};}
 fort_scope_access rw{0,4,sections,4,sections,0,nullptr};
 fort_scope_external_binding_v1 binding{buffer,FORT_SCOPE_REAL64,3,8,rw};
 fort_scope_external_call_v1 call{1,0,&binding,1,0,0};fort_scope_external_report_v1 report{};int phase=0;
 check(fort_scope_source_execute_v2(ctx,&call,1,submit,&phase,&report));assert(report.started&&report.committed);
 auto first=stats(ctx);assert(first.upload_bytes==payload&&first.download_bytes==0);
 assert(first.uploads==(compact?1u:4u));
 assert(fort_scope_source_execute_v1(ctx,&call,1,submit,&phase,&report)==FORT_SCOPE_UNINITIALIZED);
 const std::size_t changed_lo[]={13,2,1},changed_hi[]={14,3,2},changed_index=13+nx*(2+ny);
 fort_scope_section changed{changed_lo,changed_hi};
 fort_scope_access host_access{0,4,sections,writer?1u:0u,writer?&changed:nullptr,0,nullptr};
 fort_scope_plan_binding native{buffer,host_access};
 check(fort_scope_original_host_begin_v1(ctx,&native,1));
 for(std::size_t i=0;i<n;++i){
  uint64_t actual;std::memcpy(&actual,&host[i],8);
  assert(actual==(touched(i)?bits(double(i)+10):expected[i]));
 }
 if(writer)host[changed_index]+=7;
 check(fort_scope_host_end(ctx,buffer));
 auto native_done=stats(ctx);assert(native_done.download_bytes==payload);
 assert(native_done.downloads==(compact?1u:4u));
 phase=1;check(fort_scope_source_execute_v2(ctx,&call,1,submit,&phase,&report));assert(report.started&&report.committed);
 auto second=stats(ctx);assert(second.launches==2);
 assert(second.upload_bytes==payload+(writer?8u:0u));
 assert(second.uploads==first.uploads+(writer?1u:0u));
 assert(second.download_bytes==payload);
 check(fort_scope_observe_all_v1(ctx));auto published=stats(ctx);
 assert(published.download_bytes==2*payload&&published.downloads==(compact?2u:8u));
 check(fort_scope_observe_v1(ctx,&buffer,1));assert(stats(ctx).download_bytes==published.download_bytes);
 void* pointer=nullptr;assert(fort_scope_device_begin(ctx,buffer,&rw,&pointer)==FORT_SCOPE_UNINITIALIZED);
 check(fort_scope_close(ctx));if(borrowed)check(fort_scope_domain_close_v1(domain));
 for(std::size_t i=0;i<n;++i){
  if(touched(i))expected[i]=bits((double(i)+10+(writer&&i==changed_index?7:0))*2);
  uint64_t actual;std::memcpy(&actual,&host[i],8);assert(actual==expected[i]);
 }
 std::puts("SOURCE_STORAGE_FRAGMENTED_COPIES_OK");
}
'''


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit and supported GNU compiler unavailable")
    directory = tmp_path_factory.mktemp("source_storage_compact_cuda")
    source, executable = directory / "client.cu", directory / "run"
    source.write_text(CLIENT)
    argv = [nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp", "-I", str(RUNTIME),
            *cuda_compile_options(), str(RUNTIME / "scoped_runtime.cu"), str(source), "-o", str(executable)]
    (directory / "command.json").write_text(json.dumps(argv) + "\n")
    result = subprocess.run(argv, capture_output=True, text=True, timeout=180)
    (directory / "build.stdout").write_text(result.stdout)
    (directory / "build.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    return executable


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["readonly", "writer"])
@pytest.mark.parametrize("ownership", ["private", "borrowed"])
@pytest.mark.parametrize("transport", ["direct", "compact", "budget"])
def test_fragmented_source_storage_preserves_unknown_holes_and_readonly_replicas(binary, mode, ownership, transport):
    result = subprocess.run([str(binary), mode, ownership, transport], capture_output=True, text=True, timeout=30,
                            env={**os.environ, "FORT_RUNTIME_TRACE": "1", "OMP_NUM_THREADS": "4"})
    prefix = binary.parent / (ownership + "-" + mode + "-" + transport)
    prefix.with_suffix(".stdout").write_text(result.stdout)
    prefix.with_suffix(".stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SOURCE_STORAGE_FRAGMENTED_COPIES_OK" in result.stdout


@pytest.fixture(scope="module")
def collective_configured_binary(tmp_path_factory):
    """Exercise the public owner configuration, without manual cost setters."""
    from compiler.driver.options import CompilerOptions
    from compiler.driver.pipeline import prepare_function
    from compiler.emission.common.resources import read_common_header, read_scoped_runtime
    from compiler.emission.cuda.scoped import generate_scoped
    from compiler.frontend import lower_file
    from compiler.offload.config import OffloadConfig
    from compiler.offload.physical_copy_fit import physical_copies_from_measurements
    from compiler.tests.test_module_allocatable_scopes_runtime import _run
    from compiler.tests.test_physical_copy_profile import observations
    from compiler.tests.test_scoped_batch_sources import FLAT
    from compiler.tests.test_scoped_batch_sources_cuda import actual_profile

    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14")
    if not nvcc or not host:
        pytest.skip("CUDA toolkit and supported GNU compiler unavailable")
    directory = tmp_path_factory.mktemp("collective_copy_configuration")
    profile = actual_profile(directory, nvcc, host)
    # Deliberately synthetic, independently fitted selector evidence. Actual
    # hardware identities enable the compatibility check; these are not prices
    # for performance claims. No collective computation model is supplied.
    rows = observations()
    for row in rows:
        if row["name"] == "direct":
            row["seconds"] = [sample * 1e5 for sample in row["seconds"]]
    profile["scoped"]["physical_copies"] = physical_copies_from_measurements(rows, profile)
    assert "collective" not in profile["scoped"]
    (directory / "profile.json").write_text(json.dumps(profile, indent=2))
    source = directory / "original.f90"
    source.write_text(FLAT)
    options = CompilerOptions(opt_level=0, gpu_policy="sections", memory_model="scoped", scope_transfers="auto")
    function, plan = prepare_function(lower_file(source, "advance"), options=options)
    artifacts, manifest = read_scoped_runtime()
    emitted = generate_scoped(function, plan, OffloadConfig("sections", profile, 4, True, scope_transfers="auto"),
                              "common_functions.cuh", runtime_id=manifest["runtime_id"])
    assert not emitted.report["planning"]["profile_available"]
    assert not emitted.report["automatic_estimate_available"]
    for name, content in artifacts.items():
        (directory / name).write_text(content)
    (directory / "common_functions.cuh").write_text(read_common_header())
    (directory / "entry.cu").write_text(emitted.cuda)
    (directory / "entry-report.json").write_text(json.dumps(emitted.report, indent=2))
    entry = emitted.report["transfer_configuration"]["entry"]
    client = CLIENT.replace(CONFIGURE, f'extern "C" int {entry}(fort_scope_t);\n')
    client = client.replace("if(compact||limited)configure(ctx,false);", f"check({entry}(ctx));")
    assert "fort_scope_set_copy_costs" not in client
    (directory / "client.cu").write_text(client)
    executable = directory / "run"
    argv = [nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp",
            *cuda_compile_options(), "scoped_runtime.cu", "entry.cu", "client.cu", "-o", str(executable)]
    (directory / "command.json").write_text(json.dumps(argv) + "\n")
    result = _run(argv, directory, timeout=180)
    (directory / "build.stdout").write_text(result.stdout)
    (directory / "build.stderr").write_text(result.stderr)
    return executable


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["readonly", "writer"])
@pytest.mark.parametrize("ownership", ["private", "borrowed"])
def test_collective_owner_uses_copy_costs_without_compute_calibration(collective_configured_binary, mode, ownership):
    test_fragmented_source_storage_preserves_unknown_holes_and_readonly_replicas(
        collective_configured_binary, mode, ownership, "compact")
