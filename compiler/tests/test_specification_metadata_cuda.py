"""The emitted Fortran entry block observes asynchronous device-written bounds."""
import json
import os
import shutil
import subprocess

import pytest

from compiler.numerical_contract import cuda_compile_options
from compiler.scopes.specification_entry import publication
from compiler.tests.test_scoped_runtime import RUNTIME

CLIENT = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" {extern int metadata[1]; void publish_before_child(fort_scope_t,fort_buffer_t,int*);}
void check(int status){if(status){std::fprintf(stderr,"%d %s\n",status,fort_scope_error());std::abort();}}
__global__ void work(int* meta,double* values,std::size_t n,int operation){
 const auto i=blockIdx.x*blockDim.x+threadIdx.x;
 if(operation==0){if(i==0)*meta=47;if(i<n)values[i]*=2;}
 else if(i<n)values[i]+=*meta;
}
int submit(void* raw,const fort_scope_external_view_v1* views,std::size_t count,void* stream,int* started,uint64_t* accepted){
 assert(count==2);*started=1;*accepted=0;
 work<<<(views[1].extents[0]+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
  static_cast<int*>(views[0].device),static_cast<double*>(views[1].device),views[1].extents[0],*static_cast<int*>(raw));
 if(cudaGetLastError()!=cudaSuccess)return FORT_SCOPE_EXECUTION;*accepted=1;return 0;
}
fort_scope_stats stats(fort_scope_t c){fort_scope_stats s{};check(fort_scope_stats_get(c,&s));return s;}
int main(int argc,char**argv){
 assert(argc==2);const bool missing=!std::strcmp(argv[1],"missing"),stale=!std::strcmp(argv[1],"stale");
 const std::size_t one=1,n=4099;const std::int64_t lower=1;
 std::vector<double> host(n);for(std::size_t i=0;i<n;++i)host[i]=double(i)+0.25;
 fort_scope_domain_t domain=0;fort_scope_t c=0;check(fort_scope_domain_create_v1(0,&domain));
 check(fort_scope_create_in_domain_v1(domain,&c));fort_buffer_t m=0,b=0,old=0;
 fort_scope_layout ml{1,FORT_SCOPE_INTEGER32,4,metadata,&one,&lower,1},bl{1,FORT_SCOPE_REAL64,8,host.data(),&n,&lower,1};
 check(fort_scope_register(c,1,1,&ml,1,&m));
 if(stale){old=m;check(fort_scope_retire_v1(c,m));check(fort_scope_register(c,1,2,&ml,1,&m));assert(m!=old);}
 check(fort_scope_register(c,2,1,&bl,1,&b));
 fort_scope_access write{};write.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
 fort_scope_access rw{};rw.flags=FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL;
 fort_scope_external_binding_v1 bindings[]={{m,FORT_SCOPE_INTEGER32,1,4,write},{b,FORT_SCOPE_REAL64,1,8,rw}};
 fort_scope_external_call_v1 call{1,0,bindings,2,0,0};fort_scope_external_report_v1 report{};int operation=0;
 check(fort_scope_source_execute_v1(c,&call,1,submit,&operation,&report));assert(report.committed);
 const auto before=stats(c);assert(before.download_bytes==0);assert(metadata[0]==3);
 int extent=-1;publish_before_child(c,missing?0:stale?old:m,&extent);
 assert(extent==47&&metadata[0]==47);
 const auto observed=stats(c);assert(observed.download_bytes==4+(missing||stale?n*8:0));
 if(!missing&&!stale)for(std::size_t i=0;i<n;++i)assert(host[i]==double(i)+0.25);
 bindings[0].access={};bindings[0].access.flags=FORT_SCOPE_READ_ALL;operation=1;report={};
 check(fort_scope_source_execute_v1(c,&call,1,submit,&operation,&report));assert(report.committed);
 const auto continued=stats(c);assert(continued.upload_bytes==before.upload_bytes);
 assert(continued.download_bytes==observed.download_bytes);
 check(fort_scope_observe_v1(c,&b,1));
 for(std::size_t i=0;i<n;++i)assert(host[i]==2*(double(i)+0.25)+47);
 const auto final=stats(c);assert(final.download_bytes==observed.download_bytes+n*8);assert(final.launches==2);
 check(fort_scope_close(c));check(fort_scope_domain_close_v1(domain));
 std::printf("ORIGINAL_AUTOMATIC_EXTENT_AND_METADATA_RESIDENCY_OK %s\n",argv[1]);
}
'''


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not all((nvcc, host, fc)):
        pytest.skip("CUDA and GNU toolchains required")
    directory = tmp_path_factory.mktemp("device_written_specification")
    block = "\n".join(publication("context", [{"caller_slot": 1}]))
    bridge = directory / "bridge.f90"
    bridge.write_text("""module metadata_bridge
use iso_c_binding
integer(c_int),target,bind(C,name='metadata')::tile(1)=[3]
contains
subroutine publish_before_child(context,fort_buffer_1,observed)bind(C)
integer(c_int64_t),value::context,fort_buffer_1
integer(c_int),intent(out)::observed
""" + block + """
call original_child(observed)
end subroutine
subroutine original_child(observed)
integer(c_int),intent(out)::observed
real(8)::scratch(tile(1))
observed=size(scratch)
end subroutine
end module
""")
    client = directory / "client.cu"
    client.write_text(CLIENT)
    binary = directory / "candidate"
    commands = [
        [fc, "-fopenmp", "-fcheck=all", "-J", str(directory), "-c", str(RUNTIME / "scoped_memory.f90"),
         "-o", str(directory / "memory.o")],
        [fc, "-fopenmp", "-fcheck=all", "-I", str(directory), "-J", str(directory), "-c", str(bridge),
         "-o", str(directory / "bridge.o")],
        [nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp", *cuda_compile_options(),
         "-I", str(RUNTIME), str(RUNTIME / "scoped_runtime.cu"), str(client), str(directory / "memory.o"),
         str(directory / "bridge.o"), "-lgfortran", "-o", str(binary)],
    ]
    for i, command in enumerate(commands):
        (directory / f"command-{i}.json").write_text(json.dumps(command) + "\n")
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=180)
        for suffix in ("stdout", "stderr"):
            (directory / f"command-{i}.{suffix}").write_text(getattr(result, suffix))
        assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["valid", "missing", "stale"])
def test_pending_metadata_write_is_observed_before_original_bound_evaluation(executable, mode):
    result = subprocess.run([str(executable), mode], cwd=executable.parent,
                            env={**os.environ, "FORT_RUNTIME_TRACE": "1"},
                            capture_output=True, text=True, timeout=30)
    for suffix in ("stdout", "stderr"):
        (executable.parent / f"{mode}.{suffix}").write_text(getattr(result, suffix))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ORIGINAL_AUTOMATIC_EXTENT_AND_METADATA_RESIDENCY_OK" in result.stdout
