"""Actual generated producer/external/consumer transport with native recovery."""
import json
import os
import re
import shutil
import subprocess

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.device_tasks import task_scope_identity
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_external_device_tasks import PROVIDER_INTERFACE, SOURCE, digest, fixture
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_source_scopes import run

SOURCE = SOURCE.replace("output=3.0_8*input", "integer::i\ndo i=2,size(input)-1\noutput(i)=3.0_8*input(i)\nenddo").replace(
    "do i=1,n", "do i=2,n+1")

DRIVER = """program exercise
use external_fixture
implicit none
real(8),allocatable::a(:),b(:),expected_a(:),expected_b(:)
integer::n,k,i,j
integer,parameter::sizes(3)=[0,1,257]
do j=1,3
n=sizes(j)
allocate(a(-1:n),b(-1:n),expected_a(-1:n),expected_b(-1:n))
do k=1,2
do i=-1,n
a(i)=real(i+5+k,8)
b(i)=-real(i+20+k,8)
enddo
expected_a=a
expected_b=b
do i=0,n-1
expected_b(i)=2*expected_a(i)
expected_a(i)=3*expected_b(i)
expected_b(i)=expected_a(i)+1
enddo
call advance(a,b,n)
if(any(a/=expected_a))error stop 'full A or halo'
if(any(b/=expected_b))error stop 'full B or halo'
enddo
deallocate(a,b,expected_a,expected_b)
enddo
print *, 'EXTERNAL_FIELDS_OK'
end program
"""

PROVIDER = r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <new>
struct Config { int64_t extent; double coefficient; };
struct Provider { Config config; fort_scope_domain_t domain; uint64_t lease=0; void *scratch=nullptr;
                  cudaEvent_t completion=nullptr; cudaStream_t stream=nullptr; bool pending=false; int device=0; };
__global__ void transform(const double *input,double *output,int64_t count,double coefficient) {
    const int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count)output[i+1]=coefficient*input[i+1];
}
extern "C" int provider_create(fort_scope_domain_t domain,const Config *config,void **handle) {
    if(!config||!handle)return FORT_SCOPE_ARGUMENT;
    *handle=nullptr;
    if(config->extent<1)return FORT_SCOPE_BOUNDARY;
    auto *p=new(std::nothrow)Provider{}; if(!p)return FORT_SCOPE_RESOURCE;
    p->config=*config;p->domain=domain;
    int status=fort_scope_domain_device_get_v1(domain,&p->device);
    if(!status)status=fort_scope_domain_workspace_reserve_v1(domain,32,&p->lease);
    int previous=0;bool selected=false;
    if(!status && cudaGetDevice(&previous)!=cudaSuccess)status=FORT_SCOPE_RESOURCE;
    if(!status) { if(cudaSetDevice(p->device)!=cudaSuccess)status=FORT_SCOPE_RESOURCE;else selected=true; }
    if(!status && cudaMalloc(&p->scratch,32)!=cudaSuccess)status=FORT_SCOPE_RESOURCE;
    if(!status && cudaEventCreateWithFlags(&p->completion,cudaEventDisableTiming)!=cudaSuccess)status=FORT_SCOPE_RESOURCE;
    if(status) {
        if(p->completion)cudaEventDestroy(p->completion);
        if(p->scratch)cudaFree(p->scratch);
        if(p->lease)fort_scope_domain_workspace_release_v1(domain,p->lease);
        delete p;
    } else *handle=p;
    if(selected && cudaSetDevice(previous)!=cudaSuccess)return FORT_SCOPE_EXECUTION;
    return status;
}
static int callback(void *opaque,const fort_scope_external_view_v1 *views,size_t count,void *stream,int *started) {
    auto &p=*static_cast<Provider *>(opaque);*started=0;
    if(count!=2||!stream)return FORT_SCOPE_ARGUMENT;
    for(size_t i=0;i<count;++i)
        if(views[i].type!=FORT_SCOPE_REAL64||views[i].rank!=1||views[i].strides[0]!=8||
           views[i].extents[0]!=size_t(p.config.extent+2))return FORT_SCOPE_BOUNDARY;
    if(std::getenv("FORT_EXTERNAL_TEST_REJECT"))return FORT_SCOPE_RESOURCE;
    p.stream=static_cast<cudaStream_t>(stream);p.pending=true;*started=1;
    transform<<<unsigned((p.config.extent+127)/128),128,0,p.stream>>>(
        static_cast<const double *>(views[0].device),static_cast<double *>(views[1].device),p.config.extent,p.config.coefficient);
    if(cudaGetLastError()!=cudaSuccess)return FORT_SCOPE_EXECUTION;
    if(cudaEventRecord(p.completion,p.stream)!=cudaSuccess)return FORT_SCOPE_EXECUTION;
    if(std::getenv("FORT_EXTERNAL_TEST_FAIL_STARTED"))return FORT_SCOPE_EXECUTION;
    return FORT_SCOPE_OK;
}
extern "C" int provider_execute(fort_scope_t context,void *handle,fort_buffer_t input,fort_buffer_t output,
                                 fort_scope_external_report_v1 *report) {
    if(!handle)return FORT_SCOPE_ARGUMENT;
    auto &p=*static_cast<Provider *>(handle);
    size_t lower[]={1},upper[]={size_t(p.config.extent+1)};
    fort_scope_section box{lower,upper};
    fort_scope_access read{0,1,&box,0,nullptr,0,nullptr},overwrite{0,0,nullptr,1,&box,1,&box};
    fort_scope_external_binding_v1 bindings[2]={{input,FORT_SCOPE_REAL64,1,8,read},{output,FORT_SCOPE_REAL64,1,8,overwrite}};
    fort_scope_external_call_v1 call{1,0,bindings,2,p.lease,32};
    return fort_scope_external_execute_v1(context,&call,callback,&p,report);
}
extern "C" int provider_destroy(void **handle) {
    if(!handle)return FORT_SCOPE_ARGUMENT;
    if(!*handle)return FORT_SCOPE_OK;
    auto *p=static_cast<Provider *>(*handle);int previous=0;
    if(cudaGetDevice(&previous)!=cudaSuccess||cudaSetDevice(p->device)!=cudaSuccess)return FORT_SCOPE_EXECUTION;
    // A failed record still requires joining the attempted stream before free.
    if(p->pending && cudaStreamSynchronize(p->stream)!=cudaSuccess)return FORT_SCOPE_EXECUTION;
    if(cudaFree(p->scratch)!=cudaSuccess||cudaEventDestroy(p->completion)!=cudaSuccess)return FORT_SCOPE_EXECUTION;
    const int status=fort_scope_domain_workspace_release_v1(p->domain,p->lease);
    if(status)return status;
    delete p;*handle=nullptr;
    return cudaSetDevice(previous)==cudaSuccess?FORT_SCOPE_OK:FORT_SCOPE_EXECUTION;
}
'''


@pytest.fixture(scope="module")
def external_binaries(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/Fortran toolchain unavailable")
    directory = tmp_path_factory.mktemp("source_external")
    path, analysis, _, _, task, provider_manifest, provider_source = fixture(directory, SOURCE)
    provider_source.write_text(PROVIDER_INTERFACE)
    provider_cuda = directory / "provider.cu"
    provider_cuda.write_text(PROVIDER)
    provider_record = json.loads(provider_manifest.read_text())
    provider_record["artifacts"] = {str(provider_source): digest(provider_source), str(provider_cuda): digest(provider_cuda)}
    provider_manifest.write_text(json.dumps(provider_record))
    task["provider_manifest"]["sha256"] = digest(provider_manifest)
    task = json.loads(json.dumps(task))
    analysis.contracts["external_fixture::advance"]["device_tasks"] = [task]
    for resource in task["resources"]:
        for effect in resource["effects"]:
            effect["lower"] = [1]
            effect["upper"][0]["offset"] = -1
    task["configuration"][0]["values"][0]["offset"] = -2
    provider_record["numerical_contract"]["task_contracts"] = [task_scope_identity(task)]
    provider_manifest.write_text(json.dumps(provider_record))
    task["provider_manifest"]["sha256"] = digest(provider_manifest)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): digest(path)},
             "captures": {"argument::a": dict(FACT), "argument::b": dict(FACT)}}
    outputs, manifest = ScopeBuilder([path], "external_fixture::advance", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"), contracts=analysis.contracts).run()
    assert not manifest["scopes"][0]["boundaries"]
    assert len(manifest["scopes"][0]["external_device_tasks"]) == 1
    build = directory / "build"
    build.mkdir()
    for name, text in outputs.items():
        target = build / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    (build / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    interface = build / "external_provider.f90"
    interface.write_text(PROVIDER_INTERFACE)
    provider = build / "provider.cu"
    provider.write_text(PROVIDER)
    objects = []
    for role in ("common_runtime", "provider", "shared_entry", "original_source"):
        if role == "provider":
            paths = [(interface,"fortran"),(provider,"cuda")]
        else:
            paths = [(build/item["path"],item["language"]) for item in manifest["build_sources"] if item["role"] == role]
        for target, language in paths:
            obj = build / f"object-{len(objects)}.o"
            command = ([nvcc,"-std=c++17","-ccbin",host,"-arch=sm_86","-Xcompiler=-fopenmp","-I",str(build)]
                       if language == "cuda" else
                       [fortran,"-std=f2018","-fopenmp","-fcheck=all","-J",str(build),"-I",str(build)])
            run([*command,"-c",str(target),"-o",str(obj)],cwd=build)
            objects.append(str(obj))
    driver = build / "driver.f90"
    driver.write_text(DRIVER)
    candidate, native = build / "candidate", build / "native"
    run([fortran,"-std=f2018","-fopenmp","-fcheck=all",str(driver),*objects,
         "-L/usr/local/cuda/lib64","-Wl,-rpath,/usr/local/cuda/lib64","-lcudart","-lstdc++","-o",str(candidate)],cwd=build)
    run([fortran,"-std=f2018","-fopenmp","-fcheck=all",str(path),str(driver),"-o",str(native)],cwd=build)
    return build,candidate,native


def execute(external_binaries, mode):
    build,candidate,native = external_binaries
    environment = {**os.environ,"OMP_NUM_THREADS":"4","OMP_DYNAMIC":"FALSE","FORT_RUNTIME_TRACE":"1"}
    for key in ("CUDA_VISIBLE_DEVICES","FORT_EXTERNAL_TEST_REJECT","FORT_EXTERNAL_TEST_FAIL_STARTED"):
        environment.pop(key,None)
    if mode == "hidden":
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    if mode == "reject":
        environment["FORT_EXTERNAL_TEST_REJECT"] = "1"
    if mode == "poison":
        environment["FORT_EXTERNAL_TEST_FAIL_STARTED"] = "1"
    baseline = run([str(native)],cwd=build,env=environment)
    result = subprocess.run([str(candidate)],cwd=build,env=environment,text=True,capture_output=True,timeout=60)
    (build/f"{mode}.stdout").write_text(result.stdout)
    (build/f"{mode}.stderr").write_text(result.stderr)
    if mode == "poison":
        assert result.returncode != 0
        assert "external device work failed; replay prohibited" in result.stderr
        assert "EXTERNAL_FIELDS_OK" not in result.stdout
        return result
    assert result.returncode == 0,result.stderr
    assert result.stdout == baseline.stdout
    assert "EXTERNAL_FIELDS_OK" in result.stdout
    return result


@pytest.mark.native
def test_original_fields_and_halos_remain_native_when_device_hidden(external_binaries):
    result = execute(external_binaries,"hidden")
    assert "FORT_SCOPED launch " not in result.stderr
    assert "FORT_SCOPED external_commit " not in result.stderr


@pytest.mark.cuda
def test_generated_external_call_retains_both_mirrors_without_intermediate_copy(external_binaries):
    result = execute(external_binaries,"enabled")
    lines = result.stderr.splitlines()
    assert sum(line.startswith("FORT_SCOPED launch ") for line in lines) == 8,result.stderr
    assert sum(line.startswith("FORT_SCOPED external_commit ") for line in lines) == 4,result.stderr
    for event,expected in (("upload",4128),("download",8256)):
        assert sum(int(re.search(r"\bbytes=(\d+)\b",line)[1]) for line in lines
                   if line.startswith("FORT_SCOPED "+event+" ")) == expected,result.stderr


@pytest.mark.cuda
def test_external_prework_rejection_publishes_prefix_and_runs_original_call_once(external_binaries):
    result = execute(external_binaries,"reject")
    assert "FORT_SCOPED external_commit " not in result.stderr
    assert sum(line.startswith("FORT_SCOPED launch ") for line in result.stderr.splitlines()) == 8,result.stderr


@pytest.mark.cuda
def test_submitted_external_failure_never_replays_original_source(external_binaries):
    execute(external_binaries,"poison")
