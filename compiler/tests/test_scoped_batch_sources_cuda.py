"""Independent source artifacts execute complete batched fields on real CUDA."""

from __future__ import annotations

import json
import os
import shutil
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.tests.test_module_allocatable_scopes_runtime import ROOT, _run
from compiler.tests.test_scoped_batch_sources import CHAIN, FLAT, transfer_profile
from compiler.tests.test_source_scopes import FACT

PROBE = r'''#include <cuda_runtime.h>
#include <cstdio>
int main() {
    int device=0,runtime=0,driver=0; cudaDeviceProp p{};
    if (cudaGetDeviceProperties(&p,device) || cudaRuntimeGetVersion(&runtime) || cudaDriverGetVersion(&driver)) return 2;
    std::printf("{\"gpu_name\":\"%s\",\"gpu_uuid\":\"",p.name);
    for (unsigned char value : p.uuid.bytes) std::printf("%02x",value);
    std::printf("\",\"compute_capability\":\"%d.%d\",\"async_engine_count\":%d,\"runtime\":%d,\"driver\":%d}\n",
                p.major,p.minor,p.asyncEngineCount,runtime,driver);
}
'''

DRIVER = """program caller
use original,only:step
implicit none
real(8),allocatable::a(:,:),b(:,:),c(:,:)
integer,parameter::ns(3)=[0,67,135],ms(3)=[7,39,71]
integer::shape,repeat,n,m,i,j,lx,ly
do shape=1,3
n=ns(shape)
m=ms(shape)
lx=-7-shape
ly=3-2*shape
allocate(a(lx:lx+n-1,ly:ly+m-1),b(lx:lx+n-1,ly:ly+m-1),c(lx:lx+n-1,ly:ly+m-1))
do repeat=1,2
do j=ly,ly+m-1
do i=lx,lx+n-1
a(i,j)=real(2*i-j+16*repeat,8)*0.125d0
enddo
enddo
b=-33.d0
c=-71.d0
call step(a,b,c,n,m)
print '(a,3i6)', 'CALL_OK',shape,repeat,n
print '(100000f16.4)', b
print '(100000f16.4)', c
enddo
deallocate(a,b,c)
enddo
print *, 'FIELDS_OK'
end program
"""


def actual_profile(directory, nvcc, host):
    """Bind synthetic correctness costs to verified real hardware/tool identities.

    These deliberately favorable costs exercise selection; they are never
    performance evidence or a substitute for the benchmark calibration.
    """
    probe = directory / "probe.cu"
    probe.write_text(PROBE)
    binary = directory / "probe"
    _run([nvcc, "-std=c++17", "-ccbin", host, str(probe), "-o", str(binary)], directory)
    observed = json.loads(_run([str(binary)], directory).stdout)
    value = transfer_profile()
    value["hardware"].update({name: observed[name] for name in (
        "gpu_name", "gpu_uuid", "compute_capability", "async_engine_count")})
    value["hardware"]["cpu_name"] = next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                                          if line.startswith("model name"))
    value["toolchain"].update(nvcc_version=_run([nvcc,"--version"],directory).stdout,
        host_cxx_version=_run([host,"--version"],directory).stdout,
        cuda_runtime_version=observed["runtime"], driver_version=observed["driver"])
    value["rates"].update(cpu_flops_per_second=1e4,cpu_memory_bytes_per_second=1e4,
                          gpu_flops_per_second=1e13,gpu_memory_bytes_per_second=1e13)
    for direction in ("h2d", "d2h"):
        value["rates"][direction+"_pageable"].update(latency_seconds=1e-3,bandwidth_bytes_per_second=1e6)
        value["rates"][direction+"_pinned"].update(latency_seconds=1e-7,bandwidth_bytes_per_second=1e12)
    value["scoped"]["transfers"]["costs"].update(preparation_operation_seconds=1e-12,
        staging_cold_seconds=[1e-9]*4,staging_reuse_seconds=[1e-9]*4,event_record_seconds=1e-9,
        event_wait_seconds=1e-9,ready_event_seconds=1e-9,pack_bytes_per_second=1e13,unpack_bytes_per_second=1e13)
    # The flat fixture must exercise a complete two-unit callback. This price
    # makes repeated ordinary access setup costlier than the extra batch launch.
    value["scoped"]["costs"]["device_access_seconds"]=1e-4
    return value


def source_case(label):
    if label == "flat":
        source = (FLAT.replace("module numerical", "module original")
                  .replace("a(:,:)", "a(-2:,3:)").replace("b(:,:),c(:,:)", "b(-2:,3:),c(-2:,3:)")
                  .replace("do j=2,m-1", "do j=4,m+1").replace("do i=2,n-1", "do i=-1,n-4")
                  .replace("end module", """subroutine inspect(c)
real(8),intent(inout)::c(:,:)
if(size(c,1)>0.and.size(c,2)>0) c(1,1)=sum(c)
end subroutine
subroutine step(a,b,c,n,m)
real(8),intent(in)::a(:,:)
real(8),intent(inout)::b(:,:),c(:,:)
integer,intent(in)::n,m
call advance(a,b,c,n,m)
call inspect(c)
end subroutine
end module"""))
    else:
        source = CHAIN.replace("intent(out)::b", "intent(inout)::b").replace("intent(out)::c", "intent(inout)::c")
        if label == "halo":
            source = source.replace("a(i+1,j)+real(i,8)", "a(i+1,j)+a(i,j-1)+a(i,j+1)+real(i,8)")
        if label == "nonterminal":
            source = source.replace("call second(b,c,n,m)\nend subroutine", "call second(b,c,n,m)\n"
                "call observe(b,c)\ncall first(a,b,n,m)\ncall second(b,c,n,m)\nend subroutine")
            source = source.replace("end module", "subroutine observe(b,c)\nreal(8),intent(in)::b(:,:)\n"
                "real(8),intent(inout)::c(:,:)\nif(size(c,1)>0.and.size(c,2)>0) c(1,1)=sum(b)+sum(c)\nend subroutine\nend module")
    return source


def negative_package(directory, original):
    """Public extraction keeps original coordinates with whole-storage indices."""
    normalized=directory/"normalized.f90"
    text=(FLAT.replace("do j=2,m-1","do j=4,m+1").replace("do i=2,n-1","do i=-1,n-4")
          .replace("(i-1,j)","(i+2,j-2)").replace("(i+1,j)","(i+4,j-2)").replace("(i,j)","(i+3,j-2)"))
    normalized.write_text(text)
    digest=sha256(original.read_bytes()).hexdigest()
    return {"schema_version":1,"source_inputs":{str(original):digest},"entries":[{
        "procedure":"original::advance","source_sha256":digest,"path":str(normalized),
        "entry":"numerical::advance","sha256":sha256(normalized.read_bytes()).hexdigest(),
        "normalization":"whole_storage_rebased_v1","participation":"serial_coordinator",
        "capture_safe":True,"preserves_source_order":True,
        "parameters":[{"name":name,"resource":"argument::"+name,
                       **({"physical_origin":[0,0]} if name in {"a","b","c"} else {})}
                      for name in ("a","b","c","n","m")]}]}


@pytest.fixture(scope="module")
def batch_binaries(tmp_path_factory):
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++"), shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, C++ and Fortran toolchains required")
    directory = tmp_path_factory.mktemp("source-batches")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT/"compiler",checkout/"compiler",ignore=shutil.ignore_patterns("__pycache__",".*cache","CODE_MAP.md","MEMORY_MODEL_PLAN.md"))
    profile = actual_profile(directory,nvcc,host)
    flags = [nvcc,"-O2","-std=c++17","-ccbin",host,"-arch=sm_86","-Xcompiler=-fopenmp"]
    fflags = [fortran,"-O3","-std=f2018","-fopenmp","-fcheck=all,array-temps"]
    binaries, cache, reuse = {}, {}, []
    for label, transfers, policy in [("flat","pipelined","sections"),("chain","pipelined","sections"),
                                       ("halo","pipelined","sections"),("nonterminal","pipelined","sections"),
                                       ("automatic","auto","auto")]:
        case = directory/label
        case.mkdir()
        original = case/"original.f90"
        original.write_text(source_case(label))
        captures = case/"captures.json"
        captures.write_text(json.dumps({"schema_version":1,"participation":"serial",
            "sources":{str(original):sha256(original.read_bytes()).hexdigest()},
            "captures":{"argument::"+name:FACT for name in ("a","b","c")}}))
        costs = case/"profile.json"
        costs.write_text(json.dumps(profile))
        numerical_sources=[]
        if label == "flat":
            package=case/"numerical-sources.json"
            package.write_text(json.dumps(negative_package(case,original)))
            numerical_sources=["--numerical-sources",str(package)]
        output = case/"output"
        public = json.loads(_run([sys.executable,"-m","compiler","--form-scopes","--input",str(original),
            "--kernel","step","--scope-facts",str(captures),"--memory-model","scoped","--gpu-policy",policy,
            "--scope-transfers",transfers,"--calibration-profile",str(costs),"--opt-level","0",
            *numerical_sources,"--json","--output-dir",str(output)],checkout,env={**os.environ,"PYTHONPATH":str(checkout)}).stdout)
        assert public["supported"], public
        manifest = public["scopes"]
        assert manifest["runtime"]["runtime_id"] == profile["scoped"]["runtime_id"]
        objects=[]
        for role in ("common_runtime","shared_entry","original_source"):
            for item in manifest["build_sources"]:
                if item["role"] != role:
                    continue
                artifact=item["path"]
                target=output/(artifact.replace("/","_")+".o")
                if item["language"] == "cuda":
                    headers={name:sha256((output/name).read_bytes()).hexdigest() for name in sorted(manifest["artifacts_sha256"])
                             if name.endswith((".h",".hpp",".cuh"))}
                    key=(manifest["artifacts_sha256"][artifact],tuple(headers.items()),tuple(flags))
                    reused=key in cache
                    if reused:
                        target=cache[key]
                    else:
                        _run([*flags,"-I",str(output),"-c",str(output/artifact),"-o",str(target)],output)
                        cache[key]=target
                    reuse.append({"case":label,"artifact":artifact,"sha256":key[0],"headers":headers,
                                  "flags":flags,"reused":reused,"object":str(target)})
                else:
                    _run([*fflags,"-c",str(output/artifact),"-o",str(target)],output)
                objects.append(str(target))
        driver=output/"driver.f90"
        driver.write_text(DRIVER)
        binary=output/"verify"
        _run([*fflags,str(driver),*objects,"-L/usr/local/cuda/lib64","-Wl,-rpath,/usr/local/cuda/lib64",
              "-lcudart","-lstdc++","-o",str(binary)],output)
        native=output/"native"
        _run([*fflags,str(original),str(driver),"-o",str(native)],output)
        reference=_run([str(native)],output,env={**os.environ,"OMP_NUM_THREADS":"4","OMP_DYNAMIC":"FALSE"}).stdout
        (output/"native.stdout").write_text(reference)
        binaries[label]=(binary,output,reference,manifest)
    (directory/"object-reuse.json").write_text(json.dumps(reuse,indent=2)+"\n")
    return binaries


@pytest.mark.cuda
@pytest.mark.parametrize("label",["flat","chain","halo","nonterminal","automatic"])
def test_complete_batch_fields_and_halos_match_native_from_independent_checkout(batch_binaries,label):
    binary,output,reference,manifest=batch_binaries[label]
    result=_run([str(binary)],output,env={**os.environ,"OMP_NUM_THREADS":"4","OMP_DYNAMIC":"FALSE","FORT_RUNTIME_TRACE":"1"})
    (output/"scoped.stdout").write_text(result.stdout)
    (output/"scoped.stderr").write_text(result.stderr)
    assert result.stdout == reference
    assert result.stdout.count("CALL_OK") == 6
    assert "array temporary" not in result.stderr.lower()
    events=[json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in result.stderr.splitlines()
            if line.startswith("FORT_SCOPED evidence ")]
    batches=[event for event in events if event["event"] == "batch_statistics" and event["applied"]]
    assert batches, result.stderr
    assert all(event["batches"] >= 2 and event["completed_batches"] == event["batches"] for event in batches)
    assert all(event["actual_launches"] == 2*event["batches"] for event in batches)
    assert all(not event["owner_cost_available"] for event in batches)
    if label == "flat":
        assert all(event["actual_download_bytes"] == 0 for event in batches)
    elif label != "nonterminal":
        assert all(event["actual_download_bytes"] > 0 for event in batches)
        assert manifest["scopes"][0]["batch_subchains"][0]["terminal_exports"]
    else:
        assert any(event["actual_download_bytes"] == 0 for event in batches)
        assert any(event["actual_download_bytes"] > 0 for event in batches)
        first,last=manifest["scopes"][0]["batch_subchains"]
        assert not first["terminal_exports"]
        assert last["terminal_exports"]
    if label == "halo":
        assert all(event["prefix_upload_bytes"] > 0 for event in batches)
    shapes=[(67,39),(67,39),(135,71),(135,71)]
    if label == "nonterminal":
        assert len(batches) == 8
        for (n,m), first,last in zip(shapes,batches[::2],batches[1::2],strict=True):
            assert first["actual_upload_bytes"] == n*(m-2)*8
            assert first["actual_download_bytes"] == 0
            assert last["actual_upload_bytes"] == 0
            assert last["actual_download_bytes"] == 2*(n-2)*(m-2)*8
    else:
        assert len(batches) == 4
        for (n,m),event in zip(shapes,batches,strict=True):
            assert event["actual_upload_bytes"] == ((n*m-4) if label == "halo" else n*(m-2))*8
            assert event["actual_download_bytes"] == (0 if label == "flat" else 2*(n-2)*(m-2)*8)
    stats=[event for event in events if event["event"] == "transfer_statistics"]
    assert stats
    assert all(event["complete"] for event in stats)
    assert max(event["process_peak_bytes"] for event in stats) <= 64*1024*1024
