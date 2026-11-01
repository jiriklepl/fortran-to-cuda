"""Actual generated native team preserves a predecessor's read-only replica."""

import json
import os
import re
import shutil
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.tests.test_source_native_pricing import live_emitted
from compiler.tests.test_source_scopes import run


def synthetic_gpu_ranking(original):
    """Exercise dispatch only; these invented observations are not calibration."""
    from compiler.emission.common.resources import read_scoped_runtime
    from compiler.offload.source_protocol import profile_with_source_measurements
    from compiler.tests.test_source_native_calibration import native_rows
    from compiler.tests.test_source_protocol_calibration import MAX_BYTES, observations, teams

    profile = deepcopy(original)
    identity = deepcopy(profile["source_tasks"]["identity"])
    profile.pop("source_tasks")
    profile.pop("source_submit_model", None)
    profile["rates"].update(cpu_flops_per_second=1e3, cpu_memory_bytes_per_second=1e3,
                            gpu_flops_per_second=1e15, gpu_memory_bytes_per_second=1e15)
    records = observations() + native_rows()
    records[0] = identity
    for row in records[1:]:
        row["seconds"] = [1e-9]*5
    native = teams()
    for row in native:
        row.update(schedule_kind=identity["schedule_kind"], schedule_chunk=identity["schedule_chunk"],
                   seconds=[.25]*5)
    return profile_with_source_measurements(profile, records, runtime_id=read_scoped_runtime()[1]["runtime_id"],
                                            max_bytes=MAX_BYTES, team_records=native)


@pytest.mark.cuda
@pytest.mark.parametrize("synthetic_gpu", [False, True], ids=["measured-native-only", "synthetic-gpu-dispatch"])
def test_generated_native_source_team_between_gpu_workers_keeps_readonly_replica(tmp_path, monkeypatch, synthetic_gpu):
    profile_path = os.environ.get("FORT_TEST_SOURCE_PROFILE")
    if not profile_path:
        pytest.skip("independent fresh native-live source calibration required")
    profile = json.loads(Path(profile_path).read_text())
    if "native_transactions" not in profile.get("source_tasks", {}):
        pytest.skip("profile has no measured native transaction protocol")
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not all((nvcc, host, fc)):
        pytest.skip("CUDA and GNU toolchains required")
    if synthetic_gpu:
        profile = synthetic_gpu_ranking(profile)
    # Use the exact independently measured profile without altering any rates.
    monkeypatch.setattr("compiler.tests.test_source_native_pricing.live_profile", lambda: profile)
    original, (outputs, manifest) = live_emitted(tmp_path, True)
    scope, = manifest["scopes"]
    task, = scope["source_device_tasks"]
    assert task["version"] == 4
    coordinator, = scope["module_coordinators"]
    mapping = coordinator["calls"][0]["handle_bindings"]
    assert sorted(mapping.values()) == [1, 2]
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        artifact = build / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    (build / "scope-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (build / "profile-input.json").write_text(json.dumps({"path": profile_path,
        "sha256": sha256(Path(profile_path).read_bytes()).hexdigest(),
        "qualification": "invented ranking tests dispatch only" if synthetic_gpu else "unchanged measured native-only profile",
        "gpu_cost_qualified": False}, indent=2) + "\n")
    if synthetic_gpu:
        (build / "synthetic-ranking-profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    objects = []
    commands = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["role"] != role:
                continue
            source, target = build / artifact["path"], build / f"object-{index}.o"
            assert sha256(source.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
            if artifact["language"] == "cuda":
                command = [nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp",
                           *artifact["numerical_contract"]["required_cuda_options"], "-I", str(build)]
            else:
                command = [fc, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build)]
            commands.append([*command, "-c", str(source), "-o", str(target)])
            objects.append(str(target))
    reference = build / "reference.f90"
    reference.write_text(original.read_text().replace("module stencil", "module reference_stencil", 1))
    bridge = build / "bridge.f90"
    # ENTRY controls forward exact existing canonical handles and their separate
    # storage identities. This uses the real checked child, not its provider only.
    identities = ",".join(str(mapping[str(slot)]) + "_c_int64_t" for slot in sorted(map(int, mapping)))
    handles = ",".join("ha" if mapping[str(slot)] == 1 else "hb" for slot in sorted(map(int, mapping)))
    bridge.write_text(f"""subroutine checked_bridge(a,b,n,t,context,ha,hb)bind(C)
use iso_c_binding
use stencil,only:{coordinator['entry']}
implicit none
integer(c_int),value::n,t
integer(c_int64_t),intent(inout)::context,ha,hb
real(c_double),intent(in),target::a(-3:n+4)
real(c_double),intent(inout),target::b(-3:n+4)
logical::enabled,epoch,query
enabled=.true.
epoch=.false.
query=.false.
call {coordinator['entry']}(a,b,n,t,context,enabled,epoch,query,{handles},{identities})
if(.not.enabled.or.context==0)error stop 'native source discarded caller domain'
end subroutine
subroutine reference_bridge(a,b,n,t)bind(C)
use iso_c_binding
use reference_stencil
implicit none
integer(c_int),value::n,t
real(c_double),intent(in)::a(-3:n+4)
real(c_double),intent(inout)::b(-3:n+4)
call advance(a,b,n,t)
end subroutine
""")
    client = build / "client.cu"
    client.write_text("#define EXPECT_SOURCE_GPU " + str(int(synthetic_gpu)) + "\n" + r'''
#include "scoped_runtime.h"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" void checked_bridge(const double*,double*,int,int,fort_scope_t*,fort_buffer_t*,fort_buffer_t*);
extern "C" void reference_bridge(const double*,double*,int,int);
static void check(int s){if(s){std::fprintf(stderr,"%d %s\n",s,fort_scope_error());std::abort();}}
__global__ void producer(double* a,std::size_t n){auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]+=1.0/32;}
__global__ void consumer(const double* a,double* b,std::size_t n){auto i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)b[i]+=a[i]*.5;}
static int submit(void* opaque,const fort_scope_external_view_v1* v,std::size_t count,void* stream,int* started,std::uint64_t* accepted){
 *started=0;*accepted=0;const bool write_b=*static_cast<bool*>(opaque);if(count!=(write_b?2:1))return FORT_SCOPE_ARGUMENT;
 *started=1;const auto n=v[0].extents[0];
 if(write_b)consumer<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(static_cast<const double*>(v[0].device),static_cast<double*>(v[1].device),n);
 else producer<<<1,32,0,static_cast<cudaStream_t>(stream)>>>(static_cast<double*>(v[0].device),n);
 if(cudaGetLastError()!=cudaSuccess)return FORT_SCOPE_EXECUTION;*accepted=1;return FORT_SCOPE_OK;
}
static fort_scope_stats stats(fort_scope_t c){fort_scope_stats s{};check(fort_scope_stats_get(c,&s));return s;}
int main(){
 const int n=7,t=3;const std::size_t extent=n+8;const std::int64_t lower=-3;
 std::vector<double>a(extent),b(extent),expected_a(extent),expected_b(extent);
 for(std::size_t i=0;i<extent;++i){a[i]=double(int(i)-5)/64;b[i]=double(31-int(i))/32;}
 expected_a=a;expected_b=b;
 fort_scope_domain_t domain=0;fort_scope_t context=0;fort_buffer_t ha=0,hb=0;
 check(fort_scope_domain_create_v1(0,&domain));check(fort_scope_create_in_domain_v1(domain,&context));
 fort_scope_layout la{1,FORT_SCOPE_REAL64,8,a.data(),&extent,&lower,1},lb{1,FORT_SCOPE_REAL64,8,b.data(),&extent,&lower,1};
 check(fort_scope_register(context,1,1,&la,1,&ha));check(fort_scope_register(context,2,1,&lb,1,&hb));
 fort_scope_access read{};read.flags=FORT_SCOPE_READ_ALL;fort_scope_access rw{};rw.flags=FORT_SCOPE_READ_ALL|FORT_SCOPE_WRITE_ALL;
 for(int repeat=0;repeat<2;++repeat){
  bool output=false;fort_scope_external_binding_v1 p{ha,FORT_SCOPE_REAL64,1,8,rw};fort_scope_external_call_v1 pc{1,0,&p,1,0,0};
  fort_scope_external_report_v1 report{};check(fort_scope_source_execute_v1(context,&pc,1,submit,&output,&report));
  for(auto& value:expected_a)value+=1.0/32;
  reference_bridge(expected_a.data(),expected_b.data(),n,t);
  const auto before=stats(context);
  std::fprintf(stderr,"LIVE_ORIGINAL_BEGIN %d\n",repeat);
  checked_bridge(a.data(),b.data(),n,t,&context,&ha,&hb);
  std::fprintf(stderr,"LIVE_ORIGINAL_END %d\n",repeat);
  auto native=stats(context);
  if(EXPECT_SOURCE_GPU){
   assert(native.launches==before.launches+1&&native.upload_bytes==before.upload_bytes+(repeat?0:extent*8));
   assert(native.download_bytes==before.download_bytes);
   check(fort_scope_host_begin(context,ha,&read));check(fort_scope_host_end(context,ha));
   check(fort_scope_host_begin(context,hb,&read));check(fort_scope_host_end(context,hb));
   native=stats(context);
  }else{
   assert(native.launches==before.launches&&native.uploads==before.uploads);
   assert(native.download_bytes==before.download_bytes+extent*8);
  }
  assert(a==expected_a&&b==expected_b);
  output=true;fort_scope_external_binding_v1 c[2]={{ha,FORT_SCOPE_REAL64,1,8,read},{hb,FORT_SCOPE_REAL64,1,8,rw}};
  fort_scope_external_call_v1 cc{1,0,c,2,0,0};check(fort_scope_source_execute_v1(context,&cc,1,submit,&output,&report));
  const auto after=stats(context);assert(after.upload_bytes==native.upload_bytes+(EXPECT_SOURCE_GPU?0:extent*8));
  assert(after.download_bytes==native.download_bytes);
  for(std::size_t i=0;i<extent;++i)expected_b[i]+=expected_a[i]*.5;
  check(fort_scope_host_begin(context,hb,&read));check(fort_scope_host_end(context,hb));assert(b==expected_b);
 }
 assert(stats(context).launches==(EXPECT_SOURCE_GPU?6:4));
 check(fort_scope_close(context));check(fort_scope_domain_close_v1(domain));
 std::puts("GENERATED_NATIVE_LIVE_FULL_FIELDS_HALOS_READ_REPLICA_OK");
}
''')
    for index, source in enumerate((reference, bridge)):
        target = build / f"extra-{index}.o"
        commands.append([fc, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build),
                         "-c", str(source), "-o", str(target)])
        objects.append(str(target))
    commands.append([nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp", "-I", str(build),
                     "-c", str(client), "-o", str(build / "client.o")])
    executable = build / "client"
    commands.append([fc, "-fopenmp", str(build / "client.o"), *objects, "-L/usr/local/cuda/lib64",
                     "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(executable)])
    for index, argv in enumerate(commands):
        (build / f"command-{index}.json").write_text(json.dumps(argv) + "\n")
        result = run(argv, cwd=build)
        (build / f"command-{index}.stdout").write_text(result.stdout)
        (build / f"command-{index}.stderr").write_text(result.stderr)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "OMP_THREAD_LIMIT": "4", "OMP_MAX_ACTIVE_LEVELS": "1", "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    result = run([str(executable)], cwd=build, env=environment)
    (build / "client.stdout").write_text(result.stdout)
    (build / "client.stderr").write_text(result.stderr)
    assert "GENERATED_NATIVE_LIVE_FULL_FIELDS_HALOS_READ_REPLICA_OK" in result.stdout
    cases = re.findall(r"LIVE_ORIGINAL_BEGIN (\d+)\n(.*?)LIVE_ORIGINAL_END \1", result.stderr, re.S)
    assert len(cases) == 2
    for _, events in cases:
        expected = (r"available=1 gpu=1 reason=0" if synthetic_gpu else
                    r"available=1 gpu=0 reason=0" if profile["source_tasks"]["available"] else r"available=0 gpu=0 reason=1")
        assert re.search(r"FORT_SOURCE_QUOTE .*" + expected, events), events
        for name in ("prepare", "start", "commit"):
            assert events.count("FORT_SCOPED source_native_" + name + " ") == (0 if synthetic_gpu else 1)
        assert events.count("FORT_SCOPED launch ") == int(synthetic_gpu)
        if not synthetic_gpu:
            assert "FORT_SCOPED upload " not in events
