"""Valid graph AUTO CPU choices and readonly native mirrors on actual CUDA.

Rates are deliberately synthetic and bound to real hardware/tool identities.
This checks placement policy and coherence, never measured performance.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil

import pytest

from compiler.numerical_contract import require_numerical_build_contract
from compiler.tests.test_graph_native_choice import emit, source_case
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run

DRIVER = """program verify
use native_choice
implicit none
real(8),allocatable::a(:),b(:),c(:),d(:)
integer,allocatable::counter(:)
integer,parameter::sizes(3)=[0,1,257]
integer::shape,n,repeat,mode,i,unit
logical::flag
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
n=sizes(shape)
allocate(a(-3:n+4),b(-7:n),c(-2:n+5),d(-9:n-2),counter(-11:n-4))
do mode=1,2
flag=mode==2
do repeat=1,2
do i=lbound(a,1),ubound(a,1)
a(i)=real(i+5,8)*0.25d0+repeat
enddo
b=-99.d0
c=-77.d0
d=-55.d0
counter=0
call advance(a,b,c,d,counter,n,flag)
write(unit)n,mode,repeat,lbound(a),ubound(a),lbound(b),ubound(b),lbound(c),ubound(c)
write(unit)lbound(d),ubound(d),lbound(counter),ubound(counter),a,b,c,d,counter
enddo
enddo
deallocate(a,b,c,d,counter)
enddo
close(unit)
end program
"""


def policy_profile(observed, policy):
    profile = copy.deepcopy(observed)
    profile["rates"].update(cpu_flops_per_second=1e5, cpu_memory_bytes_per_second=1e12,
                            gpu_flops_per_second=1e13, gpu_memory_bytes_per_second=1e13)
    profile["scoped"]["costs"]["launch_enqueue_seconds"] = 1e-1
    for direction in ("h2d", "d2h"):
        profile["rates"][direction + "_pageable"].update(latency_seconds=1e-6,
                                                        bandwidth_bytes_per_second=1e12)
    if policy == "cpu":
        profile["rates"].update(cpu_flops_per_second=1e13, gpu_flops_per_second=1e8,
                                gpu_memory_bytes_per_second=1e8)
    return profile


@pytest.fixture(scope="module")
def native_choice_binaries(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA, C++ and Fortran toolchains required")
    directory = tmp_path_factory.mktemp("source_native_choices")
    observed = actual_profile(directory, nvcc, host)
    architecture = observed["hardware"]["compute_capability"].replace(".", "")
    flags = [fortran, "-O2", "-cpp", "-std=f2018", "-fopenmp", "-ffp-contract=off", "-fcheck=all,array-temps"]
    cases, runtime_objects = {}, {}
    for style in ("combined", "joined"):
        for policy in ("cpu", "mixed"):
            case = directory / (style + "_" + policy)
            profile = policy_profile(observed, policy)
            outputs, manifest, _ = emit(case, style, profile=profile)
            assert not manifest["scopes"][0]["boundaries"]
            for name, text in outputs.items():
                artifact = case / name
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(text)
            objects = []
            for role in ("common_runtime", "shared_entry", "original_source"):
                for item in manifest["build_sources"]:
                    if item["role"] != role:
                        continue
                    artifact = case / item["path"]
                    target = case / (item["path"].replace("/", "_") + ".o")
                    key = item["path"], manifest["artifacts_sha256"][item["path"]]
                    # Fortran modules must still be generated in each case.
                    if role == "common_runtime" and key in runtime_objects and item["language"] == "cuda":
                        objects.append(runtime_objects[key])
                        continue
                    if item["language"] == "cuda":
                        contract = item["numerical_contract"]
                        require_numerical_build_contract(contract)
                        compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host,
                            "-arch=sm_" + architecture, "-Xcompiler=-fopenmp",
                            *contract["required_cuda_options"],
                            *("-Xcompiler=" + option for option in contract["required_host_options"])]
                    else:
                        compiler = flags
                    run([*compiler, "-I", str(case), "-c", str(artifact), "-o", str(target)], cwd=case)
                    objects.append(str(target))
                    if role == "common_runtime":
                        runtime_objects[key] = str(target)
            driver = case / "driver.f90"
            driver.write_text(DRIVER)
            original = case / "native_original.f90"
            original.write_text(source_case(style))
            native, candidate = case / "native", case / "candidate"
            run([*flags, str(original), str(driver), "-o", str(native)], cwd=case)
            run([*flags, str(driver), *objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64",
                 "-lcudart", "-lstdc++", "-o", str(candidate)], cwd=case)
            (case / "manifest.json").write_text(json.dumps(manifest, indent=2))
            (case / "synthetic-policy-profile.json").write_text(json.dumps(profile, indent=2))
            cases[style, policy] = case, native, candidate, manifest
    return cases


@pytest.mark.cuda
@pytest.mark.parametrize("style", ["combined", "joined"])
@pytest.mark.parametrize("policy", ["cpu", "mixed"])
def test_valid_cpu_choices_retain_context_and_readonly_device_mirrors(native_choice_binaries, style, policy):
    case, native, candidate, manifest = native_choice_binaries[style, policy]
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    run([str(native)], cwd=case, env=env)
    expected = (case / "fields.bin").read_bytes()
    result = run([str(candidate)], cwd=case, env=env)
    (case / "candidate.stderr").write_text(result.stderr)
    assert (case / "fields.bin").read_bytes() == expected
    assert "array temporary" not in result.stderr.lower()
    lines = result.stderr.splitlines()
    events = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in lines
              if line.startswith("FORT_SCOPED evidence ")]
    decisions = [event for event in events if event["event"] == "decision"]
    assert decisions
    assert all(event["available"] == 1 for event in decisions)
    assert all(event["reason"] != "hardware_or_calibration_incompatible" for event in decisions)
    closes = [line for line in lines if line.startswith("FORT_SCOPED close ")]
    hints = [json.loads(line.removeprefix("FORT_SCOPED native_hint ")) for line in lines
             if line.startswith("FORT_SCOPED native_hint ")]
    native_hits = [event for event in hints if event.get("phase") == "lookup" and event.get("hit") is True]
    assert all(event.get("event") == "native_hint" and event.get("reason") == "conservative_native_owner_hint"
               and event.get("estimates_available") is False and event.get("observation_eligible") is False
               for event in native_hits)
    assert len(closes) == 12-len(native_hits)  # Each non-hint invocation retains one coherent context.
    if style == "combined":
        assert any(line.startswith("FORT_SCOPED plan_native_begin ") for line in lines)
        assert all(item["operations"]["combined_numerical_team"]["original_cpu_alternative"]["available"]
                   for item in manifest["scopes"][0]["planning_segments"])
    if policy == "cpu":
        assert all(event["gpu_units"] == 0 for event in decisions)
        assert any(event["cpu_units"] for event in decisions)
        for action in ("initialize", "upload", "download", "allocate", "launch", "graph_submit"):
            assert not any(line.startswith("FORT_SCOPED " + action) for line in lines)
    else:
        assert any(event["gpu_units"] for event in decisions)
        assert any(event["cpu_units"] for event in decisions)
        contexts = {event["context"] for event in decisions}
        assert any(any(event["context"] == context and event["gpu_units"] for event in decisions)
            and any(event["context"] == context and event["cpu_units"]
                    for event in decisions) for context in contexts)
        if style == "combined":
            assert any(event["gpu_units"] and event["cpu_units"] for event in decisions)
            assert any(region["workers"] == 3 for region in manifest["scopes"][0]["query_regions"])
        ids = {item["resource"]: item["registration_identity"]
               for item in manifest["scopes"][0]["resource_bindings"]["resources"]}
        uploads = [line for line in lines if line.startswith("FORT_SCOPED upload ")]
        a_uploads = [line for line in uploads if re.search(r"\bbuffer=" + str(ids["argument::a"]) + r"\b", line)]
        assert len(a_uploads) == 4
        assert all("bytes=2056 " in line for line in a_uploads)
        # Producer overwrites B; its CPU reader publishes a host mirror without
        # invalidating B's GPU copy, so the consumer never uploads B.
        assert not any(re.search(r"\bbuffer=" + str(ids["argument::b"]) + r"\b", line) for line in uploads)
