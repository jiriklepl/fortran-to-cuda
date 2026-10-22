"""Actual CUDA source query widening with explicit synthetic policy costs.

This is correctness and global-decision evidence, never a performance result.
Both programs use the same source and hardware-bound synthetic rates. Only
metadata-only query-region issuance is disabled in the local comparison.
The dyadic multiply plus add keeps the native/GPU reference bit exact while
providing a clear policy margin with the corrected floating-operation count.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pytest

from compiler.driver.options import CompilerOptions
from compiler.numerical_contract import require_numerical_build_contract
from compiler.offload.config import OffloadConfig
from compiler.scopes import query_regions
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run

SOURCE = """module query_tiles
implicit none
integer,protected::tiles(1)=[1]
end module
module wide_query
use query_tiles
implicit none
contains
subroutine update(a,n,value)
real(8),intent(inout)::a(:)
integer,intent(in)::n
real(8),intent(in)::value
integer::i,tnx
real(8)::coefficient
tnx=tiles(1)
coefficient=value/2.0_8
!$omp parallel do private(i) shared(a,n,tnx,coefficient) default(none)
do i=1,n-tnx+1
a(i)=2.0_8*a(i)+coefficient
enddo
!$omp end parallel do
end subroutine
subroutine finish(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i) shared(a,n) default(none)
do i=1,n
a(i)=2.0_8*a(i)
enddo
!$omp end parallel do
end subroutine
subroutine advance(a,b,n,flag,counter)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
logical,intent(in)::flag
integer,intent(inout)::counter
if(flag)then
""" + "".join(f"call update({'a' if index % 2 else 'b'},n,{index}.0_8)\n" for index in range(1, 24)) + """call finish(a,n)
else
call update(a,n,-3.0_8)
call update(b,n,-4.0_8)
endif
counter=counter+1
end subroutine
end module
"""

DRIVER = """program verify
use wide_query
implicit none
real(8),allocatable::a(:),b(:)
integer,parameter::sizes(3)=[0,1,1024]
integer::shape,n,mode,repeat,i,unit,counter
logical::flag
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
n=sizes(shape)
allocate(a(-3:n+4),b(-7:n))
do mode=2,1,-1
flag=mode==2
counter=0
do repeat=1,2
do i=lbound(a,1),ubound(a,1)
a(i)=real(i+5,8)*0.25d0+repeat
enddo
do i=lbound(b,1),ubound(b,1)
b(i)=real(i+9,8)*0.5d0+repeat
enddo
call advance(a,b,n,flag,counter)
write(unit)n,mode,repeat,counter,lbound(a),ubound(a),lbound(b),ubound(b),a,b
enddo
enddo
deallocate(a,b)
enddo
close(unit)
end program
"""


def synthetic_profile(observed, mode):
    profile = copy.deepcopy(observed)
    profile["rates"].update(cpu_flops_per_second=2e6, cpu_memory_bytes_per_second=1e12,
        cpu_worker_flops_per_second=2e6, cpu_worker_memory_bytes_per_second=1e12,
        gpu_flops_per_second=1e13, gpu_memory_bytes_per_second=1e13)
    for direction in ("h2d", "d2h"):
        profile["rates"][direction + "_pageable"].update(latency_seconds=3e-3, bandwidth_bytes_per_second=1e12)
    profile["scoped"]["costs"].update(cold_driver_startup_seconds=1.0,
        gpu_setup_seconds=1e-6, allocation_seconds=1e-6, release_seconds=1e-6,
        launch_enqueue_seconds=1e-6)
    if mode == "cpu":
        profile["rates"].update(cpu_flops_per_second=1e13, cpu_worker_flops_per_second=1e13,
            gpu_flops_per_second=1e8, gpu_memory_bytes_per_second=1e8)
    return profile


@pytest.fixture(scope="module")
def query_binaries(tmp_path_factory):
    prepared = os.environ.get("FORT_QUERY_PREBUILT")
    if prepared:
        return {mode: tuple(Path(value) for value in values) for mode, values in
            json.loads(Path(prepared).read_text()).items()}
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA, C++ and Fortran toolchains required")
    directory = tmp_path_factory.mktemp("wide_source_queries")
    observed_path = os.environ.get("FORT_QUERY_PROFILE")
    observed = json.loads(Path(observed_path).read_text()) if observed_path else actual_profile(directory, nvcc, host)
    architecture = observed["hardware"]["compute_capability"].replace(".", "")
    flags = [fortran, "-O2", "-cpp", "-std=f2018", "-fopenmp", "-ffp-contract=off", "-fcheck=all,array-temps"]
    binaries, common_objects = {}, {}
    source = directory / "original.f90"
    source.write_text(SOURCE)
    for mode in ("local", "wide", "cpu", "mismatch", "postwork"):
        profile = synthetic_profile(observed, mode)
        case = directory / mode
        case.mkdir()
        builder = ScopeBuilder([source], "wide_query::advance",
            facts={"schema_version": 1, "participation": "serial",
                "sources": {str(source): sha256(source.read_bytes()).hexdigest()},
                "captures": {"argument::a": dict(FACT), "argument::b": dict(FACT), "query_tiles::tiles": dict(FACT)}},
            options=CompilerOptions(opt_level=0), config=OffloadConfig("auto", profile))
        if mode == "local":
            original = query_regions.discover_query_regions

            def no_regions(owner, original=original):
                original(owner)
                owner.emission.query_drafts = ()

            with patch.object(query_regions, "discover_query_regions", side_effect=no_regions):
                outputs, manifest = builder.run()
        else:
            outputs, manifest = builder.run()
            assert max(region["workers"] for region in manifest["scopes"][0]["query_regions"]) == 24
        if mode in {"mismatch", "postwork"}:
            finish = next(item.scoped for name, item in builder.generated.items()
                          if name.startswith("wide_query::finish#"))
            matched = 0
            for name, text in tuple(outputs.items()):
                if not name.endswith("shared_entry.cu") or 'extern "C" int ' + finish["entry"] + "(" not in text:
                    continue
                if mode == "mismatch":
                    text, changed = re.subn(r"(fort_access\.record(?:_compute)?\(FORT_SCOPE_PLAN_WORKER, )(\d+)ULL",
                        lambda match: match[1] + str(int(match[2]) + 1) + "ULL", text)
                    assert changed == 1
                else:
                    marker = "    return FORT_SCOPE_OK;\n    }();\n    if (fort_entry_result && fort_entry_work_started)"
                    assert marker in text
                    text = text.replace(marker, "    return fort_scope_report_error(FORT_SCOPE_PLAN_MISMATCH, &FAULT&);\n"
                        "    }();\n    if (fort_entry_result && fort_entry_work_started)".replace("&FAULT&", '"injected postwork mismatch"'), 1)
                    # Owning original teams use the collective entry. Inject
                    # after its queue acceptance, inside the existing wrapper
                    # that upgrades every postwork status to EXECUTION.
                    pattern = (r"(?P<space>[ \t]*)return FORT_SCOPE_OK;"
                        r"(?P<end>\n[ \t]*\}\(\);\n[ \t]*if \(shared->status == FORT_SCOPE_OK && shared->work_started\))")
                    text, changed = re.subn(pattern, lambda match:
                        match["space"] + 'if (shared->work_started) return fort_scope_report_error('
                        'FORT_SCOPE_PLAN_MISMATCH, "injected collective postwork mismatch");\n' + match[0], text)
                    assert changed >= 1
                    text, queued = re.subn(r"(shared->work_started = true;\n[ \t]*)return FORT_SCOPE_OK;",
                        r'\1return fort_scope_report_error(FORT_SCOPE_PLAN_MISMATCH, "injected queued postwork mismatch");', text)
                    assert queued == 1
                outputs[name] = text
                matched += 1
            assert matched == 1
            (case / "fault-injection.json").write_text(json.dumps({"kind": mode,
                "purpose": "test-only source entry failure; never application qualification",
                "emitted_artifact_hashes": manifest["artifacts_sha256"],
                "compiled_artifact_hashes": {name: sha256(text.encode()).hexdigest() for name, text in outputs.items()}}, indent=2))
        (case / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (case / "synthetic-policy-profile.json").write_text(json.dumps(profile, indent=2))
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
                key = item["path"], sha256(artifact.read_bytes()).hexdigest()
                if item["language"] == "cuda" and key in common_objects:
                    objects.append(common_objects[key])
                    continue
                if item["language"] == "cuda":
                    contract = item["numerical_contract"]
                    require_numerical_build_contract(contract)
                    compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_" + architecture,
                        "-Xcompiler=-fopenmp", *contract["required_cuda_options"],
                        *("-Xcompiler=" + option for option in contract["required_host_options"])]
                else:
                    compiler = flags
                run([*compiler, "-I", str(case), "-c", str(artifact), "-o", str(target)], cwd=case)
                objects.append(str(target))
                if item["language"] == "cuda":
                    common_objects[key] = str(target)
        driver = case / "driver.f90"
        driver.write_text(DRIVER if mode != "postwork" else """program verify
use wide_query
implicit none
real(8)::a(-3:1028),b(-7:1024)
integer::counter
a=1.0_8
b=2.0_8
counter=0
call advance(a,b,1024,.true.,counter)
print *, 'UNSAFE_POSTWORK_RETURN',counter
end program
""")
        native, candidate = case / "native", case / "candidate"
        run([*flags, str(source), str(driver), "-o", str(native)], cwd=case)
        run([*flags, str(driver), *objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64",
            "-lcudart", "-lstdc++", "-o", str(candidate)], cwd=case)
        binaries[mode] = case, native, candidate
    (directory / "prebuilt.json").write_text(json.dumps(
        {mode: [str(value) for value in values] for mode, values in binaries.items()}, indent=2))
    return binaries


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_wide_query_amortizes_cold_first_task_and_preserves_original_paths(query_binaries, disabled):
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        env["CUDA_VISIBLE_DEVICES"] = ""
    all_decisions = {}
    for mode in ("local", "wide", "cpu"):
        case, native, candidate = query_binaries[mode]
        run([str(native)], cwd=case, env=env)
        expected = (case / "fields.bin").read_bytes()
        result = run([str(candidate)], cwd=case, env=env)
        assert (case / "fields.bin").read_bytes() == expected
        (case / ("disabled.stderr" if disabled else "enabled.stderr")).write_text(result.stderr)
        assert "array temporary" not in result.stderr.lower()
        events = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in result.stderr.splitlines()
            if line.startswith("FORT_SCOPED evidence ")]
        decisions = [event for event in events if event["event"] == "decision"]
        assert decisions
        all_decisions[mode] = decisions
        if disabled or mode in {"local", "cpu"}:
            assert all(event["gpu_units"] == 0 for event in decisions)
            if not disabled:
                assert all(event["available"] == 1 for event in decisions)
            for action in ("initialize", "upload", "download", "allocate", "launch", "graph_submit"):
                assert "FORT_SCOPED " + action + " " not in result.stderr
        else:
            assert any(event["gpu_units"] >= 8 for event in decisions)
            selected = [event for event in decisions if event["gpu_units"]]
            assert selected[0]["cold_driver_startup_debt_seconds"] == 1.0
            assert selected[0]["warm_gate_seconds"] <= (selected[0]["warm_gate_counterfactual_seconds"]
                - selected[0]["warm_gate_required_saving_seconds"])
            assert any("FORT_SCOPED " + action + " " in result.stderr for action in ("launch", "graph_submit"))
            uploads = re.findall(r"^FORT_SCOPED upload .*", result.stderr, re.M)
            assert len(uploads) == 4  # Two actual bindings in two n=1024 invocations.
    assert len(all_decisions["wide"]) < len(all_decisions["local"])


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_prework_mismatch_keeps_prefix_and_executes_current_original_once(query_binaries, disabled):
    case, native, candidate = query_binaries["mismatch"]
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        env["CUDA_VISIBLE_DEVICES"] = ""
    run([str(native)], cwd=case, env=env)
    expected = (case / "fields.bin").read_bytes()
    result = run([str(candidate)], cwd=case, env=env)
    (case / ("disabled.stderr" if disabled else "enabled.stderr")).write_text(result.stderr)
    assert (case / "fields.bin").read_bytes() == expected
    if not disabled:
        assert any("FORT_SCOPED " + action + " " in result.stderr for action in ("launch", "graph_submit"))


@pytest.mark.cuda
def test_postwork_failure_is_fatal_and_never_replays_current_original(query_binaries):
    case, _native, candidate = query_binaries["postwork"]
    import subprocess

    result = subprocess.run([str(candidate)], cwd=case, capture_output=True, text=True, timeout=30,
        env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"})
    (case / "candidate.stderr").write_text(result.stderr)
    assert result.returncode != 0
    assert "UNSAFE_POSTWORK_RETURN" not in result.stdout
    assert ("unsafe replay prohibited" in result.stderr or "after execution or invalid state" in result.stderr
        or "inherited collective entry executed or invalid state" in result.stderr)
    assert "FORT_SCOPED close " not in result.stderr
