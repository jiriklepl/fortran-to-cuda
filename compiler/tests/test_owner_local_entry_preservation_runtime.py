"""Preserved entry definitions feed real source teams through one local graph."""
import json
import os
import re
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_owner_local_call_groups_runtime import DRIVER, SOURCE, build_local_group_binaries


def team(statement):
    return f"""!$omp parallel private(bi,i,li,tmp)
!$omp do schedule(runtime)
do bi=1,n,4
 do i=bi,min(bi+3,n)
  li=i-bi+1
  tmp(li)=c(i)
 enddo
 do i=bi,min(bi+3,n)
  li=i-bi+1
  {statement}
 enddo
enddo
!$omp end do
!$omp end parallel"""


TILED = SOURCE.replace("integer::prefix_visits=0,suffix_visits=0",
    "integer::prefix_visits=0,suffix_visits=0\nlogical::write_enabled=.false.\nreal(8)::probe_value=0")
for target, statement in (("b(i)=b(i)+c(i)", "b(i)=b(i)+tmp(li)"),
                          ("a(i)=b(i)+c(i)", "a(i)=b(i)+tmp(li)"),
                          ("a(i)=a(i)+c(i)", "a(i)=a(i)+tmp(li)")):
    old = "integer::i\ndo i=1,n\n" + target + "\nenddo"
    before = "" if target != "a(i)=b(i)+c(i)" else (
        "if(n>=7)probe_value=sum(b(3:7))\n"
        "if(write_enabled.and.n>=6)c(4:6)=c(4:6)+0.5_8\n")
    TILED = TILED.replace(old, "integer::bi,i,li\nreal(8)::tmp(4)\n" + before + team(statement))
TILED = TILED.replace("real(8),intent(in)::b(:),c(:)",
                       "real(8),intent(in)::b(:)\nreal(8),intent(inout)::c(:)")
TILED = TILED.replace("call unknown_prefix()\nif(flag)then",
    "prefix_visits=prefix_visits+1\nif(flag)then\n"
    "if(n>=3)b(lbound(b,1)+1:lbound(b,1)+2)=b(lbound(b,1)+1:lbound(b,1)+2)+0.125_8")

EXPECTED = DRIVER.replace("alias=trim(mode)=='alias'", "alias=.false.\nwrite_enabled=trim(mode)=='writer'")
EXPECTED = EXPECTED.replace("  do i=1,n\n   if(alias)then", "  if(n>=3)eb(8:9)=eb(8:9)+0.125_8\n"
    "  do i=1,n\n   eb(i+6)=eb(i+6)+c(i-9)\n  enddo\n"
    "  if(write_enabled.and.n>=6)ec(-5:-3)=ec(-5:-3)+0.5_8\n"
    "  do i=1,n\n   if(alias)then")
EXPECTED = EXPECTED.replace("    eb(i+6)=eb(i+6)+c(i-9)\n    ea(i-4)=eb(i+6)+2.0_8*c(i-9)",
                            "    ea(i-4)=eb(i+6)+2.0_8*ec(i-9)")
EXPECTED = EXPECTED.replace(" write(error_unit,'(A,1X,I0)')'LOCAL_GROUP_BEGIN',repeat",
                            " probe_value=0\n write(error_unit,'(A,1X,I0)')'LOCAL_GROUP_BEGIN',repeat")
EXPECTED = EXPECTED.replace("  if(any(a/=ea).or.any(c/=ec))", "  if(n>=7)then\n"
    "   if(probe_value/=sum(eb(9:13)))error stop 'native readonly observation'\n"
    "  endif\n  if(any(a/=ea).or.any(c/=ec))")


def test_three_checked_child_providers_publish_every_exact_artifact_in_the_owning_graph(tmp_path):
    path = tmp_path/"teams.f90"
    path.write_text(TILED)
    fact = {"storage": "stable", "initialized": "whole", "escapes": False,
            "allocation_changes": False, "lifetime": "invocation"}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::"+name: dict(fact) for name in ("a", "b", "c")}}
    contracts = {"live_groups::run": {"native_ancestor": {"version": 2,
        "lifetime": "owner_local_call_groups", "device_roots": ["live_groups::"+name
            for name in ("first", "second", "third")]}}}
    builder = ScopeBuilder([path], "live_groups::run", facts=facts, contracts=contracts,
                           options=CompilerOptions(), config=OffloadConfig(policy="sections"))
    outputs, manifest = builder.run()
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    second = next(member for member in manifest["scopes"][0]["module_coordinators"]
                  if member["procedure"] == "live_groups::second")
    assert second["boundaries"] == [], second["boundaries"]
    replacement = outputs[manifest["sources"][str(path)]["replacement"]]
    assert "bind(C, name='fort_scope_task_graph_enable_v1')" in replacement
    plan = manifest["scopes"][0]["checked_execution_plan"]
    tasks = [task for member in plan["storage_epoch"]["members"] for task in member.get("source_device_tasks", [])]
    assert len(tasks) == 3
    sources = {item["path"]: item for item in manifest["build_sources"]}
    for task in tasks:
        provider = task["execution_binding"]["provider"]
        for name, expected in ((provider["entry"]+".cu", provider["cuda_source_sha256"]),
                               (provider["module"]+".f90", provider["fortran_source_sha256"])):
            assert name in outputs
            assert sources[name]["role"] == "shared_entry"
            assert sha256(outputs[name].encode()).hexdigest() == expected
            assert manifest["artifacts_sha256"][name] == expected


@pytest.fixture(scope="module")
def preserved_group_binaries(tmp_path_factory):
    result = build_local_group_binaries(tmp_path_factory, source=TILED, driver_source=EXPECTED,
                                       initialized="whole", lifetime="invocation")
    _build, manifest, _candidate, _native = result
    plan = manifest["scopes"][0]["checked_execution_plan"]
    epoch = plan["storage_epoch"]["selected_position_epoch"]
    assert epoch["version"] == 3
    assert epoch["entry_preservation"]["version"] == 2
    assert set(epoch["entry_preservation"]["preserved_resources"]) == {
        "argument::a", "argument::b", "argument::c"}
    tasks = [task for member in plan["storage_epoch"]["members"] for task in member.get("source_device_tasks", [])]
    assert len(tasks) == 3
    for task in tasks:
        metadata = task["execution_binding"]["provider"]["metadata"]
        assert task["version"] == 3
        assert metadata["version"] == 10
        assert metadata["field_transport"]["version"] == 2
        assert metadata["field_transport"]["definitions_authorized"] is False
        assert metadata["access_envelopes"]["initialization_authorized"] is False
    return result


@pytest.mark.native
def test_preserved_entry_source_teams_and_original_abi_compile_without_cuda(preserved_group_binaries):
    _build, _manifest, candidate, native = preserved_group_binaries
    assert candidate.is_file()
    assert native.is_file()


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["resident", "writer", "late", "zero", "unallocated", "hidden"])
def test_preserved_source_teams_keep_coherence_and_never_reseed_retired_entry(preserved_group_binaries, mode):
    build, manifest, candidate, native = preserved_group_binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
           "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("FORT_SCOPE_TEST_FAIL_ALLOC_AFTER", None)
    if mode == "hidden":
        env["CUDA_VISIBLE_DEVICES"] = ""
    if mode == "late":
        # Runtime field-allocation statistics exclude tile scratch. The first
        # team allocates B/C; A is the third field allocation, after one GPU
        # commit and the original native probe. Reject A before its work.
        env["FORT_SCOPE_TEST_FAIL_ALLOC_AFTER"] = "2"
    results = []
    for role, binary in (("native", native), ("candidate", candidate)):
        argv = [str(binary), mode]
        (build/f"{mode}-{role}.command.json").write_text(json.dumps({"argv": argv,
            "CUDA_VISIBLE_DEVICES": env.get("CUDA_VISIBLE_DEVICES"),
            "FORT_SCOPE_TEST_FAIL_ALLOC_AFTER": env.get("FORT_SCOPE_TEST_FAIL_ALLOC_AFTER")}, indent=2)+"\n")
        result = subprocess.run(argv, cwd=build, env=env, capture_output=True, text=True, timeout=30)
        for suffix, text in (("stdout", result.stdout), ("stderr", result.stderr)):
            (build/f"{mode}-{role}.{suffix}").write_text(text)
        (build/f"{mode}-{role}.completion.json").write_text(json.dumps({"returncode": result.returncode})+"\n")
        assert result.returncode == 0, result.stdout+result.stderr
        results.append(result)
    reference, result = results
    assert result.stdout == reference.stdout
    launches = [line for line in result.stderr.splitlines() if line.startswith("FORT_SCOPED launch ")]
    copies = {kind: [dict(re.findall(r"(\w+)=([^ ]+)", line)) for line in result.stderr.splitlines()
        if line.startswith("FORT_SCOPED "+kind+" ")] for kind in ("upload", "download")}
    resources = manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]["resources"]
    slots = {item["resource"]: item["registration_identity"] for item in resources}
    ledger = {"mode": mode, "launches": len(launches), "domains": result.stderr.count('"event":"domain_create"'),
        "upload_bytes": sum(int(item["bytes"]) for item in copies["upload"]),
        "download_bytes": sum(int(item["bytes"]) for item in copies["download"]), "copies": copies,
        "field_allocations": sum(line.startswith("FORT_SCOPED allocate ") for line in result.stderr.splitlines()),
        "source_commits": sum(line.startswith("FORT_SCOPED external_commit ") for line in result.stderr.splitlines()),
        "canonical_slots": slots}
    (build/f"{mode}-ledger.json").write_text(json.dumps(ledger, indent=2)+"\n")
    if mode in {"hidden", "zero", "unallocated"}:
        assert not launches, result.stderr
        assert not copies["upload"], result.stderr
        assert not copies["download"], result.stderr
        if mode in {"zero", "unallocated"}:
            assert not re.search(r"FORT_SCOPED (initialize|allocate) ", result.stderr), result.stderr
    elif mode in {"resident", "writer"}:
        assert len(launches) == 6, result.stderr
        assert ledger["domains"] == 2, result.stderr
        c_uploads = [item for item in copies["upload"] if int(item["buffer"]) == slots["argument::c"]]
        expected = [152, 24, 152, 24] if mode == "writer" else [152, 152]
        assert [int(item["bytes"]) for item in c_uploads] == expected, result.stderr
        b_uploads = [item for item in copies["upload"] if int(item["buffer"]) == slots["argument::b"]]
        assert [int(item["bytes"]) for item in b_uploads] == [152, 152], result.stderr
        assert ledger["download_bytes"] == 608, result.stderr
    else:
        # The second team falls back before submission and retires the old
        # domain. Source-v2 transport lets the third team acquire the native
        # result in a fresh domain without reusing retired entry definitions.
        assert len(launches) == 2, result.stderr
        launched = [dict(re.findall(r"(\w+)=([^ ]+)", line)) for line in launches]
        assert len({item["domain"] for item in launched}) == 2, result.stderr
        assert len({item["implementation"] for item in launched}) == 2, result.stderr
        assert ledger["domains"] == 2, result.stderr
        assert ledger["field_allocations"] == 4, result.stderr
        assert ledger["source_commits"] == 2, result.stderr
        assert ledger["upload_bytes"] == 608, result.stderr
        assert ledger["download_bytes"] == 304, result.stderr
