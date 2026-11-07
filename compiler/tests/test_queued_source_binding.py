"""Checked graph emission and strict transport integrity for queued source work."""
import json
import os
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_device_tasks import source_device_task
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes import source_device_tasks
from compiler.scopes.queued_source_binding import execution_record
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_tile_private import SOURCE
from compiler.tests.test_tile_private_team import team_fixture


def issued(directory):
    _path, analysis, routine, nodes = team_fixture(directory, SOURCE)
    proof = source_device_task(analysis, routine, nodes)
    provider = source_device_tasks._provider(proof, transport="queued_single")
    resources = {binding.root: {"resource": binding.root, "descriptor": binding.public(),
                 "device_replica_authorized": True, "registration_identity": index}
                 for index, (binding, _written) in enumerate(proof.fields, 1)}
    binding = {"owner": routine.qualified, "canonical_resources": {root: root for root in resources},
               "registration_slots": {root: value["registration_identity"] for root, value in resources.items()},
               "provider": source_device_tasks._provider_record(provider)}
    record = execution_record(proof, binding)
    member = {"procedure": routine.qualified, "source_graph": proof.public()["source_graph"]}
    manifest = {"artifacts_sha256": {provider.entry + ".cu": sha256(provider.cuda_source.encode()).hexdigest(),
                                   provider.module + ".f90": sha256(provider.fortran_source.encode()).hexdigest()}}
    return record, member, {"procedure": routine.qualified, "calls": []}, analysis.inputs.identity(), resources, manifest


def validate(values):
    from elmm_pipeline.source_task_record import validate_source_task

    record, member, procedure, inputs, resources, manifest = values
    validate_source_task(record, member, procedure, inputs, resources, owner=procedure["procedure"], manifest=manifest)


def reseal(record):
    from elmm_pipeline.checked_execution_plan import _digest

    record["identity"] = _digest({key: value for key, value in record.items() if key != "identity"})


def test_real_original_numerical_proof_and_checked_queued_artifact_round_trip(tmp_path):
    values = issued(tmp_path)
    validate(values)
    record = values[0]
    metadata = record["execution_binding"]["provider"]["metadata"]
    assert metadata["version"] == 2
    assert metadata["consumer_wiring"] == "checked_source_device_task_v1"
    assert record["source_proof"]["transaction"]["runtime"] == "fort_scope_source_execute_v1"
    assert record["transport"]["backend"] == "fort_scope_graph_enqueue_v1+fort_scope_graph_flush_v1"
    assert record["source_proof_role"] == "original numerical work and scalar readability only"
    assert metadata["effects"]["initialization_authorized"] is False


@pytest.mark.parametrize("fault", ["version", "extra", "pricing", "backend", "source_role", "completion",
                                  "replay_status", "metadata_standalone", "code_key", "artifact", "canonical",
                                  "slot", "initialization", "overwrite", "units", "launches", "queue",
                                  "report", "abi_bool", "abi_bytes", "parameter", "empty_queue"])
def test_resealed_transport_cannot_gain_direct_costs_replay_or_stale_storage(tmp_path, fault):
    values = issued(tmp_path)
    validate(values)
    record, _member, _procedure, _inputs, _resources, manifest = values
    binding = record["execution_binding"]
    metadata = binding["provider"]["metadata"]
    if fault == "version":
        record["version"] = True
    elif fault == "extra":
        record["placement"] = {}
    elif fault == "pricing":
        record["pricing"] = {}
    elif fault == "backend":
        record["transport"]["backend"] = "fort_scope_source_execute_v1"
    elif fault == "source_role":
        record["source_proof_role"] = "direct transaction authority"
    elif fault == "completion":
        record["transport"]["committed"] = "device completion"
    elif fault == "replay_status":
        record["transport"]["recoverable_preaccept_statuses"].append(6)
    elif fault == "metadata_standalone":
        metadata.update(version=1, consumer_wiring=False)
    elif fault == "code_key":
        metadata["code_key"] = "1"
    elif fault == "artifact":
        manifest["artifacts_sha256"].clear()
    elif fault == "canonical":
        roots = list(binding["canonical_resources"])
        binding["canonical_resources"][roots[0]] = roots[1]
    elif fault == "slot":
        root = next(iter(binding["registration_slots"]))
        binding["registration_slots"][root] += 1
    elif fault == "initialization":
        metadata["effects"]["initialization_authorized"] = True
    elif fault == "overwrite":
        metadata["effects"]["overwrites_authorized"] = True
    elif fault == "units":
        metadata["restriction"]["units"] = 2
    elif fault == "launches":
        metadata["restriction"]["expected_launches"] = 2
    elif fault == "queue":
        metadata["restriction"]["queue_enabled"] = False
    elif fault == "report":
        metadata["report"]["fields"][2] = "started"
    elif fault == "abi_bool":
        metadata["launch_abi"]["parameters"] = True
    elif fault == "abi_bytes":
        metadata["launch_abi"]["bytes"] = 0
    elif fault == "parameter":
        metadata["parameters"][0]["kind"] = "c_double"
    elif fault == "empty_queue":
        metadata["restriction"]["empty_queue_at_entry"] = False
    reseal(record)
    with pytest.raises(ValueError, match="queued|source"):
        validate(values)


def emit(directory, monkeypatch, *, queued=True, source=SOURCE, entry="stencil::advance", captures=("a", "b")):
    directory.mkdir()
    path = directory / "source.f90"
    path.write_text(source)
    if queued:
        monkeypatch.setattr(source_device_tasks, "admit_source_device_task",
                            lambda member, nodes: source_device_tasks._admit_source_device_task(
                                member, nodes, transport="queued_single"))
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: dict(FACT) for name in captures}}
    builder = ScopeBuilder([path], entry, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"), facts=facts)
    outputs, manifest = builder.run()
    return outputs, manifest, facts


def test_genuine_checked_owner_manifest_and_new_report_compile(tmp_path, monkeypatch):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    outputs, manifest, facts = emit(tmp_path / "inputs", monkeypatch)
    assert manifest["scope_count"] == 1
    task, = manifest["scopes"][0]["source_device_tasks"]
    assert task["kind"] == "queued_single_original_tile_private_team"
    validate_checked_execution_plans(manifest, capture_facts=facts)
    original = next(text for name, text in outputs.items() if name.startswith("sources/"))
    assert "fort_tile_queue_report_v1" in original
    assert "%started" not in original
    assert "fort_scope_abandon" in original
    assert original.index("if ((t) > 0)") < original.index("fort_source_status = fort_tile_task_")
    assert original.index("if ((1) <= (n))") < original.index("fort_source_status = fort_tile_task_")
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        artifact = build / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (tmp_path / "facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert fortran is not None, "this finite gate requires its original Fortran tool"
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["language"] != "fortran" or artifact["role"] != role:
                continue
            argv = [fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-c",
                    str(build / artifact["path"]), "-J", str(build), "-I", str(build),
                    "-o", str(build / f"source-{index}.o")]
            result = subprocess.run(argv, cwd=build, capture_output=True, text=True, timeout=30)
            (build / f"compile-{index}.json").write_text(json.dumps({"argv": argv, "returncode": result.returncode}, indent=2))
            (build / f"compile-{index}.stdout").write_text(result.stdout)
            (build / f"compile-{index}.stderr").write_text(result.stderr)
            assert result.returncode == 0, result.stdout + result.stderr


def test_direct_default_does_not_select_queued_transport(tmp_path, monkeypatch):
    _outputs, manifest, _facts = emit(tmp_path / "inputs", monkeypatch, queued=False)
    task, = manifest["scopes"][0]["source_device_tasks"]
    assert task["kind"] == "original_tile_private_team"
    assert task["execution_binding"]["provider"]["metadata"]["version"] == 10


def test_actual_generated_caller_report_branches_never_replay_accepted_work(tmp_path, monkeypatch):
    """Compile the real caller against the existing CPU runtime and a report double."""
    outputs, manifest, _facts = emit(tmp_path / "inputs", monkeypatch)
    task, = manifest["scopes"][0]["source_device_tasks"]
    entry = task["execution_binding"]["provider"]["entry"]
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        artifact = build / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    cpp = shutil.which("g++-14") or shutil.which("g++")
    assert fortran is not None
    assert cpp is not None
    commands = []

    def compile_command(argv):
        result = subprocess.run(argv, cwd=build, capture_output=True, text=True, timeout=60)
        index = len(commands)
        commands.append({"argv": argv, "returncode": result.returncode})
        (build / f"command-{index}.stdout").write_text(result.stdout)
        (build / f"command-{index}.stderr").write_text(result.stderr)
        (build / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr

    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["language"] != "fortran" or artifact["role"] != role:
                continue
            target = build / f"source-{index}.o"
            compile_command([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-c",
                             str(build / artifact["path"]), "-J", str(build), "-I", str(build), "-o", str(target)])
            objects.append(target)
    (build / "provider-double.cpp").write_text('''
#include "scoped_runtime.h"
#include <cstdio>
#include <cstdlib>
#include <string>
struct Report { unsigned version; int status; unsigned accepted,committed; };
static_assert(sizeof(Report)==16);
extern "C" int ENTRY(fort_scope_t,fort_buffer_t,fort_buffer_t,const int*,const int*,const int*,const int*,const int*,const int*,Report* r) {
 const std::string mode=std::getenv("QUEUED_REPORT_MODE");
 std::fputs("PROVIDER_CALLED\\n",stderr);std::fflush(stderr);
 *r={1,0,1,1};int status=0;
 if(mode=="arg")status=1;
 if(mode=="resource")status=4;
 if(mode=="boundary")status=5;
 if(mode=="uninit")status=7;
 if(mode=="plan")status=9;
 if(status){r->accepted=0;r->committed=0;}
 if(mode=="accepted"){status=6;r->committed=0;}
 if(mode=="fatal_preaccept"){status=6;r->accepted=0;r->committed=0;}
 if(mode=="bad_version")r->version=2;
 if(mode=="success_no_commit")r->committed=0;
 if(mode=="commit_without_accept")r->accepted=0;
 if(mode=="success_empty"){r->accepted=0;r->committed=0;}
 if(mode=="negative_flag")r->accepted=unsigned(-1);
 r->status=status;
 if(mode=="bad_status"){status=4;r->accepted=0;r->committed=0;}
 return status;
}
'''.replace("ENTRY", entry))
    (build / "main.f90").write_text('''
program control
use stencil,only:advance
implicit none
character(32)::mode
integer::n,i
real(8),allocatable::a(:),b(:),before(:),expected(:)
call get_environment_variable('QUEUED_REPORT_MODE',mode)
n=7
if(trim(mode)=='zero')n=0
allocate(a(-3:n+4),b(-3:n+4),before(-3:n+4),expected(-3:n+4))
do i=-3,n+4
a(i)=real(i*i+2*i+7,8)/8.0_8
b(i)=real(300-i,8)/32.0_8
enddo
before=b
expected=b
if(trim(mode)/='ok'.and.trim(mode)/='zero')then
do i=1,n
expected(i)=expected(i)+3.0_8*(a(i+1)-a(i-1))
enddo
endif
call advance(a,b,n,3)
if(any(b/=expected))error stop 'native fallback count or success omission differs'
if(trim(mode)=='ok'.or.trim(mode)=='zero')then
if(any(b/=before))error stop 'unexpected original execution'
endif
do i=-3,n+4
if(a(i)/=real(i*i+2*i+7,8)/8.0_8)error stop 'readonly field changed'
enddo
print *,'RETURNED_FIELDS_HALOS_AND_NATIVE_ONCE_OK'
end program
''')
    for source, target, extra in (("scoped_runtime.cu", "runtime.o", ["-DFORT_SCOPE_CPU_TEST", "-x", "c++"]),
                                  ("provider-double.cpp", "provider-double.o", [])):
        compile_command([cpp, "-std=c++17", "-O0", "-fopenmp", "-ffp-contract=off", *extra,
                         "-I", str(build), "-c", str(build / source), "-o", str(build / target)])
        objects.append(build / target)
    binary = build / "report-control"
    compile_command([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-I", str(build),
                     str(build / "main.f90"), *map(str, objects), "-lstdc++", "-o", str(binary)])
    recoverable = ("arg", "resource", "boundary", "uninit", "plan")
    fatal = ("accepted", "fatal_preaccept", "bad_version", "success_no_commit", "commit_without_accept",
             "success_empty", "negative_flag", "bad_status")
    rows = []
    for mode in ("ok", "zero", *recoverable, *fatal):
        env = {**os.environ, "QUEUED_REPORT_MODE": mode, "FORT_RUNTIME_TRACE": "1", "CUDA_VISIBLE_DEVICES": ""}
        result = subprocess.run([str(binary)], env=env, cwd=build, capture_output=True, text=True, timeout=15)
        (build / (mode + ".stdout")).write_text(result.stdout)
        (build / (mode + ".stderr")).write_text(result.stderr)
        rows.append({"mode": mode, "returncode": result.returncode})
        (build / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
        assert result.stderr.count("PROVIDER_CALLED") == (0 if mode == "zero" else 1)
        if mode in fatal:
            assert result.returncode != 0
            assert "RETURNED_FIELDS_HALOS" not in result.stdout
            assert "queued source transaction failed; native replay prohibited" in result.stderr
        else:
            assert result.returncode == 0, result.stdout + result.stderr
            assert "RETURNED_FIELDS_HALOS_AND_NATIVE_ONCE_OK" in result.stdout


@pytest.mark.parametrize("transport", [True, "unknown", None])
def test_unknown_transport_cannot_be_a_checked_patch(transport):
    with pytest.raises(CompilationError, match="checked source transport"):
        source_device_tasks._provider(None, transport=transport)


def test_queued_patch_rejects_direct_pricing_or_placement():
    with pytest.raises(CompilationError, match="direct prices"):
        source_device_tasks._QueuedSourceDevicePatch(None, (), (), 1, 1, pricing=object())


def test_checked_child_maps_original_fields_to_one_parent_epoch(tmp_path, monkeypatch):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    source = SOURCE.rsplit("end module", 1)[0] + """subroutine outer(left,right,n,t)
integer,intent(in)::n,t
real(8),intent(in)::left(-3:)
real(8),intent(inout)::right(-3:)
call advance(left,right,n,t)
end subroutine
end module
"""
    _outputs, manifest, facts = emit(tmp_path / "inputs", monkeypatch, source=source,
                                     entry="stencil::outer", captures=("left", "right"))
    assert manifest["scope_count"] == 1
    task, = manifest["scopes"][0]["source_device_tasks"]
    assert task["source_proof"]["procedure"] == "stencil::advance"
    assert task["execution_binding"]["canonical_resources"] == {
        "argument::a": "argument::left", "argument::b": "argument::right"}
    validate_checked_execution_plans(manifest, capture_facts=facts)
