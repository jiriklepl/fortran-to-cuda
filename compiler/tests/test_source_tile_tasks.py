"""Complete original teams are checked graph tasks with ordinary coherence."""
from __future__ import annotations

import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT, walk_graph
from compiler.tests.test_tile_private import SOURCE


def emit(tmp_path, *, policy="sections", child=False):
    source = SOURCE
    if child:
        source = source.rsplit("end module", 1)[0] + """subroutine outer(left,right,n,t)
integer,intent(in)::n,t
real(8),intent(in)::left(-3:)
real(8),intent(inout)::right(-3:)
call advance(left,right,n,t)
end subroutine
end module
"""
    path = tmp_path / "source.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + root: dict(FACT) for root in (("left", "right") if child else ("a", "b"))}}
    builder = ScopeBuilder([path], "stencil::" + ("outer" if child else "advance"), facts=facts,
        options=CompilerOptions(), config=OffloadConfig(policy=policy, scope_execution="graph"))
    outputs, report = builder.run()
    return path, outputs, report


@pytest.mark.parametrize("child", [False, True])
def test_sections_consumes_whole_team_proof_on_same_graph_without_definition_credit(tmp_path, child):
    path, outputs, report = emit(tmp_path, child=child)
    scope, = report["scopes"]
    task, = scope["source_device_tasks"]
    assert scope["estimate_available"] is False
    assert "unpriced" in scope["planning_reason"]
    assert task["definitions_authorized"] is False
    assert task["placement_authorized"] is False
    assert [item["access"] for item in task["resources"]] == ["read", "readwrite"]
    members = scope["checked_execution_plan"]["storage_epoch"]["members"]
    member, = [member for member in members if member.get("source_device_tasks")]
    assert member["source_device_tasks"] == [task]
    canonical = {"argument::a": "argument::left", "argument::b": "argument::right"} if child else {
        "argument::a": "argument::a", "argument::b": "argument::b"}
    assert task["execution_binding"]["canonical_resources"] == canonical
    graph = scope["graph_execution"]["procedures"]
    selected = [task for procedure in graph for task in walk_graph(procedure["task_graph"]) if ":source_device:" in task["id"]]
    assert len(selected) == 1
    assert selected[0]["implementations"] == ["gpu"]
    assert selected[0]["lifetime_resources"] == []
    assert selected[0]["metadata_reads"] == sorted(canonical.values())
    replacement = outputs[report["sources"][str(path)]["replacement"]]
    positive = replacement.index("if ((1) <= (n)) then")
    descriptor = replacement.index("= lbound(tmp,", positive)
    assert positive < descriptor
    assert "fort_scope_numerical_environment_supported()" in replacement
    # A module ENTRY preserves two exclusive routes: the checked prework
    # fallback, and the unchanged original ABI when no controls are present.
    assert replacement.count("!$omp parallel private(bi,i,li,tmp)") == (2 if child else 1)
    assert "if (.not. fort_source_done) then\n!$omp parallel private(bi,i,li,tmp)" in replacement
    if child:
        assert "end block fort_source_task_" in replacement
        assert "else\n!$omp parallel private(bi,i,li,tmp)" in replacement
        coordinator, = scope["module_coordinators"]
        assert not coordinator["planning_segments"]
        call, = coordinator["calls"]
        assert call["caller"] == "stencil::outer"
        assert call["resource_mappings"] == canonical
        forwarded = call["handle_bindings"]
        assert len(forwarded) == 2
        assert set(forwarded.values()) == {1, 2}
        assert all(int(local) > 2 for local in forwarded)
        assert coordinator["control_version"] == 4
        for local, number in forwarded.items():
            assert replacement.count(f"fort_buffer_{local} = fort_buffer_{number}") == 1
            assert f"fort_storage_identity_{local} = {number}_" in replacement
            assert f", fort_storage_identity_{local}, &" in replacement
        # Public registrations describe canonical storage, even though the
        # reusable source helper has its own private naming slots.
        assert set(task["execution_binding"]["registration_slots"].values()) == {1, 2}
    provider = task["execution_binding"]["provider"]
    assert provider["metadata"]["version"] == 7
    assert provider["metadata"]["access_envelopes"]["initialization_authorized"] is False
    assert provider["entry"] + ".cu" in outputs
    assert "fort_scope_source_execute_v1(" in outputs[provider["entry"] + ".cu"]
    assert "#define CUDA_CALLABLE" in outputs["common_functions.cuh"]
    assert "namespace generated_kernels::numeric" in outputs["numeric.hpp"]


def test_auto_keeps_unpriced_original_team_without_source_provider_emission(tmp_path):
    _, outputs, report = emit(tmp_path, policy="auto")
    assert all(not scope.get("source_device_tasks") for scope in report["scopes"])
    assert not any("fort_scope_source_execute_v1(" in text for name, text in outputs.items() if name.endswith(".cu"))


@pytest.mark.native
@pytest.mark.parametrize("child", [False, True])
def test_original_and_checked_child_Fortran_ABIs_compile_with_real_runtime_module(tmp_path, child):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    path, outputs, report = emit(tmp_path, child=child)
    directory = tmp_path / "build"
    directory.mkdir()
    for name, source in outputs.items():
        selected = directory / name
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text(source)
    task, = report["scopes"][0]["source_device_tasks"]
    sources = ["fort_scoped_memory.f90", task["execution_binding"]["provider"]["module"] + ".f90",
               report["sources"][str(path)]["replacement"]]
    for index, name in enumerate(sources):
        result = subprocess.run([fortran, "-fopenmp", "-ffp-contract=off", "-J", str(directory), "-I", str(directory),
                                 "-c", str(directory / name), "-o", str(directory / f"source-{index}.o")],
                                capture_output=True, text=True, timeout=60)
        (directory / f"compile-{index}.stdout").write_text(result.stdout)
        (directory / f"compile-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stderr
