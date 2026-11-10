"""Whole original record-array teams share the existing native host contract."""
from copy import copy, deepcopy
from dataclasses import replace

import pytest

from compiler.frontend.native_call_effects import prove_native_group_effects
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_native_call_graph import SOURCE, generate

GROUP = """!$omp parallel do private(i)
! Keep original documentation and the original join.
do i=-1,2
b(i,1,1)=a(1,1,1)+points(i)%flux+points(i)%aux
enddo
!$omp end parallel do"""


def source(*, write=False):
    group = GROUP
    if write:
        group = group.replace("b(i,1,1)=a(1,1,1)", "b(i,1,1)=b(i,1,1)+a(1,1,1)")
    return SOURCE.replace("type(record),target::points", "type(record)::points").replace(
        "call update(points,a)", group + "\nvisits=visits+1")


def issue(tmp_path, text=None):
    path = tmp_path / "source.f90"
    path.write_text(source() if text is None else text)
    analysis = SourceEffects([path])
    analysis.seal_session()
    routine = analysis.routines["native_graph::advance"]
    group, = [node for node in grouped_original_sequence(routine.execution.content, {})
              if isinstance(node, NativeGroupCandidate)]
    return analysis, group.original_nodes, prove_native_group_effects(analysis, routine.qualified, group.original_nodes)


def test_group_projects_records_and_managed_arrays_without_private_storage(tmp_path):
    analysis, selection, proof = issue(tmp_path, source(write=True))
    assert proof.validate(analysis, "native_graph::advance", selection) is proof
    roles = {binding.root: (role, actions) for binding, role, actions in proof.resources}
    assert roles == {
        "native_graph::a": ("managed", frozenset({"read"})),
        "native_graph::b": ("managed", frozenset({"read", "write"})),
        "native_graph::points": ("reservation", frozenset({"read"})),
    }
    assert proof.storage_structures[0].public()["private_bindings"]
    assert proof.public()["selection_kind"] == "joined_group"
    assert not proof.public()["definitions_authorized"]


@pytest.mark.parametrize("mutation", ["completion", "registry", "selection", "resources", "sections"])
def test_changed_group_authority_cannot_acquire_host_access(tmp_path, mutation):
    analysis, selection, proof = issue(tmp_path)
    if mutation == "completion":
        object.__setattr__(proof, "completion", copy(proof.completion))
    elif mutation == "registry":
        analysis._native_graph_completions.clear()
    elif mutation == "selection":
        selection = selection[1:]
    elif mutation == "sections":
        object.__setattr__(proof, "sections", replace(proof.sections, reason="changed source geometry"))
    else:
        object.__setattr__(proof, "resources", ())
    with pytest.raises(CompilationError):
        proof.validate(analysis, "native_graph::advance", selection)


@pytest.mark.parametrize(("before", "after"), [
    ("!$omp end parallel do", "!$omp end parallel do nowait"),
    ("b(i,1,1)=a(1,1,1)+points(i)%flux+points(i)%aux", "call unknown(points(i)%aux)"),
    ("real(8)::flux,aux", "real(8)::flux\nreal(8),allocatable::aux"),
])
def test_unproved_group_or_storage_is_rejected(tmp_path, before, after):
    with pytest.raises((CompilationError, ValueError)):
        issue(tmp_path, source().replace(before, after))


@pytest.mark.parametrize("write", [False, True])
def test_normal_group_emission_uses_existing_coherence_and_strict_reader(tmp_path, write):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.original_host_access_record import validate_original_host_accesses

    path, _builder, outputs, manifest = generate(tmp_path, source(write=write))
    owner, = manifest["scopes"]
    assert not owner["boundaries"], owner["boundaries"]
    operation, = [op for unit in owner["planning_segments"] for op in unit["operations"]["native_operations"]
                  if op["kind"] == "complete original native group"]
    assert operation["preserves_original_source"]
    assert operation["original_host_access"]["proof"]["selection_kind"] == "joined_group"
    assert set(operation["host_metadata"]) == {"native_graph::points"}
    changed = outputs[manifest["sources"][str(path)]["replacement"]]
    assert "fort_scope_original_host_begin_v1" in changed
    assert changed.count("! Keep original documentation and the original join.") == 1
    validate_checked_execution_plans(manifest)
    broken = deepcopy(manifest)
    for procedure in broken["scopes"][0]["checked_execution_plan"]["procedures"]:
        for unit in procedure.get("units", []):
            for op in unit["operations"]["native_operations"]:
                if op["kind"] == "complete original native group":
                    op["original_host_access"]["proof"]["completion_identity"] = None
                    with pytest.raises(ValueError, match="original host CALL proof authority differs"):
                        validate_original_host_accesses(procedure, unit, broken)


def test_group_retains_original_caller_handle_mapping_in_checked_child(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans

    from compiler.tests.test_native_group_may_access_cuda import group_source

    _, _, _, manifest = generate(tmp_path, group_source(child=True), capture_updates={
        "native_graph::b": {"storage": "stable", "initialized": "none", "escapes": False,
                            "allocation_changes": False}})
    validate_checked_execution_plans(manifest)
    plan = manifest["scopes"][0]["checked_execution_plan"]
    child, = [row for row in plan["procedures"] if row["procedure"] == "native_graph::consume"]
    group, = [operation for unit in child["units"] for operation in unit["operations"]["native_operations"]
              if operation["kind"] == "complete original native group"]
    assert group["original_host_access"]["plan_kind"] == 6
    assert group["resource_effects"] == {"argument::x": ["read"], "argument::y": ["write"]}
