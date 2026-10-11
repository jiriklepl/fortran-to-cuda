"""Readonly original host conditions do not become managed payload authority."""

from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.reached_windows import plan_reached_window, prove_reached_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "metadata_user::advance"
ROOTS = ("argument::a", "argument::b", "argument::c")
METADATA = """module original_settings
implicit none
integer,parameter::width=2
integer::flags(width)=[0,0]
logical::switches(2)=[.true.,.false.]
end module
"""
BRIDGE = """module settings_bridge
use original_settings,only:exposed=>flags,choices=>switches
implicit none
private
public::exposed,choices
end module
"""
SOURCE = """module metadata_user
use settings_bridge,only:local_flags=>exposed,local_choices=>choices
implicit none
integer::visits=0
contains
subroutine advance(a,b,c,n,enabled)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:),c(-2:)
integer,intent(in)::n
logical,intent(in)::enabled
integer::i,bias
bias=0
if(enabled) then
if(any(local_flags==7).and.local_flags(1)>0.and.all(local_choices)) then
bias=3
else
bias=-2
endif
endif
do i=-2,n-3
b(i)=a(i)+real(bias+i,8)
enddo
do i=-2,n-3
c(i)=2.0_8*b(i)
enddo
visits=visits+1
end subroutine
end module
"""


def condition_fixture(directory, *, source=SOURCE, metadata=METADATA, builder=False, operations=256):
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / name for name in ("settings.f90", "bridge.f90", "worker.f90"))
    for path, text in zip(paths, (metadata, BRIDGE, source), strict=True):
        path.write_text(text)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {root: {**deepcopy(FACT), "initialized": "whole" if index == 0 else "none"}
                          for index, root in enumerate(ROOTS)}}
    owner = (ScopeBuilder(paths, ENTRY, facts=facts, options=CompilerOptions(),
                          config=OffloadConfig(policy="sections", scope_execution="reached")) if builder else None)
    analysis = owner.analysis if owner is not None else SourceEffects(paths)
    analysis.operation_limit = operations
    routine = analysis.routines[ENTRY]
    selection = tuple(node for node in routine.execution.content
                      if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    assert len(selection) == 2
    return paths, analysis, facts, selection, owner


def native_conditions(proof):
    return [step for step in proof.public()["prefix_eligibility"]["steps"]
            if step["original_source_kind"] == "original_native_condition_reads"]


def test_renamed_mutable_metadata_stays_an_original_guarded_host_read(tmp_path):
    paths, analysis, facts, selection, _owner = condition_fixture(tmp_path)
    facts_before = deepcopy(facts)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    assert proof.public()["schema_version"] == 11
    assert proof.public()["prefix_eligibility"]["activation_path_possible"]
    assert not proof.public()["prefix_eligibility"]["disabled_path_possible"]
    step, = native_conditions(proof)
    assert step["guard"]  # The original enabled guard is retained.
    assert {resource["resource"] for resource in step["resources"]} == {
        "original_settings::flags", "original_settings::switches"}
    assert all(resource["original_bounds"] == [[1, 2]] for resource in step["resources"])
    assert all(resource["source"] == str(paths[0]) for resource in step["resources"])
    for key in ("capture_authority", "managed_definition_authority", "allocation_freshness_authority",
                "gpu_legality_established"):
        assert step[key] is False
    assert {state["resource"] for state in proof.public()["states"]} == set(ROOTS)
    assert facts == facts_before
    assert analysis.stable_module_allocatables == frozenset()
    assert proof.validate(analysis, ENTRY, selection) is proof
    # The complete ordinary contract gains no extra managed roots either.
    complete = prove_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    assert {state["resource"] for state in complete.public()["states"]} == set(ROOTS)


def test_emission_keeps_condition_once_and_metadata_out_of_registration(tmp_path):
    paths, analysis, facts, selection, builder = condition_fixture(tmp_path, builder=True)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    owner = ReachedWindowOwner(builder, proof, selection)
    assert owner.run()
    outputs, report = owner.builder.finish()
    text = outputs[report["sources"][str(paths[-1])]["replacement"]]
    assert text.count("if(any(local_flags==7).and.local_flags(1)>0.and.all(local_choices)) then") == 1
    assert text.count("if(enabled) then") == 1
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    registered = {resource["resource"] for resource in scope["resource_bindings"]["resources"]}
    assert registered == set(ROOTS)
    assert registered.isdisjoint({"original_settings::flags", "original_settings::switches"})
    assert "flags" not in facts["captures"]


@pytest.mark.parametrize(("declaration", "condition"), [
    ("real(4)::flags(2)=[0.0,0.0]", "any(local_flags>0.0)"),
    ("real(8)::flags(2)=[0.0_8,0.0_8]", "any(local_flags>0.0_8)"),
    ("integer(8)::flags(2)=[0_8,0_8]", "any(local_flags>0_8)"),
    ("logical::flags(2)=[.true.,.false.]", "any(local_flags)"),
])
def test_fixed_original_numeric_and_logical_metadata(tmp_path, declaration, condition):
    metadata = METADATA.replace("integer::flags(width)=[0,0]", declaration)
    source = SOURCE.replace("any(local_flags==7).and.local_flags(1)>0.and.all(local_choices)", condition)
    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path, source=source, metadata=metadata)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    assert proof.public()["prefix_eligibility"]["activation_path_possible"]
    step, = native_conditions(proof)
    assert len(step["resources"]) == 1


def test_fixed_private_local_metadata_is_not_a_capture(tmp_path):
    source = SOURCE.replace("integer::i,bias", "integer::i,bias\ninteger::local_table(2)=[0,1]").replace(
        "any(local_flags==7).and.local_flags(1)>0.and.all(local_choices)", "any(local_table>0)")
    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path, source=source)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    step, = native_conditions(proof)
    resource, = step["resources"]
    assert resource["resource"] == ENTRY + "::local_table"
    assert resource["resource"] not in facts["captures"]


def test_readonly_source_predicate_keeps_its_complete_original_proof(tmp_path):
    source = SOURCE.replace("contains\nsubroutine advance", """contains
pure logical function accepts(x)
integer,intent(in)::x(2)
accepts=any(x==7)
end function
subroutine advance""").replace("any(local_flags==7).and.local_flags(1)>0.and.all(local_choices)", "accepts(local_flags)")
    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path, source=source)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    assert proof.public()["prefix_eligibility"]["activation_path_possible"]
    assert native_conditions(proof)
    assert analysis._native_functions


def test_managed_undefined_array_does_not_get_native_condition_exemption(tmp_path):
    source = SOURCE.replace("any(local_flags==7).and.local_flags(1)>0.and.all(local_choices)", "any(b>0.0_8)")
    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path, source=source)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    prefix = proof.public()["prefix_eligibility"]
    assert prefix["activation_path_possible"]  # Original enabled=false path is still eligible.
    assert prefix["disabled_path_possible"]
    point, = prefix["disable_points"]
    assert point["guard"]
    assert "definition" in point["reason"]
    assert not native_conditions(proof)
    assert facts["captures"]["argument::b"]["initialized"] == "none"


def test_untracked_array_assignment_is_not_a_condition(tmp_path):
    source = SOURCE.replace("bias=0", "bias=sum(local_flags)")
    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path, source=source)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    assert not proof.public()["prefix_eligibility"]["activation_path_possible"]
    assert not native_conditions(proof)
    assert "untracked array" in proof.public()["prefix_eligibility"]["disable_points"][0]["reason"]


def test_active_condition_still_closes_before_untracked_metadata_read(tmp_path):
    source = SOURCE.replace("visits=visits+1", """if(any(local_flags==7)) then
visits=visits+1
endif""")
    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path, source=source)
    active_if = analysis.routines[ENTRY].execution.content[-1]
    proof = plan_reached_window(analysis, ENTRY, (*selection, active_if), ROOTS, facts)
    close, = proof.public()["planned_closures"]["close_points"]
    assert close["original_source_kind"] == "If_Construct"
    assert "untracked array" in close["reason"]


@pytest.mark.parametrize("attribute", ["target", "volatile", "asynchronous", "pointer", "allocatable"])
def test_uncertain_metadata_association_is_never_admitted(tmp_path, attribute):
    declaration = (f"integer,{attribute}::flags(:)" if attribute in {"pointer", "allocatable"}
                   else f"integer,{attribute}::flags(2)")
    metadata = METADATA.replace("integer::flags(width)=[0,0]", declaration)
    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path, metadata=metadata)
    try:
        proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    except CompilationError:
        return  # Incomplete original effects remain a hard rejection.
    prefix = proof.public()["prefix_eligibility"]
    assert prefix["activation_path_possible"]
    assert prefix["disabled_path_possible"]
    point, = prefix["disable_points"]
    assert point["guard"]
    assert "fixed unaliased native metadata" in point["reason"]
    assert not native_conditions(proof)


@pytest.mark.parametrize("extra", ["common /shared/ flags", "equivalence(flags(1),alias)", "!$omp threadprivate(flags)"])
def test_original_metadata_storage_ownership_is_required(tmp_path, extra):
    metadata = METADATA.replace("end module", "integer::alias\n" + extra + "\nend module")
    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path, metadata=metadata)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    prefix = proof.public()["prefix_eligibility"]
    assert prefix["activation_path_possible"]
    assert prefix["disabled_path_possible"]
    assert not native_conditions(proof)
    point, = prefix["disable_points"]
    assert point["guard"]
    reason = point["reason"]
    assert "storage association" in reason or "OpenMP ownership" in reason


def test_side_effectful_predicate_cannot_borrow_readonly_condition_authority(tmp_path):
    source = SOURCE.replace("contains\nsubroutine advance", """contains
logical function changes_flags()
local_flags(1)=local_flags(1)+1
changes_flags=any(local_flags==7)
end function
subroutine advance""").replace("any(local_flags==7).and.local_flags(1)>0.and.all(local_choices)", "changes_flags()")
    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path, source=source)
    with pytest.raises(CompilationError, match="complete.*effects|read.only|function"):
        plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)


@pytest.mark.parametrize("mutation", ["file", "binding", "declaring_scope", "rename", "private", "bounds_kind"])
def test_registered_condition_proof_reauthenticates_original_authority(tmp_path, mutation):
    paths, analysis, facts, selection, _owner = condition_fixture(tmp_path)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    module = analysis.modules["original_settings"]
    if mutation == "file":
        paths[0].write_text(METADATA + "\n! changed source\n")
    elif mutation == "binding":
        module.bindings["flags"] = copy(module.bindings["flags"])
    elif mutation == "declaring_scope":
        module.bindings["flags"].declaring_scope = analysis.modules["settings_bridge"]
    elif mutation == "rename":
        analysis.modules["settings_bridge"].imports["exposed"] = ("original_settings", "switches")
    elif mutation == "private":
        module.access["flags"] = False
    else:
        module.kinds.values["width"] = 3
    with pytest.raises(CompilationError):
        proof.validate(analysis, ENTRY, selection)


def test_dormant_condition_budget_failure_is_hard(tmp_path):
    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path, operations=1)
    with pytest.raises(CompilationError, match="budget|bounded"):
        plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)


def test_dormant_metadata_scan_exhaustion_is_not_a_prefix_disable(tmp_path, monkeypatch):
    from compiler.frontend import native_group_structure

    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path)
    assert native_group_structure.NATIVE_GROUP_SCAN_LIMIT == 65536
    monkeypatch.setattr(native_group_structure, "NATIVE_GROUP_SCAN_LIMIT", 64)
    with pytest.raises(CompilationError, match="aggregate coordinate/source scan budget exhausted"):
        plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)


def test_mapped_child_condition_has_no_dormant_metadata_exception(tmp_path):
    parent = """subroutine advance(a,b,c,n,enabled)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:),c(-2:)
integer,intent(in)::n
logical,intent(in)::enabled
integer::i
call nested(a,b,c,n,enabled)
do i=-2,n-3
b(i)=b(i)+1.0_8
enddo
do i=-2,n-3
c(i)=c(i)+1.0_8
enddo
end subroutine
"""
    source = SOURCE.replace("subroutine advance(", "subroutine nested(").replace("end module", parent + "end module")
    _paths, analysis, facts, _selection, _owner = condition_fixture(tmp_path, source=source)
    routine = analysis.routines[ENTRY]
    selection = tuple(routine.execution.content)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    child, = proof.public()["planned_closures"]["mapped_child_plans"]
    assert child["procedure"] == "metadata_user::nested"
    assert not child["plan"]["prefix_eligibility"]["steps"]
    close, = child["plan"]["planned_closures"]["close_points"]
    assert close["original_source_kind"] == "If_Construct"
    assert "untracked array" in close["reason"]


@pytest.mark.parametrize("declaration", ["integer::flags(257)", "integer::flags(width+unknown)"])
def test_fixed_metadata_bounds_keep_existing_conservative_limits(tmp_path, declaration):
    metadata = METADATA.replace("integer::flags(width)=[0,0]", declaration)
    _paths, analysis, facts, selection, _owner = condition_fixture(tmp_path, metadata=metadata)
    with pytest.raises(CompilationError, match="256.element|constant explicit bounds"):
        plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
