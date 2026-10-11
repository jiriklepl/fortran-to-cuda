"""Unsupported original native calls disable only reached dormant paths."""

from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.native_prefix import (
    prove_native_prefix,
    reject_unsupported_native_prefix_call,
)
from compiler.frontend.preowner_transfers import PreownerUnsupported
from compiler.frontend.reached_windows import plan_reached_window, prove_reached_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "observer_user::advance"
ROOTS = ("argument::a", "argument::b", "argument::c")
IO_BODY = "write(*,*) n"
TEAM_BODY = """!$omp parallel private(local)
local=n
!$omp end parallel"""
NATIVE = """module native_observers
implicit none
contains
subroutine notify(n)
integer,intent(in)::n
integer::local
{body}
end subroutine
subroutine quiet(n)
integer,intent(in)::n
integer::local
local=n
end subroutine
end module
"""
BRIDGE = """module observer_bridge
use native_observers,only:exported=>notify
implicit none
private
public::exported
end module
"""
SOURCE = """module observer_user
use observer_bridge,only:local_hook=>exported
implicit none
contains
subroutine advance(a,b,c,n,enabled)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:),c(-2:)
integer,intent(in)::n
logical,intent(in)::enabled
integer::i
if(enabled) then
call local_hook(n)
endif
do i=-2,n-3
b(i)=a(i)+real(i+4,8)
enddo
do i=-2,n-3
c(i)=2.0_8*b(i)
enddo
end subroutine
end module
"""


def call_fixture(directory, *, body=IO_BODY, source=SOURCE, native=None, builder=False, call_procedure=ENTRY):
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / name for name in ("observers.f90", "bridge.f90", "worker.f90"))
    for path, text in zip(paths, (native or NATIVE.format(body=body), BRIDGE, source), strict=True):
        path.write_text(text)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {root: deepcopy(FACT) for root in ROOTS}}
    owner = (ScopeBuilder(paths, ENTRY, facts=facts, options=CompilerOptions(),
                          config=OffloadConfig(policy="sections", scope_execution="reached")) if builder else None)
    analysis = owner.analysis if owner is not None else SourceEffects(paths)
    routine = analysis.routines[ENTRY]
    call_routine = analysis.routines[call_procedure]
    branch = next((node for node in call_routine.execution.content if type(node).__name__ == "If_Construct"), None)
    call = next(node for node in (branch.content if branch is not None else call_routine.execution.content)
                if type(node).__name__ == "Call_Stmt")
    selection = tuple(node for node in routine.execution.content
                      if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    assert len(selection) == 2
    return paths, analysis, facts, call, selection, owner


@pytest.mark.parametrize("body", [IO_BODY, TEAM_BODY], ids=["renamed-io", "renamed-native-team"])
def test_unsupported_native_call_preserves_the_inactive_owner_path(tmp_path, body):
    _paths, analysis, facts, call, selection, _owner = call_fixture(tmp_path, body=body)
    original_facts = deepcopy(facts)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    record = proof.public()
    prefix = record["prefix_eligibility"]
    assert prefix["activation_path_possible"]
    assert prefix["disabled_path_possible"]
    assert prefix["disabled_is_absorbing"]
    assert not prefix["reopen"]
    point, = prefix["disable_points"]
    assert point["original_source_kind"] == "Call_Stmt"
    assert analysis.structure(ENTRY).source_nodes(point["node_id"]) == (call,)
    assert point["selected_node_ids"] == [point["node_id"]]
    assert point["guard"]
    assert any("enabled" in item.lower() for item in point["guard"])
    assert point["disabled_is_absorbing"]
    assert point["context_exists"] is False
    assert point["reopen"] is False
    assert not prefix["steps"]
    assert not record["native_prefix_proof_identities"]
    assert not record["execution_authorized"]
    assert not record["gpu_legality_established"]
    assert {item["resource"] for item in record["states"]} == set(ROOTS)
    assert facts == original_facts
    assert analysis.stable_module_allocatables == frozenset()
    assert not getattr(analysis, "_native_prefix_proofs", {})
    assert proof.validate(analysis, ENTRY, selection) is proof


@pytest.mark.parametrize("body", [IO_BODY, TEAM_BODY], ids=["io", "native-team"])
def test_disable_is_inside_original_if_before_exact_call_without_replay(tmp_path, body):
    paths, analysis, facts, _call, selection, builder = call_fixture(tmp_path, body=body, builder=True)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    owner = ReachedWindowOwner(builder, proof, selection)
    assert owner.run()
    outputs, report = owner.builder.finish()
    text = outputs[report["sources"][str(paths[-1])]["replacement"]]
    condition = text.index("if(enabled) then")
    disable = text.index(f"{owner.prefix_eligible} = .false.")
    call = text.index("call local_hook(n)")
    spend = text.index(f"{owner.spent} = .true.")
    assert condition < disable < call < spend
    assert text.count("call local_hook(n)") == text.count("if(enabled) then") == 1
    assert text.count(f"{owner.prefix_eligible} = .true.") == 1
    assert text.count(f"{owner.spent} = .true.") == 1
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    assert {item["resource"] for item in scope["resource_bindings"]["resources"]} == set(ROOTS)
    assert not scope["reopen_after_boundary"]
    assert paths[-1].read_text() == SOURCE


@pytest.mark.parametrize("body", [IO_BODY, TEAM_BODY], ids=["io", "native-team"])
def test_preflight_decline_never_upgrades_full_native_prefix_effects(tmp_path, body):
    _paths, analysis, facts, call, selection, _owner = call_fixture(tmp_path, body=body)
    original = analysis.summarize("native_observers::notify")
    with pytest.raises(PreownerUnsupported):
        reject_unsupported_native_prefix_call(analysis, ENTRY, call, consume=lambda _count: None)
    assert analysis.summarize("native_observers::notify") == original
    assert not getattr(analysis, "_native_prefix_proofs", {})
    with pytest.raises(CompilationError):
        prove_native_prefix(analysis, "native_observers::notify")
    with pytest.raises(CompilationError):
        prove_reached_window(analysis, ENTRY, selection, ROOTS, facts)


def test_supported_original_call_still_uses_the_existing_effect_proof(tmp_path):
    _paths, analysis, _facts, call, _selection, _owner = call_fixture(tmp_path, body="local=n")
    assert reject_unsupported_native_prefix_call(analysis, ENTRY, call, consume=lambda _count: None) is None
    assert not getattr(analysis, "_native_prefix_proofs", {})
    complete = prove_native_prefix(analysis, "native_observers::notify")
    assert complete.validate(analysis, "native_observers::notify") is complete
    assert complete.summary["complete"]


def test_transitive_source_call_is_classified_without_borrowing_effects(tmp_path):
    native = NATIVE.format(body="call report_value(n)").replace("subroutine quiet(n)", """subroutine report_value(n)
integer,intent(in)::n
write(*,*) n
end subroutine
subroutine quiet(n)""")
    _paths, analysis, facts, call, selection, _owner = call_fixture(tmp_path, native=native)
    with pytest.raises(PreownerUnsupported):
        reject_unsupported_native_prefix_call(analysis, ENTRY, call, consume=lambda _count: None)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    assert proof.public()["prefix_eligibility"]["activation_path_possible"]
    assert proof.public()["prefix_eligibility"]["disabled_path_possible"]
    assert not proof.public()["native_prefix_proof_identities"]


@pytest.mark.parametrize("body", [IO_BODY, TEAM_BODY], ids=["io", "native-team"])
def test_known_unsupported_call_declines_before_eager_effect_or_prefix_closure(tmp_path, monkeypatch, body):
    from compiler.frontend import native_prefix

    _paths, analysis, facts, call, selection, _owner = call_fixture(tmp_path, body=body)
    graph = analysis.structure(ENTRY)
    call_id = next(identity for identity in graph.nodes if graph.source_nodes(identity) == (call,))
    segment_summary = analysis.segment_summary

    def guarded_summary(procedure, nodes, **kwargs):
        selected = tuple(nodes)
        if procedure == ENTRY and (selected == (call_id,) or len(selected) == 1 and selected[0] is call):
            raise AssertionError("unsupported original call reached expensive effect summary")
        return segment_summary(procedure, nodes, **kwargs)

    def forbidden_prefix(*_args, **_kwargs):
        raise AssertionError("unsupported original call reached full native-prefix closure")

    monkeypatch.setattr(analysis, "segment_summary", guarded_summary)
    monkeypatch.setattr(native_prefix, "prove_native_prefix", forbidden_prefix)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    assert proof.public()["prefix_eligibility"]["activation_path_possible"]
    assert proof.public()["prefix_eligibility"]["disabled_path_possible"]
    assert not proof.public()["native_prefix_proof_identities"]


def test_active_original_call_never_uses_dormant_support_preflight(tmp_path, monkeypatch):
    from compiler.frontend import native_prefix

    branch = "if(enabled) then\ncall local_hook(n)\nendif\n"
    source = SOURCE.replace(branch, "").replace("enddo\ndo i=-2,n-3", "enddo\n" + branch + "do i=-2,n-3")
    _paths, analysis, facts, _call, _selection, _owner = call_fixture(tmp_path, source=source)

    def forbidden_preflight(*_args, **_kwargs):
        raise AssertionError("dormant preflight consulted while owner is active")

    monkeypatch.setattr(native_prefix, "reject_unsupported_native_prefix_call", forbidden_preflight)
    selection = tuple(analysis.routines[ENTRY].execution.content)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    assert not proof.public()["prefix_eligibility"]["disable_points"]
    assert proof.public()["planned_closures"]["closed_path_possible"]
    assert proof.public()["planned_closures"]["close_points"]


def test_mapped_child_call_never_uses_dormant_support_preflight(tmp_path, monkeypatch):
    from compiler.frontend import native_prefix

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
    _paths, analysis, facts, _call, _selection, _owner = call_fixture(
        tmp_path, source=source, call_procedure="observer_user::nested")

    def forbidden_preflight(*_args, **_kwargs):
        raise AssertionError("dormant preflight consulted for mapped original child")

    monkeypatch.setattr(native_prefix, "reject_unsupported_native_prefix_call", forbidden_preflight)
    selection = tuple(analysis.routines[ENTRY].execution.content)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    child, = proof.public()["planned_closures"]["mapped_child_plans"]
    assert child["procedure"] == "observer_user::nested"
    assert not child["plan"]["prefix_eligibility"]["disable_points"]
    assert child["plan"]["planned_closures"]["closed_path_possible"]
    assert child["plan"]["planned_closures"]["close_points"]


@pytest.mark.parametrize("mutation", ["file", "actual_binding", "callee_binding", "declaring_scope", "rename"])
def test_source_or_binding_mutation_is_not_a_support_decline(tmp_path, mutation):
    paths, analysis, facts, call, selection, _owner = call_fixture(tmp_path)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    if mutation == "file":
        paths[0].write_text(NATIVE.format(body=IO_BODY) + "\n! changed original\n")
    elif mutation == "actual_binding":
        scope = analysis.routines[ENTRY].scope
        scope.bindings["n"] = copy(scope.bindings["n"])
    elif mutation == "callee_binding":
        scope = analysis.routines["native_observers::notify"].scope
        scope.bindings["n"] = copy(scope.bindings["n"])
    elif mutation == "declaring_scope":
        analysis.routines["native_observers::notify"].scope.bindings["n"].declaring_scope = (
            analysis.modules["native_observers"])
    else:
        analysis.modules["observer_bridge"].imports["exported"] = ("native_observers", "quiet")
    for operation in (
        lambda: reject_unsupported_native_prefix_call(analysis, ENTRY, call, consume=lambda _count: None),
        lambda: proof.validate(analysis, ENTRY, selection),
    ):
        with pytest.raises(CompilationError) as error:
            operation()
        assert not isinstance(error.value, PreownerUnsupported)


def component_fixture(directory, *, dtype="real(8)"):
    initial = ".true." if dtype == "logical" else "1.0"
    declarations = f"""module original_settings
implicit none
type tuning
{dtype}::gain={initial},alternate={initial}
real(8),allocatable::unrelated(:)
end type
type(tuning)::retained
end module
"""
    native = declarations + NATIVE.format(body=IO_BODY).replace(
        "integer,intent(in)::n", f"{dtype},intent(in)::n").replace("integer::local", f"{dtype}::local")
    source = SOURCE.replace("implicit none\ncontains", """use original_settings,only:settings=>retained
implicit none
contains""", 1).replace("call local_hook(n)", "call local_hook(settings%gain)")
    return call_fixture(directory, native=native, source=source)


@pytest.mark.parametrize("dtype", ["real(4)", "real(8)", "logical"])
def test_renamed_fixed_component_is_authenticated_without_whole_type_layout_authority(tmp_path, dtype):
    _paths, analysis, facts, call, selection, _owner = component_fixture(tmp_path, dtype=dtype)
    schema = analysis.modules["original_settings"].component_types["tuning"]
    assert "unrelated" in schema.members
    assert "allocatable" in str(schema.members["unrelated"][1]).lower()
    with pytest.raises(PreownerUnsupported, match="Write_Stmt"):
        reject_unsupported_native_prefix_call(analysis, ENTRY, call, consume=lambda _count: None)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    point, = proof.public()["prefix_eligibility"]["disable_points"]
    assert analysis.structure(ENTRY).source_nodes(point["node_id"]) == (call,)
    assert {state["resource"] for state in proof.public()["states"]} == set(ROOTS)
    assert not proof.public()["native_prefix_proof_identities"]
    assert not getattr(analysis, "_native_prefix_proofs", {})


@pytest.mark.parametrize("member", ["gain", "unrelated"], ids=["selected-member", "unused-sibling"])
def test_consulted_component_schema_mutation_is_hard_before_support_decline(tmp_path, member):
    _paths, analysis, facts, call, selection, _owner = component_fixture(tmp_path)
    proof = plan_reached_window(analysis, ENTRY, selection, ROOTS, facts)
    schema = analysis.modules["original_settings"].component_types["tuning"]
    # The replacement stays a compatible REAL8 scalar when it is selected;
    # rejection must authenticate the original schema, not just its signature.
    schema.members[member] = schema.members["alternate"]
    for operation in (
        lambda: reject_unsupported_native_prefix_call(analysis, ENTRY, call, consume=lambda _count: None),
        lambda: proof.validate(analysis, ENTRY, selection),
    ):
        with pytest.raises(CompilationError) as error:
            operation()
        assert not isinstance(error.value, PreownerUnsupported)


def test_copied_original_call_has_no_classification_authority(tmp_path):
    _paths, analysis, _facts, call, _selection, _owner = call_fixture(tmp_path)
    with pytest.raises(CompilationError) as error:
        reject_unsupported_native_prefix_call(analysis, ENTRY, copy(call), consume=lambda _count: None)
    assert not isinstance(error.value, PreownerUnsupported)


def test_shared_scan_budget_failure_remains_hard(tmp_path):
    _paths, analysis, _facts, call, _selection, _owner = call_fixture(tmp_path)

    def exhausted(_count):
        raise CompilationError("sentinel exhausted aggregate coordinate/source scan budget")

    with pytest.raises(CompilationError, match="sentinel exhausted aggregate") as error:
        reject_unsupported_native_prefix_call(analysis, ENTRY, call, consume=exhausted)
    assert not isinstance(error.value, PreownerUnsupported)
    assert not getattr(analysis, "_native_prefix_proofs", {})


@pytest.mark.parametrize("limit", ["operation_limit", "procedure_limit", "depth_limit"])
def test_local_call_graph_budget_failures_remain_hard(tmp_path, limit):
    native = NATIVE.format(body="call intermediate(n)").replace("subroutine quiet(n)", """subroutine intermediate(n)
integer,intent(in)::n
call report_value(n)
end subroutine
subroutine report_value(n)
integer,intent(in)::n
write(*,*) n
end subroutine
subroutine quiet(n)""")
    _paths, analysis, _facts, call, _selection, _owner = call_fixture(tmp_path, native=native)
    setattr(analysis, limit, 1)
    with pytest.raises(CompilationError, match="budget|bounded|depth|limit") as error:
        reject_unsupported_native_prefix_call(analysis, ENTRY, call, consume=lambda _count: None)
    assert not isinstance(error.value, PreownerUnsupported)
    assert not getattr(analysis, "_native_prefix_proofs", {})
