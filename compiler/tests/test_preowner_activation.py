"""Original native allocation/copy prefix and absorbing source-point eligibility."""

from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.reached_windows import plan_reached_window, prove_reached_window
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "renamed_prefix::advance"
ROOTS = ("argument::a", "argument::b", "renamed_prefix::workspace")
PREOWNER_SOURCE = """module renamed_prefix
implicit none
real(8),allocatable::workspace(:)
integer::prefix_visits=0,tail_visits=0,bad_visits=0,condition_visits=0
contains
subroutine advance(a,b,n,iterations,bad,skip)
real(8),intent(in)::a(-2:)
real(8),allocatable,intent(inout)::b(:)
integer,intent(in)::n,iterations
logical,intent(in)::bad,skip
real(8),allocatable::temporary(:)
integer::stage,i
prefix_visits=prefix_visits+1
if(allocated(workspace)) then
deallocate(workspace)
endif
allocate(workspace(-2:n-3))
workspace=a
b=a
do stage=1,iterations
condition_visits=condition_visits+1
if(bad) then
call opaque_prefix()
bad_visits=bad_visits+1
endif
if(skip .and. stage==1) then
cycle
endif
do i=-2,n-3
b(i)=workspace(i)+real(i+4,8)
enddo
do i=-2,n-3
workspace(i)=2.0_8*b(i)
enddo
call move_alloc(workspace,temporary)
call move_alloc(temporary,workspace)
tail_visits=tail_visits+1
enddo
end subroutine
end module
"""


def preowner_window_fixture(directory, source=PREOWNER_SOURCE, *, policy="sections", device_budget=None,
                            captures=None, operations=256):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "prefix_stage.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {ROOTS[0]: deepcopy(FACT), **{
                 root: {**deepcopy(FACT), "initialized": "none", "lifetime": "selected_window"}
                 for root in ROOTS[1:]}}}
    if captures:
        for root, values in captures.items():
            facts["captures"][root].update(values)
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy=policy, scope_execution="reached"))
    builder.analysis.operation_limit = operations
    outer = next(node for node in builder.entry.execution.content
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    originals = tuple(node for node in outer.content if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    assert len(originals) == 2
    proof = plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, facts)
    return path, builder, proof, originals


def emit_preowner_window(directory, source=PREOWNER_SOURCE, *, policy="sections", device_budget=None):
    path, builder, proof, originals = preowner_window_fixture(directory, source, policy=policy,
                                                            device_budget=device_budget)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    return path, outputs, report, owner


@pytest.mark.parametrize("private_key", ["source_sections", "requires_registered_handle"])
def test_external_capture_facts_cannot_supply_private_activation_authority(tmp_path, private_key):
    _path, builder, _proof, _originals = preowner_window_fixture(tmp_path)
    binding = builder.visible_binding(builder.entry, ROOTS[1])
    builder.facts["captures"][binding.root][private_key] = False
    with pytest.raises(CompilationError, match="compiler-owned"):
        builder.capture_fact(binding)


def test_original_allocation_and_copy_establish_only_source_defined_activation(tmp_path):
    path, builder, proof, originals = preowner_window_fixture(tmp_path)
    record = proof.public()
    prefix = record["prefix_eligibility"]
    assert record["schema_version"] == 8
    assert prefix["activation_path_possible"]
    assert prefix["disabled_path_possible"]
    assert not prefix["selected_window_whole_assertion_is_prefix_read_authority"]
    assert [step["original_source_kind"] for step in prefix["steps"]] == [
        "Deallocate_Stmt", "Allocate_Stmt", "Assignment_Stmt", "Assignment_Stmt"]
    for step in prefix["steps"]:
        assert not step["allocation_presence_proves_payload"]
        assert step["ordered_events"][-1]["fact_initialized"] == "none"
    assert all(state["initialized"] == "whole" for state in record["states"])
    assert all(state["active_path_return_is_conditional"] for state in record["states"])
    assert proof.validate(builder.analysis, ENTRY, originals) is proof
    assert builder.facts["captures"][ROOTS[1]]["initialized"] == "none"
    assert builder.facts["captures"][ROOTS[2]]["initialized"] == "none"
    assert path.read_text() == PREOWNER_SOURCE
    with pytest.raises(CompilationError, match="whole allocatable|unknown selected|unknown effects|complete.*effects"):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)


def test_emitted_prefix_and_condition_once_and_spend_before_eligibility(tmp_path):
    path, outputs, report, owner = emit_preowner_window(tmp_path)
    text = outputs[report["sources"][str(path)]["replacement"]]
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    assert text.count("workspace=a") == text.count("b=a") == 1
    assert text.count("allocate(workspace(-2:n-3))") == 1
    assert text.count("call opaque_prefix()") == 1
    assert text.count("condition_visits=condition_visits+1") == 1
    assert text.count("if(skip .and. stage==1) then") == 1
    assert text.count(f"{owner.prefix_eligible} = .true.") == 1
    disabled = text.index(f"{owner.prefix_eligible} = .false.")
    assert text.index("if(bad) then") < disabled < text.index("call opaque_prefix()")
    spend = text.index(f"{owner.spent} = .true.")
    eligibility = text.index(f"if ({owner.prefix_eligible}) then", spend)
    serial = text.index(f"{owner.enabled} = fort_scope_serial_caller()", eligibility)
    assert spend < eligibility < serial
    assert text.index("b=a") < spend
    assert "prefix_eligibility" in scope["reached_window"]
    assert not scope["reopen_after_boundary"]


@pytest.mark.parametrize("destination", ROOTS[1:])
def test_invocation_lifetime_does_not_gain_prefix_descriptor_authority(tmp_path, destination):
    _path, builder, proof, originals = preowner_window_fixture(tmp_path, captures={destination: {"lifetime": "invocation"}})
    prefix = proof.public()["prefix_eligibility"]
    assert not prefix["activation_path_possible"]
    assert "selected-window" in prefix["disable_points"][0]["reason"]
    owner = ReachedWindowOwner(builder, proof, originals)
    assert not owner.run()
    assert owner.builder.edits == {}
    assert not owner.builder.scopes


def test_selected_window_whole_is_not_an_earlier_copy_read_fact(tmp_path):
    _path, _builder, proof, _originals = preowner_window_fixture(tmp_path, captures={
        ROOTS[0]: {"lifetime": "selected_window", "initialized": "whole"}})
    prefix = proof.public()["prefix_eligibility"]
    assert not prefix["activation_path_possible"]
    assert "earlier pre-owner copy read" in prefix["disable_points"][-1]["reason"]


def test_source_established_selected_window_none_can_feed_later_copy(tmp_path):
    source = PREOWNER_SOURCE.replace("b=a", "b=workspace")
    _path, _builder, proof, _originals = preowner_window_fixture(tmp_path, source)
    steps = proof.public()["prefix_eligibility"]["steps"]
    reads = [event for event in steps[-1]["ordered_events"] if event["kind"] == "read"]
    assert reads == [{"kind": "read", "resource": ROOTS[2], "coverage": "whole",
                      "payload_provenance": "ordered source definitions from none"}]
    assert all(state["initialized"] == "whole" for state in proof.public()["states"])


def test_allocate_alone_does_not_define_payload(tmp_path):
    source = PREOWNER_SOURCE.replace("workspace=a\n", "")
    _path, _builder, proof, _originals = preowner_window_fixture(tmp_path, source)
    state = next(item for item in proof.public()["states"] if item["resource"] == ROOTS[2])
    assert state["initialized"] == "none"
    assert proof.public()["planned_closures"]["close_points"]
    assert not proof.public()["prefix_eligibility"]["payload_from_allocation_presence"]


@pytest.mark.parametrize("allocation", ["allocate(workspace(-2:n-3),stat=i)",
                                        "allocate(workspace,source=a)",
                                        "allocate(workspace,mold=a)"])
def test_allocation_options_disable_without_descriptor_queries(tmp_path, allocation):
    source = PREOWNER_SOURCE.replace("allocate(workspace(-2:n-3))", allocation)
    _path, _builder, proof, _originals = preowner_window_fixture(tmp_path, source)
    assert not proof.public()["prefix_eligibility"]["activation_path_possible"]
    assert not proof.public()["planned_closures"]["active_node_ids"]


@pytest.mark.parametrize("control", ["cycle", "exit", "return"])
def test_early_prefix_control_is_absorbing_at_original_action(tmp_path, control):
    source = PREOWNER_SOURCE.replace("cycle\nendif", f"{control}\nendif")
    _path, _builder, proof, _originals = preowner_window_fixture(tmp_path, source)
    points = proof.public()["prefix_eligibility"]["disable_points"]
    point = next(item for item in points if item["original_source_kind"] == control.capitalize() + "_Stmt")
    assert point["disabled_is_absorbing"]
    assert point["context_exists"] is False
    assert not point["reopen"]


def test_single_line_prefix_control_disables_before_complete_if(tmp_path):
    source = PREOWNER_SOURCE.replace("if(skip .and. stage==1) then\ncycle\nendif",
                                    "if(skip .and. stage==1) cycle")
    _path, _builder, proof, _originals = preowner_window_fixture(tmp_path, source)
    prefix = proof.public()["prefix_eligibility"]
    assert not prefix["activation_path_possible"]
    point = next(item for item in prefix["disable_points"] if item["original_source_kind"] == "If_Stmt")
    assert point["disabled_is_absorbing"]


@pytest.mark.parametrize("mutation", ["source", "binding", "declaring_scope", "imports", "budget"])
def test_prefix_authority_and_budget_failures_remain_hard(tmp_path, mutation):
    path, builder, proof, originals = preowner_window_fixture(tmp_path)
    if mutation == "source":
        path.write_text(PREOWNER_SOURCE + "\n! changed original\n")
    elif mutation == "binding":
        builder.entry.scope.bindings["a"] = copy(builder.entry.scope.bindings["a"])
    elif mutation == "declaring_scope":
        builder.entry.scope.bindings["a"].declaring_scope = builder.analysis.modules["renamed_prefix"]
    elif mutation == "imports":
        builder.entry.scope.imports["alias"] = ("other", "field")
    else:
        builder.analysis.operation_limit = 1
    with pytest.raises(CompilationError, match="changed|authority|budget|complete bounded original structure"):
        proof.validate(builder.analysis, ENTRY, originals)


def test_foreign_prefix_proof_has_no_emission_authority(tmp_path):
    path, builder, proof, originals = preowner_window_fixture(tmp_path)
    foreign = ScopeBuilder([path], ENTRY, facts=builder.facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    with pytest.raises(CompilationError, match="registered"):
        ReachedWindowOwner(foreign, proof, originals)


@pytest.mark.parametrize("mutation", ["source", "facts"])
def test_native_only_fast_result_revalidates_source_and_facts(tmp_path, mutation):
    path, builder, proof, originals = preowner_window_fixture(tmp_path, captures={
        ROOTS[1]: {"lifetime": "invocation"}})
    owner = ReachedWindowOwner(builder, proof, originals)
    if mutation == "source":
        path.write_text(PREOWNER_SOURCE + "\n! changed after owner construction\n")
    else:
        owner.builder.facts["captures"][ROOTS[0]]["initialized"] = "none"
    with pytest.raises(CompilationError, match="changed|facts|authority"):
        owner.run()
