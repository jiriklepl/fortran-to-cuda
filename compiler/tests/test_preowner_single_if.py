"""One original single-line IF keeps its dormant false path eligible."""

import json
import re
from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.reached_windows import plan_reached_window, prove_reached_window
from compiler.frontend.structured_effects import _freeze
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_configured_sources import inputs
from compiler.tests.test_preowner_activation import (
    ENTRY,
    PREOWNER_SOURCE,
    ROOTS,
    emit_preowner_window,
    preowner_window_fixture,
)
from compiler.tests.test_source_scopes import FACT

BAD_BRANCH = """if(bad) then
call opaque_prefix()
bad_visits=bad_visits+1
endif
"""
SKIP_BRANCH = """if(skip .and. stage==1) then
cycle
endif
"""
OBSERVERS = """module original_observers
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
module observer_bridge
use original_observers,only:exported=>notify
implicit none
private
public::exported
end module
"""
TEAM = """!$omp parallel private(local)
local=n
!$omp end parallel"""


def single_if_source(action="call absent_notice(n)", condition="bad", *, body=None):
    source = PREOWNER_SOURCE.replace(BAD_BRANCH, f"if({condition}) {action}\n").replace(SKIP_BRANCH, "")
    if body is not None:
        source = source.replace("module renamed_prefix\n", """module renamed_prefix
use observer_bridge,only:local_notice=>exported
""")
        source = OBSERVERS.format(body=body) + source
    return source


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def compact(text):
    return re.sub(r"\s+", "", text.lower())


def conditional_point(proof):
    return next(point for point in proof.public()["prefix_eligibility"]["disable_points"]
                if point.get("conditional_action") is not None)


def original_if(builder, point):
    graph = builder.analysis.structure(ENTRY)
    parent, = graph.source_nodes(point["node_id"])
    assert type(parent).__name__ == "If_Stmt"
    return graph, parent


@pytest.mark.parametrize(("action", "body", "reason"), [
    ("call local_notice(n)", "write(*,*) n", "Write_Stmt"),
    ("call local_notice(n)", TEAM, "complete original PARALLEL DO"),
    ("call absent_notice(n)", None, "source call is unresolved or ambiguous: absent_notice"),
], ids=["renamed-reexported-io", "renamed-reexported-team", "unknown-call"])
def test_original_false_path_survives_one_unsupported_single_if_action(tmp_path, action, body, reason):
    source = single_if_source(action, body=body)
    path, builder, proof, originals = preowner_window_fixture(tmp_path, source)
    facts = deepcopy(builder.facts)
    record = proof.public()
    prefix = record["prefix_eligibility"]
    assert prefix["activation_path_possible"]
    assert prefix["disabled_path_possible"]
    assert prefix["disabled_is_absorbing"]
    assert not prefix["reopen"]
    point = conditional_point(proof)
    graph, parent = original_if(builder, point)
    condition, original_action = parent.items
    details = point["conditional_action"]
    assert point["selected_node_ids"] == [point["node_id"]]
    assert point["original_source_kind"] == "If_Stmt"
    assert point["original_source_sha256"] == canonical_hash([str(parent)])
    assert point["guard"] == list(graph.nodes[point["node_id"]].guard)
    assert graph.source_nodes(details["condition_node_id"]) == (condition,)
    assert graph.source_nodes(details["action_node_id"]) == (original_action,)
    assert details["condition_source_sha256"] == canonical_hash([str(condition)])
    assert details["action_source_sha256"] == canonical_hash([str(original_action)])
    assert details["action_source_kind"] == "Call_Stmt"
    assert details["condition_evaluated_once"] is True
    assert details["action_evaluated_once"] is True
    first, last = point["span"]
    original_span = "".join(path.read_text().splitlines(keepends=True)[first - 1:last])
    assert details["original_span_sha256"] == sha256(original_span.encode()).hexdigest()
    assert first == last
    assert reason in point["reason"]
    assert point["context_exists"] is False
    assert point["disabled_is_absorbing"] is True
    assert point["reopen"] is False
    assert {state["resource"] for state in record["states"]} == set(ROOTS)
    assert not record["execution_authorized"]
    assert not record["gpu_legality_established"]
    assert builder.facts == facts
    assert proof.validate(builder.analysis, ENTRY, originals) is proof
    with pytest.raises(CompilationError):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)


@pytest.mark.parametrize(("action", "body"), [
    ("call local_notice(n)", "write(*,*) n"),
    ("call local_notice(n)", TEAM),
    ("call absent_notice(n)", None),
], ids=["io", "team", "unknown"])
def test_emission_disables_only_inside_true_branch_then_executes_action_once(tmp_path, action, body):
    source = single_if_source(action, body=body)
    path, outputs, report, owner = emit_preowner_window(tmp_path, source)
    text = outputs[report["sources"][str(path)]["replacement"]]
    normalized = compact(text)
    start = normalized.index("if(bad)then")
    disable = normalized.index(compact(f"{owner.prefix_eligible} = .false."), start)
    original_action = normalized.index(compact(action), disable)
    ending = normalized.index("endif", original_action)
    spend = normalized.index(compact(f"{owner.spent} = .true."), ending)
    assert start < disable < original_action < ending < spend
    assert normalized.count("if(bad)then") == 1
    assert normalized.count(compact(action)) == 1
    assert normalized.count(compact(f"{owner.prefix_eligible} = .true.")) == 1
    assert normalized.count(compact(f"{owner.spent} = .true.")) == 1
    eligibility = normalized.index(compact(f"if ({owner.prefix_eligible}) then"), spend)
    enable = normalized.index(compact(f"{owner.enabled} = fort_scope_serial_caller()"), eligibility)
    assert spend < eligibility < enable
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    assert not scope["reopen_after_boundary"]
    assert scope["reached_window"]["prefix_eligibility"]["disable_points"] == [conditional_point(owner.window_proof)]
    assert path.read_text() == source


def test_continued_original_if_binds_full_span_and_emits_condition_and_action_once(tmp_path):
    source = single_if_source().replace("if(bad) call absent_notice(n)", """if(bad .and. &
 stage>=1) &
 call absent_notice(n)""")
    path, builder, proof, originals = preowner_window_fixture(tmp_path, source)
    point = conditional_point(proof)
    _graph, parent = original_if(builder, point)
    first, last = point["span"]
    assert last - first == 2
    raw_span = "".join(path.read_text().splitlines(keepends=True)[first - 1:last])
    assert raw_span == "if(bad .and. &\n stage>=1) &\n call absent_notice(n)\n"
    assert point["conditional_action"]["original_span_sha256"] == sha256(raw_span.encode()).hexdigest()
    condition, action = parent.items
    assert point["conditional_action"]["condition_source_sha256"] == canonical_hash([str(condition)])
    assert point["conditional_action"]["action_source_sha256"] == canonical_hash([str(action)])
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    normalized = compact(outputs[report["sources"][str(path)]["replacement"]])
    start = normalized.index("if(bad.and.stage>=1)then")
    disable = normalized.index(compact(f"{owner.prefix_eligible} = .false."), start)
    call = normalized.index("callabsent_notice(n)", disable)
    ending = normalized.index("endif", call)
    assert start < disable < call < ending
    assert normalized.count("bad.and.stage>=1") == 1
    assert normalized.count("callabsent_notice(n)") == 1
    assert path.read_text() == source


@pytest.mark.parametrize("action", ["cycle", "exit", "return"])
def test_single_if_control_action_keeps_false_path_and_never_reopens(tmp_path, action):
    condition = "skip .and. stage==1"
    source = single_if_source(action, condition)
    _path, builder, proof, originals = preowner_window_fixture(tmp_path / "proof", source)
    point = conditional_point(proof)
    assert proof.public()["prefix_eligibility"]["activation_path_possible"]
    assert point["conditional_action"]["action_source_kind"] == action.capitalize() + "_Stmt"
    assert point["disabled_is_absorbing"]
    assert not point["reopen"]
    assert proof.validate(builder.analysis, ENTRY, originals) is proof
    path, outputs, report, owner = emit_preowner_window(tmp_path / "emission", source)
    normalized = compact(outputs[report["sources"][str(path)]["replacement"]])
    start = normalized.index("if(" + compact(condition) + ")then")
    disable = normalized.index(compact(f"{owner.prefix_eligible} = .false."), start)
    ending = normalized.index("endif", disable)
    assert normalized[disable:ending].endswith(action)
    assert normalized.count(compact(f"{owner.prefix_eligible} = .true.")) == 1
    assert normalized.count(compact(f"{owner.spent} = .true.")) == 1
    assert normalized.index(compact(f"{owner.spent} = .true.")) > ending


def test_fixed_metadata_predicate_remains_at_original_point_once(tmp_path):
    condition = "any(route==7).and.route(1)>0"
    source = single_if_source(condition=condition)
    _path, builder, proof, _originals = preowner_window_fixture(tmp_path / "proof", source)
    point = conditional_point(proof)
    conditions = [step for step in proof.public()["prefix_eligibility"]["steps"]
                  if step["original_source_kind"] == "original_native_condition_reads"
                  and step["node_id"] == point["conditional_action"]["condition_node_id"]]
    step, = conditions
    assert [resource["resource"] for resource in step["resources"]] == ["renamed_prefix::route"]
    for key in ("capture_authority", "managed_definition_authority", "allocation_freshness_authority",
                "gpu_legality_established"):
        assert step[key] is False
    assert "renamed_prefix::route" not in builder.facts["captures"]
    path, outputs, report, owner = emit_preowner_window(tmp_path / "emission", source)
    normalized = compact(outputs[report["sources"][str(path)]["replacement"]])
    start = normalized.index("if(" + compact(condition) + ")then")
    assert normalized.count(compact(condition)) == 1
    assert normalized.index("condition_visits=condition_visits+1") < start
    assert start < normalized.index(compact(f"{owner.prefix_eligible} = .false."), start)
    scope, = report["scopes"]
    registered = {item["resource"] for item in scope["resource_bindings"]["resources"]}
    assert registered <= set(ROOTS)
    assert "renamed_prefix::route" not in registered
    assert set(scope["reached_window"]["managed_resources"]) == set(ROOTS)


def test_unallocated_array_condition_rejects_missing_lifetime_authority(tmp_path):
    source = single_if_source(condition="allocated(temporary).and.any(temporary>0.0_8)")
    path = tmp_path / "prefix_stage.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {ROOTS[0]: deepcopy(FACT), **{
                 root: {**deepcopy(FACT), "initialized": "none", "lifetime": "selected_window"}
                 for root in ROOTS[1:]}}}
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    outer = next(node for node in builder.entry.execution.content
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    originals = tuple(node for node in outer.content if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    assert len(originals) == 2
    with pytest.raises(CompilationError, match="storage lifetime requires capture proof:.*temporary"):
        plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, facts)
    assert builder.edits == {}
    assert not builder.scopes
    assert path.read_text() == source


@pytest.mark.parametrize("mutation", ["file", "binding", "condition", "action", "span"])
def test_registered_single_if_rejects_changed_original_authority(tmp_path, mutation):
    source = single_if_source()
    path, builder, proof, originals = preowner_window_fixture(tmp_path, source)
    _graph, parent = original_if(builder, conditional_point(proof))
    if mutation == "file":
        path.write_text(source + "\n! changed after registration\n")
        assert proof.validate(builder.analysis, ENTRY, originals) is proof
        with GenerationTransaction(builder) as transaction:
            owner = ReachedWindowOwner(transaction.builder, proof, originals)
            with pytest.raises(CompilationError, match="source changed"):
                owner.run()
    elif mutation == "binding":
        with pytest.raises(CompilationError):
            builder.entry.scope.bindings["bad"] = copy(builder.entry.scope.bindings["bad"])
    elif mutation == "condition":
        with pytest.raises(CompilationError):
            parent.items = (copy(parent.items[0]), parent.items[1])
    elif mutation == "action":
        with pytest.raises(CompilationError):
            parent.items = (parent.items[0], copy(parent.items[1]))
    else:
        with pytest.raises(CompilationError):
            parent.item.fort_original_span = (parent.item.span[0] + 1, parent.item.span[1] + 1)
    assert not builder.outputs
    assert not builder.edits
    assert not builder.scopes


@pytest.mark.parametrize("field", ["condition_node_id", "action_node_id", "condition_source_sha256",
                                   "action_source_sha256", "original_span_sha256", "action_source_kind",
                                   "condition_evaluated_once", "action_evaluated_once"])
def test_mutated_conditional_action_record_never_authorizes_emission(tmp_path, field):
    _path, builder, proof, originals = preowner_window_fixture(tmp_path, single_if_source())
    record = proof.public()
    record.pop("proof_identity")
    point = next(item for item in record["prefix_eligibility"]["disable_points"] if item.get("conditional_action"))
    details = point["conditional_action"]
    details[field] = False if field.endswith("evaluated_once") else "forged-original-authority"
    object.__setattr__(proof, "_record", _freeze(record))
    with pytest.raises(CompilationError):
        ReachedWindowOwner(builder, proof, originals)


@pytest.mark.parametrize("limit", ["operations", "depth"])
def test_single_if_budget_exhaustion_remains_hard_not_a_disabled_path(tmp_path, limit):
    _path, builder, proof, originals = preowner_window_fixture(tmp_path, single_if_source())
    if limit == "operations":
        builder.analysis.operation_limit = 1
    else:
        builder.analysis.depth_limit = 1
    with pytest.raises(CompilationError, match="budget|bounded|depth"):
        proof.validate(builder.analysis, ENTRY, originals)


def test_single_if_shared_scan_exhaustion_is_not_a_support_decline(tmp_path, monkeypatch):
    from compiler.frontend import native_group_structure

    _path, builder, proof, originals = preowner_window_fixture(tmp_path, single_if_source())
    assert native_group_structure.NATIVE_GROUP_SCAN_LIMIT == 65536
    monkeypatch.setattr(native_group_structure, "NATIVE_GROUP_SCAN_LIMIT", 64)
    with pytest.raises(CompilationError, match="aggregate coordinate/source scan budget exhausted"):
        proof.validate(builder.analysis, ENTRY, originals)


@pytest.mark.parametrize(("statement", "reason"), [
    ("900 if(bad) call absent_notice(n)", "labeled"),
    ("if(bad) call absent_notice(n); bad_visits=bad_visits+1", "semicolon"),
], ids=["label", "shared-semicolon-line"])
def test_unsafe_original_edit_span_keeps_all_native_source_untouched(tmp_path, statement, reason):
    source = single_if_source().replace("if(bad) call absent_notice(n)", statement)
    path, builder, proof, originals = preowner_window_fixture(tmp_path, source)
    prefix = proof.public()["prefix_eligibility"]
    assert not prefix["activation_path_possible"]
    point, = prefix["disable_points"]
    assert point["original_source_kind"] == "If_Stmt"
    assert reason in point["reason"]
    assert "conditional_action" not in point
    if reason == "labeled":
        with pytest.raises(CompilationError, match="source statement labels remain native"):
            ReachedWindowOwner(builder, proof, originals)
    else:
        owner = ReachedWindowOwner(builder, proof, originals)
        assert not owner.run()
    assert builder.edits == {}
    assert not builder.scopes
    assert path.read_text() == source


def test_configured_single_if_crossing_cpp_lines_cannot_rewrite_original_span(tmp_path):
    source = single_if_source().replace("if(bad) call absent_notice(n)", """if(bad) &
#ifdef DOUBLE_PRECISION
call absent_notice(n)
#endif""")
    original, _prepared, document = inputs(tmp_path, source=source)
    facts = {"schema_version": 1, "participation": "serial", "sources": document["source_inputs"],
             "captures": {ROOTS[0]: deepcopy(FACT), **{
                 root: {**deepcopy(FACT), "initialized": "none", "lifetime": "selected_window"}
                 for root in ROOTS[1:]}}}
    builder = ScopeBuilder([original], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"),
                           analysis_sources=document)
    outer = next(node for node in builder.entry.execution.content
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    originals = tuple(node for node in outer.content if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    assert len(originals) == 2
    proof = plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, facts)
    assert not proof.public()["prefix_eligibility"]["activation_path_possible"]
    point, = proof.public()["prefix_eligibility"]["disable_points"]
    assert "preprocessor" in point["reason"]
    assert "conditional_action" not in point
    owner = ReachedWindowOwner(builder, proof, originals)
    assert not owner.run()
    assert owner.builder.edits == {}
    assert original.read_text() == source
