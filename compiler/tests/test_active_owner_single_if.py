"""An original scalar IF can close one active path without closing its sibling."""

import json
import re
from copy import copy
from hashlib import sha256

import pytest

from compiler.frontend.reached_windows import plan_reached_window, prove_reached_window
from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.tests.test_planned_reached_closures import (
    ENTRY,
    PLANNED_SOURCE,
    ROOTS,
    planned_window_fixture,
)

ORIGINAL_BRANCH = """if(bad) then
call replace_storage(b)
else
call face_update(b)
endif"""
SINGLE_SOURCE = PLANNED_SOURCE.replace(ORIGINAL_BRANCH, "if(bad) call replace_storage(b)")


def compact(text):
    return re.sub(r"\s+", "", text.lower())


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def close_point(builder, proof):
    point, = proof.public()["planned_closures"]["close_points"]
    graph = builder.analysis.structure(ENTRY)
    parent, = graph.source_nodes(point["node_id"])
    assert type(parent).__name__ == "If_Stmt"
    return point, graph, parent


@pytest.mark.parametrize("condition", ["bad", ".true.", ".false."])
def test_scalar_single_if_keeps_false_path_and_closes_before_true_action_once(tmp_path, condition):
    source = SINGLE_SOURCE.replace("if(bad) call replace_storage(b)",
                                   f"if({condition}) call replace_storage(b)")
    path, builder, proof, originals = planned_window_fixture(tmp_path, source)
    point, graph, parent = close_point(builder, proof)
    predicate, action = parent.items
    details = point["conditional_action"]
    assert point["selected_node_ids"] == [point["node_id"]]
    assert point["guard"] == list(graph.nodes[point["node_id"]].guard)
    assert point["original_source_sha256"] == canonical_hash([str(parent)])
    assert graph.source_nodes(details["condition_node_id"]) == (predicate,)
    assert graph.source_nodes(details["action_node_id"]) == (action,)
    assert details["condition_source_sha256"] == canonical_hash([str(predicate)])
    assert details["action_source_sha256"] == canonical_hash([str(action)])
    assert details["action_source_kind"] == "Call_Stmt"
    assert details["condition_evaluated_once"] is True
    assert details["action_evaluated_once"] is True
    assert point["closed_is_absorbing"] is True
    assert point["reopen"] is False
    planned = proof.public()["planned_closures"]
    assert planned["closed_path_possible"]
    assert planned["active_path_may_return"]
    assert not planned["conditional_return_is_unconditional_definition_summary"]
    for state in proof.public()["states"]:
        assert state["coverage_on_return"] == "unknown_after_close"
        assert state["active_path_coverage_on_return"] == "whole"
        assert state["active_path_return_is_conditional"]
        assert state["closed_path_storage_authority"] == "unavailable"
    assert proof.validate(builder.analysis, ENTRY, originals) is proof
    with pytest.raises(CompilationError):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)

    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    text = compact(outputs[report["sources"][str(path)]["replacement"]])
    start = text.index("if(" + compact(condition) + ")then")
    close = text.index("fort_scope_close(", start)
    disable = text.index(compact(f"{owner.enabled} = .false."), close)
    call = text.index("callreplace_storage(b)", disable)
    ending = text.index("endif", call)
    assert start < close < disable < call < ending
    assert text.count("if(" + compact(condition) + ")then") == 1
    assert text.count("callreplace_storage(b)") == 1
    assert text.count("condition_visits=condition_visits+1") == 1
    assert text.count("callnative_prefix(a)") == 1
    assert text.count("callmove_alloc(a,temporary)") == 1
    assert text.count(compact(f"{owner.spent} = .true.")) == 1
    assert len(owner.units) == 2
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    boundary, = scope["boundaries"]
    emitted_point, = scope["reached_window"]["planned_closures"]["close_points"]
    assert boundary["original_node_ids"] == emitted_point["selected_node_ids"]
    assert emitted_point["conditional_action"] == details
    assert not scope["reopen_after_boundary"]
    assert path.read_text() == source


@pytest.mark.parametrize("shape", ["continued", "enclosing-branch"])
def test_exact_single_if_span_and_enclosing_join_preserve_active_consumer(tmp_path, shape):
    if shape == "continued":
        statement = "if(bad .and. &\n stage==1) &\n call replace_storage(b)"
    else:
        statement = """if(stage==1) then
if(bad) call replace_storage(b)
else
call face_update(b)
endif"""
    source = SINGLE_SOURCE.replace("if(bad) call replace_storage(b)", statement)
    path, builder, proof, originals = planned_window_fixture(tmp_path, source)
    point, _graph, parent = close_point(builder, proof)
    first, last = point["span"]
    raw = "".join(path.read_text().splitlines(keepends=True)[first - 1:last])
    assert point["conditional_action"]["original_span_sha256"] == sha256(raw.encode()).hexdigest()
    assert last - first == (2 if shape == "continued" else 0)
    assert proof.public()["planned_closures"]["active_path_may_return"]
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    text = compact(outputs[report["sources"][str(path)]["replacement"]])
    condition = compact(str(parent.items[0]))
    start = text.index(f"if({condition})then")
    close = text.index("fort_scope_close(", start)
    call = text.index("callreplace_storage(b)", close)
    assert start < close < call
    assert text.count(f"if({condition})then") == 1
    assert text.count("callreplace_storage(b)") == 1
    assert len(report["scopes"][0]["gpu_leaves"]) == 2
    assert not report["scopes"][0]["reopen_after_boundary"]
    if shape == "enclosing-branch":
        assert text.index("if(stage==1)then") < start
        assert text.index("callface_update(b)") > call
        assert text.count("callface_update(b)") == 1
    assert path.read_text() == source


def test_forced_original_parent_close_stays_unconditional(tmp_path):
    path, builder, _proof, originals = planned_window_fixture(tmp_path, SINGLE_SOURCE)
    parent = originals[2]
    assert type(parent).__name__ == "If_Stmt"
    proof = plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts,
                                close_before=(((parent,), "forced original parent boundary"),))
    point, _graph, _parent = close_point(builder, proof)
    assert "conditional_action" not in point
    assert point["reason"] == "forced original parent boundary"
    assert not proof.public()["planned_closures"]["active_path_may_return"]
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.index("fort_scope_close(") < text.index("if(bad) call replace_storage(b)")
    assert text.count("if(bad) call replace_storage(b)") == 1
    assert len(owner.units) == 1
    assert path.read_text() == SINGLE_SOURCE


@pytest.mark.parametrize("mutation", ["source", "binding", "condition", "action", "span", "facts"])
def test_conditional_close_rejects_changed_original_authority_without_emission(tmp_path, mutation):
    path, builder, proof, originals = planned_window_fixture(tmp_path, SINGLE_SOURCE)
    _point, _graph, parent = close_point(builder, proof)
    if mutation == "source":
        path.write_text(SINGLE_SOURCE + "\n! changed after registration\n")
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
    elif mutation == "span":
        first, last = parent.item.span
        with pytest.raises(CompilationError):
            parent.item.fort_original_span = (first + 1, last + 1)
    else:
        builder.facts["captures"][ROOTS[0]]["initialized"] = "none"
        with pytest.raises(CompilationError):
            ReachedWindowOwner(builder, proof, originals)
    assert not builder.edits
    assert not builder.outputs
    assert not builder.scopes


@pytest.mark.parametrize("limit", ["operations", "shared-source-scan"])
def test_conditional_close_keeps_source_budget_failures_hard(tmp_path, monkeypatch, limit):
    from compiler.frontend import native_group_structure

    _path, builder, proof, originals = planned_window_fixture(tmp_path, SINGLE_SOURCE)
    if limit == "operations":
        builder.analysis.operation_limit = 1
    else:
        monkeypatch.setattr(native_group_structure, "NATIVE_GROUP_SCAN_LIMIT", 1)
    with pytest.raises(CompilationError, match="budget|bounded"):
        ReachedWindowOwner(builder, proof, originals)
    assert not builder.edits
    assert not builder.scopes
