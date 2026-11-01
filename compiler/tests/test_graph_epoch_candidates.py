"""Numerical priority changes search order, never storage or execution authority."""

from hashlib import sha256

import pytest

from compiler.ir import CompilationError
from compiler.scopes.graph import GraphOwner
from compiler.scopes.graph_epoch_candidates import candidates, original_candidates
from compiler.scopes.graph_epochs import discover_graph_storage_epoch
from compiler.tests.test_control_local_graph_epochs import CONDITIONAL_SOURCE
from compiler.tests.test_graph_storage_epochs import SOURCE, builder_at


def search(tmp_path, source=SOURCE):
    path, builder = builder_at(tmp_path, source)
    owner = GraphOwner(builder)
    roots = tuple(builder.facts["captures"])
    ordered = tuple(candidates(owner, roots))
    return path, builder, owner, ordered


def signature(rows):
    return [(selected, cut) for selected, cut, *_rest in rows]


def test_connected_preparation_precedes_a_longer_lifetime_changing_prefix(tmp_path):
    source = SOURCE.replace(
        "call relocate(left,right)",
        "call relocate(left,right)\ncall consumer(left,right,n)\ncall consumer(left,right,n)",
    )
    path, builder, owner, ordered = search(tmp_path, source)
    original = tuple(original_candidates(owner.graph))
    selected, cut, priority = ordered[0]
    assert priority["priority"] == "connected prepared calls"
    assert len(selected) == 2
    assert owner.graph.nodes[cut].details["procedure"] == "epochs::relocate"
    assert len(original[0][0]) > len(selected)
    assert set(signature(ordered)) == set(original)
    assert len(signature(ordered)) == len(set(signature(ordered)))
    assert not owner.patches
    assert not owner.boundaries
    assert getattr(owner, "position_epoch", None) is None
    assert path.read_text() == source
    assert not builder.generated
    assert not builder.scopes
    assert all(row["prepared"] for row in owner.position_epoch_search["numerical_preparations"])


@pytest.mark.parametrize("count", [33, 150])
def test_connected_group_has_no_source_work_count_limit(tmp_path, count):
    source = SOURCE.replace("call consumer(left,right,n)", "\n".join(["call consumer(left,right,n)"] * (count - 1)))
    _path, builder = builder_at(tmp_path, source)
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    assert len(epoch.selected) == count
    assert len(epoch.public()["calls"]) == count
    assert len(owner.position_epoch_search["attempts"]) == 1
    assert len(owner.position_epoch_search["numerical_preparations"]) == 2
    assert len(builder.analysis._descriptor_window_proofs) == 1


def test_unrelated_local_allocation_is_only_a_priority_hint(tmp_path):
    source = SOURCE.replace(
        "subroutine relocate(left,right)",
        "subroutine scratch()\nreal(8),allocatable::temporary(:)\nallocate(temporary(8))\n"
        "temporary=0\ndeallocate(temporary)\nend subroutine\nsubroutine relocate(left,right)",
    )
    source = source.replace(
        "call relocate(left,right)", "call scratch()\ncall consumer(left,right,n)\ncall relocate(left,right)"
    )
    _path, _builder, owner, ordered = search(tmp_path, source)
    selected, cut, priority = ordered[0]
    assert len(selected) == 2
    assert owner.graph.nodes[cut].details["procedure"] == "epochs::scratch"
    assert "Allocate_Stmt" in priority["reason"]
    assert not owner.patches
    assert getattr(owner, "position_epoch", None) is None
    # The longer source selection crossing unrelated storage remains eligible.
    assert any(len(selection) > len(selected) for selection, _cut, _row in ordered)
    assert set(signature(ordered)) == set(original_candidates(owner.graph))


def test_conditional_lifetime_change_retains_original_branch_partition(tmp_path):
    _path, builder = builder_at(tmp_path, CONDITIONAL_SOURCE)
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    record = epoch.public()
    assert record["version"] == 2
    assert len(record["calls"]) == 2
    (cut,) = record["control_local_regions"]["retirement_subtrees"]
    assert owner.graph.nodes[cut["node_id"]].guard
    assert len(epoch.boundaries()) == 2
    assert len(owner.position_epoch_search["attempts"]) == 1


def readonly_source():
    source = SOURCE.replace(
        "subroutine relocate(left,right)",
        "subroutine observe(b,n,checksum)\ninteger,intent(in)::n\nreal(8),intent(in)::b(:)\n"
        "real(8),intent(out)::checksum\ninteger::i\nchecksum=0\ndo i=1,n\n"
        "checksum=checksum+b(i)*i\nenddo\nend subroutine\nsubroutine relocate(left,right)",
    )
    return (
        source.replace("subroutine advance(left,right,n,steps)", "subroutine advance(left,right,n,steps,checksum)")
        .replace("integer,intent(in)::n,steps", "integer,intent(in)::n,steps\nreal(8),intent(out)::checksum")
        .replace("call producer(right,n)", "call producer(right,n)\ncall observe(right,n,checksum)")
    )


def test_nontrivial_readonly_host_work_can_connect_numerical_children(tmp_path):
    _path, builder = builder_at(tmp_path, readonly_source())
    outputs, report = builder.run()
    assert report["scope_count"] == 1
    epoch = report["scopes"][0]["checked_execution_plan"]["storage_epoch"]["selected_position_epoch"]
    assert [call["procedure"] for call in epoch["calls"]] == ["epochs::producer", "epochs::observe", "epochs::consumer"]
    search = report["graph_owner_attempt"]["storage_epoch_search"]
    assert search["execution_authority"] is False
    assert search["attempts"][0]["priority"] == "connected prepared calls"
    assert outputs


def test_absent_numerical_anchor_preserves_complete_original_search(tmp_path):
    source = SOURCE.replace("do i=1,n\nb(i)=real(i,8)\nenddo", "b(1)=real(n,8)").replace(
        "do i=1,n\na(i)=2.0_8*b(i)\nenddo", "a(1)=2.0_8*b(1)"
    )
    _path, _builder, owner, ordered = search(tmp_path, source)
    assert signature(ordered) == list(original_candidates(owner.graph))
    assert all(row["priority"] == "original complete search" for _selected, _cut, row in ordered)
    assert not owner.patches


def test_source_priority_does_not_require_or_grant_execution_actual_evaluation(tmp_path):
    source = SOURCE.replace("call producer(right,n)", "call producer(right,n+1)")
    _path, builder, owner, ordered = search(tmp_path, source)
    selected, _cut, priority = ordered[0]
    assert priority["priority"] == "connected prepared calls"
    (original,) = owner.graph.source_nodes(selected[0])
    with pytest.raises(CompilationError, match="original-position checked evaluation"):
        builder.execution_source_call(owner.routine, original)
    assert not owner.patches
    assert getattr(owner, "position_epoch", None) is None


def test_original_lifetime_call_can_remain_an_opaque_graph_boundary(tmp_path):
    source = (
        SOURCE.replace(
            "module epochs\ncontains",
            "module epochs\ntype::box\nreal(8),allocatable::payload(:)\nend type\ntype(box)::state\ncontains",
        )
        .replace(
            "subroutine relocate(left,right)",
            "subroutine scratch(value)\ntype(box),intent(inout)::value\nallocate(value%payload(8))\n"
            "end subroutine\nsubroutine relocate(left,right)",
        )
        .replace("call relocate(left,right)", "call scratch(state)\ncall relocate(left,right)")
    )
    _path, _builder, owner, ordered = search(tmp_path, source)
    selected, cut, priority = ordered[0]
    assert priority["priority"] == "connected prepared calls"
    assert len(selected) == 2
    assert owner.graph.nodes[cut].kind == "boundary"
    assert "unsupported components" in owner.graph.nodes[cut].details["reason"]
    assert not owner.patches
    assert getattr(owner, "position_epoch", None) is None
    assert set(signature(ordered)) == set(original_candidates(owner.graph))


def test_failed_priority_frontier_preserves_complete_original_search(tmp_path, monkeypatch):
    from compiler.scopes.graph_epoch_candidates import _Priority

    def unavailable(_self, _identity):
        raise CompilationError("injected search-only frontier decline")

    monkeypatch.setattr(_Priority, "frontier", unavailable)
    _path, builder, owner, ordered = search(tmp_path)
    assert signature(ordered) == list(original_candidates(owner.graph))
    assert owner.position_epoch_search["priority_search_decline"] == "injected search-only frontier decline"
    assert not owner.patches
    # The unchanged ordinary semantic issuer can still admit the original pair.
    epoch = discover_graph_storage_epoch(owner)
    assert len(epoch.selected) == 2
    assert len(epoch.public()["calls"]) == 2
    assert builder.analysis._descriptor_window_proofs


def test_reordering_preserves_admitted_numerical_sources(tmp_path, monkeypatch):
    from compiler.scopes import graph_epoch_candidates

    path, builder = builder_at(tmp_path, readonly_source())
    first_outputs, first_report = builder.run()

    def original_order(owner, _roots):
        owner.position_epoch_search = {"attempts": []}
        for selected, cut in original_candidates(owner.graph):
            yield selected, cut, {"priority": "test original order", "reason": None}

    monkeypatch.setattr(graph_epoch_candidates, "candidates", original_order)
    _path, second = builder_at(tmp_path, path.read_text())
    second_outputs, second_report = second.run()
    assert first_report["scope_count"] == second_report["scope_count"] == 1
    assert first_report["scopes"][0]["gpu_leaves"] == second_report["scopes"][0]["gpu_leaves"]

    # The manifest deliberately reports search diagnostics; generated program
    # artifacts and numerical choices remain identical for this same interval.
    def artifacts(outputs):
        return {
            name: sha256(text.encode()).hexdigest() for name, text in outputs.items() if name != "scope-manifest.json"
        }

    assert artifacts(first_outputs) == artifacts(second_outputs)
