"""Reached source proofs retain original authority and bounded control flow."""

import copy
import json

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.source_effects import SOURCE_SUMMARY_VERSION, SourceEffects
from compiler.ir import CompilationError


def source(tmp_path, body, *, specification="", helpers="", operations=256, cache=None):
    path = tmp_path / "graph.f90"
    path.write_text("module varied\nimplicit none\ncontains\nsubroutine advance(a,b,n,flag)\n"
                    "real(8),intent(inout)::a(:),b(:)\ninteger,intent(in)::n\n"
                    "logical,intent(in)::flag\ninteger::i\n" + specification + "\n" + body
                    + "\nend subroutine\n" + helpers + "\nend module\n")
    return path, SourceEffects([path], operations=operations, summary_cache=cache)


def assignments(analysis):
    return list(walk(analysis.routines["varied::advance"].execution, F.Assignment_Stmt))


def test_unknown_else_does_not_poison_reached_then(tmp_path):
    _path, analysis = source(tmp_path, "if(flag) then\na(1)=b(1)\nelse\ncall opaque(a)\nendif")
    assert not analysis.summarize("varied::advance")["complete"]
    graph = analysis.structure("varied::advance")
    assert graph.available
    assert {node.kind for node in graph.nodes.values()} >= {"sequence", "branch", "boundary", "entry", "operation"}
    summary = analysis.segment_summary("varied::advance", assignments(analysis))
    assert summary["complete"], summary["reasons"]
    assert summary["summary_version"] == SOURCE_SUMMARY_VERSION == 13
    assert summary["structured_identity"] == graph.identity
    assert summary["includes_entry"] is False
    assert summary["definition_changes"] == []
    assert all(effect["guard_frames"] == [{"procedure": "varied::advance", "condition": "flag"}]
               for effect in summary["ordered_effects"])


def test_entire_oversized_closure_keeps_bounded_reached_segments(tmp_path):
    _path, analysis = source(tmp_path, "\n".join(f"a({i})=b({i})" for i in range(1, 10)), operations=12)
    assert not analysis.summarize("varied::advance")["complete"]
    graph = analysis.structure("varied::advance")
    assert graph.available
    for node in assignments(analysis):
        summary = analysis.segment_summary("varied::advance", (node,))
        assert summary["complete"], summary["reasons"]
        assert len(summary["ordered_effects"]) == 2
    assert not analysis.segment_summary("varied::advance", assignments(analysis))["complete"]


def test_structural_containers_do_not_spend_the_source_operation_budget(tmp_path):
    body = 'do i=1,n\nif(flag) then\na(i)=b(i)\nelse\nendif\nenddo\n'
    _, analysis = source(tmp_path, body*3, operations=12)
    graph = analysis.structure('varied::advance')
    assert graph.available, graph.reasons
    assert graph.version == 5
    assert graph.operation_count == 9
    assert graph.operation_limit == 12
    assert len(graph.nodes) > analysis.operation_limit+2
    assert len(graph.nodes) <= graph.node_limit
    for node in assignments(analysis):
        assert analysis.segment_summary('varied::advance', (node,))['complete']


def test_structural_accounting_keeps_the_operation_limit(tmp_path):
    _, analysis = source(tmp_path, 'a(1)=b(1)\n'*13, operations=12)
    graph = analysis.structure('varied::advance')
    assert not graph.available
    assert any('operation budget exhausted' in reason for reason in graph.reasons)


def test_condition_and_loop_header_ids_preserve_original_protected_expressions(tmp_path):
    _path, analysis = source(tmp_path, "if(flag) then\na(1)=1\nelse if(b(n)>0) then\n"
        "do i=1,n\na(i)=2\nenddo\nendif")
    routine = analysis.routines["varied::advance"]
    graph = analysis.structure(routine.qualified)
    header = walk(routine.execution, F.Else_If_Stmt)[0]
    identity = graph.node_id(header, "condition")
    assert graph.source_nodes(identity) == (header.items[0],)
    summary = analysis.segment_summary(routine.qualified, identity)
    assert summary["complete"], summary["reasons"]
    assert summary["ordered_effects"][0]["resource"] == "argument::b"
    assert summary["ordered_effects"][0]["guard_frames"] == [{"procedure": routine.qualified, "condition": ".not.(flag)"}]
    do = walk(routine.execution, F.Nonlabel_Do_Stmt)[0]
    loop_summary = analysis.segment_summary(routine.qualified, graph.node_id(do, "header"))
    assert loop_summary["complete"], loop_summary["reasons"]
    assert any(effect["resource"] == "argument::n" for effect in loop_summary["ordered_effects"])


def test_original_out_definition_is_once_at_reached_child_entry(tmp_path):
    helpers = """subroutine leaf(x)
real(8),intent(out)::x(:)
x(1)=7
end subroutine
"""
    _path, analysis = source(tmp_path, "a(1)=2\ncall leaf(a)\nb(1)=a(1)", helpers=helpers)
    routine = analysis.routines["varied::advance"]
    call = walk(routine.execution, F.Call_Stmt)[0]
    summary = analysis.segment_summary(routine.qualified, (call,))
    assert summary["complete"], summary["reasons"]
    events = [effect for effect in summary["ordered_effects"] if effect["kind"] == "definition_change"]
    assert len(events) == 1 and events[0]["resource"] == "argument::a"
    operation, = [op for op in summary["operations"] if op["kind"] == "call"]
    assert operation["definition_events"][0]["position"] == "callee entry after original actual evaluation"
    assert operation["resource_mappings"][0]["formal_resource"] == "argument::x"


def test_foreign_nodes_copies_mutated_source_and_unordered_selections_reject(tmp_path):
    path, analysis = source(tmp_path, "a(1)=b(1)\na(2)=b(2)")
    first, second = assignments(analysis)
    for foreign in (F.Assignment_Stmt(str(first)), copy.copy(first)):
        with pytest.raises(CompilationError, match="original source authority"):
            analysis.segment_summary("varied::advance", (foreign,))
    with pytest.raises(CompilationError, match="original source order"):
        analysis.segment_summary("varied::advance", (second, first))
    graph = analysis.structure("varied::advance")
    with pytest.raises(CompilationError, match="overlap"):
        analysis.segment_summary("varied::advance", (graph.root_id, graph.node_id(first)))
    path.write_text(path.read_text().replace("a(1)=b(1)", "a(1)=3*b(1)"))
    with pytest.raises(CompilationError, match="changed"):
        analysis.structure("varied::advance")


def test_mutated_original_ast_cannot_borrow_its_old_graph(tmp_path):
    _path, analysis = source(tmp_path, "a(1)=b(1)")
    analysis.structure("varied::advance")
    routine = analysis.routines["varied::advance"]
    routine.execution.content.pop()
    with pytest.raises(CompilationError, match="source-backed .*authority"):
        analysis.structure(routine.qualified)


def test_original_span_and_graph_records_cannot_be_replaced_to_grant_authority(tmp_path):
    _path, analysis = source(tmp_path, "a(1)=b(1)")
    original, = assignments(analysis)
    graph = analysis.structure("varied::advance")
    identity = graph.node_id(original)
    with pytest.raises(TypeError):
        graph._originals[identity] = (F.Assignment_Stmt(str(original)),)
    with pytest.raises(TypeError):
        graph.nodes[identity].details["evaluation"] = "condition"
    original.item.fort_original_span = (1, 1)
    with pytest.raises(CompilationError, match="source span authority"):
        analysis.segment_summary("varied::advance", (identity,))


def test_segments_from_mutually_exclusive_arms_do_not_form_one_reached_sequence(tmp_path):
    _path, analysis = source(tmp_path, "if(flag) then\na(1)=b(1)\nelse\na(2)=b(2)\nendif")
    with pytest.raises(CompilationError, match="reached guarded segment"):
        analysis.segment_summary("varied::advance", assignments(analysis))


def test_capture_locals_and_entry_effects_are_explicit(tmp_path):
    _path, analysis = source(tmp_path, "local(1)=a(1)\nb(1)=local(1)", specification="real(8)::local(3)")
    ordinary = analysis.segment_summary("varied::advance", assignments(analysis))
    captured = analysis.segment_summary("varied::advance", assignments(analysis), capture_locals=True)
    assert ordinary["complete"] and captured["complete"]
    assert "varied::advance::local" not in {effect["resource"] for effect in ordinary["ordered_effects"]}
    assert "varied::advance::local" in {effect["resource"] for effect in captured["ordered_effects"]}
    assert ordinary["demand_identity"] != captured["demand_identity"]
    captured["operations"].clear()
    assert analysis.segment_summary("varied::advance", assignments(analysis), capture_locals=True)["operations"]


def test_descriptor_stable_allocatable_inout_and_transitive_forwarding(tmp_path):
    helpers = """subroutine middle(x)
real(8),allocatable,intent(inout)::x(:)
call leaf(x)
end subroutine
subroutine leaf(y)
real(8),allocatable,intent(inout)::y(:)
y(lbound(y,1))=2
end subroutine
"""
    path, _analysis = source(tmp_path, "call middle(a)", helpers=helpers)
    path.write_text(path.read_text().replace("real(8),intent(inout)::a(:),b(:)",
        "real(8),allocatable,intent(inout)::a(:),b(:)"))
    analysis = SourceEffects([path])
    proof = analysis.descriptor_stability("varied::advance")
    assert proof["complete"], proof
    assert all(item["stable"] and item["original_descriptor_required"] for item in proof["resources"])
    summary = analysis.summarize("varied::advance")
    assert summary["complete"], summary["reasons"]


@pytest.mark.parametrize("body", ["allocate(a(3))", "deallocate(a)", "a=3", "call unknown(a)",
    "if(flag) then\na(1)=2\nelse\ncall unknown(a)\nendif"])
def test_descriptor_changes_unknown_effects_and_whole_assignments_reject(tmp_path, body):
    path, _analysis = source(tmp_path, body)
    path.write_text(path.read_text().replace("real(8),intent(inout)::a(:),b(:)",
        "real(8),allocatable,intent(inout)::a(:),b(:)"))
    analysis = SourceEffects([path])
    proof = analysis.descriptor_stability("varied::advance")
    assert not proof["complete"]
    assert any(not item["stable"] for item in proof["resources"])


def test_empty_out_dummy_still_has_descriptor_entry_change(tmp_path):
    path, _analysis = source(tmp_path, "continue")
    path.write_text(path.read_text().replace("real(8),intent(inout)::a(:),b(:)",
        "real(8),allocatable,intent(out)::a(:),b(:)"))
    proof = SourceEffects([path]).descriptor_stability("varied::advance")
    assert not proof["complete"]
    assert all(not item["stable"] for item in proof["resources"])


def test_reached_native_read_does_not_repeat_original_out_entry_event(tmp_path):
    path, _analysis = source(tmp_path, "a(1)=2\na(1)=a(1)+1")
    path.write_text(path.read_text().replace("real(8),intent(inout)::a(:),b(:)",
        "real(8),intent(out)::a(:)\nreal(8),intent(inout)::b(:)"))
    analysis = SourceEffects([path])
    selection = (assignments(analysis)[1],)
    sections = analysis.native_sections_for_nodes("varied::advance", selection)
    assert sections.available, sections.reason
    output, = sections.resources
    assert len(output.reads) == len(output.overwrites) == 1
    summary = analysis.segment_summary("varied::advance", selection)
    assert summary["definition_changes"] == []
    with_entry = analysis.native_sections_for_nodes("varied::advance", selection, include_entry=True)
    assert not with_entry.available and "INTENT(OUT) reads" in with_entry.reason


def test_reached_dynamic_dummy_uses_original_descriptor_after_bound_scalar_changed(tmp_path):
    from compiler.scopes.access import build_native_access
    path, _analysis = source(tmp_path, "n=n-1\na(1)=a(1)+1")
    path.write_text(path.read_text().replace("real(8),intent(inout)::a(:),b(:)",
        "real(8),intent(inout)::a(n:),b(:)").replace("integer,intent(in)::n", "integer,intent(inout)::n"))
    analysis = SourceEffects([path])
    sections = analysis.native_sections_for_nodes("varied::advance", assignments(analysis))
    assert sections.available, sections.reason
    output, = sections.resources
    assert output.descriptor_lower_bounds
    assert output.descriptor_origin == "original_dummy"
    assert output.lower_bound_expressions == ()
    assert output.public()["original_dummy_lower_bounds"] is True
    assert "argument::n" not in {item.resource for item in output.dependencies}
    with pytest.raises(CompilationError, match="original reached dummy descriptor"):
        build_native_access(output, "handle", "fort_exact")
    code = build_native_access(output, "handle", "fort_exact", logical_lower_bounds={"argument::a": ("original_a_lower",)})
    assert "int(original_a_lower, c_int64_t)" in "\n".join(code.prepare)


def test_structural_cache_is_rebound_to_original_ast_and_invalidates_contracts(tmp_path):
    cache = tmp_path / "cache"
    path, analysis = source(tmp_path, "if(flag) then\na(1)=b(1)\nelse\ncall unknown(a)\nendif", cache=cache)
    graph = analysis.structure("varied::advance")
    assert list(cache.glob("fort_source_summary_v2_*.json"))
    fresh = SourceEffects([path], summary_cache=cache)
    reused = fresh.structure("varied::advance")
    assert reused.identity == graph.identity
    assert fresh._summary_cache.stats["disk_hits"] == 1
    assert reused.source_nodes(reused.node_id(assignments(fresh)[0]))[0] is assignments(fresh)[0]
    with pytest.raises(CompilationError, match="original source authority"):
        fresh.segment_summary("varied::advance", assignments(analysis))
    changed = SourceEffects([path], contracts={"unrelated": {"identity": "new"}}, summary_cache=cache)
    assert changed.structure("varied::advance").identity != graph.identity
    assert changed._summary_cache.stats["disk_hits"] == 0


def test_structural_cache_tampering_is_rejected_even_with_fresh_payload_digest(tmp_path):
    from hashlib import sha256
    cache = tmp_path / "cache"
    path, analysis = source(tmp_path, "a(1)=b(1)", cache=cache)
    original = analysis.structure("varied::advance")
    record_path, = cache.glob("fort_source_summary_v2_*.json")
    record = json.loads(record_path.read_text())
    record["payload"]["structure"]["nodes"][0]["definition_changes"] = ["argument::b"]
    record["payload_sha256"] = sha256(json.dumps(record["payload"], ensure_ascii=True, allow_nan=False,
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    record_path.write_text(json.dumps(record))
    fresh = SourceEffects([path], summary_cache=cache)
    assert fresh.structure("varied::advance").identity == original.identity
    assert fresh._cache_rejections == 1


def test_diamond_graph_reuses_leaves_but_counts_each_reached_call(tmp_path):
    helpers = """subroutine left(x)
real(8),intent(inout)::x(:)
call leaf(x)
end subroutine
subroutine right(x)
real(8),intent(inout)::x(:)
call leaf(x)
end subroutine
subroutine leaf(x)
real(8),intent(inout)::x(:)
x(1)=x(1)+1
end subroutine
"""
    _path, analysis = source(tmp_path, "\n".join(["call left(a)", "call right(a)"] * 70), helpers=helpers)
    assert not analysis.summarize("varied::advance")["complete"]
    assert analysis.structure("varied::advance").available
    leaf = analysis.structure("varied::leaf")
    assert analysis.structure("varied::leaf") is leaf
    calls = walk(analysis.routines["varied::advance"].execution, F.Call_Stmt)
    first = analysis.segment_summary("varied::advance", calls[:2])
    assert first["complete"], first["reasons"]
    assert len([effect for effect in first["ordered_effects"] if effect["kind"] == "write"]) == 2
    assert not analysis.segment_summary("varied::advance", calls)["complete"]


def test_bounded_graph_exhaustion_is_not_a_complete_empty_proof(tmp_path):
    _path, analysis = source(tmp_path, "\n".join("a(1)=b(1)" for _ in range(8)), operations=3)
    graph = analysis.structure("varied::advance")
    assert not graph.available and graph.reasons
    assert graph.nodes[graph.root_id].kind == "boundary"
    with pytest.raises(CompilationError, match="skeleton unavailable"):
        analysis.segment_summary("varied::advance", assignments(analysis)[:1])


def test_nested_native_unknown_child_cannot_establish_descriptor_stability(tmp_path):
    helpers = """subroutine middle(x)
real(8),intent(inout)::x(:)
call unresolved(x)
end subroutine
"""
    path, _analysis = source(tmp_path, "call middle(a)", helpers=helpers)
    path.write_text(path.read_text().replace("real(8),intent(inout)::a(:),b(:)",
        "real(8),allocatable,intent(inout)::a(:),b(:)"))
    proof = SourceEffects([path]).descriptor_stability("varied::advance")
    assert not proof["complete"]
    assert all(not resource["stable"] for resource in proof["resources"])
