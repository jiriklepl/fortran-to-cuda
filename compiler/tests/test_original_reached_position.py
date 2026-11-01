"""A root graph position can filter entry facts without an IF-exit contract."""
import json
import sys
from copy import copy
from hashlib import sha256

import pytest

from compiler.frontend.entry_preservation import prove_preserved_entry_definitions
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_position import prove_original_reached_position
from compiler.ir import CompilationError
from compiler.tests.test_owner_local_call_groups import SOURCE, fixture

PROCEDURE = "local_groups::run"


def prepared(tmp_path, *, prefix="a(:)=a(:)+1.0_8", middle="call second(a,b,n)", source=None):
    if source is None:
        source = SOURCE.replace("call unknown_prefix()", prefix).replace("if(flag)then\ncall first", "call first")
        source = source.replace("endif\ncall unknown_suffix()", "call unknown_suffix()")
        source = source.replace("call second(a,b,n)", middle)
    path, builder = fixture(tmp_path, source, lifetime="invocation")
    for fact in builder.facts["captures"].values():
        fact["initialized"] = "whole"
    analysis = builder.analysis
    graph = analysis.structure(PROCEDURE)
    children = graph.nodes[graph.root_id].children
    first = next(i for i in children if str(graph.source_nodes(i)[0]).lower().startswith("call first("))
    last = next(i for i in children if str(graph.source_nodes(i)[0]).lower().startswith("call third("))
    originals = tuple(graph.source_nodes(i)[0] for i in children[children.index(first):children.index(last)+1])
    bindings = tuple(builder.entry.scope.bindings[name] for name in ("a", "b"))
    return path, analysis, originals, bindings, builder.facts


def position(parts):
    return prove_original_reached_position(parts[1], PROCEDURE, parts[2])


def preserved(parts, anchor=None):
    return prove_preserved_entry_definitions(parts[1], PROCEDURE, parts[2], parts[3], parts[4],
                                             position=position(parts) if anchor is None else anchor)


def test_structured_root_interval_preserves_only_independent_entry_facts(tmp_path):
    parts = prepared(tmp_path, middle="if(flag)then\ncall second(a,b,n)\nelse\na(:)=b(:)\nendif")
    anchor = position(parts)
    assert anchor.validate(parts[1], PROCEDURE, parts[2]) is anchor
    assert position(parts) is anchor
    record = anchor.public()
    assert record["version"] == 1
    assert len(record["selected_node_ids"]) == 3
    assert record["activation_node_id"] == record["selected_node_ids"][0]
    assert record["parent_sequence"] == parts[1].structure(PROCEDURE).root_id
    for flag in ("execution_authorized", "entry_definition_authority", "descriptor_lifetime_authority",
                 "payload_access_authorized", "exit_authority", "cross_exit_retention_authorized", "alias_authority"):
        assert record[flag] is False
    proof = preserved(parts, anchor)
    assert proof.public()["version"] == 3
    assert proof.public()["reached_position_identity"] == anchor.identity
    assert "sequence_exit_identity" not in proof.public()
    assert {binding.root for binding in proof.preserved} == {"argument::a", "argument::b"}
    assert proof.validate(parts[1], PROCEDURE, parts[2], parts[3], parts[4]) is proof


def test_single_root_unit_and_following_unmanaged_source_need_no_exit_or_count_authority(tmp_path):
    parts = prepared(tmp_path)
    anchor = prove_original_reached_position(parts[1], PROCEDURE, parts[2][:1])
    assert len(anchor.public()["selected_node_ids"]) == 1
    assert anchor.public()["exit_authority"] is False
    assert anchor.public()["cross_exit_retention_authorized"] is False


def lexical_interval(routine):
    nodes = tuple(routine.execution.content)
    first = next(i for i, node in enumerate(nodes) if str(node).lower().startswith("call first("))
    last = next(i for i, node in enumerate(nodes) if str(node).lower().startswith("call third("))
    return nodes[first:last+1]


def test_original_documentation_comments_do_not_become_semantic_root_units(tmp_path):
    source = SOURCE.replace("call unknown_prefix()", "a(:)=a(:)+1.0_8")
    source = source.replace("if(flag)then\ncall first", "call first").replace("endif\ncall unknown_suffix()", "call unknown_suffix()")
    source = source.replace("call second(a,b,n)", "! original documentation\n\ncall second(a,b,n)\n! another original comment")
    _path, builder = fixture(tmp_path, source, lifetime="invocation")
    originals = lexical_interval(builder.entry)
    assert any(type(node).__name__ == "Comment" for node in originals)
    proof = prove_original_reached_position(builder.analysis, PROCEDURE, originals)
    assert len(proof.public()["original_units"]) == 3
    assert proof.validate(builder.analysis, PROCEDURE, originals) is proof
    foreign = tuple(copy(node) if type(node).__name__ == "Comment" else node for node in originals)
    with pytest.raises(CompilationError):
        prove_original_reached_position(builder.analysis, PROCEDURE, foreign)


def test_complete_deferred_source_group_is_one_root_position_unit_without_team_execution_authority(tmp_path):
    source = SOURCE.replace("call unknown_prefix()", "a(:)=a(:)+1.0_8")
    source = source.replace("if(flag)then\ncall first", "call first").replace("endif\ncall unknown_suffix()", "call unknown_suffix()")
    source = source.replace("logical,intent(in)::flag", "logical,intent(in)::flag\ninteger::i")
    team = ("!$omp parallel private(i)\n"
            + "!$omp do\ndo i=1,n\na(i)=b(i)\nenddo\n!$omp end do nowait\n"*76
            + "!$omp end parallel")
    source = source.replace("call second(a,b,n)", team)
    path = tmp_path/"deferred.f90"
    path.write_text(source)
    analysis = SourceEffects([path], operations=12)
    analysis.seal_session()
    originals = lexical_interval(analysis.routines[PROCEDURE])
    proof = prove_original_reached_position(analysis, PROCEDURE, originals)
    record = proof.public()
    assert len(record["original_units"]) == 3
    middle = record["original_units"][1]
    unit = analysis.structure(PROCEDURE).source_nodes(middle["node_id"])
    assert len(unit) > 1
    assert middle["original_source_sha256"] == sha256("\n".join(str(node) for node in unit).encode()).hexdigest()
    assert proof.validate(analysis, PROCEDURE, originals) is proof
    assert record["execution_authorized"] is False
    partial = tuple(node for node in originals if not str(node).lower().startswith("!$omp end"))
    with pytest.raises(CompilationError):
        prove_original_reached_position(analysis, PROCEDURE, partial)


def test_exact_first_root_position_has_a_vacuous_prefix_without_inventing_definitions(tmp_path):
    parts = prepared(tmp_path, prefix="")
    proof = preserved(parts)
    assert proof is not None
    assert proof.public()["segments"] == []
    assert proof.public()["new_definition_authority"] is False
    parts[4]["captures"]["argument::a"]["initialized"] = "none"
    other = preserved(parts)
    assert [binding.root for binding in other.preserved] == ["argument::b"]


@pytest.mark.parametrize("selection", ["empty", "duplicate", "reversed", "gap", "foreign"])
def test_nonoriginal_or_noncontiguous_intervals_are_rejected(tmp_path, selection):
    parts = prepared(tmp_path)
    originals = parts[2]
    if selection == "empty":
        originals = ()
    elif selection == "duplicate":
        originals = (originals[0], originals[0])
    elif selection == "reversed":
        originals = originals[::-1]
    elif selection == "gap":
        originals = (originals[0], originals[2])
    else:
        other = tmp_path/"other"
        other.mkdir()
        originals = prepared(other)[2]
    with pytest.raises(CompilationError):
        prove_original_reached_position(parts[1], PROCEDURE, originals)


def test_branch_child_is_not_a_root_position(tmp_path):
    _path, builder = fixture(tmp_path, SOURCE.replace("call unknown_prefix()", "a(:)=0.0_8"))
    from compiler.scopes.call_groups import _candidates

    originals = next(_candidates(builder))
    with pytest.raises(CompilationError, match="root"):
        prove_original_reached_position(builder.analysis, PROCEDURE, originals)


@pytest.mark.parametrize("prefix", ["call unknown_prefix()", "deallocate(a)", "allocate(a(n))", "a=2.0_8"])
def test_new_position_does_not_bypass_unknown_effects_or_descriptor_replacement(tmp_path, prefix):
    assert preserved(prepared(tmp_path, prefix=prefix)) is None


def test_private_allocation_stays_a_conservative_preservation_frontier_without_separate_effect_proof(tmp_path):
    source = SOURCE.replace("real(8),allocatable,intent(inout)::a(:),b(:)",
                            "real(8),allocatable,intent(inout)::a(:),b(:)\nreal(8),allocatable,save::scratch(:)")
    source = source.replace("call unknown_prefix()", "if(.not.allocated(scratch))then\nallocate(scratch(n))\nscratch=0.0_8\nendif")
    source = source.replace("if(flag)then\ncall first", "call first").replace("endif\ncall unknown_suffix()", "call unknown_suffix()")
    parts = prepared(tmp_path, source=source)
    assert position(parts)
    assert preserved(parts) is None


@pytest.mark.parametrize("factory", ["projection", "native_prefix", "segment"])
def test_private_analyses_cannot_borrow_the_position_registry(tmp_path, factory):
    parts = prepared(tmp_path)
    anchor = position(parts)
    if factory == "projection":
        other = parts[1].private_projection()
    elif factory == "native_prefix":
        from compiler.frontend.native_prefix import _fork

        other = _fork(parts[1], ())
    else:
        from compiler.scopes.segments import _native_analysis

        other = _native_analysis(parts[1])
    assert not getattr(other, "_original_reached_positions", {})
    with pytest.raises(CompilationError):
        anchor.validate(other, PROCEDURE, parts[2])
    fresh = prove_original_reached_position(other, PROCEDURE, parts[2])
    assert fresh is not anchor


@pytest.mark.parametrize("mutation", ["copy", "public", "selection", "record_retag", "owner"])
def test_registered_position_cannot_be_rebound(tmp_path, mutation):
    parts = prepared(tmp_path)
    anchor = position(parts)
    if mutation == "copy":
        anchor = copy(anchor)
    elif mutation == "public":
        anchor = anchor.public()
    elif mutation == "selection":
        object.__setattr__(anchor, "_originals", parts[2][::-1])
    elif mutation == "record_retag":
        record = anchor.public()
        record.pop("identity")
        record["exit_authority"] = True
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        object.__setattr__(anchor, "_record", encoded)
        object.__setattr__(anchor, "identity", sha256(encoded.encode()).hexdigest())
        parts[1]._original_reached_positions[anchor.identity] = anchor
    else:
        parts[1]._original_reached_positions_owner = object()
    with pytest.raises(CompilationError):
        preserved(parts, anchor)


@pytest.mark.parametrize("mutation", ["registry", "attachment", "source"])
def test_last_publication_callback_cannot_detach_or_change_a_position(tmp_path, monkeypatch, mutation):
    parts = prepared(tmp_path)
    anchor = position(parts)
    original = parts[1].verify_publication

    def changed():
        original()
        if mutation == "registry":
            parts[1]._original_reached_positions[anchor.identity] = copy(anchor)
        elif mutation == "attachment":
            object.__setattr__(anchor, "_originals", tuple(reversed(parts[2])))
        else:
            parts[0].write_text(parts[0].read_text()+"! changed after publication callback\n")

    monkeypatch.setattr(parts[1], "verify_publication", changed)
    with pytest.raises(CompilationError):
        anchor.validate(parts[1], PROCEDURE, parts[2])


@pytest.mark.parametrize("mutation", ["selection", "registry", "registry_copy", "owner", "position_pointer"])
def test_entry_final_publication_cannot_invalidate_its_retained_position(tmp_path, monkeypatch, mutation):
    parts = prepared(tmp_path)
    anchor = position(parts)
    proof = preserved(parts, anchor)
    original = parts[1].verify_publication
    callbacks = []

    def changed():
        original()
        if sys._getframe(1).f_code is type(proof).validate.__code__:
            callbacks.append("entry-final-publication")
            if mutation == "selection":
                object.__setattr__(anchor, "_originals", tuple(reversed(parts[2])))
            elif mutation == "registry":
                parts[1]._original_reached_positions.pop(anchor.identity)
            elif mutation == "registry_copy":
                parts[1]._original_reached_positions = dict(parts[1]._original_reached_positions)
            elif mutation == "owner":
                parts[1]._original_reached_positions_owner = object()
            else:
                object.__setattr__(proof, "_position", copy(anchor))

    monkeypatch.setattr(parts[1], "verify_publication", changed)
    with pytest.raises(CompilationError, match="position changed|native proof attachment changed"):
        proof.validate(parts[1], PROCEDURE, parts[2], parts[3], parts[4])
    assert callbacks == ["entry-final-publication"]


@pytest.mark.parametrize("mutation", ["native_selection", "native_registry", "reachability_barriers",
                                      "reachability_registry", "effect_graph_selection", "effect_graph_registry"])
def test_entry_final_publication_cannot_change_consumed_native_receipts(tmp_path, monkeypatch, mutation):
    parts = prepared(tmp_path)
    proof = preserved(parts)
    native, = proof._preservations
    original = parts[1].verify_publication
    callbacks = []

    def changed():
        original()
        if sys._getframe(1).f_code is type(proof).validate.__code__:
            callbacks.append("entry-final-publication")
            if mutation == "native_selection":
                object.__setattr__(native, "_selection", ())
            elif mutation == "native_registry":
                parts[1]._native_preservation_proofs.pop(native.identity)
            elif mutation == "reachability_barriers":
                object.__setattr__(native.reachability, "barriers", (object(),))
            elif mutation == "reachability_registry":
                parts[1]._resource_reachability_proofs.pop(native.reachability.identity)
            elif mutation == "effect_graph_selection":
                object.__setattr__(native.reachability.native_graph, "_selection", ())
            else:
                parts[1]._native_effect_graphs.pop(native.reachability.native_graph.identity)

    monkeypatch.setattr(parts[1], "verify_publication", changed)
    with pytest.raises(CompilationError, match="retained native source receipt changed"):
        proof.validate(parts[1], PROCEDURE, parts[2], parts[3], parts[4])
    assert callbacks == ["entry-final-publication"]
