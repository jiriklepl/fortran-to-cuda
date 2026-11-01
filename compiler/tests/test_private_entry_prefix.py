"""A fresh private guard filters no incoming fact without ordinary effects."""
import json
import sys
from copy import copy
from hashlib import sha256

import pytest

from compiler.frontend.entry_preservation import prove_preserved_entry_definitions
from compiler.frontend.native_preservation import prove_native_preservation
from compiler.frontend.private_entry_prefix import prove_private_entry_prefix
from compiler.frontend.source_position import prove_original_reached_position
from compiler.ir import CompilationError
from compiler.tests.test_allocation_epoch_eligibility import SOURCE, clock_source, fixture, issue

PROCEDURE = "gate_fixture::advance"


def prepared(tmp_path, *, clock=False, extra="", sealed=True, incoming_allocatable=False, optional=False):
    source = clock_source() if clock else SOURCE
    source = source.replace("advance(n,mode)", "advance(n,mode,incoming)")
    source = source.replace("integer,intent(in)::n,mode", "integer,intent(in)::n,mode\nreal(8),intent(inout)::incoming(:)")
    if incoming_allocatable:
        source = source.replace("real(8),intent(inout)::incoming(:)", "real(8),allocatable,intent(inout)::incoming(:)")
    if optional:
        source = source.replace("advance(n,mode,incoming)", "advance(n,mode,incoming,toggle)")
        source = source.replace("integer,intent(in)::n,mode",
                                "integer,intent(in)::n,mode\nlogical,optional,intent(in)::toggle\nlogical::update")
        extra = "if(present(toggle))then\nupdate=.not.toggle\nelse\nupdate=.true.\nendif\n"+extra
    source = source.replace("called=called+1", "called=called+1\n"+extra)
    parts = fixture(tmp_path, source, sealed=sealed)
    analysis, routine = parts[1:3]
    graph = analysis.structure(PROCEDURE)
    children = graph.nodes[graph.root_id].children
    last = next(i for i, identity in enumerate(children)
                if str(graph.source_nodes(identity)[0]).lower().startswith("if (eligible)"))
    prefix = tuple(graph.source_nodes(identity)[0] for identity in children[:last])
    return parts, (prefix,), routine.scope.bindings["incoming"]


def project(data):
    parts, segments, _incoming = data
    return prove_private_entry_prefix(parts[1], PROCEDURE, segments, issue(parts))


def entry(data, *, initialized="whole", projection=True, bindings=None):
    parts, segments, incoming = data
    analysis = parts[1]
    graph, selected = analysis._selected_source(PROCEDURE, segments[0])
    following = graph.nodes[graph.root_id].children[len(selected)]
    originals = graph.source_nodes(following)
    position = prove_original_reached_position(analysis, PROCEDURE, originals)
    bindings = (incoming,) if bindings is None else bindings
    facts = {"captures": {binding.root: {"initialized": initialized, "storage": "stable", "lifetime": "invocation",
                                         "allocation_changes": False, "escapes": False} for binding in bindings}}
    proof = prove_preserved_entry_definitions(analysis, PROCEDURE, originals, bindings, facts, position=position,
                                              prefix_projection=project(data) if projection else None)
    return proof, originals, bindings, facts


@pytest.mark.parametrize("sealed", [False, True])
def test_complete_fresh_guard_projects_private_effects_and_retains_original_native_predicates(tmp_path, sealed):
    data = prepared(tmp_path, sealed=sealed)
    parts, segments, incoming = data
    source = parts[0].read_bytes()
    proof = project(data)
    assert proof.validate(parts[1], PROCEDURE, segments) is proof
    assert project(data) is proof
    record = proof.public()
    assert record["version"] == 3
    assert [step["kind"] for step in record["steps"]] == [
        "ordinary_native", "fresh_private_allocation", "fresh_private_allocation",
        "fresh_private_literal_overwrite", "ordinary_native", "original_end_if", "ordinary_native"]
    for flag in ("new_definition_authority", "execution_authorized", "payload_access_authorized",
                 "descriptor_lifetime_authority", "alias_authority", "cross_exit_retention_authorized"):
        assert record[flag] is False
    ordinary = tuple(prove_native_preservation(parts[1], PROCEDURE, segment, (incoming,))
                     for segment in proof.native_segments)
    assert all(result.complete and not result.invalidations for result in ordinary)
    assert parts[0].read_bytes() == source
    assert parts[1]._allocation_authorizations == {}


@pytest.mark.parametrize(("clock", "optional"), [(False, False), (True, False), (True, True)])
def test_projection_steps_identify_each_exact_original_graph_node(tmp_path, clock, optional):
    data = prepared(tmp_path, clock=clock, optional=optional)
    proof = project(data)
    analysis = data[0][1]
    graph = analysis.structure(PROCEDURE)
    semantic_ids = []
    for public, (kind, node, proofs) in zip(proof.public()["steps"], proof._steps, strict=True):
        assert public["original_source_sha256"] == sha256(str(node).encode()).hexdigest()
        assert public["proof_identities"] == [item.identity for item in proofs]
        if kind in {"original_comment", "original_end_if"}:
            assert public["source_node_id"] is None
        else:
            current, identities = analysis._selected_source(PROCEDURE, (node,))
            assert current is graph
            assert len(identities) == 1
            assert public["source_node_id"] == identities[0]
            semantic_ids.extend(identities)
    assert len(set(semantic_ids)) == len(semantic_ids)


def test_projection_rejects_changed_step_graph_identity_even_with_rehashed_record(tmp_path):
    data = prepared(tmp_path)
    proof = project(data)
    record = proof.public()
    record.pop("identity")
    record["steps"][0]["source_node_id"] = record["steps"][1]["source_node_id"]
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
    object.__setattr__(proof, "_record", encoded)
    object.__setattr__(proof, "identity", sha256(encoded.encode()).hexdigest())
    data[0][1]._private_entry_prefixes[proof.identity] = proof
    with pytest.raises(CompilationError, match="changed its original projection"):
        proof.validate(data[0][1], PROCEDURE, data[1])


def test_original_documentation_comment_has_no_semantic_graph_identity(tmp_path):
    data = prepared(tmp_path, extra="! document the unchanged original prefix")
    parts, segments, incoming = data
    comment = next(node for node in parts[2].execution.content if type(node).__name__ == "Comment")
    data = parts, (segments[0]+(comment,),), incoming
    proof = project(data)
    comments = [step for step in proof.public()["steps"] if step["kind"] == "original_comment"]
    assert len(comments) == 1
    assert comments[0]["source_node_id"] is None
    assert proof.validate(data[0][1], PROCEDURE, data[1]) is proof


def test_scalar_clocks_keep_exact_intrinsic_and_descriptor_receipts_without_array_permissions(tmp_path):
    data = prepared(tmp_path, clock=True, extra="if(mode>0)call system_clock(count=rate)")
    proof = project(data)
    clocks = [step for step in proof.public()["steps"] if step["kind"] == "original_scalar_clock"]
    assert len(clocks) == 2
    assert all(len(step["proof_identities"]) == 2 for step in clocks)
    assert all(prove_native_preservation(data[0][1], PROCEDURE, segment, (data[2],)).complete
               for segment in proof.native_segments)


def test_entry4_preserves_independent_incoming_state_without_promoting_private_storage(tmp_path):
    data = prepared(tmp_path, clock=True, extra="if(mode>0)call system_clock(count=rate)")
    assert entry(data, projection=False)[0] is None
    bindings = (data[2], *data[0][4])
    proof, originals, bindings, facts = entry(data, bindings=bindings)
    assert proof.preserved == (data[2],)
    assert proof.public()["version"] == 4
    assert proof.public()["prefix_projection"]["identity"] == project(data).identity
    assert proof.public()["original_prefix_segments"]
    assert proof.public()["new_definition_authority"] is False
    assert proof.validate(data[0][1], PROCEDURE, originals, bindings, facts) is proof
    assert entry(data, initialized="none")[0] is None


def test_entry4_composes_both_optional_scalar_arms_without_reading_optional_or_array_payloads(tmp_path):
    data = prepared(tmp_path, clock=True, optional=True)
    original = data[0][0].read_bytes()
    assert entry(data, projection=False)[0] is None
    proof, originals, bindings, facts = entry(data)
    assert proof.preserved == (data[2],)
    record = proof.public()
    assert record["version"] == 4
    assert record["prefix_projection"]["version"] == 3
    assert sum(step["kind"] == "original_optional_scalar_control" for step in record["prefix_projection"]["steps"]) == 1
    assert proof.validate(data[0][1], PROCEDURE, originals, bindings, facts) is proof
    assert entry(data, initialized="none")[0] is None
    assert data[0][0].read_bytes() == original


def test_entry_final_publication_fences_the_typed_optional_scalar_control(tmp_path, monkeypatch):
    data = prepared(tmp_path, optional=True)
    proof, originals, bindings, facts = entry(data)
    analysis = data[0][1]
    child = next(p for p, name in proof._projection._retained if name == "_optional_scalar_controls")
    original = analysis.verify_publication

    def changed():
        original()
        if sys._getframe(1).f_code is type(proof).validate.__code__:
            object.__setattr__(child, "_record", "{}")

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError, match="retained prefix projection changed"):
        proof.validate(analysis, PROCEDURE, originals, bindings, facts)


@pytest.mark.parametrize("extra", ["call unknown(incoming)", "a=1.0_8"])
def test_entry_projection_cannot_bypass_another_unknown_or_private_mutation(tmp_path, extra):
    assert entry(prepared(tmp_path, extra=extra))[0] is None


@pytest.mark.parametrize("mutation", ["projection_pointer", "projection_fields", "projection_registry",
                                      "projection_owner", "gate_fields", "local_fields", "clock_fields", "source_binding"])
def test_entry_final_publication_fences_the_consumed_projection_and_its_dependencies(tmp_path, monkeypatch, mutation):
    data = prepared(tmp_path, clock=True)
    proof, originals, bindings, facts = entry(data)
    analysis = data[0][1]
    projection = proof._projection
    original = analysis.verify_publication
    callbacks = []

    def changed():
        original()
        if sys._getframe(1).f_code is type(proof).validate.__code__:
            callbacks.append("entry-final")
            if mutation == "projection_pointer":
                object.__setattr__(proof, "_projection", copy(projection))
            elif mutation == "projection_fields":
                object.__setattr__(projection, "native_segments", ())
            elif mutation == "projection_registry":
                analysis._private_entry_prefixes.pop(projection.identity)
            elif mutation == "projection_owner":
                analysis._private_entry_prefixes_owner = object()
            elif mutation == "gate_fields":
                object.__setattr__(projection._gate, "_resources", ())
            elif mutation == "source_binding":
                object.__setattr__(projection.private_bindings[0], "attributes", frozenset({"save", "allocatable", "target"}))
            else:
                registry_name = "_local_allocation_proofs" if mutation == "local_fields" else "_scalar_clock_proofs"
                child = next(p for p, name in projection._retained if name == registry_name)
                object.__setattr__(child, "_record", "{}")

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError):
        proof.validate(analysis, PROCEDURE, originals, bindings, facts)
    assert callbacks == ["entry-final"]


@pytest.mark.parametrize("extra", ["call unknown(incoming)", "deallocate(incoming)", "call kill(incoming)"])
def test_unrelated_effects_remain_in_the_ordinary_native_filter(tmp_path, extra):
    data = prepared(tmp_path, extra=extra, incoming_allocatable=extra.startswith("deallocate"))
    projection = project(data)
    native = tuple(prove_native_preservation(data[0][1], PROCEDURE, segment, (data[2],))
                   for segment in projection.native_segments)
    assert any(not result.complete or result.invalidations for result in native)
    assert any(extra.split("(")[0].upper() in str(node).upper()
               for segment in projection.native_segments for node in segment)


def test_ordinary_payload_write_preserves_only_existing_definedness(tmp_path):
    data = prepared(tmp_path, extra="incoming=1.0_8")
    projection = project(data)
    native = tuple(prove_native_preservation(data[0][1], PROCEDURE, segment, (data[2],))
                   for segment in projection.native_segments)
    assert all(result.complete and not result.invalidations for result in native)
    assert projection.public()["new_definition_authority"] is False
    assert any(str(node).upper() == "INCOMING = 1.0_8"
               for segment in projection.native_segments for node in segment)


def test_private_write_after_guard_is_not_exempted_by_local_declaration_or_matching_name(tmp_path):
    data = prepared(tmp_path, extra="a=1.0_8")
    projection = project(data)
    assert any(str(node).upper() == "A = 1.0_8" for segment in projection.native_segments for node in segment)
    assert any(not prove_native_preservation(data[0][1], PROCEDURE, segment, (data[2],)).complete
               for segment in projection.native_segments)


@pytest.mark.parametrize("mutation", ["omit", "reorder", "duplicate", "foreign_gate", "copy_gate"])
def test_incomplete_original_prefix_or_foreign_gate_cannot_issue_projection(tmp_path, mutation):
    data = prepared(tmp_path)
    parts, segments, _incoming = data
    gate = issue(parts)
    if mutation == "omit":
        segments = (segments[0][1:],)
    elif mutation == "reorder":
        segments = (tuple(reversed(segments[0])),)
    elif mutation == "duplicate":
        segments = (segments[0]+segments[0][:1],)
    elif mutation == "copy_gate":
        gate = copy(gate)
    else:
        other = tmp_path/"other"
        other.mkdir()
        gate = issue(prepared(other)[0])
    with pytest.raises(CompilationError):
        prove_private_entry_prefix(parts[1], PROCEDURE, segments, gate)


@pytest.mark.parametrize("mutation", ["copy", "record", "selection", "gate", "registry", "owner"])
def test_projection_cannot_be_retagged_or_detached_from_original_source(tmp_path, mutation):
    data = prepared(tmp_path)
    proof = project(data)
    if mutation == "copy":
        proof = copy(proof)
    elif mutation == "record":
        record = proof.public()
        record.pop("identity")
        record["new_definition_authority"] = True
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"))
        object.__setattr__(proof, "_record", encoded)
        object.__setattr__(proof, "identity", sha256(encoded.encode()).hexdigest())
        data[0][1]._private_entry_prefixes[proof.identity] = proof
    elif mutation == "selection":
        object.__setattr__(proof, "native_segments", ())
    elif mutation == "gate":
        object.__setattr__(proof, "_gate", copy(proof._gate))
    elif mutation == "registry":
        data[0][1]._private_entry_prefixes.pop(proof.identity)
    else:
        data[0][1]._private_entry_prefixes_owner = object()
    with pytest.raises(CompilationError):
        proof.validate(data[0][1], PROCEDURE, data[1])


@pytest.mark.parametrize("mutation", ["gate", "local", "registry", "source", "binding"])
def test_final_publication_cannot_change_retained_sources_or_typed_dependencies(tmp_path, monkeypatch, mutation):
    data = prepared(tmp_path)
    proof = project(data)
    analysis = data[0][1]
    original = analysis.verify_publication
    callbacks = []

    def changed():
        original()
        if sys._getframe(1).f_code is type(proof).validate.__code__:
            callbacks.append("projection-final")
            if mutation == "gate":
                object.__setattr__(proof._gate, "_resources", tuple(reversed(proof.private_bindings)))
            elif mutation == "local":
                local = next(p for p, name in proof._retained if name == "_local_allocation_proofs")
                object.__setattr__(local, "binding", data[2])
            elif mutation == "registry":
                analysis._allocation_gates.pop(proof._gate.identity)
            elif mutation == "binding":
                object.__setattr__(proof.private_bindings[0], "attributes", frozenset({"save", "allocatable", "target"}))
            else:
                data[0][0].write_text(data[0][0].read_text()+"! late original source mutation\n")

    monkeypatch.setattr(analysis, "verify_publication", changed)
    with pytest.raises(CompilationError):
        proof.validate(analysis, PROCEDURE, data[1])
    assert callbacks == ["projection-final"]
