"""Returned selections borrow exact storage membership only during emission."""
import json
from copy import copy
from dataclasses import replace

import pytest

from compiler.ir import CompilationError
from compiler.scopes.conditional_return import ConditionalUnitDescriptorProof
from compiler.scopes.storage_admission import StorageAdmission, _supported, active_admission, admit_emitted_storage
from compiler.scopes.storage_epoch import StorageEpochProof
from compiler.tests.test_inherited_lexical_namespace import emit_package, scalar_element_source
from compiler.tests.test_storage_emission_admission import query


@pytest.fixture
def final_plan(tmp_path):
    path, builder, _outputs, _report, _manifest = emit_package(tmp_path, scalar_element_source())
    owner = builder._conditional_return_draft.owner
    plan = owner.execution_plan
    assert getattr(owner, "position_epoch", None) is None
    assert _supported(plan)
    assert sum(len(member.units) for member in plan.members) >= 4
    return path, owner, plan


def test_many_hierarchical_units_keep_two_full_records_and_public_calls_stay_full(final_plan, monkeypatch):
    _path, _owner, plan = final_plan
    proof, calls = plan.storage_epoch(), []
    original = StorageEpochProof.validate
    expected = [(member, tuple(unit.arrays.values()), proof.admits_many(plan, member, unit.arrays.values()))
                for member in plan.members for unit in plan.units(member)]

    def full(self, current):
        calls.append(current)
        return original(self, current)

    monkeypatch.setattr(StorageEpochProof, "validate", full)
    before = proof._record
    with StorageAdmission(plan):
        for _ in range(12):
            for member, bindings, roots in expected:
                assert admit_emitted_storage(proof, plan, member, bindings) == roots
        assert calls == [plan]
        proof.validate(plan)
        assert calls == [plan, plan]
    assert calls == [plan] * 3
    assert proof._record == before
    assert active_admission() is None


@pytest.mark.parametrize("fault", [
    "draft_owner", "draft_issuer", "draft_selected", "draft_receipt", "draft_consumers",
    "draft_consumer_token", "native_registry", "descriptor_consumer", "descriptor_consumers_map",
    "descriptor_record", "selected_proof", "selected_cut", "selected_originals", "selected_owner",
    "closed_selection", "closed_prefix", "closed_controls", "closed_initializations", "selection_context",
    "selection_bundle", "prefix_proof", "returned_proof", "returned_record", "return_calls",
    "companion_calls", "lexical_registry", "lexical_record", "internal_record", "callsite_record",
    "counted_controls", "graph_native_nodes", "graph_native_scope", "graph_native_payload",
    "graph_node", "graph_details", "epoch", "numerical_proof", "handoff", "handoff_registry",
    "fact", "config", "session", "session_field", "source",
])
def test_returned_proof_and_control_mutations_reject_before_membership(final_plan, fault):
    path, owner, plan = final_plan
    member, bindings = query(plan)
    draft = owner.builder._conditional_return_draft
    middle = owner.members["hierarchical::middle"]
    lexical, = middle.source_lexical_calls
    source_member = next(m for m in plan.members if getattr(m, "source_companion_calls", ()))
    called = next(m for m in plan.members if getattr(m, "call_sites", ()))

    def inspect():
        with StorageAdmission(plan) as admission:
            descriptor = next(p for p, _state, _registry in admission._retained
                              if type(p) is ConditionalUnitDescriptorProof)
            if fault == "draft_owner":
                draft.owner = copy(owner)
            elif fault == "draft_issuer":
                draft._issuer = object()
            elif fault == "draft_selected":
                draft.selected_root = copy(draft.selected_root)
            elif fault == "draft_receipt":
                draft._selected_root_receipt = copy(draft._selected_root_receipt)
            elif fault == "draft_consumers":
                draft._consumers = list(draft._consumers)
            elif fault == "draft_consumer_token":
                draft._consumers[0] = draft._consumers[0][0], object()
            elif fault == "native_registry":
                assert draft._native_preservations
                draft._native_preservations.clear()
            elif fault == "descriptor_consumer":
                descriptor._analysis._conditional_descriptor_consumers[descriptor.identity] = object()
            elif fault == "descriptor_consumers_map":
                descriptor._analysis._conditional_descriptor_consumers = dict(
                    descriptor._analysis._conditional_descriptor_consumers)
            elif fault == "descriptor_record":
                object.__setattr__(descriptor, "_record", descriptor._record + " ")
            elif fault == "selected_proof":
                object.__setattr__(owner, "hierarchical_selection", copy(owner.hierarchical_selection))
            elif fault in {"selected_cut", "selected_originals", "selected_owner"}:
                selection = owner.hierarchical_selection
                name, value = {"selected_cut": ("cut_before", selection.selected[0]),
                               "selected_originals": ("_originals", ()),
                               "selected_owner": ("_owner", copy(owner))}[fault]
                object.__setattr__(selection, name, value)
            elif fault in {"closed_selection", "closed_prefix", "closed_controls", "closed_initializations"}:
                object.__setattr__(owner, fault, tuple([*getattr(owner, fault), object()]))
            elif fault == "selection_context":
                dict.__setitem__(owner.selection_context, "last_line", -1)
            elif fault == "selection_bundle":
                object.__setattr__(owner, "selection_bundle", object())
            elif fault == "prefix_proof":
                owner.emission.selection_prefix = object()
            elif fault == "returned_proof":
                object.__setattr__(owner, "return_contract", copy(owner.return_contract))
            elif fault == "returned_record":
                object.__setattr__(owner.return_contract, "_record", owner.return_contract._record + " ")
            elif fault == "return_calls":
                object.__setattr__(owner, "return_calls", list(owner.return_calls))
            elif fault == "companion_calls":
                call = source_member.source_companion_calls[0]
                object.__setattr__(call, "callee", owner)
            elif fault == "lexical_registry":
                dict.clear(middle._issued_lexical_calls)
            elif fault == "lexical_record":
                dict.__setitem__(lexical.record, "target", "foreign")
            elif fault == "internal_record":
                dict.__setitem__(middle.internal_calls[0], "execution", "foreign")
            elif fault == "callsite_record":
                dict.__setitem__(called.call_sites[0], "first_line", -1)
            elif fault == "counted_controls":
                list.append(middle.counted_controls, {"header_identity": "foreign"})
            elif fault.startswith("graph_native_"):
                nodes, scope = middle.graph_native_tasks[0]
                if fault == "graph_native_nodes":
                    list.__setitem__(middle.graph_native_tasks, 0, ((), scope))
                elif fault == "graph_native_scope":
                    object.__setattr__(scope, "tree", [])
                else:
                    dict.__setitem__(scope.native[0].summary, "complete", False)
            elif fault in {"graph_node", "graph_details"}:
                node = next(iter(owner.graph.nodes.values()))
                object.__setattr__(node, "kind" if fault == "graph_node" else "details",
                                   "foreign" if fault == "graph_node" else {"foreign": True})
            elif fault == "epoch":
                object.__setattr__(owner, "position_epoch", object())
            elif fault == "numerical_proof":
                object.__setattr__(plan._numerical_authorities[0], "identity", "foreign")
            elif fault == "handoff":
                object.__setattr__(plan._proof_handoffs[0], "key", "foreign")
            elif fault == "handoff_registry":
                handoff = plan._proof_handoffs[0]
                getattr(handoff.analysis, handoff.registry).pop(handoff.key)
            elif fault == "fact":
                owner.builder.facts["captures"]["hierarchical::right"]["initialized"] = "none"
            elif fault == "config":
                owner.builder.config = replace(owner.builder.config)
            elif fault == "session":
                object.__setattr__(owner.builder.analysis, "_source_session", copy(owner.builder.analysis._source_session))
            elif fault == "session_field":
                session = owner.builder.analysis._source_session
                object.__setattr__(session, "objects", dict(session.objects))
            else:
                path.write_text(path.read_text() + "! source changed during emission\n")
            admission.admits_many(plan.storage_epoch(), plan, member, bindings)
            pytest.fail("a changed proof must not return detached canonical membership")
    with pytest.raises(CompilationError):
        inspect()
    assert active_admission() is None


@pytest.mark.parametrize("family", ["selected_position_epoch", "supported_path_epoch",
                                   "private_allocation_obligations", "original_worksharing_storage_obligations"])
def test_unqualified_lifetime_families_keep_the_full_path(final_plan, family):
    _path, _owner, plan = final_plan
    proof = plan.storage_epoch()
    record = json.loads(proof._record)
    record[family] = {}
    object.__setattr__(proof, "_record", json.dumps(record))
    assert not _supported(plan)


@pytest.mark.parametrize("fault", ["descriptor_consumer", "draft_consumer", "native_registry", "selection_context"])
def test_final_live_source_callback_cannot_change_returned_ownership(final_plan, monkeypatch, fault):
    _path, owner, plan = final_plan
    member, bindings = query(plan)
    inputs, calls = owner.builder.analysis.inputs, []
    original = type(inputs).verify_live

    def inspect():
        with StorageAdmission(plan) as admission:
            descriptor = next(p for p, _state, _registry in admission._retained
                              if type(p) is ConditionalUnitDescriptorProof)

            def changed(self):
                result = original(self)
                if self is inputs:
                    calls.append(self)
                    if len(calls) == 3:
                        draft = owner.builder._conditional_return_draft
                        if fault == "descriptor_consumer":
                            descriptor._analysis._conditional_descriptor_consumers.pop(descriptor.identity)
                        elif fault == "draft_consumer":
                            draft._consumers[0] = draft._consumers[0][0], object()
                        elif fault == "native_registry":
                            draft._native_preservations.clear()
                        else:
                            dict.__setitem__(owner.selection_context, "last_line", -1)
                return result

            monkeypatch.setattr(type(inputs), "verify_live", changed)
            admission.admits_many(plan.storage_epoch(), plan, member, bindings)
            pytest.fail("the final callback must be fenced before membership returns")
    with pytest.raises(CompilationError):
        inspect()
    assert len(calls) == 3
    assert active_admission() is None
