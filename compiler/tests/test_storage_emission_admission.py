"""Scoped admission never borrows a live mutable capture or canonical name."""
import gc
import sys
import weakref
from copy import copy
from dataclasses import dataclass, replace

import pytest

from compiler.ir import CompilationError
from compiler.scopes.storage_admission import (
    StorageAdmission,
    _attachments,
    _Identity,
    _mapping,
    active_admission,
    admit_emitted_storage,
    validate_emitted_storage,
)
from compiler.scopes.storage_epoch import StorageEpochProof
from compiler.tests.test_graph_epoch_capture_batch import issued


@pytest.fixture
def final_plan(tmp_path):
    path, owner, _epoch, _bindings = issued(tmp_path)
    assert owner.run(owner.closed_selection)
    owner.builder._graph_storage_epoch = owner.position_epoch
    return path, owner, owner.execution_plan


def query(plan):
    member = next(member for member in plan.members if plan.units(member))
    bindings = tuple(plan.units(member)[0].arrays.values())
    return member, bindings


def test_many_membership_queries_keep_only_two_full_record_checks(final_plan, monkeypatch):
    _path, _owner, plan = final_plan
    member, bindings = query(plan)
    proof = plan.storage_epoch()
    expected = proof.admits_many(plan, member, bindings)
    original, calls = StorageEpochProof.validate, []

    def full(self, current):
        calls.append(current)
        return original(self, current)

    monkeypatch.setattr(StorageEpochProof, "validate", full)
    admission = StorageAdmission(plan)
    with admission:
        for _index in range(270):
            assert admission.validate(proof, plan) is proof
            assert admission.admits_many(proof, plan, member, bindings) == expected
        assert calls == [plan]
    assert calls == [plan, plan]
    assert active_admission() is None
    proof.validate(plan)
    assert calls == [plan, plan, plan]  # Standalone remains fresh.


@pytest.mark.parametrize("fault", ["fact", "fact_copy", "fact_map", "canonical", "binding", "unit",
                                   "descriptor", "registry", "namespace", "session", "config"])
def test_transient_changes_are_rejected_before_another_query_can_restore_them(final_plan, fault):
    _path, owner, plan = final_plan
    member, bindings = query(plan)
    proof = plan.storage_epoch()
    builder = owner.builder
    reached = []
    def inspect():
        with StorageAdmission(plan) as admission:
            if fault == "fact":
                builder.facts["captures"]["argument::a"]["initialized"] = "none"
            elif fault == "fact_copy":
                builder.facts["captures"]["argument::a"] = dict(builder.facts["captures"]["argument::a"])
            elif fault == "fact_map":
                builder.facts["captures"] = dict(builder.facts["captures"])
            elif fault == "canonical":
                object.__setattr__(member, "canonical_bindings", dict(member.canonical_bindings))
            elif fault == "binding":
                object.__setattr__(bindings[0], "root", "argument::other")
            elif fault == "unit":
                unit = plan.units(member)[0]
                object.__setattr__(unit, "arrays", dict(unit.arrays))
            elif fault == "descriptor":
                object.__setattr__(owner.position_epoch, "_descriptor", copy(owner.position_epoch._descriptor))
            elif fault == "registry":
                builder.analysis._descriptor_window_proofs.pop(owner.position_epoch._descriptor.identity)
            elif fault == "namespace":
                original = builder.analysis.modules
                replacement = dict(original)
                with pytest.raises(CompilationError, match="original source registry is immutable"):
                    builder.analysis.modules = replacement
                assert builder.analysis.modules is original
                admission.admits_many(proof, plan, member, bindings)
                # Inject internal corruption only to reach the consumer fence.
                object.__setattr__(builder.analysis, "modules", replacement)
            elif fault == "session":
                original = builder.analysis._source_session
                replacement = copy(original)
                with pytest.raises(CompilationError, match="source context ownership is immutable"):
                    builder.analysis._source_session = replacement
                assert builder.analysis._source_session is original
                admission.admits_many(proof, plan, member, bindings)
                # Inject internal corruption only to reach the consumer fence.
                object.__setattr__(builder.analysis, "_source_session", replacement)
            else:
                builder.config = replace(builder.config)
            reached.append("query")
            admission.admits_many(proof, plan, member, bindings)
            pytest.fail("a later callback must never get an opportunity to restore this substitution")
    with pytest.raises(CompilationError):
        inspect()
    assert reached == ["query"]
    assert active_admission() is None


def test_binding_iterator_cannot_seed_a_result_from_transient_canonical_state(final_plan):
    _path, owner, plan = final_plan
    member, bindings = query(plan)

    def changed():
        object.__setattr__(member, "canonical_bindings", dict(member.canonical_bindings))
        yield from bindings

    with pytest.raises(CompilationError), StorageAdmission(plan) as admission:
        admission.admits_many(plan.storage_epoch(), plan, member, changed())
    assert owner.execution_plan is plan
    assert active_admission() is None


@pytest.mark.parametrize("fault", ["plan", "proof", "member", "binding"])
def test_scope_cannot_be_consumed_by_a_copied_or_foreign_authority(final_plan, fault):
    _path, _owner, plan = final_plan
    member, bindings = query(plan)
    proof = plan.storage_epoch()
    with StorageAdmission(plan) as admission, pytest.raises(CompilationError):
        admission.admits_many(copy(proof) if fault == "proof" else proof,
                copy(plan) if fault == "plan" else plan,
                copy(member) if fault == "member" else member,
                (copy(bindings[0]),) if fault == "binding" else bindings)


def test_scope_rejects_reentry_reuse_and_detached_consumption(final_plan):
    _path, _owner, plan = final_plan
    member, bindings = query(plan)
    scope = StorageAdmission(plan)
    with scope, pytest.raises(CompilationError, match="nested"), StorageAdmission(plan):
        pass
    with pytest.raises(CompilationError):
        scope.admits_many(plan.storage_epoch(), plan, member, bindings)
    with pytest.raises(CompilationError, match="reused"), scope:
        pass
    assert active_admission() is None


def test_exception_unwinds_the_private_scope_without_replacing_the_original_error(final_plan):
    _path, _owner, plan = final_plan
    with pytest.raises(RuntimeError, match="original emission stopped"), StorageAdmission(plan):
        raise RuntimeError("original emission stopped")
    assert active_admission() is None
    with StorageAdmission(plan):
        pass


def test_final_live_source_publication_remains_fresh(final_plan):
    path, _owner, plan = final_plan
    with pytest.raises(CompilationError), StorageAdmission(plan):
        path.write_text(path.read_text()+"! changed during source emission\n")
    assert active_admission() is None


@pytest.mark.parametrize("fault", ["fact", "canonical", "proof", "source"])
def test_final_publication_callback_cannot_change_a_previously_checked_attachment(final_plan, monkeypatch, fault):
    path, owner, plan = final_plan
    member, _bindings = query(plan)
    analysis = owner.builder.analysis
    original = analysis.verify_publication

    def publication():
        original()
        frame = sys._getframe(1)
        if frame.f_code.co_name != "__exit__" or not frame.f_code.co_filename.endswith("storage_admission.py"):
            return
        if fault == "fact":
            owner.builder.facts["captures"]["argument::a"]["initialized"] = "none"
        elif fault == "canonical":
            object.__setattr__(member, "canonical_bindings", dict(member.canonical_bindings))
        elif fault == "proof":
            object.__setattr__(owner.position_epoch, "_descriptor", copy(owner.position_epoch._descriptor))
        else:
            path.write_text(path.read_text()+"! changed by the final publication callback\n")

    monkeypatch.setattr(analysis, "verify_publication", publication)
    with pytest.raises(CompilationError), StorageAdmission(plan):
        pass
    assert active_admission() is None


@pytest.mark.parametrize("fault", ["fact", "proof", "canonical", "registry"])
def test_initial_full_validator_cannot_seed_a_changed_receipt_before_scope_activation(final_plan, monkeypatch, fault):
    _path, owner, plan = final_plan
    original = StorageEpochProof.validate

    def changed(self, current):
        result = original(self, current)
        if fault == "fact":
            owner.builder.facts["captures"]["argument::a"]["initialized"] = "none"
        elif fault == "proof":
            object.__setattr__(owner.position_epoch, "_descriptor", copy(owner.position_epoch._descriptor))
        elif fault == "canonical":
            member, _bindings = query(plan)
            object.__setattr__(member, "canonical_bindings", dict(member.canonical_bindings))
        else:
            owner.builder.analysis._descriptor_window_proofs.pop(owner.position_epoch._descriptor.identity)
        return result

    monkeypatch.setattr(StorageEpochProof, "validate", changed)
    with pytest.raises(CompilationError), StorageAdmission(plan):
        pytest.fail("the substituted initial authority must not activate an admission scope")
    assert active_admission() is None


@pytest.mark.parametrize("fault", ["record", "map", "bypass_map", "retained"])
def test_scope_derived_data_cannot_be_replaced_or_mutated_into_a_new_authority(final_plan, fault):
    _path, _owner, plan = final_plan
    member, bindings = query(plan)
    reached = []

    def inspect():
        with StorageAdmission(plan) as admission:
            if fault == "record":
                admission._record = "{}"
            elif fault == "map":
                admission._admitted = dict(admission._admitted)
            elif fault == "bypass_map":
                key = id(member), id(bindings[0])
                dict.__setitem__(admission._admitted, key, (member, bindings[0], "argument::forged"))
            else:
                admission._retained = ()
            reached.append("query")
            admission.admits_many(plan.storage_epoch(), plan, member, bindings)
    with pytest.raises(CompilationError):
        inspect()
    assert reached == ["query"]
    assert active_admission() is None


@pytest.mark.parametrize("kind", ["field", "mapping"])
def test_identity_receipts_retain_the_original_object_until_the_scope_releases_it(kind):
    class Payload:
        pass

    @dataclass
    class Holder:
        value: object

    value = Payload()
    original = weakref.ref(value)
    holder = Holder(value) if kind == "field" else {"resource": value}
    receipt = _attachments(holder) if kind == "field" else _mapping(holder)
    del value
    for _index in range(100):
        if kind == "field":
            holder.value = Payload()
        else:
            holder["resource"] = Payload()
    gc.collect()
    assert original() is not None  # An integer-only receipt loses this owner.
    assert receipt != (_attachments(holder) if kind == "field" else _mapping(holder))
    del receipt
    gc.collect()
    assert original() is None


def test_identity_receipt_payload_cannot_be_replaced_even_through_object_setattr():
    original, replacement = [], []
    receipt = _Identity(original)
    assert receipt == _Identity(original)
    assert receipt != _Identity(replacement)
    with pytest.raises(AttributeError):
        object.__setattr__(receipt, "value", replacement)
    assert receipt.value is original
    assert not hasattr(receipt, "__dict__")


def test_repeated_live_slot_replacement_cannot_recycle_an_admitted_resource_identity(final_plan):
    _path, owner, plan = final_plan
    member, bindings = query(plan)
    root = next(iter(owner.resources))
    binding, slot = owner.resources[root]
    original_identity = id(owner.resources[root])
    with StorageAdmission(plan) as admission:
        for index in range(100):
            dict.__setitem__(owner.resources, root, (binding, slot+100+index))
            assert id(owner.resources[root]) != original_identity
        with pytest.raises(CompilationError):
            admission.admits_many(plan.storage_epoch(), plan, member, bindings)
        # Equivalent replacement does not restore the original object either.
        for _index in range(100):
            dict.__setitem__(owner.resources, root, (binding, slot))
            assert id(owner.resources[root]) != original_identity
        with pytest.raises(CompilationError):
            admission.admits_many(plan.storage_epoch(), plan, member, bindings)
        # Restore the exact retained object so normal cleanup can be exercised.
        initial_resources = admission._initial[4][1]
        original = next(value.value for key, value in initial_resources if key == root)
        dict.__setitem__(owner.resources, root, original)
    assert active_admission() is None


@pytest.mark.parametrize("failure", ["none", "body", "entry"])
def test_no_strong_receipts_or_detached_record_survive_scope_exit(final_plan, monkeypatch, failure):
    _path, _owner, plan = final_plan
    scope = StorageAdmission(plan)
    if failure == "entry":
        def reject(self, current):
            raise CompilationError("initial full validator declined")
        monkeypatch.setattr(StorageEpochProof, "validate", reject)
        with pytest.raises(CompilationError, match="initial"), scope:
            pytest.fail("failed entry must never activate")
    elif failure == "body":
        with pytest.raises(RuntimeError), scope:
            raise RuntimeError("body stopped")
    else:
        with scope:
            pass
    assert active_admission() is None
    assert scope._token is None
    assert scope._live is False
    assert all(getattr(scope, name) is None for name in
               ("_plan", "_proof", "_owner", "_initial", "_retained", "_record", "_admitted"))


def test_real_group_emission_scopes_only_its_final_storage_record_checks(tmp_path, monkeypatch):
    _path, owner, _epoch, _bindings = issued(tmp_path)
    original_enter, original_exit = StorageAdmission.__enter__, StorageAdmission.__exit__
    original_full = StorageEpochProof.validate
    events = []

    def enter(self):
        events.append("begin")
        return original_enter(self)

    def leave(self, *args):
        try:
            return original_exit(self, *args)
        finally:
            events.append("end")

    def full(self, plan):
        if events and events[-1] != "end":
            events.append("full")
        return original_full(self, plan)

    monkeypatch.setattr(StorageAdmission, "__enter__", enter)
    monkeypatch.setattr(StorageAdmission, "__exit__", leave)
    monkeypatch.setattr(StorageEpochProof, "validate", full)
    assert owner.run(owner.closed_selection)
    assert events == ["begin", "full", "full", "end"]
    assert len(owner.execution_plan.members) == 4
    assert len(owner.builder.scopes) == 1
    assert active_admission() is None


def test_only_emission_call_sites_use_admission_public_storage_apis_stay_full(final_plan, monkeypatch):
    _path, _owner, plan = final_plan
    member, bindings = query(plan)
    proof = plan.storage_epoch()
    original, calls = StorageEpochProof.validate, []

    def full(self, current):
        calls.append(current)
        return original(self, current)

    monkeypatch.setattr(StorageEpochProof, "validate", full)
    with StorageAdmission(plan):
        assert calls == [plan]
        for _index in range(3):
            validate_emitted_storage(proof, plan)
            admit_emitted_storage(proof, plan, member, bindings)
        assert calls == [plan]
        proof.validate(plan)
        proof.admits_many(plan, member, bindings)
        assert calls == [plan, plan, plan]
    assert calls == [plan]*4
    validate_emitted_storage(proof, plan)
    admit_emitted_storage(proof, plan, member, bindings)
    assert calls == [plan]*6


def test_checked_plan_keeps_independent_position_epoch_validation_inside_scope(final_plan, monkeypatch):
    _path, owner, plan = final_plan
    epoch = owner.position_epoch
    original, calls = type(epoch).validate_plan, []

    def position(self, current):
        calls.append(current)
        return original(self, current)

    monkeypatch.setattr(type(epoch), "validate_plan", position)
    with StorageAdmission(plan):
        plan.validate()
        assert calls == [plan]
