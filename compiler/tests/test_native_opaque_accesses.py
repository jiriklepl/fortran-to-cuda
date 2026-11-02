"""Typed native READ permits retain their original source and consumer ownership."""
from copy import copy
from dataclasses import replace

import pytest
from fparser.two.utils import walk

from compiler.ir import CompilationError
from compiler.scopes.native_opaque_accesses import NativeReadonlyPermit, readonly_accesses
from compiler.scopes.segments import fragment
from compiler.tests.test_native_host_only_reads import source_builder


def original_read(tmp_path):
    builder = source_builder(tmp_path)
    selected = next(node for node in walk(builder.entry.execution)
                    if type(node).__name__ == "Assignment_Stmt" and "SUM(edge_data)" in str(node))
    native = fragment(builder, (selected,), native_host_reads=True)
    access, = native.native_opaque_accesses.values()
    return builder, native, access


def test_typed_read_keeps_full_source_effects_without_numeric_credit(tmp_path):
    _builder, native, access = original_read(tmp_path)
    assert access.validate(native) is access
    root = access.binding.root
    assert native.effects[root] == {"read"}
    assert root not in native.coherence_effects
    assert root not in native.coherence_overwrites
    assert access.permit.public()["allocation_changes"] is False
    assert access.permit.public()["managed_definitions"] is False


@pytest.mark.parametrize("mode", ["W", "RW", "read"])
def test_mode_cannot_authorize_writes(tmp_path, mode):
    _builder, native, access = original_read(tmp_path)
    with pytest.raises(CompilationError, match="typed source permit"):
        replace(access, mode=mode).validate(native)


@pytest.mark.parametrize("fault", ["copied", "removed", "owner", "foreign_builder", "actual", "generation", "source"])
def test_original_consumer_registration_and_source_remain_required(tmp_path, fault):
    builder, native, access = original_read(tmp_path)
    permit = access.permit
    analysis = permit._analysis
    if fault == "copied":
        permit = replace(permit)
    elif fault == "removed":
        analysis._native_opaque_read_permits.pop(permit.identity)
    elif fault == "owner":
        analysis._native_opaque_read_permits_owner = object()
    elif fault == "foreign_builder":
        foreign = copy(builder)
        with pytest.raises(CompilationError, match="consumer or actual"):
            readonly_accesses(foreign, analysis, permit._selection, permit._summary, {access.binding.root: access.binding})
        return
    elif fault == "actual":
        builder._native_host_array_actuals = {access.binding.root: access.binding}
    elif fault == "generation":
        assert permit._context is not None
        object.__setattr__(analysis, "_source_derivation_arena", type(permit._arena)(permit._context))
    elif fault == "source":
        path = builder.entry.scope.path
        path.write_text(path.read_text() + "\n! changed original\n")
    with pytest.raises(CompilationError):
        permit.validate(native)


@pytest.mark.parametrize("fault", ["binding", "record", "context", "arena", "mode", "effects", "marker"])
def test_last_source_callback_cannot_change_permit_or_operation(tmp_path, monkeypatch, fault):
    _builder, native, access = original_read(tmp_path)
    permit = access.permit
    inputs = permit._analysis.inputs
    original = type(inputs).verify_live
    reached = []

    def mutate(self):
        result = original(self)
        if self is inputs and not reached:
            reached.append(True)
            if fault == "binding":
                object.__setattr__(access.binding, "name", "changed")
            elif fault == "record":
                object.__setattr__(permit, "_record", "{}")
            elif fault == "context":
                object.__setattr__(permit, "_context", None)
            elif fault == "arena":
                object.__setattr__(permit, "_arena", None)
            elif fault == "mode":
                object.__setattr__(access, "mode", "RW")
            elif fault == "effects":
                native.effects[access.binding.root].add("write")
            elif fault == "marker":
                permit._analysis._native_opaque_read_permits_owner = None
        return result

    monkeypatch.setattr(type(inputs), "verify_live", mutate)
    with pytest.raises(CompilationError, match="attachments changed"):
        permit.validate(native)
    assert reached == [True]


def test_failed_provisional_permit_publishes_neither_registry_nor_branch_handle(tmp_path, monkeypatch):
    builder = source_builder(tmp_path)
    selected = next(node for node in walk(builder.entry.execution)
                    if type(node).__name__ == "Assignment_Stmt" and "SUM(edge_data)" in str(node))
    original = NativeReadonlyPermit._validate
    attempted = []

    def reject(self, operation=None, *, issued=True):
        result = original(self, operation, issued=issued)
        if not issued:
            attempted.append(self)
            raise CompilationError("provisional publication rejected")
        return result

    monkeypatch.setattr(NativeReadonlyPermit, "_validate", reject)
    with pytest.raises(CompilationError, match="provisional publication rejected"):
        fragment(builder, (selected,), native_host_reads=True)
    permit, = attempted
    assert permit.identity not in permit._analysis._native_opaque_read_permits
    assert permit._arena is not None
    assert permit._arena._find_handle(("_native_opaque_read_permits", id(permit._analysis), permit.identity)) is None


def test_registered_read_permit_survives_supported_generation_fork(tmp_path):
    from compiler.scopes.generation_transaction import GenerationTransaction

    builder, native, access = original_read(tmp_path)
    builder._native_read_fixture = native
    with GenerationTransaction(builder) as transaction:
        projected = transaction.builder._native_read_fixture
        current, = projected.native_opaque_accesses.values()
        assert current.permit is access.permit
        assert current.validate(projected) is current
    assert access.validate(native) is access


@pytest.mark.parametrize("adopt", [False, True])
def test_new_read_permit_follows_supported_generation_commit_or_rollback(tmp_path, adopt):
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.tests.test_native_host_only_reads import HOST_SOURCE

    source = HOST_SOURCE.replace("b(-2)=b(-2)+sum(edge_data)",
                                "b(-2)=b(-2)+sum(edge_data)\nb(-1)=b(-1)+sum(edge_data)")
    builder = source_builder(tmp_path, source)
    selected = [node for node in walk(builder.entry.execution)
                if type(node).__name__ == "Assignment_Stmt" and "SUM(edge_data)" in str(node)]
    first, second = selected
    native = fragment(builder, (first,), native_host_reads=True)
    retained, = native.native_opaque_accesses.values()
    analysis = retained.permit._analysis
    carrier = copy(builder)
    carrier.analysis = analysis
    summary = analysis.segment_summary(builder.entry.qualified, (second,), capture_locals=True)
    binding = retained.binding
    with GenerationTransaction(carrier) as transaction:
        access = readonly_accesses(transaction.builder, analysis, (second,), summary, {binding.root: binding})[binding.root]
        assert access.validate() is access
        transaction.builder._native_read_fixture = access
        if adopt:
            transaction.adopt()
    if adopt:
        assert carrier._native_read_fixture.permit is access.permit
        assert access.validate() is access
    else:
        with pytest.raises(CompilationError):
            access.validate()
    assert retained.validate(native) is retained


@pytest.mark.parametrize("fault", ["binding", "mode", "permit"])
def test_container_without_operation_rechecks_its_own_fields_after_callback(tmp_path, monkeypatch, fault):
    _builder, _native, access = original_read(tmp_path)
    inputs = access.permit._analysis.inputs
    original = type(inputs).verify_live
    reached = []

    def mutate(self):
        result = original(self)
        if self is inputs and not reached:
            reached.append(True)
            value = {"binding": copy(access.binding), "mode": "RW", "permit": replace(access.permit)}[fault]
            object.__setattr__(access, fault, value)
        return result

    monkeypatch.setattr(type(inputs), "verify_live", mutate)
    with pytest.raises(CompilationError, match="attachments changed"):
        access.validate()
    assert reached == [True]


def test_public_serialization_cannot_lose_operation_fence_in_a_second_validation(tmp_path, monkeypatch):
    _builder, native, access = original_read(tmp_path)
    assert access.public(native)["mode"] == "R"
    inputs = access.permit._analysis.inputs
    original = type(inputs).verify_live
    calls, changed = [], []

    def mutate(self):
        result = original(self)
        if self is inputs:
            calls.append(True)
            if len(calls) == 2:
                changed.append(True)
                native.effects[access.binding.root].add("write")
        return result

    monkeypatch.setattr(type(inputs), "verify_live", mutate)
    try:
        record = access.public(native)
    except CompilationError:
        assert changed  # A reached mutation may reject; silent publication may not.
    else:
        assert record["mode"] == "R"
        assert native.effects[access.binding.root] == {"read"}
