"""Rollback invalidates reusable workers without refunding source search."""

from copy import copy

import pytest

from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.ordinary_consumption import _OrdinaryPairConsumption
from compiler.scopes.region_dispatch import InlineRegions
from compiler.tests.support.borrowed_numerical_team import (
    HELPER_SOURCE,
    INITIALIZER,
    OWNER_SOURCE,
    SCALER,
    borrowed_team_fixture,
    emit_borrowed_team_fixture,
)


def test_restore_discards_new_worker_and_preserves_shared_cache_and_spent_work(tmp_path):
    _paths, builder = borrowed_team_fixture(tmp_path)
    projection = copy(builder)
    coordinators, reasons, owners = builder.coordinators, builder.coordinator_reasons, builder.region_owners
    checkpoint = builder.scope_checkpoint()
    old = builder.coordinator(INITIALIZER)
    assert old is not None
    assert old.whole_team
    old.scope._validate()
    registry = old.builder.inline
    attempts, operations = registry.attempts, registry.operations
    assert attempts == 1
    assert operations > 0

    builder.restore_scope_checkpoint(checkpoint)
    assert projection.coordinators is builder.coordinators is coordinators
    assert projection.coordinator_reasons is builder.coordinator_reasons is reasons
    assert projection.region_owners is builder.region_owners is owners
    assert INITIALIZER not in coordinators
    assert INITIALIZER not in reasons
    assert owners[INITIALIZER] is registry
    assert not registry.original_numerical_regions
    assert not registry.used
    assert (registry.attempts, registry.operations) == (attempts, operations)

    current = builder.coordinator(INITIALIZER)
    assert current is not None
    assert current is not old
    assert current.builder.inline is not registry
    assert owners[INITIALIZER] is current.builder.inline
    assert current.builder.inline.attempts == attempts + 1
    assert current.builder.inline.operations == 2 * operations
    current.scope._validate()


@pytest.mark.parametrize("consume", [False, True])
def test_restore_retains_exact_preexisting_owner_and_pair_after_replacement(tmp_path, consume):
    _paths, builder = borrowed_team_fixture(tmp_path)
    if consume:
        with _OrdinaryPairConsumption(builder):
            _retained_owner(builder)
    else:
        _retained_owner(builder)


def _retained_owner(builder):
    retained = builder.coordinator(INITIALIZER)
    assert retained is not None
    registry = retained.builder.inline
    pairs = dict(registry.original_numerical_regions)
    checkpoint = builder.scope_checkpoint()
    discarded = builder.coordinator(SCALER)
    assert discarded is not None
    proxy = copy(retained.builder)
    replacement = InlineRegions(proxy)
    proxy.inline = replacement
    replacement._reserve(retained.scope.original_nodes)
    builder.region_owners[INITIALIZER] = replacement
    charged = replacement.attempts, replacement.operations
    builder.restore_scope_checkpoint(checkpoint)
    assert builder.coordinators[INITIALIZER] is retained
    assert SCALER not in builder.coordinators
    assert builder.region_owners[INITIALIZER] is registry
    assert registry.original_numerical_regions == pairs
    assert all(registry.original_numerical_regions[key] is pair for key, pair in pairs.items())
    assert (registry.attempts, registry.operations) == charged
    retained.scope._validate()


def test_rejected_original_section_then_repeated_whole_calls_keep_lexical_owner(tmp_path):
    source = OWNER_SOURCE.replace("call scale(b,scale_factor)",
        "call scale(b(:,:,:),scale_factor)\ncall scale(b,scale_factor)", 1)
    paths, outputs, report, builder = emit_borrowed_team_fixture(tmp_path, owner_source=source)
    attempt = report["lexical_owner_attempt"]
    assert attempt["outcome"] == "accepted"
    assert attempt["error"] is None
    owner, = report["scopes"]
    assert owner["owner_variant"] is not None
    boundary, = owner["boundaries"]
    assert "native rectangular source-call mapping" in boundary["reason"]
    assert "whole original array actuals" in boundary["reason"]
    assert boundary["reopen"] is False
    scaler = builder.coordinators[SCALER]
    assert scaler.whole_team
    scaler.scope._validate()
    calls = [call for call in report["resolved_calls"]
             if call["procedure"] == SCALER and call["actual_arguments"][0] in {"b", "c", "d"}]
    assert len(calls) == 3
    assert {call["resource_mapping"]["argument::values"] for call in calls} == {
        "argument::b", "argument::c", "argument::d"}
    original_actual = "call scale(b(:,:,:),scale_factor)"
    emitted = outputs[report["sources"][str(paths[1])]["replacement"]].lower().replace(" ", "")
    assert emitted.count(original_actual.replace(" ", "")) == 1
    assert "fort_scope_close(" in emitted[:emitted.index(original_actual.replace(" ", ""))]


@pytest.mark.parametrize("cap", ["attempts", "operations"])
def test_rebuilding_discarded_helper_keeps_attempt_and_operation_caps(tmp_path, cap):
    _paths, builder = borrowed_team_fixture(tmp_path)
    checkpoint = builder.scope_checkpoint()
    prior_attempts = prior_operations = 0
    for _index in range(33):
        coordinator = builder.coordinator(INITIALIZER)
        registry = builder.region_owners[INITIALIZER]
        assert registry.attempts >= prior_attempts
        assert registry.operations >= prior_operations
        assert registry.attempts <= 32
        assert registry.operations <= 256
        if coordinator is None:
            assert "bounded inline region/operation budget exhausted" in builder.coordinator_reasons[INITIALIZER]
            break
        assert registry.attempts == prior_attempts + 1
        if cap == "operations":
            # Charge repeated actual original-group scans, independently of
            # numerical eligibility. A rollback cannot refund this work.
            cost = registry.operations - prior_operations
            registry._reserve(coordinator.scope.original_nodes * ((256 - registry.operations) // cost))
        prior_attempts, prior_operations = registry.attempts, registry.operations
        builder.restore_scope_checkpoint(checkpoint)
    else:
        pytest.fail("rebuilding refunded the source search budget")
    assert builder.coordinator_attempts == {INITIALIZER}
    if cap == "attempts":
        assert registry.attempts == 32
    else:
        assert registry.attempts < 32
        assert registry.operations == 256


def test_distinct_coordinator_search_limit_is_not_refunded_by_restore(tmp_path):
    additions = "\n".join(f"subroutine marker_{index}(value)\ninteger,intent(inout)::value\n"
        "value=value+1\nend subroutine" for index in range(65))
    helper = HELPER_SOURCE.rsplit("end module", 1)[0] + additions + "\nend module\n"
    _paths, builder = borrowed_team_fixture(tmp_path, helper_source=helper)
    assert builder.analysis.procedure_limit == 64
    checkpoint = builder.scope_checkpoint()
    for index in range(64):
        name = f"amber_planes::marker_{index}"
        assert builder.coordinator(name) is None
        assert name in builder.coordinator_attempts
        builder.restore_scope_checkpoint(checkpoint)
    assert len(builder.coordinator_attempts) == 64
    excluded = "amber_planes::marker_64"
    assert builder.coordinator(excluded) is None
    assert excluded not in builder.coordinator_attempts
    assert excluded not in builder.region_owners
    assert excluded not in builder.coordinators


def test_restore_discards_a_diamond_of_workers_and_rebuilds_one_shared_leaf(tmp_path):
    additions = """
subroutine route_left(values)
real(8),contiguous,intent(out)::values(:,:,:)
call initialize_planes(values,0)
end subroutine
subroutine route_right(values)
real(8),contiguous,intent(out)::values(:,:,:)
call initialize_planes(values,0)
end subroutine
subroutine route_pair(left,right)
real(8),contiguous,intent(out)::left(:,:,:),right(:,:,:)
call route_left(left)
call route_right(right)
end subroutine
"""
    helper = HELPER_SOURCE.rsplit("end module", 1)[0] + additions + "end module\n"
    _paths, builder = borrowed_team_fixture(tmp_path, helper_source=helper)
    checkpoint = builder.scope_checkpoint()
    old = builder.coordinator("amber_planes::route_pair")
    assert old is not None
    retained = dict(builder.coordinators)
    assert all(retained[name] is not None for name in (
        "amber_planes::route_pair", "amber_planes::route_left", "amber_planes::route_right", INITIALIZER))
    builder.restore_scope_checkpoint(checkpoint)
    assert not builder.coordinators
    current = builder.coordinator("amber_planes::route_pair")
    assert current is not None
    assert current is not old
    assert all(builder.coordinators[name] is not coordinator for name, coordinator in retained.items())
    leaf = builder.coordinators[INITIALIZER]
    assert leaf.builder.inline.attempts == 2
    assert current.leaves == leaf.leaves
    assert len(current.leaves) == 1
    leaf.scope._validate()
    current.worker()


def test_new_demand_after_restore_still_rejects_changed_original_file(tmp_path):
    paths, builder = borrowed_team_fixture(tmp_path)
    checkpoint = builder.scope_checkpoint()
    assert builder.coordinator(INITIALIZER) is not None
    builder.restore_scope_checkpoint(checkpoint)
    paths[0].write_text(paths[0].read_text().replace("values(:,:,plane)=seed", "values(:,:,plane)=seed+1"))
    with pytest.raises(CompilationError, match="source changed"):
        builder.coordinator(INITIALIZER)
    assert INITIALIZER not in builder.coordinators
    assert not builder.outputs
    assert not builder.edits
    assert not builder.scopes


def test_generation_transaction_isolates_and_adopts_search_charges(tmp_path):
    _paths, builder = borrowed_team_fixture(tmp_path)
    with GenerationTransaction(builder) as transaction:
        candidate = transaction.builder
        assert candidate.coordinator_attempts is not builder.coordinator_attempts
        assert candidate.coordinator(INITIALIZER) is not None
        assert candidate.coordinator_attempts == {INITIALIZER}
        assert not builder.coordinator_attempts
    assert not builder.coordinator_attempts
    assert not builder.coordinators
    assert builder.region_owners == {builder.entry.qualified: builder.inline}
    with GenerationTransaction(builder) as transaction:
        assert transaction.builder.coordinator(INITIALIZER) is not None
        transaction.adopt()
    assert builder.coordinator_attempts == {INITIALIZER}
    assert builder.coordinators[INITIALIZER].root is builder
    builder.coordinators[INITIALIZER].scope._validate()
