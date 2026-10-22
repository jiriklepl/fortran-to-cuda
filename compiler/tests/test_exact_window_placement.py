"""Only the issued placement for the retained checked plan can govern emission."""

from contextlib import contextmanager
from copy import copy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.ir import CompilationError
from compiler.offload.exact_window import prepare
from compiler.offload.exact_window_calibration import fixture_source
from compiler.scopes.execution_plan import build_execution_plan
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.ordinary_consumption import _OrdinaryPairConsumption
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_exact_window import observations


@contextmanager
def issued(directory, *, request=None):
    path = directory / "original.f90"
    path.write_text(fixture_source())
    facts = {
        "schema_version": 1, "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
        "captures": {"argument::" + name: {
            "storage": "stable", "initialized": "whole", "escapes": False,
            "allocation_changes": False,
        } for name in ("a", "b", "d", "out")},
    }
    builder = ScopeBuilder(
        [path], "exact_fixture::advance", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="auto", scope_execution="reached",
                             exact_window={"calibration": True} if request is None else request))
    nodes = builder.inline.prepare(builder.entry.execution.content)
    builder.generated.update(builder.inline.generated)
    builder.numerical_ir.update(builder.inline.ir)
    builder.numerical_reasons.update({name: None for name in builder.inline.regions})
    with _OrdinaryPairConsumption(builder):
        owner = LexicalOwner(builder)
        owner.scan(nodes)
        owner.validate_plan(nodes)
        owner.leaves = {leaf for member in owner.members.values() for unit in member.units for leaf in unit.leaves}
        plan = build_execution_plan(owner)
        prepare(owner)
        yield owner, plan, plan.exact_window_placement()


@pytest.mark.parametrize("replacement", ["dictionary", "copied", "foreign_issuer", "deleted"])
def test_replaced_or_deleted_slot_cannot_change_pricing_authority(tmp_path, replacement):
    with issued(tmp_path) as (owner, plan, placement):
        invalid = {"dictionary": {}, "copied": copy(placement),
                   "foreign_issuer": replace(placement, _issuer=object()), "deleted": None}[replacement]
        owner.emission.exact_window = invalid
        try:
            with pytest.raises(CompilationError, match="placement"):
                plan.automatic_native_reason()
            with pytest.raises(CompilationError, match="placement"):
                _ = owner.price_units
        finally:
            owner.emission.exact_window = placement
        assert plan.exact_window_placement() is placement
        assert plan.automatic_native_reason() is None
        assert owner.price_units is False


def test_same_portable_identity_does_not_authorize_a_foreign_plan(tmp_path):
    with (issued(tmp_path) as (_first_owner, first, first_placement),
          issued(tmp_path) as (second_owner, second, second_placement)):
        assert first.execution_identity == second.execution_identity
        assert first is not second
        with pytest.raises(CompilationError, match="attachment"):
            first_placement.validate(second)
        second_owner.emission.exact_window = first_placement
        try:
            with pytest.raises(CompilationError, match="attachment"):
                second.automatic_native_reason()
        finally:
            second_owner.emission.exact_window = second_placement
        assert second.exact_window_placement() is second_placement


def test_nested_evidence_mutation_invalidates_issued_resident_selection(tmp_path):
    with issued(tmp_path) as (_owner, _plan, capture):
        evidence = observations(capture.public()["contract"])
    with issued(tmp_path, request={"evidence": evidence}) as (owner, plan, placement):
        assert placement.public()["contract"] == evidence["contract"]
        assert placement.public()["selection"] == "resident", placement.public()["reason"]
        elapsed = evidence["observations"][0]["seconds"]
        evidence["observations"][0]["seconds"] = elapsed + .01
        try:
            with pytest.raises(CompilationError, match="request or evidence changed"):
                plan.automatic_native_reason()
            with pytest.raises(CompilationError, match="request or evidence changed"):
                _ = owner.exact_window
        finally:
            evidence["observations"][0]["seconds"] = elapsed
        assert plan.exact_window_placement() is placement
        assert placement.public()["selection"] == "resident"


def test_calibration_request_cannot_be_replaced_after_issuance(tmp_path):
    with issued(tmp_path) as (owner, plan, placement):
        previous = owner.builder.config
        owner.builder.config = replace(previous, exact_window={})
        try:
            with pytest.raises(CompilationError, match="request or evidence changed"):
                plan.exact_window_placement()
        finally:
            owner.builder.config = previous
        assert plan.exact_window_placement() is placement


def test_request_changed_during_reconstruction_cannot_bind_an_earlier_decision(tmp_path, monkeypatch):
    import compiler.offload.exact_window as exact_window

    with issued(tmp_path) as (_owner, _plan, capture):
        request = {"evidence": observations(capture.public()["contract"])}
    reconstruct = exact_window.evidence_decision

    def changed_request(evidence, contract):
        decision = reconstruct(evidence, contract)
        assert decision["selection"] == "resident"
        request["evidence"]["observations"][0]["seconds"] += .01
        return decision

    monkeypatch.setattr(exact_window, "evidence_decision", changed_request)
    with pytest.raises(CompilationError, match="request or evidence changed"), issued(tmp_path, request=request):
        pass


@pytest.mark.parametrize("mutation", ["source", "configuration", "capture_facts"])
def test_valid_placement_does_not_replace_source_configuration_or_fact_validation(tmp_path, mutation):
    with issued(tmp_path) as (owner, plan, placement):
        builder, path = owner.builder, owner.routine.scope.path
        source, config = path.read_text(), builder.config
        fact = builder.facts["captures"]["argument::a"]["initialized"]
        try:
            if mutation == "source":
                path.write_text(source + "\n! changed after placement\n")
            elif mutation == "configuration":
                builder.config = replace(config, host_threads=2)
            else:
                builder.facts["captures"]["argument::a"]["initialized"] = "none"
            with pytest.raises(CompilationError):
                plan.validate()
        finally:
            path.write_text(source)
            builder.config = config
            builder.facts["captures"]["argument::a"]["initialized"] = fact
        assert plan.validate() is plan
        assert plan.exact_window_placement() is placement


@pytest.mark.parametrize("adopt", [False, True], ids=["rollback", "adopt"])
def test_transaction_preserves_exact_registered_owner_plan_attachment(tmp_path, adopt):
    with issued(tmp_path) as (owner, plan, placement):
        parent = owner.builder
        parent.coordinators["retained_owner"] = owner
    with GenerationTransaction(parent) as transaction:
        candidate = transaction.builder.coordinators["retained_owner"]
        candidate_plan = candidate.execution_plan
        candidate_placement = candidate_plan.exact_window_placement()
        assert candidate is not owner
        assert candidate_plan is not plan
        assert candidate_placement is not placement
        assert candidate_plan.validate() is candidate_plan
        assert candidate_placement._plan is candidate_plan
        assert transaction.builder._exact_window_placements[candidate_plan.identity] is candidate_placement
        with pytest.raises(CompilationError, match="attachment"):
            placement.validate(candidate_plan)
        if adopt:
            transaction.adopt()
    retained = parent.coordinators["retained_owner"]
    retained_plan = retained.execution_plan
    retained_placement = retained_plan.exact_window_placement()
    assert retained.builder is parent
    assert retained_plan.validate() is retained_plan
    assert retained_placement._plan is retained_plan
    assert parent._exact_window_placements[retained_plan.identity] is retained_placement
    if adopt:
        assert retained is not owner
        assert retained is not candidate
        assert retained_placement is not placement
        assert retained_placement is not candidate_placement
        assert retained_plan.execution_identity == plan.execution_identity
    else:
        assert retained is owner
        assert retained_plan is plan
        assert retained_placement is placement
    with pytest.raises(CompilationError, match="attachment"):
        candidate_placement.validate(retained_plan)
