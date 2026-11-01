"""Lifetime attachments cannot widen the checked source/storage authority."""
from copy import copy
from dataclasses import replace

import pytest

from compiler.ir import CompilationError
from compiler.tests.test_checked_execution_plan import scanned
from compiler.tests.test_graph_source_owner import EPOCH_SOURCE, emit


def test_epoch_borrows_only_exact_canonical_bindings_in_the_issued_plan(tmp_path):
    with scanned(tmp_path) as (owner, plan):
        epoch = plan.storage_epoch()
        binding = next(iter(plan.units(owner)[0].arrays.values()))
        assert epoch.admits(plan, owner, binding) == owner.canonical(binding).root
        with pytest.raises(CompilationError, match="admitted original"):
            epoch.admits(plan, owner, copy(binding))
        with pytest.raises(CompilationError, match="own this original"):
            epoch.admits(plan, copy(owner), binding)


def test_epoch_rejects_a_different_plan_even_for_identical_source(tmp_path):
    (tmp_path / "first").mkdir()
    (tmp_path / "second").mkdir()
    with scanned(tmp_path / "first") as (_owner, first):
        epoch = first.storage_epoch()
        with scanned(tmp_path / "second") as (_other, second), pytest.raises(CompilationError, match="issuing checked"):
            epoch.validate(second)


def test_epoch_batch_rejects_mixed_forged_bindings_and_changed_lifetime_facts(tmp_path):
    with scanned(tmp_path) as (owner, plan):
        epoch = plan.storage_epoch()
        bindings = tuple(plan.units(owner)[0].arrays.values())
        assert epoch.admits_many(plan, owner, bindings) == tuple(owner.canonical(value).root for value in bindings)
        with pytest.raises(CompilationError, match="admitted original"):
            epoch.admits_many(plan, owner, (*bindings, copy(bindings[0])))
        fact = owner.builder.facts["captures"]["argument::a"]
        old = fact["allocation_changes"]
        try:
            fact["allocation_changes"] = True
            with pytest.raises(CompilationError):
                epoch.admits_many(plan, owner, bindings)
        finally:
            fact["allocation_changes"] = old


def test_serialized_epoch_does_not_change_lifetime_or_definition_authority(tmp_path):
    with scanned(tmp_path) as (_owner, plan):
        epoch = plan.storage_epoch()
        public = epoch.public()
        public["cross_cut_retention_authorized"] = True
        public["resources"].clear()
        assert epoch.validate(plan) is epoch
        assert epoch.public()["resources"]
        assert epoch.public()["definitions_authorized"] is False
        assert epoch.public()["cross_cut_retention_authorized"] is False
        with pytest.raises(CompilationError, match="issuing checked"):
            replace(epoch, _issuer=object()).validate(plan)


def test_lifetime_fact_changes_are_rejected_before_emission(tmp_path):
    with scanned(tmp_path) as (owner, plan):
        fact = owner.builder.facts["captures"]["argument::a"]
        old = fact["allocation_changes"]
        try:
            fact["allocation_changes"] = True
            with pytest.raises(CompilationError):
                plan.storage_epoch().validate(plan)
        finally:
            fact["allocation_changes"] = old
        plan.validate()


def test_unknown_native_cuts_are_retained_in_the_shared_lifetime_contract(tmp_path):
    source, outputs, report = emit(tmp_path, EPOCH_SOURCE)
    owner, = report["scopes"]
    epoch = owner["checked_execution_plan"]["storage_epoch"]
    assert epoch["cross_cut_retention_authorized"] is False
    assert epoch["members"][0]["retirement_cuts"] == owner["boundaries"]
    assert len(epoch["members"][0]["retirement_cuts"]) >= 2
    generated = outputs[report["sources"][str(source)]["replacement"]]
    assert "fort_scope_resident_create_v1(0_c_int, fort_context)" in generated
    assert "fort_scope_resident_close_v1(" in generated


def test_checked_child_registration_opens_a_domain_before_dummy_rebasing(tmp_path):
    from compiler.tests.test_graph_source_owner import reusable_helper_source

    source, outputs, report = emit(tmp_path, reusable_helper_source(True))
    generated = outputs[report["sources"][str(source)]["replacement"]]
    assert "fort_scope_resident_create_v1(" in generated
    assert " = fort_scope_create(" not in generated
