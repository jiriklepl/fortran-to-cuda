"""Complete interval inventories cannot borrow authority from a scanned prefix."""

from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.reached_windows import prove_reached_window
from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.resource_demand import resource_demand_from_plan, resource_demand_from_reached_window
from compiler.tests.test_checked_execution_plan import scanned
from compiler.tests.test_lexical_source_owner import MODULE_SOURCE, NATIVE_METADATA_SOURCE
from compiler.tests.test_reached_window_owner import window_fixture


def test_plan_demand_includes_later_outputs_original_guards_and_native_sections(tmp_path):
    with scanned(tmp_path) as (owner, plan):
        demand = resource_demand_from_plan(plan)
        assert resource_demand_from_plan(plan) is demand
        assert demand.validate_plan(plan) is demand
        assert {item.root for item in demand.managed_bindings} == {
            "argument::a", "argument::b", "argument::out"}
        roots = {item.root for item in demand.bindings}
        assert {"argument::n", "argument::escape", "local_owner::step::i"} <= roots
        assert owner.units[0].arrays.keys() != {item.root for item in demand.managed_bindings}
        record = demand.public()
        assert not record["prospective"]
        assert not record["complete_effect_summary"]
        assert not record["execution_authorized"]
        assert not record["placement_authorized"]
        assert record["cost_coverage"] == "not_established_by_inventory"
        assert record["existing_cost_requirements"] == plan.public()["cost_requirements"]
        assert "automatic_estimate_available" not in record
        record["resources"].clear()
        assert demand.public()["resources"]


@pytest.mark.parametrize("replacement", ["copy", "issuer"])
def test_only_exact_issued_demand_has_authority(tmp_path, replacement):
    with scanned(tmp_path) as (_owner, plan):
        demand = resource_demand_from_plan(plan)
        other = copy(demand) if replacement == "copy" else replace(demand, _issuer=object())
        with pytest.raises(CompilationError, match="exact issued"):
            other.validate()


def test_equal_source_identity_cannot_lend_demand_to_foreign_plan(tmp_path):
    with scanned(tmp_path) as (_owner, first):
        demand = resource_demand_from_plan(first)
    with scanned(tmp_path) as (_owner, second):
        assert first.execution_identity == second.execution_identity
        with pytest.raises(CompilationError, match="different checked plan"):
            demand.validate_plan(second)


@pytest.mark.parametrize("mutation", ["source", "configuration", "facts"])
def test_inventory_reauthenticates_external_inputs(tmp_path, mutation):
    with scanned(tmp_path) as (owner, plan):
        demand = resource_demand_from_plan(plan)
        builder, path = owner.builder, owner.routine.scope.path
        source, config = path.read_text(), builder.config
        defined = builder.facts["captures"]["argument::a"]["initialized"]
        try:
            if mutation == "source":
                path.write_text(source + "\n! changed after sealing\n")
            elif mutation == "configuration":
                builder.config = replace(builder.config, host_threads=2)
            else:
                builder.facts["captures"]["argument::a"]["initialized"] = "none"
            with pytest.raises(CompilationError):
                demand.validate()
        finally:
            path.write_text(source)
            builder.config = config
            builder.facts["captures"]["argument::a"]["initialized"] = defined


def test_reached_demand_exists_before_scan_and_covers_final_plan(tmp_path):
    _path, builder, proof, originals = window_fixture(tmp_path)
    demand = resource_demand_from_reached_window(builder, proof)
    assert demand.public()["prospective"]
    assert demand.public()["requires_final_plan_validation"]
    assert demand.public()["prospective_cost_requirements"]
    assert {binding.root for binding in demand.managed_bindings} == {"argument::a", "argument::b"}
    assert not builder.coordinators
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    assert demand.validate_plan(owner.execution_plan) is demand


def test_prefix_inventory_cannot_cover_later_numerical_resource(tmp_path):
    with scanned(tmp_path) as (owner, plan):
        original_loop = next(node for node in owner.routine.execution.content
                             if type(node).__name__ == "Block_Nonlabel_Do_Construct")
        proof = prove_reached_window(owner.builder.analysis, owner.routine.qualified,
                                     (original_loop,), ("argument::a", "argument::b"), owner.builder.facts)
        demand = resource_demand_from_reached_window(owner.builder, proof)
        assert "argument::out" not in {binding.root for binding in demand.bindings}
        with pytest.raises(CompilationError, match="misses final interval resources.*argument::out"):
            demand.validate_plan(plan)


def test_reached_factory_rejects_copied_or_foreign_proof(tmp_path):
    _path, builder, proof, _originals = window_fixture(tmp_path)
    with pytest.raises(CompilationError, match="registered exact original"):
        resource_demand_from_reached_window(builder, copy(proof))
    _path, other, _proof, _originals = window_fixture(tmp_path)
    with pytest.raises(CompilationError, match="registered exact original"):
        resource_demand_from_reached_window(other, proof)


def test_public_records_cannot_be_imported_as_demands(tmp_path):
    with scanned(tmp_path) as (_owner, plan), pytest.raises(CompilationError, match="exact checked execution plan"):
        resource_demand_from_plan(plan.public())


def test_mapped_child_uses_canonical_parent_storage_and_keeps_child_controls(tmp_path):
    with scanned(tmp_path, source=MODULE_SOURCE) as (_owner, plan):
        assert len(plan.members) == 2
        demand = resource_demand_from_plan(plan)
        child = [row for row in demand.public()["resources"] if row["procedure"] == "stage_child::adjust"]
        assert any(row["resource"] == "argument::x" and row["canonical_resource"] == "argument::b"
                   and "managed" in row["roles"] for row in child)
        assert any(row["resource"] == "argument::escape" for row in child)
        assert demand.validate_plan(plan) is demand


def test_admitted_opaque_metadata_keeps_its_original_allocation_identity(tmp_path):
    with scanned(tmp_path, source=NATIVE_METADATA_SOURCE) as (owner, plan):
        original = owner.builder.analysis.modules["local_owner"].bindings["metadata"]
        before = getattr(owner.builder.analysis, "_native_metadata", False)
        demand = resource_demand_from_plan(plan)
        assert demand.validate_plan(plan) is demand
        assert any(binding is original for binding in demand.bindings)
        assert all(not hasattr(binding, "native_metadata_object") for binding in demand.bindings)
        assert getattr(owner.builder.analysis, "_native_metadata", False) is before
        assert original not in demand.managed_bindings
        assert not demand.public()["execution_authorized"]


@pytest.mark.parametrize("adopt", [False, True], ids=["rollback", "adopt"])
def test_transaction_copies_registry_and_exact_plan_together(tmp_path, adopt):
    with scanned(tmp_path) as (owner, plan):
        demand = resource_demand_from_plan(plan)
        parent = owner.builder
        parent.coordinators["retained_owner"] = owner
    with GenerationTransaction(parent) as transaction:
        candidate = transaction.builder.coordinators["retained_owner"].execution_plan
        copied = candidate.resource_demand()
        assert copied is not demand
        assert copied._plan is candidate
        assert copied.validate_plan(candidate) is copied
        assert transaction.builder._resource_demands[copied.identity] is copied
        if adopt:
            transaction.adopt()
    final = parent.coordinators["retained_owner"].execution_plan
    retained = final.resource_demand()
    assert retained.validate_plan(final) is retained
    assert (retained is demand) is (not adopt)
