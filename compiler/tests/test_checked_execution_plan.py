"""Placement and emission consume one admitted original operation inventory."""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.ir import CompilationError
from compiler.scopes.execution_plan import _digest, build_execution_plan
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.ordinary_consumption import _OrdinaryPairConsumption
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_lexical_source_owner import SOURCE, emit
from compiler.tests.test_source_scopes import FACT


@contextmanager
def scanned(directory, before_plan=None, *, config=None, source=SOURCE):
    path = directory / "owner.f90"
    path.write_text(source.replace("call opaque(b,n)", "b(-2)=b(-2)+sum(a)"))
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: deepcopy(FACT) for name in ("a", "b", "out")}}
    builder = ScopeBuilder([path], "local_owner::step", facts=facts, options=CompilerOptions(),
                           config=config or OffloadConfig(policy="sections", scope_execution="reached"))
    nodes = builder.inline.prepare(builder.entry.execution.content)
    builder.generated.update(builder.inline.generated)
    builder.numerical_ir.update(builder.inline.ir)
    builder.numerical_reasons.update({name: None for name in builder.inline.regions})
    with _OrdinaryPairConsumption(builder):
        owner = LexicalOwner(builder)
        owner.scan(nodes)
        owner.validate_plan(nodes)
        assert not owner.boundaries
        if before_plan is not None:
            before_plan(owner)
        yield owner, build_execution_plan(owner)


def test_plan_binds_original_guarded_native_correction_and_shared_resources(tmp_path):
    with scanned(tmp_path) as (owner, plan):
        assert plan.validate() is plan
        record = plan.public()
        identity = record.pop("identity")
        assert identity == _digest(record)
        assert plan.members == (owner,)
        assert plan.units(owner) == tuple(owner.units)
        assert record["numerical_invocations"] == 2
        assert {row["resource"] for row in record["resources"]} == {
            "argument::a", "argument::b", "argument::out"}
        units = record["procedures"][0]["units"]
        assert [row["ordinal"] for row in units] == list(range(len(units)))
        correction = next(row for row in units if row["operations"]["native_operations"])
        operation, = correction["operations"]["native_operations"]
        assert operation["resource_effects"]["argument::a"] == ["read"]
        assert operation["resource_effects"]["argument::b"] == ["read", "write"]
        assert owner.graph.nodes[correction["selected_node_ids"][0]].guard
        assert record["policy"]["transfers"]["observed"] is False
        # Returned JSON cannot change either admitted operations or emission.
        correction["operations"]["native_operations"].clear()
        assert plan.validate() is plan
        assert plan.public()["procedures"][0]["units"] != units


def test_execution_identity_reuses_semantics_across_independent_source_graphs(tmp_path):
    with scanned(tmp_path) as (first_owner, first):
        expected = first.execution_identity
        authorities = first._numerical_authorities
        assert authorities
        assert first.public()["execution_identity"] == expected
    with scanned(tmp_path) as (second_owner, second):
        assert second_owner.builder.analysis is not first_owner.builder.analysis
        assert second.execution_identity == expected
        assert all(left is not right and left.identity != right.identity
                   for left, right in zip(authorities, second._numerical_authorities, strict=True))
        # Exact in-process source/proof authority is still required.
        with pytest.raises(CompilationError, match="registered"):
            authorities[0].validate(second_owner.builder.analysis, authorities[0].procedure,
                                    second._numerical_authorities[0]._selection)


@pytest.mark.parametrize("change", ["source", "configuration"])
def test_execution_identity_changes_with_source_or_compiler_configuration(tmp_path, change):
    with scanned(tmp_path) as (_owner, plan):
        expected = plan.execution_identity
    options = ({"source": SOURCE + "\n! changed authenticated source\n"} if change == "source" else
               {"config": OffloadConfig(policy="sections", scope_execution="reached", host_threads=2)})
    with scanned(tmp_path, **options) as (_owner, changed):
        assert changed.execution_identity != expected


@pytest.mark.parametrize("mutation", ["order", "guard", "native_text", "resource", "effects",
                                      "overwrites", "patches", "control_guard", "new_field"])
def test_admitted_execution_graph_rejects_mutation_immediately(tmp_path, mutation):
    with scanned(tmp_path) as (owner, plan):
        def mutate():
            if mutation == "order":
                owner.units.reverse()
            elif mutation == "guard":
                owner.units[0].source_guard = ".true."
            elif mutation == "native_text":
                owner.units[0].native_source = "call another_operation()"
            elif mutation == "resource":
                root = next(iter(owner.resources))
                binding, number = owner.resources[root]
                owner.resources[root] = binding, number + 100
            elif mutation == "patches":
                owner.patches.append((1, 1, "call unadmitted()"))
            elif mutation == "control_guard":
                owner.control_guard = ".false."
            elif mutation == "new_field":
                owner.units[0].future_semantic_field = {"mutable": True}
            else:
                native = next(operation for unit in owner.units for operation in unit.scope.native)
                if mutation == "overwrites":
                    native.overwrites |= {"argument::b"}
                else:
                    native.effects["argument::b"] = {"read"}
        with pytest.raises(CompilationError, match="immutable"):
            mutate()
        assert plan.validate() is plan


@pytest.mark.parametrize("mutation", ["facts", "configuration", "source"])
def test_emission_revalidates_external_source_and_configuration(tmp_path, mutation):
    with scanned(tmp_path) as (owner, plan):
        original_config = owner.builder.config
        original_fact = owner.builder.facts["captures"]["argument::b"]["initialized"]
        path = owner.routine.scope.path
        original_source = path.read_text()
        try:
            if mutation == "facts":
                owner.builder.facts["captures"]["argument::b"]["initialized"] = "none"
            elif mutation == "configuration":
                owner.builder.config = replace(original_config, host_threads=2)
            else:
                path.write_text(original_source + "\n! changed after admission\n")
            with pytest.raises(CompilationError):
                plan.validate()
        finally:
            owner.builder.config = original_config
            owner.builder.facts["captures"]["argument::b"]["initialized"] = original_fact
            path.write_text(original_source)
        assert plan.validate() is plan


def test_all_new_metadata_is_frozen_detached_and_bound_to_emission_identity(tmp_path):
    retained = {}

    def add_metadata(owner):
        retained["metadata"] = {"extra": [{"required": True}]}
        owner.units[0].future_metadata = retained["metadata"]
        retained["fields"] = vars(owner.units[0])

    with scanned(tmp_path, add_metadata) as (owner, plan):
        identity = plan.public()["procedures"][0]["emission_identity"]
        executable = plan.execution_identity
        retained["metadata"]["extra"][0]["required"] = False
        retained["fields"]["source_guard"] = ".false."
        assert owner.units[0].future_metadata["extra"][0]["required"] is True
        assert owner.units[0].source_guard is None
        with pytest.raises(CompilationError, match="immutable"):
            owner.units[0].future_metadata["extra"][0]["required"] = False
        assert plan.validate() is plan
        # Emission results do not change the admitted alternative's identity.
        owner.unit_emission(owner.units[0]).native_preflight = {"available": False}
        assert plan.public()["procedures"][0]["emission_identity"] == identity
    with scanned(tmp_path, lambda owner: setattr(owner.units[0], "future_metadata",
                                                {"extra": [{"required": False}]})) as (_owner, changed):
        assert changed.public()["procedures"][0]["emission_identity"] != identity
        assert changed.execution_identity != executable


def test_unknown_opaque_metadata_cannot_silently_escape_sealing(tmp_path):
    from types import SimpleNamespace

    with (pytest.raises(CompilationError, match="ownership is unspecified"),
          scanned(tmp_path, lambda owner: setattr(owner.units[0], "future_metadata",
                                                  SimpleNamespace(mutable=True)))):
        pytest.fail("opaque mutable metadata was admitted")


def test_unknown_frontend_dataclass_is_metadata_not_registered_authority(tmp_path):
    from dataclasses import dataclass, field

    @dataclass
    class FutureProof:
        semantic: list = field(default_factory=lambda: ["original"])

        def validate(self):
            raise AssertionError("not an issued proof")

        def public(self):
            return {"role": "future-proof"}

    FutureProof.__module__ = "compiler.frontend.future_proof"
    proof = FutureProof()
    retained_alias = proof.semantic
    with scanned(tmp_path, lambda owner: setattr(owner.units[0], "future_metadata", proof)) as (owner, plan):
        expected = plan.execution_identity
        retained_alias.append("changed retained alias")
        assert owner.units[0].future_metadata.semantic == ["original"]
        with pytest.raises(CompilationError, match="immutable"):
            owner.units[0].future_metadata.semantic.append("changed after handoff")
        assert plan.validate() is plan
    with scanned(tmp_path, lambda owner: setattr(owner.units[0], "future_metadata",
                                                FutureProof(["different semantics"]))) as (_owner, changed):
        assert changed.execution_identity != expected


def test_unregistered_prepared_proof_copy_cannot_be_borrowed(tmp_path):
    def install_copy(owner):
        proof = next(iter(owner.builder.analysis._numerical_preparations.values()))
        owner.units[0].future_authority = replace(proof)

    with (pytest.raises(CompilationError, match="exact registered"),
          scanned(tmp_path, install_copy)):
        pytest.fail("unregistered proof copy was borrowed")


def test_foreign_or_replaced_plan_has_no_emission_authority(tmp_path):
    with scanned(tmp_path) as (owner, plan):
        for foreign in (replace(plan, identity="0" * 64), replace(plan, _issuer=object())):
            with pytest.raises(CompilationError, match="issuing lexical owner"):
                foreign.validate()
        with pytest.raises(CompilationError, match="immutable"):
            owner.execution_plan = replace(plan, identity="0" * 64)


def test_transaction_rebinds_and_preserves_an_already_sealed_handoff(tmp_path):
    from compiler.scopes.generation_transaction import GenerationTransaction

    with scanned(tmp_path) as (owner, plan):
        parent = owner.builder
        parent.coordinators["retained_owner"] = owner
        owner.unit_emission(owner.units[0]).native_preflight = {"available": False}
    with GenerationTransaction(parent) as transaction:
        candidate = transaction.builder.coordinators["retained_owner"]
        assert candidate is not owner
        assert candidate.builder is transaction.builder
        assert candidate.execution_plan._owner is candidate
        assert candidate.execution_plan.validate().identity == plan.identity
        assert candidate.unit_emission(candidate.units[0]).native_preflight == {"available": False}
        candidate.unit_emission(candidate.units[0]).native_preflight = {"available": True}
        with pytest.raises(CompilationError, match="immutable"):
            candidate.control_guard = ".false."
        transaction.adopt()
    adopted = parent.coordinators["retained_owner"]
    assert adopted is not candidate
    assert adopted.builder is parent
    assert adopted.execution_plan._owner is adopted
    assert adopted.execution_plan.validate().identity == plan.identity
    assert adopted.unit_emission(adopted.units[0]).native_preflight == {"available": True}
    assert owner.unit_emission(owner.units[0]).native_preflight == {"available": False}
    with pytest.raises(CompilationError, match="immutable"):
        adopted.units[0].arrays.clear()


def test_mutated_borrowed_numerical_authority_cannot_validate_or_publish(tmp_path):
    from compiler.tests.support.numerical_regions import emit_combined_team_fixture

    _paths, _outputs, _report, owner = emit_combined_team_fixture(tmp_path)
    plan = owner.execution_plan
    proof = plan._numerical_authorities[0]
    assert proof.extraction.completion is proof.completion
    with pytest.raises(CompilationError, match="immutable"):
        proof.extraction.completion["changed_after_admission"] = True
    assert plan.validate() is plan
    owner.builder.finish()


def test_immutable_handoff_avoids_rederiving_geometry_and_rechecks_registry(tmp_path, monkeypatch):
    from compiler.frontend.numerical_preparation import PreparedOriginalRegion

    with scanned(tmp_path, config=OffloadConfig(policy="sections", scope_execution="graph")) as (_owner, plan):
        proof = plan._numerical_authorities[0]

        def unexpected(*_args, **_kwargs):
            pytest.fail("immutable emission handoff recomputed original numerical geometry")

        monkeypatch.setattr(PreparedOriginalRegion, "validate", unexpected)
        assert plan.validate() is plan
        assert plan.validate() is plan
        registry = proof._provider._analysis._numerical_preparations
        assert registry.pop(proof.identity) is proof
        try:
            with pytest.raises(CompilationError, match="registered immutable proof handoff"):
                plan.validate()
        finally:
            registry[proof.identity] = proof
        assert plan.validate() is plan


def test_diagnostics_and_emission_share_the_checked_operation_tree(tmp_path):
    _path, _outputs, report = emit(tmp_path)
    scope, = report["scopes"]
    plan = scope["checked_execution_plan"]
    assert plan["entry"] == report["entry"]
    assert plan["gpu_leaves"] == scope["gpu_leaves"]
    for unit, emitted in zip(plan["procedures"][0]["units"], scope["planning_segments"], strict=True):
        assert unit["operations"] == emitted["operations"]
        assert unit["ordered_definitions"] == emitted["ordered_definitions"]
    assert plan["procedures"][0]["boundaries"] == scope["boundaries"]


def test_window_observations_are_placement_only_but_semantic_configuration_still_invalidates(tmp_path):
    from compiler.tests.test_prepared_original_regions import ENTRY, OPTIONS, fixture

    _path, analysis, _provider, selection = fixture(tmp_path)
    config = OffloadConfig(policy="auto", scope_execution="reached", exact_window={"calibration": True})
    provider = NumericalPreparationProvider(analysis, options=OPTIONS, config=config, environment_root=ENTRY)
    prepared = provider.prepare(analysis, ENTRY, selection)
    production = replace(config, exact_window={"evidence": {"untrusted": "not numerical authority"}})
    prepared.validate_configuration(options=OPTIONS, config=production, environment_root=ENTRY)
    provider.validate_configuration(options=OPTIONS, config=production, environment_root=ENTRY)
    with pytest.raises(CompilationError, match="policy changed"):
        prepared.validate_configuration(options=OPTIONS, config=replace(production, host_threads=2),
                                        environment_root=ENTRY)
