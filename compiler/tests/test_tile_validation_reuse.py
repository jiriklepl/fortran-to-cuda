"""A geometry receipt cannot replace mutable source, facts, or typed authority."""
from copy import copy

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_device_tasks import source_device_task
from compiler.frontend.tile_private import prove_tile_private_team
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_graph_source_owner import FACT
from compiler.tests.test_tile_private import SOURCE
from compiler.tests.test_tile_private_team import team_fixture


def fixture(tmp_path, *, scalar=False):
    source = SOURCE
    if scalar:
        source = source.replace("advance(a,b,n,t)", "advance(a,b,n,t,scale)")
        source = source.replace("integer,intent(in)::n,t", "integer,intent(in)::n,t\nreal(8),intent(in)::scale")
        source = source.replace("b(i)=b(i)+(tmp(li+1)-tmp(li-1))", "b(i)=b(i)+scale*(tmp(li+1)-tmp(li-1))")
    path, _analysis, routine, _nodes = team_fixture(tmp_path, source)
    facts = {"schema_version": 1, "participation": "serial", "sources": dict(_analysis.sources),
             "captures": {"argument::" + name: dict(FACT) for name in ("a", "b", "scale")}}
    builder = ScopeBuilder([path], routine.qualified, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    analysis, routine = builder.analysis, builder.entry
    nodes = tuple(routine.execution.content)
    proof = prove_tile_private_team(analysis, routine, nodes)
    return builder, routine, nodes, proof


def test_sealed_team_revalidation_preserves_exact_record_without_relowering(tmp_path, monkeypatch):
    builder, routine, nodes, proof = fixture(tmp_path)
    before = proof.public()
    assert builder.analysis._tile_team_validation_receipts
    monkeypatch.setattr("compiler.frontend.lowering.lower_tile_private",
                        lambda *_args, **_kwargs: pytest.fail("immutable team geometry was redundantly lowered"))
    assert proof.validate() is proof
    assert prove_tile_private_team(builder.analysis, routine, nodes) is proof
    assert proof.public() == before


def test_standalone_mutable_analysis_keeps_full_derivation_and_binding_checks(tmp_path, monkeypatch):
    _, analysis, routine, nodes = team_fixture(tmp_path)
    proof = prove_tile_private_team(analysis, routine, nodes)
    assert not getattr(analysis, "_tile_team_validation_receipts", {})
    import compiler.frontend.lowering as lowering
    original = lowering.lower_tile_private
    calls = []
    def lower(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(lowering, "lower_tile_private", lower)
    proof.validate()
    assert calls
    proof.fields[0][0].kind = 4
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("mutation", ["namespace", "policy", "field", "unit_registry", "team_registry", "unit_record"])
def test_receipt_does_not_retag_changed_source_authority_or_proof_objects(tmp_path, mutation):
    builder, _, _, proof = fixture(tmp_path)
    analysis = builder.analysis
    if mutation == "namespace":
        registry = analysis.modules
        replacement = dict(registry)
        with pytest.raises(CompilationError, match="original source registry is immutable"):
            analysis.modules = replacement
        assert analysis.modules is registry
        assert proof.validate() is proof
        # Bypass only the setter to retain the original receipt-consumer negative.
        object.__setattr__(analysis, "modules", replacement)
    elif mutation == "policy":
        analysis.operation_limit += 1
    elif mutation == "field":
        binding, written = proof.fields[0]
        object.__setattr__(proof, "fields", ((copy(binding), written), *proof.fields[1:]))
    elif mutation == "unit_registry":
        del analysis._tile_private_proofs[proof.units[0].identity]
    elif mutation == "team_registry":
        analysis._tile_private_team_proofs[proof.identity] = copy(proof)
    else:
        object.__setattr__(proof.units[0], "_record", "{}")
    with pytest.raises(CompilationError):
        proof.validate()


def test_receipt_rechecks_current_frozen_binding_descriptor(tmp_path):
    _, _, _, proof = fixture(tmp_path)
    # Exercise the fence even against a forced Python-level bypass of the
    # frozen source graph's ordinary mutation rejection.
    object.__setattr__(proof.fields[0][0], "kind", 4)
    with pytest.raises(CompilationError):
        proof.validate()


def test_foreign_projection_cannot_borrow_an_original_team_receipt(tmp_path):
    builder, _, _, proof = fixture(tmp_path)
    original = builder.analysis
    projected = original.private_projection()
    assert projected is not original
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        proof.validate(analysis=projected)


def test_scalar_entry_fact_is_revalidated_after_geometry_receipt(tmp_path):
    builder, routine, nodes, proof = fixture(tmp_path, scalar=True)
    task = source_device_task(builder.analysis, routine, nodes, builder=builder)
    assert task.team is proof
    assert task.validate() is task
    builder.facts["captures"]["argument::scale"]["initialized"] = "none"
    with pytest.raises(CompilationError, match="whole-defined"):
        task.validate()


def test_live_file_publication_check_is_not_replaced_by_session_geometry(tmp_path):
    builder, _, _, proof = fixture(tmp_path)
    path = next(iter(builder.analysis.sources))
    with open(path, "a") as stream:
        stream.write("\n! changed after source capture\n")
    # Captured source is immutable within the session; publication still
    # checks the actual disk bytes, exactly as before the receipt existed.
    assert proof.validate() is proof
    with pytest.raises(CompilationError):
        builder.analysis.verify_publication()


def test_exact_unit_revalidation_reuses_registered_whole_team_geometry(tmp_path, monkeypatch):
    builder, routine, _nodes, team = fixture(tmp_path)
    before = tuple(unit.public() for unit in team.units)
    monkeypatch.setattr("compiler.frontend.tile_private._derive",
                        lambda *_args: pytest.fail("registered immutable team unit was redundantly derived"))
    for unit in team.units:
        assert unit.validate(builder.analysis, routine, unit._node) is unit
    assert tuple(unit.public() for unit in team.units) == before


@pytest.mark.parametrize("fault", ["no_receipt", "team_registry", "team_units", "team_record", "receipt_issuer",
                                   "foreign_team"])
def test_unit_without_exact_team_receipt_keeps_full_source_derivation(tmp_path, monkeypatch, fault):
    from dataclasses import replace

    import compiler.frontend.tile_private as tile

    builder, _, _, team = fixture(tmp_path)
    analysis, unit = builder.analysis, team.units[0]
    if fault == "no_receipt":
        analysis._tile_team_validation_receipts.clear()
    elif fault == "team_registry":
        analysis._tile_private_team_proofs[team.identity] = copy(team)
    elif fault == "team_units":
        object.__setattr__(team, "units", tuple(copy(item) for item in team.units))
    elif fault == "team_record":
        object.__setattr__(team, "_record", "{}")
    else:
        key, receipt = next(iter(analysis._tile_team_validation_receipts.items()))
        changed = replace(receipt, issuer=object()) if fault == "receipt_issuer" else replace(receipt, proof=object())
        analysis._tile_team_validation_receipts[key] = changed
    original, calls = tile._derive, []
    def derive(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(tile, "_derive", derive)
    assert unit.validate() is unit
    assert calls  # Independent unit authority remains valid, but cannot use the altered team.


@pytest.mark.parametrize("fault", ["unit_record", "unit_node", "unit_bindings", "unit_registry", "binding_kind",
                                   "namespace", "policy"])
def test_team_receipt_never_authenticates_substituted_unit_source_or_storage(tmp_path, fault):
    builder, _, _, team = fixture(tmp_path)
    analysis, unit = builder.analysis, team.units[0]
    if fault == "unit_record":
        object.__setattr__(unit, "_record", "{}")
    elif fault == "unit_node":
        object.__setattr__(unit, "_node", copy(unit._node))
    elif fault == "unit_bindings":
        object.__setattr__(unit, "bindings", (copy(unit.bindings[0]), *unit.bindings[1:]))
    elif fault == "unit_registry":
        analysis._tile_private_proofs[unit.identity] = copy(unit)
    elif fault == "binding_kind":
        object.__setattr__(unit.bindings[0], "kind", 4)
    elif fault == "namespace":
        registry = analysis.modules
        replacement = dict(registry)
        with pytest.raises(CompilationError, match="original source registry is immutable"):
            analysis.modules = replacement
        assert analysis.modules is registry
        assert unit.validate() is unit
        # Bypass only the setter to retain the original receipt-consumer negative.
        object.__setattr__(analysis, "modules", replacement)
    else:
        analysis.operation_limit += 1
    with pytest.raises(CompilationError):
        unit.validate()


def test_mutable_unit_still_rederives_without_a_team_receipt(tmp_path, monkeypatch):
    import compiler.frontend.tile_private as tile

    _, analysis, routine, nodes = team_fixture(tmp_path)
    team = prove_tile_private_team(analysis, routine, nodes)
    original, calls = tile._derive, []
    def derive(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(tile, "_derive", derive)
    assert team.units[0].validate() is team.units[0]
    assert calls
    assert not getattr(analysis, "_tile_team_validation_receipts", {})


def test_source_provider_artifacts_and_fresh_ir_checks_are_exact_with_unit_reuse(tmp_path, monkeypatch):
    import compiler.frontend.lowering as lowering
    import compiler.frontend.tile_private as tile
    import compiler.frontend.tile_validation as validation
    from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider

    builder, routine, nodes, team = fixture(tmp_path)
    task = source_device_task(builder.analysis, routine, nodes, builder=builder)
    original_derive, original_lower, reuse = tile._derive, lowering.lower_tile_private, validation.reuse_validated_unit
    counts = {"derive": 0, "lower": 0}
    def derive(*args):
        counts["derive"] += 1
        return original_derive(*args)
    def lower(*args, **kwargs):
        counts["lower"] += 1
        return original_lower(*args, **kwargs)
    monkeypatch.setattr(tile, "_derive", derive)
    monkeypatch.setattr(lowering, "lower_tile_private", lower)
    monkeypatch.setattr(validation, "reuse_validated_unit", lambda _unit: False)
    baseline = emit_tile_task_provider(team, readable_scalars=task)
    baseline_derivations = counts["derive"]
    counts.update(derive=0, lower=0)
    monkeypatch.setattr(validation, "reuse_validated_unit", reuse)
    current = emit_tile_task_provider(team, readable_scalars=task)
    assert current.public() == baseline.public()
    assert current.cuda_source == baseline.cuda_source
    assert current.fortran_source == baseline.fortran_source
    assert baseline_derivations > 0
    assert counts["derive"] == 0
    assert counts["lower"] > 0  # No lowered IR, source/IR equality or addressing check is cached.
