"""Exact original-read admission after an unchanged native prefix."""

from collections.abc import Mapping
from copy import copy
from dataclasses import replace
from functools import partial

import pytest

from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.frontend.selection_activation import (
    SelectionActivationProof,
    prove_selection_activation,
    selection_activation_demands,
)
from compiler.ir import CompilationError
from compiler.scopes.closed_selection import ClosedIntervalOwner
from compiler.tests.test_interval_draft import SOURCE
from compiler.tests.test_original_interval_owner import builder, call_pair
from compiler.tests.test_selection_cost import build_validation, process_fixture  # noqa: F401

PREFIX = """integer::i
!$omp parallel do private(i)
do i=0,7
out(i)=0.d0
enddo
!$omp end parallel do
visits=visits+1
"""


def activation_fixture(path, *, config=None, prefix=PREFIX):
    selected = builder(path, prefix=prefix, config=config)
    selected.facts["captures"]["interval_fields::out"]["initialized"] = "none"
    site = call_pair(selected)
    owner = ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)
    return selected, owner, site


def emit_activation_fixture(path, *, config=None):
    selected, owner, site = activation_fixture(path, config=config)
    assert owner.run(site.originals)
    outputs, report = owner.builder.finish()
    return selected, owner, outputs, report


def test_original_native_prefix_does_not_seed_selected_definitions(tmp_path):
    selected, owner, outputs, report = emit_activation_fixture(tmp_path / "prefix.f90")
    proof = owner.emission.selection_prefix
    assert type(proof) is SelectionActivationProof
    assert all(state["initialized"] == "none" for state in proof.public()["states"])
    assert proof.activation_sections("interval_fields::out") is None
    assert selected.facts["captures"]["interval_fields::out"]["initialized"] == "none"
    assert owner.builder.capture(selected.analysis._binding(selected.entry.scope, "out"))["initialized"] == "none"
    assert owner.execution_plan.validate() is owner.execution_plan
    assert owner.execution_plan.numerical_invocations == 3
    text = outputs[report["sources"][str(tmp_path / "prefix.f90")]["replacement"]]
    assert text.count("out(i)=0.d0") == 1
    assert text.index("out(i)=0.d0") < text.index("block")
    assert "fort_scope_admit_source_reads_v1" in text
    assert "fort_scope_register_sections" not in text


@pytest.mark.parametrize("empty", ["then", "else"])
def test_empty_native_branch_arm_cannot_grant_intersection(tmp_path, empty):
    group = PREFIX.partition("visits=")[0]
    group = group.replace("integer::i\n", "")
    prefix = "integer::i\nif(correcting) then\n" + (group if empty == "else" else "")
    prefix += "else\n" + (group if empty == "then" else "") + "endif\n"
    selected, owner, _site = activation_fixture(tmp_path / "prefix.f90", prefix=prefix)
    proof = prove_selection_activation(owner, selection_activation_demands(owner))
    assert proof.capture_state("interval_fields::out")["initialized"] == "none"
    assert proof.activation_sections("interval_fields::out") is None


@pytest.mark.parametrize("mutation", ["facts", "configuration", "source", "selection", "token", "state"])
def test_activation_rejects_stale_or_forged_authority(tmp_path, mutation):
    selected, owner, site = activation_fixture(tmp_path / "prefix.f90")
    demanded = selection_activation_demands(owner)
    proof = prove_selection_activation(owner, demanded)
    if mutation == "facts":
        selected.facts["captures"]["interval_fields::out"]["initialized"] = "whole"
        action = partial(proof.validate_configuration, selected)
    elif mutation == "configuration":
        selected.config = replace(selected.config, host_threads=3)
        action = partial(proof.validate_configuration, selected)
    elif mutation == "source":
        path = tmp_path / "prefix.f90"
        path.write_text(path.read_text().replace("out(i)=0.d0", "out(i)=1.d0"))
        # The immutable session remains the analysis authority; publication
        # separately authenticates current bytes before an artifact escapes.
        proof.validate(selected.analysis, selected.entry.qualified, site.originals, demanded)
        action = selected.analysis.verify_publication
    elif mutation == "selection":
        action = partial(proof.validate, selected.analysis, selected.entry.qualified,
                         (copy(site.originals[0]), *site.originals[1:]), demanded)
    elif mutation == "token":
        forged = replace(proof)
        action = partial(forged.validate, selected.analysis, selected.entry.qualified, site.originals, demanded)
    else:
        from compiler.frontend.structured_effects import _freeze, _thaw
        record = _thaw(proof._record)
        record["states"][0]["initialized"] = "whole"
        object.__setattr__(proof, "_record", _freeze(record))
        action = partial(proof.validate, selected.analysis, selected.entry.qualified, site.originals, demanded)
    with pytest.raises(CompilationError):
        action()


def test_unknown_native_prefix_stays_original_without_any_definition_credit(tmp_path):
    selected, owner, site = activation_fixture(tmp_path / "prefix.f90", prefix=PREFIX + "call opaque()\n")
    assert owner.run(site.originals)
    outputs, report = owner.builder.finish()
    proof = owner.emission.selection_prefix
    assert all(state["initialized"] == "none" for state in proof.public()["states"])
    text = outputs[report["sources"][str(tmp_path / "prefix.f90")]["replacement"]]
    assert text.count("call opaque()") == 1
    assert text.index("call opaque()") < text.index("fort_scope_create")
    assert "fort_scope_admit_source_reads_v1" in text
    assert selected.facts["captures"]["interval_fields::old"]["initialized"] == "whole"


def test_automatic_prefix_interval_uses_same_native_and_resident_contract(tmp_path, build_validation):  # noqa: F811
    contracts = []
    for variant in ("native", "resident"):
        config = OffloadConfig(policy="auto", scope_execution="reached", exact_window={
            "coverage": "checked_interval", "calibration": variant,
            "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True},
            "validation_policy": {"checked_policy_fixture": True}})
        _selected, owner, _outputs, report = emit_activation_fixture(tmp_path / "prefix.f90", config=config)
        scope, = report["scopes"]
        contracts.append(scope["exact_window"]["contract"])
        assert owner.execution_plan.numerical_invocations == 3
    assert contracts[0] == contracts[1]


def test_public_discovery_adopts_bound_activation_builder(tmp_path, build_validation):  # noqa: F811
    config = OffloadConfig(policy="auto", scope_execution="reached", exact_window={
        "coverage": "checked_interval", "calibration": "resident",
        "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True},
        "validation_policy": {"checked_policy_fixture": True}})
    # Place the initialization before every useful bounded candidate. The
    # ordinary source discovery, transaction and final publication all run.
    selected, _owner, _site = activation_fixture(tmp_path / "prefix.f90", config=config,
        prefix=PREFIX + "visits=visits+1\n" * 9)
    outputs, report = selected.run()
    proof = selected.validate_selection_activation()
    assert type(proof) is SelectionActivationProof
    assert proof.capture_state("interval_fields::out")["initialized"] == "none"
    assert selected.inline.builder is selected
    assert selected.region_owners[selected.entry.qualified] is selected.inline
    assert report["scopes"]
    text = outputs[report["sources"][str(tmp_path / "prefix.f90")]["replacement"]]
    assert text.count("out(i)=0.d0") == 1
    assert "fort_scope_admit_source_reads_v1" in text
    assert "fort_scope_register_sections" not in text


def test_native_out_after_initialization_forgets_output_coverage(tmp_path):
    source = SOURCE.replace("end module", """subroutine invalidate(value)
real(8),intent(out)::value(-1:)
end subroutine
end module""")
    selected = builder(tmp_path / "prefix.f90", source=source,
                       prefix=PREFIX + "call invalidate(out)\n")
    selected.facts["captures"]["interval_fields::out"]["initialized"] = "none"
    site = call_pair(selected)
    owner = ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)
    proof = prove_selection_activation(owner, selection_activation_demands(owner))
    assert proof.capture_state("interval_fields::out")["initialized"] == "none"
    assert proof.activation_sections("interval_fields::out") is None


def test_issuer_rejects_omitted_original_prefix(tmp_path):
    _selected, owner, _site = activation_fixture(tmp_path / "prefix.f90")
    owner.closed_prefix = ()
    with pytest.raises(CompilationError, match="exact original prefix"):
        prove_selection_activation(owner, selection_activation_demands(owner))


def test_native_prefix_does_not_grant_repeated_entry_without_backedge_proof(tmp_path):
    selected = builder(tmp_path / "prefix.f90", repeated=True, prefix="visits=visits+1\n")
    site = call_pair(selected)
    owner = ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)
    with pytest.raises(CompilationError, match="backedge definition state"):
        owner._prepare_lexical(site.originals)
    assert not selected.outputs
    assert not selected.edits


@pytest.mark.parametrize("mutation", ["policy", "initialization_binding", "prefix_identity"])
def test_registered_token_rejects_mutated_internal_authority(tmp_path, mutation):
    prefix = PREFIX.replace("integer::i\n", "integer::i,prepared\nprepared=8\n")
    selected, owner, site = activation_fixture(tmp_path / "prefix.f90", prefix=prefix)
    demanded = selection_activation_demands(owner)
    proof = prove_selection_activation(owner, demanded)
    if mutation == "policy":
        object.__setattr__(proof, "_configuration", "different policy")
    elif mutation == "initialization_binding":
        binding = selected.analysis._binding(selected.entry.scope, "prepared")
        object.__setattr__(proof, "initializations", (binding,))
    else:
        object.__setattr__(proof, "_prefix", ())
    with pytest.raises(CompilationError, match="payload changed"):
        proof.validate(selected.analysis, selected.entry.qualified, site.originals, demanded)


def test_module_threadprivate_storage_remains_ineligible(tmp_path):
    source = SOURCE.replace("integer::visits=0", "integer::visits=0\n!$omp threadprivate(out)")
    selected = builder(tmp_path / "prefix.f90", source=source, prefix=PREFIX)
    site = call_pair(selected)
    owner = ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)
    with pytest.raises(CompilationError, match="module OpenMP ownership"):
        prove_selection_activation(owner, selection_activation_demands(owner))
    assert not selected.edits


def test_selected_reads_require_original_alias_safe_registration(tmp_path):
    _selected, owner, outputs, report = emit_activation_fixture(tmp_path / "prefix.f90")
    text = outputs[report["sources"][str(tmp_path / "prefix.f90")]["replacement"]]
    first_claim = text.index("fort_status = fort_scope_admit_source_reads_v1(")
    assert text.index("fort_scope_register(") < first_claim
    assert text.index("is_contiguous(") < first_claim
    assert text.index("fort_scope_layout_get(") < first_claim
    assert "0_c_int, fort_buffer_" in text
    for member in owner.execution_plan.members:
        for unit in owner.execution_plan.units(member):
            requirements = getattr(unit.scope, "source_reads", {})
            proofs = requirements.values() if isinstance(requirements, Mapping) else (requirements,)
            for proof in proofs:
                if proof is not None:
                    assert proof.validate(member.builder.analysis, member.routine.qualified,
                                          proof.prepared._selection, proof.prepared) is proof


def test_source_read_token_cannot_move_to_another_original_call(tmp_path):
    from compiler.scopes.source_reads import validate_source_reads
    _selected, owner, _outputs, _report = emit_activation_fixture(tmp_path / "prefix.f90")
    member = next(member for member in owner.execution_plan.members
                  if member.routine.qualified == "interval_fields::momentum")
    pairs = [(call, requirements[call.procedure] if isinstance(requirements, Mapping) else requirements)
             for unit in owner.execution_plan.units(member)
             for requirements in (unit.scope.source_reads,) for call in unit.scope.calls
             if call.region is not None]
    assert len(pairs) == 2
    (first, first_proof), (second, second_proof) = pairs
    assert first_proof.procedure == second_proof.procedure
    validate_source_reads(member.builder, first, first_proof)
    with pytest.raises(CompilationError, match="exact original numerical call"):
        validate_source_reads(member.builder, second, first_proof)


def test_zero_state_activation_does_not_project_explicit_shape_dummy_geometry(tmp_path):
    from compiler.tests.test_source_scopes import FACT
    source = SOURCE.replace("advance(n,correcting)", "advance(n,correcting,beta)")
    source = source.replace("integer,intent(in)::n", "real(8),intent(in)::beta(1:3)\ninteger,intent(in)::n", 1)
    selected = builder(tmp_path / "prefix.f90", source=source, prefix=PREFIX)
    selected.facts["captures"]["argument::beta"] = dict(FACT)
    site = call_pair(selected)
    owner = ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)
    proof = prove_selection_activation(owner, selection_activation_demands(owner))
    assert proof.capture_state("argument::beta")["initialized"] == "none"
    assert proof.activation_sections("argument::beta") is None
    assert proof.public()["new_lifetime_authority"] is False


def test_borrowed_native_metadata_keeps_reservation_without_numeric_capture(tmp_path):
    from compiler.tests.test_original_interval_owner import MIXED_SOURCE
    source = MIXED_SOURCE.replace("integer::visits=0", """type marker_t
integer::index
real(8)::value
end type
type(marker_t)::markers(1)
integer::visits=0""").replace("v(i)=v(i)+sum(a)", "v(i)=v(i)+markers(1)%value")
    selected = builder(tmp_path / "prefix.f90", source=source, prefix=PREFIX)
    site = call_pair(selected)
    owner = ClosedIntervalOwner(selected, site.originals, site.prefix, site.controls)
    assert owner.run(site.originals)
    outputs, report = owner.builder.finish()
    proof = owner.emission.selection_prefix
    root = "interval_fields::markers"
    assert root not in selected.facts["captures"]
    assert root not in {state["resource"] for state in proof.public()["states"]}
    assert root in owner.resources
    member = next(member for member in owner.execution_plan.members
                  if member.routine.qualified == "interval_fields::momentum")
    assert not member.boundaries
    assert any(root in unit.host_metadata for unit in owner.execution_plan.units(member))
    text = outputs[report["sources"][str(tmp_path / "prefix.f90")]["replacement"]]
    assert "FORT_SCOPE_BYTES" in text
    assert "fort_scope_admit_source_reads_v1" in text
