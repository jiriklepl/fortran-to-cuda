"""Native prefix state consumed by the original checked resident interval."""

from copy import copy
from dataclasses import replace
from functools import partial

import pytest

from compiler.frontend.selection_activation import (
    SelectionActivationProof,
    prove_selection_activation,
    selection_activation_demands,
)
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
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


def test_original_joined_initialization_registers_only_defined_sections(tmp_path):
    selected, owner, outputs, report = emit_activation_fixture(tmp_path / "prefix.f90")
    proof = owner.emission.selection_prefix
    assert type(proof) is SelectionActivationProof
    assert proof.capture_state("interval_fields::out")["initialized"] == "sections"
    section = proof.activation_sections("interval_fields::out")
    assert section.lower_bounds == (-1,)
    box, = section.overwrites
    assert box.axes[0].lower.value == 0
    assert box.axes[0].upper.value == 7
    assert not section.reads
    assert not section.writes
    assert selected.facts["captures"]["interval_fields::out"]["initialized"] == "none"
    assert owner.builder.capture(selected.analysis._binding(selected.entry.scope, "out"))["source_sections"] is section
    assert owner.execution_plan.validate() is owner.execution_plan
    assert owner.execution_plan.numerical_invocations == 3
    text = outputs[report["sources"][str(tmp_path / "prefix.f90")]["replacement"]]
    assert text.count("out(i)=0.d0") == 1
    assert text.index("out(i)=0.d0") < text.index("block")
    assert "fort_scope_register_sections" in text


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


@pytest.mark.parametrize("mutation", ["facts", "configuration", "source", "selection", "token", "geometry"])
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
        resource, = proof._sections
        object.__setattr__(proof, "_sections", (replace(resource, lower_bounds=(0,)),))
        action = partial(proof.validate, selected.analysis, selected.entry.qualified, site.originals, demanded)
    with pytest.raises(CompilationError):
        action()


def test_unknown_native_prefix_declines_before_any_owner_emission(tmp_path):
    selected, owner, site = activation_fixture(tmp_path / "prefix.f90", prefix=PREFIX + "call opaque()\n")
    with pytest.raises(CompilationError, match="complete preservation"):
        owner._prepare_lexical(site.originals)
    assert not selected.edits
    assert not selected.outputs
    assert not selected.scopes


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
    assert proof.capture_state("interval_fields::out")["initialized"] == "sections"
    assert selected.inline.builder is selected
    assert selected.region_owners[selected.entry.qualified] is selected.inline
    assert report["scopes"]
    text = outputs[report["sources"][str(tmp_path / "prefix.f90")]["replacement"]]
    assert text.count("out(i)=0.d0") == 1
    assert "fort_scope_register_sections" in text


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
        binding, = proof.initializations
        object.__setattr__(proof, "initializations", (copy(binding),))
    else:
        object.__setattr__(proof, "_prefix", ())
    with pytest.raises(CompilationError, match="payload changed"):
        proof.validate(selected.analysis, selected.entry.qualified, site.originals, demanded)
