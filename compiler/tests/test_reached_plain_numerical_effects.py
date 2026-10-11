"""Mapped serial pointwise loops retain conditional numerical effects."""

from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend import native_group_structure
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.ir import CompilationError
from compiler.tests.test_combined_numerical_team import (
    ENVIRONMENT_TEAM,
    ENVIRONMENT_USE,
    emit_combined_team_fixture,
)
from compiler.tests.test_reached_numerical_effects import (
    CHILD,
    CHILD_SOURCE,
    ENTRY,
    ROOTS,
    reached_numerical_fixture,
)


def plain_consumer(builder):
    """Use the exact intact final original DO, not its body or header."""
    return (next(node for node in reversed(builder.analysis.routines[CHILD].execution.content)
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct"),)


def environment_source():
    return CHILD_SOURCE.replace("implicit none\n", ENVIRONMENT_USE + "implicit none\n", 1).replace(
        "i=0\n", "i=0\n" + ENVIRONMENT_TEAM, 1)


def test_plain_consumer_has_exact_registered_effects_without_output_entry_definition(tmp_path):
    _paths, builder, proof, originals, _provider = reached_numerical_fixture(
        tmp_path, child_source=environment_source())
    selection = plain_consumer(builder)
    prepared, effects = proof.numerical_region(CHILD, selection)
    assert prepared.validate(builder.analysis, CHILD, selection) is prepared
    assert effects.validate(builder.analysis, CHILD, selection, prepared) is effects
    assert len(effects.steps) == 1
    output, = (resource for resource in effects.sections.resources if resource.resource == ROOTS[2])
    assert not output.reads
    assert output.overwrites
    assert builder.facts["captures"][ROOTS[2]]["initialized"] == "none"
    assert not effects.public()["input_definitions_authorized"]
    assert not effects.public()["unconditional_return_definitions_authorized"]
    assert proof.validate(builder.analysis, ENTRY, originals) is proof
    with pytest.raises(CompilationError):
        copy(prepared).validate(builder.analysis, CHILD, selection)
    binding = builder.analysis.routines[CHILD].scope.bindings["b"]
    builder.analysis.routines[CHILD].scope.bindings["b"] = replace(
        binding, attributes=binding.attributes | {"volatile"})
    with pytest.raises(CompilationError):
        prepared.validate(builder.analysis, CHILD, selection)
    with pytest.raises(CompilationError):
        proof.validate(builder.analysis, ENTRY, originals)


def test_plain_environment_guard_closes_before_original_without_whole_output_read(tmp_path):
    _paths, _outputs, report, owner = emit_combined_team_fixture(tmp_path)
    child = owner.members[CHILD]
    consumer = child.units[-1]
    call, = consumer.scope.calls
    effects = child.builder.call_numerical_effect(call)
    assert effects is not None
    assert call.region.requires_numerical_environment
    assert consumer.scope.close_guards[id(call.node)] is effects
    assert not consumer.scope.guarded
    assert not consumer.scope.native
    assert ROOTS[2] not in consumer.definition_summary.required_whole
    assert not child.boundaries
    assert len(report["scopes"][0]["gpu_leaves"]) == 3
    assert owner.builder.facts["captures"][ROOTS[2]]["initialized"] == "none"


def test_plain_region_aggregate_source_budget_cannot_become_native_support_decline(tmp_path, monkeypatch):
    _paths, builder, proof, originals, _provider = reached_numerical_fixture(tmp_path)
    assert proof.numerical_region(CHILD, plain_consumer(builder)) is not None
    monkeypatch.setattr(native_group_structure, "NATIVE_GROUP_SCAN_LIMIT", 1)
    with pytest.raises(CompilationError, match="aggregate coordinate/source scan budget exhausted"):
        proof.validate(builder.analysis, ENTRY, originals)


def test_native_indexed_scatter_keeps_original_proof_without_speculative_preparation(tmp_path, monkeypatch):
    # This is a supported native operation, not an elementwise numerical
    # candidate. In particular, attempting its lowering must not turn an
    # optional numerical provider into a new hard compilation failure.
    source = """module quartz_numerics
implicit none
contains
subroutine bridge(a,b,c,n,tile)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n,tile
integer::i
do i=-3,n
b(int(a(i)))=b(int(a(i)))+1.0_8
enddo
end subroutine
end module
"""
    calls = []

    def unexpected_prepare(self, analysis, procedure, selection, *, consume=None):
        calls.append((procedure, selection))
        raise AssertionError("native indexed scatter must keep its existing source proof")

    monkeypatch.setattr(NumericalPreparationProvider, "prepare", unexpected_prepare)
    paths, builder, proof, originals, _provider = reached_numerical_fixture(tmp_path, child_source=source)
    assert not calls
    assert builder.analysis.summarize(CHILD)["complete"]
    assert proof.numerical_region(CHILD, plain_consumer(builder)) is None
    assert proof.validate(builder.analysis, ENTRY, originals) is proof
    assert paths[1].read_text() == source
