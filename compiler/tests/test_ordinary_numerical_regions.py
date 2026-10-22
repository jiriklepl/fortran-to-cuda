"""Ordinary lexical owners consume source-issued numerical pairs, without a plan."""

from copy import copy, deepcopy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.reached_windows import plan_reached_window
from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.source import ScopeBuilder
from compiler.tests.support.numerical_regions import CHILD, FACT, MATH_SOURCE, OWNER_SOURCE, ROOTS

ORDINARY_CHILD_SOURCE = """module quartz_numerics
use,intrinsic::ieee_exceptions,only:ieee_set_halting_mode,ieee_invalid
implicit none
integer::bridge_visits=0,middle_visits=0
real(8)::coefficient(-1:0)=[2.0_8,3.0_8]
contains
subroutine bridge(a,b,c,n,tile)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n,tile
integer::i
bridge_visits=bridge_visits+1
i=0
!$omp parallel
call ieee_set_halting_mode(ieee_invalid,.false.)
!$omp end parallel
!$omp parallel private(i)
!$omp do schedule(static)
do i=-3,n
b(i)=b(i)+a(i)*coefficient(-1)
enddo
!$omp end do
!$omp do schedule(static)
do i=-2,n
b(i)=b(i)+a(i)*coefficient(0)
enddo
!$omp end do nowait
!$omp end parallel
if(n>=-3) then
call touch_face(b)
endif
!$omp parallel private(i)
!$omp do schedule(static)
do i=-3,n
c(i)=2.0_8*b(i)+real(i+1,8)
enddo
!$omp end do
!$omp end parallel
end subroutine
subroutine touch_face(b)
real(8),intent(inout)::b(-4:)
middle_visits=middle_visits+1
b(-3)=b(-3)+0.5_8
end subroutine
end module
"""


def ordinary_numerical_fixture(directory, *, precision=8, device_budget=None,
                               child_source=ORDINARY_CHILD_SOURCE, captures=None):
    """Keep the public driver ABI, with an ordinary owner at original bridge."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = tuple(directory / name for name in ("quartz_math.f90", "quartz_numerics.f90", "quartz_owner.f90"))
    for path, source in zip(paths, (MATH_SOURCE, child_source, OWNER_SOURCE), strict=True):
        path.write_text(source.replace("real(8)", f"real({precision})")
                        .replace("_8", f"_{precision}").replace(",8)", f",{precision})"))
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {ROOTS[0]: deepcopy(FACT), ROOTS[1]: deepcopy(FACT),
                          ROOTS[2]: {**deepcopy(FACT), "initialized": "none"},
                          "quartz_numerics::coefficient": deepcopy(FACT)}}
    for root, values in (captures or {}).items():
        facts["captures"].setdefault(root, deepcopy(FACT)).update(values)
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    builder = ScopeBuilder(paths, CHILD, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    return paths, builder


def emit_ordinary_numerical_fixture(directory, **options):
    paths, builder = ordinary_numerical_fixture(directory, **options)
    outputs, report = builder.run()
    return paths, outputs, report, builder


def numerical_groups(builder):
    return tuple(group.original_nodes for group in grouped_original_sequence(builder.entry.execution.content, {})
                 if isinstance(group, NativeGroupCandidate)
                 and any(type(node).__name__ == "Block_Nonlabel_Do_Construct" for node in group.original_nodes))


def test_pair_preparation_has_no_capture_or_emission_authority(tmp_path, monkeypatch):
    paths, builder = ordinary_numerical_fixture(tmp_path)
    selection, _accumulation = numerical_groups(builder)
    before = tuple(path.read_bytes() for path in paths)

    def forbidden(*_args, **_kwargs):
        pytest.fail("source pair preparation acquired capture or emission authority")

    monkeypatch.setattr(ScopeBuilder, "capture", forbidden)
    monkeypatch.setattr("compiler.emission.generate_sources", forbidden)
    region = builder.inline.prepare_numerical_region(selection)
    assert builder.validate_reached_window() is None
    assert region.prepared.offload.available
    assert len(region.effects.steps) == 2
    assert region.prepared.extraction.scalar_element_captures == ()
    assert builder.inline.prepare_numerical_region(selection) is region
    assert builder.resolved_numerical_region(builder.analysis, CHILD, selection) is region
    assert not builder.outputs
    assert not builder.edits
    assert not builder.inline.regions
    assert not builder.inline.used
    assert tuple(path.read_bytes() for path in paths) == before


@pytest.mark.parametrize("mutation", ["source", "binding", "configuration", "prepared", "effects"])
def test_ordinary_pair_rejects_stale_or_copied_authority_before_publication(tmp_path, mutation):
    paths, builder = ordinary_numerical_fixture(tmp_path)
    selection, _accumulation = numerical_groups(builder)
    region = builder.inline.prepare_numerical_region(selection)
    if mutation == "source":
        paths[1].write_text(paths[1].read_text() + "! source changed\n")
        assert builder.resolved_numerical_region(builder.analysis, CHILD, selection) is region
        with pytest.raises(CompilationError, match="source changed"):
            builder.finish()
    elif mutation == "binding":
        scope = builder.entry.scope
        binding = copy(scope.bindings["n"])
        binding.attributes |= {"volatile"}
        with pytest.raises(CompilationError):
            scope.bindings["n"] = binding
        assert builder.resolved_numerical_region(builder.analysis, CHILD, selection) is region
    else:
        if mutation == "configuration":
            builder.config = replace(builder.config, host_threads=2)
        else:
            builder.inline.original_numerical_regions[tuple(id(node) for node in selection)] = replace(
                region, **{mutation: copy(getattr(region, mutation))})
        with pytest.raises(CompilationError):
            builder.resolved_numerical_region(builder.analysis, CHILD, selection)
        with pytest.raises(CompilationError):
            builder.finish()
    assert not builder.outputs
    assert not builder.edits


def test_ordinary_pair_registry_rolls_back_with_candidate_scope_and_transaction(tmp_path):
    _paths, builder = ordinary_numerical_fixture(tmp_path)
    selection, _accumulation = numerical_groups(builder)
    checkpoint = builder.scope_checkpoint()
    builder.inline.prepare_numerical_region(selection)
    builder.restore_scope_checkpoint(checkpoint)
    assert builder.resolved_numerical_region(builder.analysis, CHILD, selection) is None
    with GenerationTransaction(builder) as transaction:
        transaction.builder.inline.prepare_numerical_region(selection)
        assert transaction.builder.inline.original_numerical_regions
        assert not builder.inline.original_numerical_regions
    assert not builder.inline.original_numerical_regions
    assert not builder.outputs
    assert not builder.edits


def test_transaction_rejects_original_binding_mutation_and_rolls_back_prepared_pair(tmp_path):
    _paths, parent = ordinary_numerical_fixture(tmp_path)
    selection, _accumulation = numerical_groups(parent)
    binding = parent.entry.scope.bindings["n"]
    replacement = copy(binding)
    replacement.attributes |= {"volatile"}

    def attempt():
        with GenerationTransaction(parent) as transaction:
            candidate = transaction.builder
            candidate.inline.prepare_numerical_region(selection)
            candidate.entry.scope.bindings["n"] = replacement
            transaction.adopt(candidate)

    with pytest.raises(CompilationError):
        attempt()
    assert parent.entry.scope.bindings["n"] is binding
    assert not parent.inline.original_numerical_regions
    assert not parent.outputs
    assert not parent.edits


def test_bound_reached_plan_cannot_inherit_an_ordinary_pair(tmp_path):
    source = """module quartz_numerics
implicit none
integer::bridge_visits=0,middle_visits=0
real(8)::coefficient(-1:0)=[2.0_8,3.0_8]
contains
subroutine bridge(a,b,c,n,tile)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n,tile
integer::i
do i=-3,n
b(i)=2.0_8*a(i)
enddo
end subroutine
end module
"""
    _paths, builder = ordinary_numerical_fixture(tmp_path, child_source=source)
    loop, = (node for node in builder.entry.execution.content
             if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    selection = (loop,)
    builder.inline.prepare_numerical_region(selection)
    proof = plan_reached_window(builder.analysis, CHILD, selection, ROOTS[:2], builder.facts)
    bound = builder.bind_reached_window(proof, selection)
    assert bound.resolved_numerical_region(bound.analysis, CHILD, selection) is None
    with pytest.raises(CompilationError, match="registered reached preparation"):
        bound.inline.prepare_numerical_region(selection)


def test_ordinary_owner_keeps_producer_native_middle_and_accumulation_together(tmp_path):
    paths, outputs, report, builder = emit_ordinary_numerical_fixture(tmp_path)
    assert builder.validate_reached_window() is None
    assert report["scope_count"] == 1, report["boundaries"]
    assert len(builder.inline.original_numerical_regions) == 2
    assert {len(region.effects.steps) for region in builder.inline.original_numerical_regions.values()} == {1, 2}
    assert all(not region.prepared.extraction.scalar_element_captures
               for region in builder.inline.original_numerical_regions.values())
    replacement = outputs[report["sources"][str(paths[1])]["replacement"]]
    assert "call touch_face(b)" in replacement.lower()
    assert "call ieee_set_halting_mode(ieee_invalid,.false.)" in replacement.lower()
    assert "fort_team_environment_supported" in replacement
    assert builder.facts["captures"][ROOTS[2]]["initialized"] == "none"
