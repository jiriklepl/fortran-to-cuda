"""Prepared original numerical closures retain ordered resident data flow."""

import re
from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.numerical_effects import prove_numerical_effects
from compiler.frontend.reached_windows import prove_reached_window
from compiler.ir import CompilationError
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.tests.support.numerical_regions import (
    CHILD,
    CHILD_SOURCE,
    ENTRY,
    ROOTS,
    emit_reached_numerical_fixture,
    reached_numerical_fixture,
)
from compiler.tests.test_inline_source_regions import original_joined_group


def test_source_preparation_is_registered_and_keeps_private_helpers_and_guards(tmp_path):
    _paths, builder, proof, originals, provider = reached_numerical_fixture(tmp_path)
    selection = original_joined_group(builder.analysis.routines[CHILD])
    prepared = provider.prepare(builder.analysis, CHILD, selection)
    assert prepared.validate(builder.analysis, CHILD, selection) is prepared
    assert prepared.public()["proof_role"] == "prepared_original_numerical_region"
    assert prepared.extraction.private_arrays == ("g",)
    assert "quartz_math::profile_cell" in prepared.extraction.numerical_helpers
    assert prepared.preconditions
    assert len(prepared.plan.regions) == 1
    assert proof.validate(builder.analysis, ENTRY, originals) is proof
    assert builder.facts["captures"][ROOTS[2]]["initialized"] == "none"


def test_source_numerical_effects_define_an_interior_without_claiming_halos(tmp_path):
    _paths, builder, _proof, _originals, provider = reached_numerical_fixture(
        tmp_path, captures={ROOTS[1]: {"initialized": "none", "lifetime": "selected_window"}})
    selection = original_joined_group(builder.analysis.routines[CHILD])
    prepared = provider.prepare(builder.analysis, CHILD, selection)
    effects = prove_numerical_effects(builder.analysis, CHILD, selection, prepared)
    assert effects.preconditions == prepared.preconditions
    b = next(item for item in effects.sections.resources if item.resource == ROOTS[1])
    assert b.overwrites
    assert all(axis.lower is not None and axis.upper is not None for box in b.overwrites for axis in box.axes)
    assert not any(item.resource.endswith("::g") for item in effects.sections.resources)
    incoming = next(item for item in effects.sections.resources if item.resource == ROOTS[0])
    transfer = effects.transfer({ROOTS[0]: incoming.reads})
    assert transfer.reads_covered
    assert transfer.definitions[ROOTS[1]] == b.overwrites
    assert builder.facts["captures"][ROOTS[1]]["initialized"] == "none"


def test_original_native_and_default_reached_proofs_remain_strict(tmp_path):
    with pytest.raises(CompilationError):
        reached_numerical_fixture(tmp_path / "planned", provider_enabled=False)
    _paths, builder, _proof, originals, _provider = reached_numerical_fixture(tmp_path / "complete")
    with pytest.raises(CompilationError):
        prove_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)


def test_emitted_mapped_chain_retains_original_native_partial_update(tmp_path):
    paths, outputs, report, owner = emit_reached_numerical_fixture(tmp_path)
    child = owner.members[CHILD]
    mapped, = owner.window_proof.public()["planned_closures"]["mapped_child_plans"]
    assert not mapped["plan"]["planned_closures"]["close_points"]
    face, = mapped["plan"]["planned_closures"]["mapped_child_plans"]
    assert face["procedure"] == "quartz_numerics::touch_face"
    assert not face["plan"]["planned_closures"]["close_points"]
    state, = face["plan"]["states"]
    assert state["resource"] == ROOTS[1]
    assert state["initialized"] == "whole"
    assert state["coverage_on_return"] == "whole"
    text = outputs[report["sources"][str(paths[1])]["replacement"]]
    assert len(report["scopes"][0]["gpu_leaves"]) == 3
    assert len(child.units) == 3
    assert not child.boundaries
    # Native and failed-preexecution branches retain original calls. The
    # successful reached path borrows the original face body; execution tests
    # check its visit count instead of counting mutually exclusive fallbacks.
    assert "call " + owner.members["quartz_numerics::touch_face"].entry in text
    assert text.count("bridge_visits=bridge_visits+1") == 1
    assert text.count("middle_visits=middle_visits+1") == 1
    assert "fort_scope_host_begin" in text
    assert "fort_scope_host_end" in text
    assert not report["scopes"][0]["reopen_after_boundary"]


def test_partial_only_definition_keeps_unmapped_caller_control_decline(tmp_path):
    _paths, _builder, proof, _originals, _provider = reached_numerical_fixture(
        tmp_path, captures={ROOTS[1]: {"initialized": "none", "lifetime": "selected_window"}})
    mapped, = proof.public()["planned_closures"]["mapped_child_plans"]
    point, = (point for point in mapped["plan"]["planned_closures"]["close_points"]
              if point["original_source_kind"] == "Call_Stmt")
    assert point["original_source_kind"] == "Call_Stmt"
    assert "unmapped" in point["reason"]
    assert "argument::n" in point["reason"]
    assert not mapped["plan"]["planned_closures"]["mapped_child_plans"]


@pytest.mark.parametrize("token", ["provider", "prepared"])
def test_copied_provider_and_preparation_have_no_registered_authority(tmp_path, token):
    _paths, builder, _proof, _originals, provider = reached_numerical_fixture(tmp_path)
    selection = original_joined_group(builder.analysis.routines[CHILD])
    prepared = provider.prepare(builder.analysis, CHILD, selection)
    if token == "provider":
        target, arguments = copy(provider), (builder.analysis,)
    else:
        target, arguments = copy(prepared), (builder.analysis, CHILD, selection)
    with pytest.raises(CompilationError):
        target.validate(*arguments)


@pytest.mark.parametrize("mutation", ["source", "binding", "configuration"])
def test_original_source_binding_and_emission_configuration_cannot_go_stale(tmp_path, mutation):
    paths, builder, proof, originals, provider = reached_numerical_fixture(tmp_path)
    selection = original_joined_group(builder.analysis.routines[CHILD])
    prepared = provider.prepare(builder.analysis, CHILD, selection)
    if mutation == "source":
        paths[0].write_text(paths[0].read_text().replace("g(1)=value", "g(1)=2.0_8*value"))
    elif mutation == "binding":
        helper = builder.analysis.numerical_helpers["quartz_math::profile_cell"]
        binding = helper.scope.bindings["value"]
        helper.scope.bindings["value"] = replace(binding, attributes=binding.attributes | {"volatile"})
    else:
        builder.config = replace(builder.config, host_threads=2)
        with pytest.raises(CompilationError):
            ReachedWindowOwner(builder, proof, originals).run()
        return
    with pytest.raises(CompilationError):
        prepared.validate(builder.analysis, CHILD, selection)


def test_nested_out_event_does_not_resurrect_the_producers_old_interior(tmp_path):
    # Deliberately invalid consumer coverage is analysis-only and never executed.
    source = CHILD_SOURCE.replace("call touch_face(b)", "call replace_one(b)")
    source = source.removesuffix("end module\n") + """subroutine replace_one(b)
real(8),intent(out)::b(-4:)
b(-3)=2.0_8
end subroutine
end module
"""
    _paths, _builder, proof, _originals, _provider = reached_numerical_fixture(tmp_path, child_source=source)
    mapped, = proof.public()["planned_closures"]["mapped_child_plans"]
    replacement, = mapped["plan"]["planned_closures"]["mapped_child_plans"]
    assert replacement["procedure"] == "quartz_numerics::replace_one"
    assert replacement["coordinate_proof"]["entry_definition_changes"] == [ROOTS[1]]
    state, = replacement["plan"]["states"]
    assert state["initialized"] == "none"
    assert state["coverage_on_return"] == "sections"
    closes = mapped["plan"]["planned_closures"]["close_points"]
    assert closes
    assert any("definition" in point["reason"] for point in closes)


def test_inactive_outer_domain_never_evaluates_inner_tile_inputs(tmp_path):
    from compiler.scopes.numerical_preflight import preparation_preflight

    child_source = """module quartz_numerics
use quartz_math,only:cell_formula=>profile_cell
implicit none
contains
subroutine bridge(a,b,c,n,m,tilei,tilej)
real(8),intent(in)::a(-4:,-2:)
real(8),intent(inout)::b(-4:,-2:),c(-4:,-2:)
integer,intent(in)::n,m,tilei,tilej
integer::i,j,ib,jb
real(8)::g(2),s
!$omp parallel do private(ib,i,j,g,s) schedule(static)
do jb=1,m,tilej
do ib=1,n,tilei
do j=jb,min(jb+tilej-1,m)
do i=ib,min(ib+tilei-1,n)
call cell_formula(g,s,a(i,j),i+3*j)
b(i,j)=s
enddo
enddo
enddo
enddo
!$omp end parallel do
end subroutine
end module
"""
    owner_source = """module quartz_owner
use quartz_numerics,only:renamed_bridge=>bridge
implicit none
contains
subroutine advance(a,b,c,n,m,tilei,tilej)
real(8),intent(in)::a(-4:,-2:)
real(8),intent(inout)::b(-4:,-2:),c(-4:,-2:)
integer,intent(in)::n,m,tilei,tilej
call renamed_bridge(a,b,c,n,m,tilei,tilej)
end subroutine
end module
"""
    _paths, _outputs, _report, owner = emit_reached_numerical_fixture(
        tmp_path, child_source=child_source, owner_source=owner_source)
    child = owner.members[CHILD]
    unit, = child.units
    _specification, code = preparation_preflight(child.builder, unit.scope.calls[0], {})
    # For m<1 the original inner DO header is never evaluated. Recognize the
    # emitted outer-domain test, independently of generated temporary numbers.
    upper_names = {match.group(1) for line in code
                   if (match := re.fullmatch(r"(\w+) = int\(m,c_int64_t\)", line))}
    assert upper_names
    stack, inner_reads = [], 0
    for line in code:
        if line.startswith("if (") and line.endswith(" then"):
            stack.append(line)
        elif line == "endif":
            stack.pop()
        if re.search(r"\btilei\b", line):
            inner_reads += 1
            assert any(re.search(r"<= " + re.escape(name) + r"\) then$", condition)
                       for condition in stack for name in upper_names)
    assert inner_reads
    assert not stack
