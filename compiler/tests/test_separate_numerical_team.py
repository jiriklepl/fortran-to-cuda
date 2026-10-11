"""Whole separate numerical teams retain their source worksharing sequence."""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.numerical_effects import prove_numerical_effects
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.combined_team import _source_partition, is_combined_group
from compiler.scopes.source import ScopeBuilder
from compiler.tests.support.numerical_regions import EmittedNumericalText
from compiler.tests.test_ordinary_numerical_regions import emit_ordinary_numerical_fixture

ENTRY = "bounded_team::step"
BODY = """!$omp parallel private(i)
!$omp do schedule(static)
do i=-3,n
b(i)=a(i)+real(i,8)
enddo
!$omp end do nowait
!$omp do schedule(static)
do i=-3,n
c(i)=b(i)+a(i)
enddo
!$omp end do nowait
!$omp end parallel
"""
SOURCE = """module bounded_team
implicit none
contains
subroutine step(a,b,c,n)
real(8),intent(in)::a(-4:)
real(8),intent(inout)::b(-4:),c(-4:)
integer,intent(in)::n
integer::i
real(8)::s
""" + BODY + "end subroutine\nend module\n"


def group_fixture(directory, source=SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "bounded_team.f90"
    path.write_text(source)
    fact = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: dict(fact) for name in ("a", "b", "c")}}
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    group, = (item for item in grouped_original_sequence(builder.entry.execution.content, {})
              if isinstance(item, NativeGroupCandidate))
    return builder, group.original_nodes


def physical_effects(builder, selection):
    provider = NumericalPreparationProvider(builder.analysis, options=builder.options, config=builder.config,
                                            environment_root=builder._numerical_environment_root)
    prepared = provider.prepare(builder.analysis, ENTRY, selection)
    return prepared, prove_numerical_effects(builder.analysis, ENTRY, selection, prepared)


@pytest.fixture(scope="module", params=[4, 8])
def separate_emission(tmp_path_factory, request):
    paths, outputs, report, builder = emit_ordinary_numerical_fixture(
        tmp_path_factory.mktemp("separate_team"), precision=request.param)
    assert builder.validate_reached_window() is None
    return EmittedNumericalText.capture(paths, outputs, report)


def test_ordinary_separate_groups_keep_one_owner_and_native_partial_bridge(separate_emission):
    report = separate_emission.manifest()
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    assert not scope["boundaries"]
    assert not scope["reopen_after_boundary"]
    groups = [segment["operations"]["combined_numerical_team"] for segment in scope["planning_segments"]
              if "combined_numerical_team" in segment["operations"]]
    assert [group["unit_count"] for group in groups] == [2, 1]
    assert all(group["schema_version"] == 2 and group["group_kind"] == "separate_parallel" for group in groups)
    assert all(not group["native_effects_authority"] for group in groups)
    text = separate_emission.child_text()
    assert text.count("call touch_face(b)") == 1
    assert text.count("b(-3)=b(-3)+0.5_") == 1
    assert text.count("call ieee_set_halting_mode(ieee_invalid,.false.)") == 1
    assert text.count("fort_scope_numerical_environment_supported() == 0") == 2


def test_native_separate_fallback_preserves_each_original_do_and_nowait(separate_emission):
    original = separate_emission.original_sources[1][1]
    text = separate_emission.child_text()
    # Each original parallel body's native alternative is retained verbatim,
    # including its worksharing directives. No extra enclosing DO is legal.
    bodies = []
    start = 0
    while (first := original.find("!$omp parallel private(i)", start)) >= 0:
        body_first = original.index("\n", first) + 1
        last = original.index("!$omp end parallel", body_first)
        bodies.append(original[body_first:last])
        start = last + len("!$omp end parallel")
    assert len(bodies) == 2
    for body in bodies:
        assert text.count("else\n" + body + "endif\n!$omp end parallel") == 1
    assert text.count("!$omp do schedule(static)") == 3
    assert text.count("!$omp end do nowait") == 1


def test_whole_nowait_group_keeps_ordered_units_but_requires_entry_reads(tmp_path):
    builder, selection = group_fixture(tmp_path)
    assert is_combined_group(builder, selection)
    partition = _source_partition(builder, selection)
    assert partition["group_kind"] == "separate_parallel"
    assert partition["unit_count"] == 2
    prepared, effects = physical_effects(builder, selection)
    assert prepared.offload.available
    reads = {item.resource: item.reads for item in effects.sections.resources}
    incomplete = effects.transfer({"argument::a": reads["argument::a"]})
    assert {item.resource for item in incomplete.uncovered_reads} == {"argument::b"}
    assert effects.transfer({"argument::a": reads["argument::a"], "argument::b": reads["argument::b"]}).reads_covered
    loop = next(node for node in selection if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    with pytest.raises(CompilationError, match="NOWAIT"):
        builder.analysis.worksharing_completion(ENTRY, selection, (loop,))


def test_private_scalar_carried_between_original_do_units_cannot_become_a_capture(tmp_path):
    body = BODY.replace("parallel private(i)", "parallel private(i,s)")
    body = body.replace("b(i)=a(i)+real(i,8)", "s=a(i)\nb(i)=s")
    body = body.replace("c(i)=b(i)+a(i)", "c(i)=s")
    # For a native execution with at least four iterations, every member's s
    # is defined by its first DO. Independent GPU units cannot carry that state.
    builder, selection = group_fixture(tmp_path, SOURCE.replace(BODY, body))
    with pytest.raises(CompilationError):
        physical_effects(builder, selection)


@pytest.mark.parametrize("change", [
    lambda body: body.replace("!$omp do schedule(static)", "call native_sibling()\n!$omp do schedule(static)", 1),
    lambda body: body.replace("b(i)=a(i)+real(i,8)",
                              "!$omp parallel\nb(i)=a(i)+real(i,8)\n!$omp end parallel", 1),
])
def test_nonflat_or_native_siblings_keep_the_existing_joined_team_route(tmp_path, change):
    builder, selection = group_fixture(tmp_path, SOURCE.replace(BODY, change(BODY)))
    assert not is_combined_group(builder, selection)
    with pytest.raises(CompilationError):
        _source_partition(builder, selection)


@pytest.mark.parametrize("clause", ["reduction(+:s)", "num_threads(4)", "if(n>0)"])
def test_separate_partition_does_not_expand_original_parallel_clause_support(tmp_path, clause):
    source = SOURCE.replace("parallel private(i)", "parallel private(i) " + clause, 1)
    builder, selection = group_fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="unsupported clause partition"):
        _source_partition(builder, selection)


def test_separate_group_unit_budget_and_semicolon_source_boundary_are_enforced(tmp_path):
    loop = "!$omp do\ndo i=-3,n\nb(i)=a(i)\nenddo\n!$omp end do\n"
    body = "!$omp parallel private(i)\n" + loop * 33 + "!$omp end parallel\n"
    builder, selection = group_fixture(tmp_path / "many", SOURCE.replace(BODY, body))
    with pytest.raises(CompilationError, match="source-unit budget exhausted"):
        _source_partition(builder, selection)
    builder, selection = group_fixture(tmp_path / "semicolon", SOURCE.replace("enddo\n", "enddo; s=1\n", 1))
    with pytest.raises(CompilationError):
        _source_partition(builder, selection)
