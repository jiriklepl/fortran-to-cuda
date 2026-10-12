"""Small numerical selections retain exact whole-source authority."""

from copy import copy

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend import numerical_preparation
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.numerical_preparation import NumericalPreparationProvider, NumericalPreparationUnsupported
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig

ENTRY = "selection_owner::advance"
OPTIONS = CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_execution="reached")
CONFIG = OffloadConfig(policy="sections", scope_execution="reached")


def selection_fixture(directory, *, padding=1, unsupported=False, team=False):
    expression = "+".join(["seed"] * 96)
    tail = ("scratch=" + expression + "\n") * padding
    if unsupported:
        tail += "if(.false.) call unavailable_native(seed)\n"
    source = """module selection_owner
implicit none
contains
subroutine advance(a,b,n,seed)
real(8),intent(in)::a(:),seed
real(8),intent(out)::b(:)
integer,intent(in)::n
integer::i
real(8)::scratch
do i=1,n
b(i)=a(i)+1.0_8
enddo
""" + tail + """end subroutine
subroutine other(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=a(i)+1.0_8
enddo
end subroutine
end module
"""
    if team:
        loop = "do i=1,n\nb(i)=a(i)+1.0_8\nenddo\n"
        source = source.replace(loop, "!$omp parallel private(i)\n!$omp do\n" + loop
                                + "!$omp end do\n!$omp do\n" + loop
                                + "!$omp end do\n!$omp end parallel\n", 1)
    path = directory / "selection.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines[ENTRY]
    loop = walk(routine.execution, F.Block_Nonlabel_Do_Construct)[0]
    provider = NumericalPreparationProvider(analysis, options=OPTIONS, config=CONFIG, environment_root=ENTRY)
    return path, analysis, provider, (loop,)


def test_small_region_among_large_unrelated_expressions_keeps_bounded_preparation(tmp_path, monkeypatch):
    _path, analysis, provider, selection = selection_fixture(tmp_path, padding=64)
    routine = analysis.routines[ENTRY]
    source_authority = numerical_preparation._source_authority

    def redundant_membership(analysis, procedure, selection, helpers, budget):
        # Reinsert the former full-owner membership traversal. Its repeated
        # expression visits exhausted preparation despite a tiny selected IR.
        tuple(numerical_preparation._walk(analysis._require_original(procedure).execution, budget))
        return source_authority(analysis, procedure, selection, helpers, budget)

    with monkeypatch.context() as previous:
        previous.setattr(numerical_preparation, "_source_authority", redundant_membership)
        with pytest.raises(CompilationError, match="source-scan budget exhausted"):
            provider.prepare(analysis, ENTRY, selection)
    assert not getattr(analysis, "_numerical_preparations", {})
    prepared = provider.prepare(analysis, ENTRY, selection)
    assert prepared.offload.available
    assert prepared._selection[0] is selection[0]
    assert len(prepared.context.following) == 64
    assert all(left is right for left, right in zip(
        prepared.context.following, routine.execution.content[1:], strict=True))
    assert prepared.validate(analysis, ENTRY, selection) is prepared


def test_original_team_member_still_requires_registered_worksharing_completion(tmp_path):
    from compiler.scopes.region_context import original_region_context
    from compiler.scopes.regions import extract_region

    _path, analysis, provider, selection = selection_fixture(tmp_path, team=True)
    routine = analysis.routines[ENTRY]
    with pytest.raises(CompilationError, match="OpenMP association"):
        provider.prepare(analysis, ENTRY, selection)
    groups = grouped_original_sequence(routine.execution.content, {})
    joined, = (group.original_nodes for group in groups if isinstance(group, NativeGroupCandidate))
    proof = analysis.worksharing_completion(ENTRY, joined, selection)
    context = original_region_context(routine, selection)
    extraction = extract_region(analysis, routine, selection, preceding=context.preceding,
                                following=context.following, cyclic_following=context.cyclic_following,
                                worksharing=proof)
    prepared = numerical_preparation._prepare(analysis, ENTRY, selection, provider,
                                              extraction=extraction, context=context)
    assert prepared.offload.available
    assert prepared.completion_proof is proof
    assert prepared.validate(analysis, ENTRY, selection) is prepared


@pytest.mark.parametrize("replacement", ["copied", "foreign", "source_id", "scalar", "empty", "expression"])
def test_preparation_requires_original_parser_selection_not_a_lookup_surrogate(tmp_path, replacement):
    path, analysis, provider, selection = selection_fixture(tmp_path)
    prepared = provider.prepare(analysis, ENTRY, selection)
    before = path.read_bytes()
    if replacement == "copied":
        supplied = (copy(selection[0]),)
    elif replacement == "foreign":
        supplied = tuple(walk(analysis.routines["selection_owner::other"].execution, F.Block_Nonlabel_Do_Construct))
    elif replacement == "source_id":
        _graph, supplied = analysis._selected_source(ENTRY, selection)
        # Structured effect APIs accept IDs; numerical preparation requires
        # the exact parser selection even when that ID is otherwise valid.
        assert analysis._selected_source(ENTRY, supplied)[1] == supplied
    elif replacement == "scalar":
        supplied = (1,)
    elif replacement == "empty":
        supplied = ()
    else:
        supplied = (walk(selection[0], F.Part_Ref)[0],)
    for demand in (lambda: provider.prepare(analysis, ENTRY, supplied),
                   lambda: prepared.validate(analysis, ENTRY, supplied)):
        with pytest.raises(CompilationError) as failure:
            demand()
        assert not isinstance(failure.value, NumericalPreparationUnsupported)
    assert analysis._numerical_preparations == {prepared.identity: prepared}
    assert path.read_bytes() == before


@pytest.mark.parametrize("changed", ["unselected_expression", "unsupported_action", "unselected_binding"])
def test_unselected_source_still_invalidates_preparation_authority(tmp_path, changed):
    _path, analysis, provider, selection = selection_fixture(tmp_path, unsupported=True)
    prepared = provider.prepare(analysis, ENTRY, selection)
    routine = analysis.routines[ENTRY]
    if changed == "unselected_expression":
        assignment = routine.execution.content[1]
        assignment.items = (*assignment.items[:2], F.Real_Literal_Constant("0.0_8"))
    elif changed == "unsupported_action":
        call, = walk(routine.execution, F.Call_Stmt)
        call.items = (F.Name("different_native"), call.items[1])
    else:
        routine.scope.bindings["seed"] = copy(routine.scope.bindings["seed"])
    for demand in (lambda: prepared.validate(analysis, ENTRY, selection),
                   lambda: provider.prepare(analysis, ENTRY, selection)):
        with pytest.raises(CompilationError) as failure:
            demand()
        assert not isinstance(failure.value, NumericalPreparationUnsupported)


def test_parent_callback_mutation_after_final_selection_check_cannot_publish_a_token(tmp_path, monkeypatch):
    _path, analysis, provider, selection = selection_fixture(tmp_path)
    assignment = analysis.routines[ENTRY].execution.content[1]
    selected_source = analysis._selected_source
    checked, changed, charged = False, False, []

    def select(procedure, nodes):
        nonlocal checked
        result = selected_source(procedure, nodes)
        # Cold preparation has installed its provisional token immediately
        # before final validation reauthenticates this exact source selection.
        checked = bool(getattr(analysis, "_numerical_preparations", {}))
        return result

    def consume(amount):
        nonlocal changed
        charged.append(amount)
        if checked and not changed:
            changed = True
            assignment.items = (*assignment.items[:2], F.Real_Literal_Constant("0.0_8"))

    monkeypatch.setattr(analysis, "_selected_source", select)
    with pytest.raises(CompilationError, match="original.*authority") as failure:
        provider.prepare(analysis, ENTRY, selection, consume=consume)
    assert not isinstance(failure.value, NumericalPreparationUnsupported)
    assert changed
    assert charged
    assert all(amount == 1 for amount in charged)
    assert not analysis._numerical_preparations
