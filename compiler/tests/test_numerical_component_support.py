"""Unsupported indexed records decline numerical trials after fresh authority checks."""

from copy import copy

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler._legacy.config import LegacyCompilerOptions as CompilerOptions
from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig
from compiler.frontend.component_bindings import ComponentAccessUnsupported, component_access
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.numerical_preparation import NumericalPreparationProvider, NumericalPreparationUnsupported
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.support.numerical_regions import CHILD, CHILD_SOURCE, emit_reached_numerical_fixture

TYPES = """module record_source
implicit none
type sample
real(8)::weight
end type
type(sample),allocatable,target::records(:)
end module
"""
SOURCE = """module record_consumer
use record_source,only:entries=>records
implicit none
integer::visits=0
contains
subroutine form_field(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i) schedule(static)
do i=1,n
b(i)=a(i)+entries(i)%weight
enddo
!$omp end parallel do
visits=visits+1
end subroutine
end module
"""
ENTRY = "record_consumer::form_field"


def fixture(directory):
    types, source = directory / "records.f90", directory / "consumer.f90"
    types.write_text(TYPES)
    source.write_text(SOURCE)
    analysis = SourceEffects([types, source])
    routine = analysis.routines[ENTRY]
    group, = (item for item in grouped_original_sequence(routine.execution.content, {})
              if isinstance(item, NativeGroupCandidate))
    provider = NumericalPreparationProvider(analysis,
        options=CompilerOptions(gpu_policy="sections", memory_model="scoped", scope_execution="reached"),
        config=OffloadConfig(policy="sections", scope_execution="reached"), environment_root=ENTRY)
    return types, source, analysis, provider, group.original_nodes


def test_indexed_record_is_a_numerical_support_decline_without_metadata_capture(tmp_path):
    _types, _source, analysis, provider, selection = fixture(tmp_path)
    routine = analysis.routines[ENTRY]
    node, = walk(selection, F.Data_Ref)
    with pytest.raises(ComponentAccessUnsupported, match="scalar variable object"):
        component_access(analysis, routine.scope, node)
    with pytest.raises(NumericalPreparationUnsupported, match="scalar variable object"):
        provider.prepare(analysis, ENTRY, selection)
    assert not getattr(analysis, "_native_metadata", False)
    assert not getattr(analysis, "_numerical_preparations", {})
    native = copy(analysis)
    native._native_metadata = True
    access = component_access(native, routine.scope, node)
    assert access.root_binding.root == "record_source::records"
    assert access.binding.native_metadata_object is access.root_binding


def test_generic_component_failure_is_not_a_support_decline(tmp_path, monkeypatch):
    _types, _source, analysis, provider, selection = fixture(tmp_path)

    def failed(*_args):
        # Wording is intentionally the same as the known support diagnostic.
        raise CompilationError("source component needs one scalar variable object")

    monkeypatch.setattr("compiler.frontend.component_bindings.component_access", failed)
    with pytest.raises(CompilationError) as failure:
        provider.prepare(analysis, ENTRY, selection)
    assert type(failure.value) is CompilationError


@pytest.mark.parametrize("change", ["source", "binding", "type_ast", "module_role"])
def test_late_callback_mutation_cannot_be_masked_by_component_decline(tmp_path, monkeypatch, change):
    from compiler.frontend import source_functions

    types, source, analysis, provider, selection = fixture(tmp_path)
    routine = analysis.routines[ENTRY]
    original_selection = analysis._selected_source
    component = component_access
    observed = selected = changed = imported_checked = False
    imported_scope = analysis.modules["record_source"]
    validate_scope = source_functions._validate_scope

    def checked_scope(current_analysis, scope, **kwargs):
        nonlocal imported_checked
        result = validate_scope(current_analysis, scope, **kwargs)
        if scope is imported_scope:
            imported_checked = True
        return result

    def selected_source(*args, **kwargs):
        nonlocal selected
        result = original_selection(*args, **kwargs)
        selected = True
        return result

    def select_component(*args):
        nonlocal observed
        observed = True
        return component(*args)

    def consume(_count):
        nonlocal changed
        if not selected or changed or change == "module_role" and not imported_checked:
            return
        changed = True
        if change == "source":
            source.write_text(source.read_text() + "! changed during numerical scan\n")
        elif change == "binding":
            routine.scope.bindings["a"] = copy(routine.scope.bindings["a"])
        elif change == "type_ast":
            declaration, = walk(analysis.modules["record_source"].node, F.Component_Decl)
            declaration.items = (F.Name("other"), *declaration.items[1:])
        else:
            analysis.modules["record_source"] = copy(analysis.modules["record_source"])

    monkeypatch.setattr(analysis, "_selected_source", selected_source)
    monkeypatch.setattr(source_functions, "_validate_scope", checked_scope)
    monkeypatch.setattr("compiler.frontend.component_bindings.component_access", select_component)
    with pytest.raises(CompilationError) as failure:
        provider.prepare(analysis, ENTRY, selection, consume=consume)
    assert changed
    assert not isinstance(failure.value, NumericalPreparationUnsupported)
    assert not getattr(analysis, "_numerical_preparations", {})
    if change in {"type_ast", "module_role"}:
        # The selector itself remains unsupported; its late type mutation is
        # detected by fresh source authentication before the decline returns.
        assert observed
        assert types.read_text() == TYPES


def test_late_parent_budget_failure_remains_fatal(tmp_path, monkeypatch):
    _types, _source, analysis, provider, selection = fixture(tmp_path)
    original = analysis._selected_source
    selected = False

    def select(*args, **kwargs):
        nonlocal selected
        result = original(*args, **kwargs)
        selected = True
        return result

    def consume(_count):
        if selected:
            raise CompilationError("parent aggregate source budget exhausted")

    monkeypatch.setattr(analysis, "_selected_source", select)
    with pytest.raises(CompilationError, match="parent aggregate") as failure:
        provider.prepare(analysis, ENTRY, selection, consume=consume)
    assert type(failure.value) is CompilationError


def test_reached_indexed_record_trial_closes_only_its_original_branch(tmp_path):
    declaration = """type sample
real(8)::weight
end type
type(sample),allocatable,target::sites(:)
"""
    branch = """if(tile==-7) then
!$omp parallel do private(i) schedule(static)
do i=-3,n
b(i)=b(i)+sites(i)%weight
enddo
!$omp end parallel do
endif
"""
    source = CHILD_SOURCE.replace("integer::bridge_visits", declaration + "integer::bridge_visits")
    source = source.replace("if(n>=-3) then", branch + "if(n>=-3) then")
    paths, outputs, report, owner = emit_reached_numerical_fixture(
        tmp_path, child_source=source, captures={"argument::c": {"initialized": "whole"}})
    mapped, = owner.window_proof.public()["planned_closures"]["mapped_child_plans"]
    assert mapped["procedure"] == CHILD
    child_plan = mapped["plan"]
    point, = child_plan["planned_closures"]["close_points"]
    assert any("tile == - 7" in guard.lower() for guard in point["guard"])
    assert point["original_source_kind"] == "complete_original_joined_group"
    assert child_plan["planned_closures"]["active_path_may_return"]
    assert child_plan["planned_closures"]["closed_path_possible"]
    assert not owner.boundaries
    assert len(report["scopes"][0]["gpu_leaves"]) == 3
    child = owner.members[CHILD]
    assert not any("sites" in root for root in child.resources)
    text = outputs[report["sources"][str(paths[1])]["replacement"]]
    assert text.count("b(i)=b(i)+sites(i)%weight") == 1
    assert text.count("bridge_visits=bridge_visits+1") == 1
    assert text.index("fort_scope_close(") < text.index("b(i)=b(i)+sites(i)%weight")
