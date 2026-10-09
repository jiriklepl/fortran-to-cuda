"""Fixed fields are numerical captures of original objects, never new owners."""

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.frontend import lower_source
from compiler.frontend.source_effects import SourceEffects
from compiler.scopes.array_operations import extract_array_operation
from compiler.scopes.regions import extract_region


SOURCE = """module renamed_fields
type settings
real(8)::dt
real(8)::face(-3:5)
real(8),allocatable::unrelated(:)
end type
type(settings)::storage
contains
subroutine advance(a,n)
real(8),intent(inout)::a(-3:)
integer,intent(in)::n
integer::i
BODY
end subroutine
end module
"""


def candidate(tmp_path, body, *, assignment=False):
    path = tmp_path / "fixed_fields.f90"
    path.write_text(SOURCE.replace("BODY", body))
    analysis = SourceEffects([path])
    routine = analysis.routines["renamed_fields::advance"]
    kind = F.Assignment_Stmt if assignment else F.Block_Nonlabel_Do_Construct
    node = next(iter(walk(routine.execution, kind)))
    extraction = extract_array_operation if assignment else extract_region
    region = extraction(analysis, routine, node)
    _, plan = prepare_function(lower_source(region.source, region.entry, source_name="fixed-field.f90"),
                               options=CompilerOptions())
    assert len(plan.regions) == 1
    return region


@pytest.mark.parametrize("body", [
    "do i=-3,n\na(i)=a(i)+storage%dt\nenddo",
    "do i=-3,5\nstorage%face(i)=a(i)*storage%dt\nenddo",
    "associate(config=>storage)\ndo i=-3,n\na(i)=a(i)+config%dt\nenddo\nend associate",
    "do i=-3,ubound(storage%face,1)\na(i)=storage%face(i)+real(lbound(storage%face,1),8)\nenddo",
])
def test_fixed_component_loops_keep_original_storage_and_logical_bounds(tmp_path, body):
    region = candidate(tmp_path, body)
    roots = {binding.root for binding in region.bindings}
    assert any(root.startswith("renamed_fields::storage%") for root in roots)
    assert "storage%" not in region.source
    assert "config%" not in region.source
    assert "fort_region_field_" in region.source
    assert "type settings" not in region.source


@pytest.mark.parametrize("body", [
    "a=storage%dt", "storage%face(:)=a(-3:5)*storage%dt",
    "a(:)=storage%face(:)+real(lbound(storage%face,1),8)",
])
def test_fixed_component_array_operations_use_canonical_resources(tmp_path, body):
    region = candidate(tmp_path, body, assignment=True)
    assert any(binding.root.startswith("renamed_fields::storage%") for binding in region.bindings)
    assert region.operation_kind == "array_assignment"


def test_component_selector_does_not_resolve_a_same_spelling_procedure(tmp_path):
    source = SOURCE.replace("end module", """real(8) function face(i)
integer,intent(in)::i
write(*,*)i
face=0.d0
end function
end module""")
    path = tmp_path / "same_spelling.f90"
    path.write_text(source.replace("BODY", "do i=-3,5\na(i)=storage%face(i)\nenddo"))
    analysis = SourceEffects([path])
    routine = analysis.routines["renamed_fields::advance"]
    node = next(iter(walk(routine.execution, F.Block_Nonlabel_Do_Construct)))
    region = extract_region(analysis, routine, node)
    assert region.numerical_helpers == ()
    _, plan = prepare_function(lower_source(region.source, region.entry, source_name="same-spelling.f90"),
                               options=CompilerOptions())
    assert len(plan.regions) == 1
