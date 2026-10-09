"""Borrowed numerical loops retain original bounds, state and proof authority."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.regions import allocation_guard, extract_region
from compiler.scopes.segments import grouped_nodes, statement_span


def case(tmp_path, body, *, declarations="", uses="", extra=""):
    path = tmp_path / "original.f90"
    path.write_text(f"""module inline_case
{uses}
implicit none
contains
subroutine step(a,n,flag)
real(8),intent(inout)::a(-3:,7:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i,j,k
{declarations}
{body}
end subroutine
{extra}
end module
""")
    analysis = SourceEffects([path])
    routine = analysis.routines["inline_case::step"]
    return analysis, routine, path


def selected(routine, index=0):
    nodes = tuple(_children(routine.execution))
    position = [i for i, node in enumerate(nodes) if _kind(node) == "Block_Nonlabel_Do_Construct"][index]
    return nodes[position], nodes[:position], nodes[position + 1:]


def extracted(analysis, routine, index=0):
    node, preceding, following = selected(routine, index)
    return extract_region(analysis, routine, node, preceding=preceding, following=following)


def original_joined_group(routine):
    original = tuple(_children(routine.execution))
    group, = [node for node in grouped_nodes(original) if isinstance(node, tuple)]
    first, last = statement_span(group[0])[0], statement_span(group[-1])[1]
    return tuple(node for node in original
                 if first <= statement_span(node)[0] and statement_span(node)[1] <= last)


def test_negative_coordinates_and_index_as_data_are_preserved(tmp_path):
    analysis, routine, _ = case(tmp_path, """do j=7,ubound(a,2)
do i=-3,ubound(a,1)
a(i,j)=a(i,j)+real(i+17*j,8)+size(a)+lbound(a,1)+ubound(a,2)
enddo
enddo""")
    original = str(routine.execution)
    region = extracted(analysis, routine)
    text = "".join(region.source.lower().split())
    assert "doi=-3," in text
    assert "doj=7," in text
    assert "(i)-fort_region_lb_a_1+1" in text
    assert "(j)-fort_region_lb_a_2+1" in text
    assert "real(i+17*j,8)" in text
    assert "size(a)" in text
    assert "fort_region_lb_a_2+(size(a,2)-1)" in text
    assert "lbound(a," not in text
    assert "ubound(a," not in text
    assert str(routine.execution) == original
    assert [(item.name, item.resource, item.lower_bound_dimension, item.runtime_lower_bound)
            for item in region.parameters] == [
                ("a", "argument::a", None, False),
                ("fort_region_lb_a_1", "argument::a", 1, True),
                ("fort_region_lb_a_2", "argument::a", 2, True),
            ]
    assert region.written_resources == {"argument::a"}
    assert set(region.private_scalars) == {"i", "j"}


def test_bounds_rewrite_avoids_intermediate_integer_overflow_and_preserves_empty_inquiries(tmp_path):
    analysis, routine, _ = case(tmp_path, """do i=1,n
a(i,7)=real(lbound(a,1)+ubound(a,1)+size(a,1),8)
enddo""")
    region = extracted(analysis, routine)
    compact = "".join(region.source.lower().split()).replace("&", "")
    assert "fort_region_lb_a_1+(size(a,1)-1)" in compact
    # Original LBOUND gives one for a zero extent, so the same formula gives
    # UBOUND=0 without evaluating an overflowing lb+size intermediate.
    for lower, extent, upper in [(1, 0, 0), (-3, 4, 0), (1, 2**31 - 1, 2**31 - 1)]:
        assert lower + (extent - 1) == upper


def test_private_temporary_defined_in_each_iteration_is_not_captured(tmp_path):
    analysis, routine, _ = case(tmp_path, """do i=-3,n
t=a(i,7)*2
a(i,7)=t+1
enddo""", declarations="real(8)::t")
    region = extracted(analysis, routine)
    assert {item.name for item in region.parameters} == {"a", "n", "fort_region_lb_a_1", "fort_region_lb_a_2"}
    assert set(region.private_scalars) == {"i", "t"}
    assert "real(8) :: t" in region.source.lower()


@pytest.mark.parametrize("declaration", ["integer(8)::t", "integer(1)::t", "real(16)::t",
                                        "real(8),volatile::t", "real(8),target::t"])
def test_private_scalar_width_and_association_cannot_be_normalized_away(tmp_path, declaration):
    analysis, routine, _ = case(tmp_path, """do i=-3,n
t=1
a(i,7)=t
enddo""", declarations=declaration)
    with pytest.raises(CompilationError, match="inline private scalar|inline original specification"):
        extracted(analysis, routine)


def test_wide_private_iterator_cannot_silently_become_default_integer(tmp_path):
    analysis, routine, path = case(tmp_path, "do i=-3,n\na(i,7)=1\nenddo")
    path.write_text(path.read_text().replace("integer::i,j,k", "integer(8)::i\ninteger::j,k"))
    analysis = SourceEffects([path])
    routine = analysis.routines["inline_case::step"]
    with pytest.raises(CompilationError, match="unsupported inline private scalar type"):
        extracted(analysis, routine)


def test_private_double_precision_is_preserved_by_the_numerical_frontend(tmp_path):
    from compiler.frontend import lower_source
    from compiler.ir import ScalarType

    analysis, routine, _ = case(tmp_path, """do i=-3,n
t=a(i,7)+0.0000000001d0
a(i,7)=t
enddo""", declarations="real(8)::t")
    region = extracted(analysis, routine)
    function = lower_source(region.source, region.entry, source_name="original.f90#inline:precision")
    assert next(symbol for symbol in function.symbols if symbol.name == "t").dtype == ScalarType.REAL


@pytest.mark.parametrize("body", [
    "t=2\ndo i=-3,n\na(i,7)=t\nt=a(i,7)+1\nenddo",
    "t=2\ndo i=-3,n\nt=t+a(i,7)\na(i,7)=t\nenddo",
    "i=-3\ndo i=i,n\na(i,7)=1\nenddo",
    "i=1\ndo i=-3,n,i\na(i,7)=1\nenddo",
    "do i=-3,n\nif(flag) t=a(i,7)\na(i,7)=t\nenddo",
])
def test_private_values_cannot_drop_an_incoming_or_conditional_definition(tmp_path, body):
    analysis, routine, _ = case(tmp_path, body, declarations="real(8)::t")
    with pytest.raises(CompilationError, match="definition inside"):
        extracted(analysis, routine)


@pytest.mark.parametrize("following", ["k=i", "if(flag) k=i", "do j=1,i\nk=j\nenddo"])
def test_original_do_final_value_live_after_region_keeps_native_owner(tmp_path, following):
    analysis, routine, _ = case(tmp_path, "do i=-3,n\na(i,7)=1\nenddo\n" + following)
    with pytest.raises(CompilationError, match="live after"):
        extracted(analysis, routine)


def test_following_definite_iterator_definition_ends_liveness(tmp_path):
    analysis, routine, _ = case(tmp_path, """do i=-3,n
a(i,7)=1
enddo
i=0
k=i""")
    assert extracted(analysis, routine).private_scalars == ("i",)


@pytest.mark.parametrize(("statement", "reason"), [
    ("call unavailable(a)", "unsupported inline numerical statement"),
    ("allocate(work(4))", "unsupported inline numerical statement"),
    ("deallocate(work)", "unsupported inline numerical statement"),
    ("a(i,7)=1\nif(n>0) return", "unsupported inline numerical statement"),
])
def test_unknown_effects_allocations_and_exits_remain_boundaries(tmp_path, statement, reason):
    analysis, routine, _ = case(tmp_path, f"do i=-3,n\n{statement}\nenddo",
                                declarations="real(8),allocatable::work(:)")
    with pytest.raises(CompilationError, match=reason):
        extracted(analysis, routine)


@pytest.mark.parametrize("inquiry", ["size", "lbound", "ubound", "real", "int", "sin"])
def test_known_shadowed_intrinsic_never_changes_meaning_in_new_module(tmp_path, inquiry):
    expression = f"{inquiry}(a,1)" if inquiry in {"size", "lbound", "ubound"} else f"{inquiry}(a(i,7))"
    arguments = "x,axis" if inquiry in {"size", "lbound", "ubound"} else "x"
    declarations = "real(8)::x(:,:)\ninteger::axis" if "," in arguments else "real(8)::x"
    extra = f"real(8) function {inquiry}({arguments})\n{declarations}\n{inquiry}=9\nend function"
    analysis, routine, _ = case(tmp_path, f"do i=-3,n\na(i,7)={expression}\nenddo", extra=extra)
    with pytest.raises(CompilationError, match="shadowed"):
        extracted(analysis, routine)


def test_unknown_wildcard_cannot_authorize_builtin_intrinsic_reinterpretation(tmp_path):
    analysis, routine, _ = case(tmp_path, "do i=-3,n\na(i,7)=size(a,1)\nenddo", uses="use missing_library")
    with pytest.raises(CompilationError, match="wildcard imports"):
        extracted(analysis, routine)


def test_named_default_integer_constant_is_resolved_in_original_scope(tmp_path):
    analysis, routine, _ = case(tmp_path, "do i=-3,n\na(i,7)=real(minimum,8)\nenddo",
                                declarations="integer,parameter::minimum=-2147483647-1")
    region = extracted(analysis, routine)
    assert "(-2147483647-1)" in "".join(region.source.split())
    assert "2147483648" not in region.source
    assert all(item.name != "minimum" for item in region.parameters)


def test_intrinsic_keyword_names_are_not_evaluated_as_uninitialized_captures(tmp_path):
    analysis, routine, _ = case(tmp_path, "do i=-3,n\na(i,7)=real(size(a,dim=1),kind=8)\nenddo",
                                declarations="integer::dim,kind")
    region = extracted(analysis, routine)
    assert not {"dim", "kind"} & {item.name for item in region.parameters}
    compact = "".join(region.source.lower().split())
    assert "size(a,dim=1)" in compact
    assert "kind=8" in compact


def test_construct_label_does_not_become_a_resolved_parameter_value(tmp_path):
    analysis, routine, _ = case(tmp_path, "region: do i=-3,n\na(i,7)=1\nenddo region",
                                declarations="integer,parameter::region=5")
    region = extracted(analysis, routine)
    compact = "".join(region.source.lower().split())
    assert "region:doi=" in compact
    assert "enddoregion" in compact
    assert all(item.name != "region" for item in region.parameters)


def test_declared_external_intrinsic_name_without_implementation_is_a_boundary(tmp_path):
    analysis, routine, _ = case(tmp_path, "do i=-3,n\na(i,7)=sin(a(i,7))\nenddo",
                                declarations="real(8),external::sin")
    with pytest.raises(CompilationError, match="shadowed|specification"):
        extracted(analysis, routine)


def test_unsupported_original_specification_cannot_disappear_during_extraction(tmp_path):
    analysis, routine, _ = case(tmp_path, "do i=-3,n\na(i,7)=1\nenddo",
                                declarations="integer::x,y\nequivalence(x,y)")
    with pytest.raises(CompilationError, match="specification"):
        extracted(analysis, routine)


@pytest.mark.parametrize("guard", [
    "allocate(work(-3:n,7:8))",
    "if(.not.allocated(work)) allocate(work(-3:n,7:8))",
    "if(.not.allocated(work)) then\nallocate(work(-3:n,7:8))\nendif",
])
def test_successful_saved_allocation_guard_preserves_storage_owner(tmp_path, guard):
    analysis, routine, path = case(tmp_path, guard + "\ndo i=-3,n\nwork(i,7)=a(i,7)\nenddo",
                                   declarations="real(8),allocatable,save::work(:,:)")
    before = path.read_text()
    region = extracted(analysis, routine)
    guard, = region.allocation_guards
    assert guard["resource"] == "inline_case::step::work"
    assert guard["allocation_proven"] is False
    assert len(guard["allocation_sites"]) == 1
    assert "ALLOCATED then checked original" in guard["runtime_guard"]
    assert "allocatable" not in region.source.lower()
    assert "save" not in region.source.lower()
    assert "allocate(" not in region.source.lower()
    assert path.read_text() == before
    assert region.public()["storage_owner"].startswith("original procedure")


@pytest.mark.parametrize("guard", [
    "allocate(work(-3:n,7:8),stat=k)",
    "if(.not.allocated(work).and.flag) allocate(work(-3:n,7:8))",
    "if(flag) allocate(work(-3:n,7:8))",
    "if(.not.allocated(work)) then\nallocate(work(-3:n,7:8))\nelse\nk=0\nendif",
])
def test_incomplete_or_fallible_saved_site_requires_a_fresh_runtime_guard(tmp_path, guard):
    analysis, routine, _ = case(tmp_path, guard + "\ndo i=-3,n\nwork(i,7)=a(i,7)\nenddo",
                                declarations="real(8),allocatable,save::work(:,:)")
    record, = extracted(analysis, routine).allocation_guards
    assert record["allocation_proven"] is False
    assert record["allocation_sites"]
    assert record["runtime_guard"] == "ALLOCATED then checked original LBOUND/UBOUND/SIZE"


def test_saved_storage_without_any_original_allocation_site_remains_native(tmp_path):
    analysis, routine, _ = case(tmp_path, "do i=-3,n\nwork(i,7)=a(i,7)\nenddo",
                                declarations="real(8),allocatable,save::work(:,:)")
    with pytest.raises(CompilationError, match="preceding original allocation site"):
        extracted(analysis, routine)


@pytest.mark.parametrize("later", [
    "call unavailable(work)",
    "if(flag) call unavailable(work)",
    "deallocate(work)",
    "work=a",
    "if(flag) work=a",
    "do j=1,1\nwork=a\nenddo",
])
def test_changed_saved_descriptor_uses_reached_runtime_guard_without_temporal_claim(tmp_path, later):
    analysis, routine, _ = case(tmp_path, "if(.not.allocated(work)) allocate(work(-3:n,7:8))\n"
                                + later + "\ndo i=-3,n\nwork(i,7)=a(i,7)\nenddo",
                                declarations="real(8),allocatable,save::work(:,:)")
    # Select the last direct loop when the intervening owner operation loops.
    index = 1 if later.startswith("do ") else 0
    record, = extracted(analysis, routine, index).allocation_guards
    assert record["allocation_proven"] is False
    assert record["runtime_guard"] == "ALLOCATED then checked original LBOUND/UBOUND/SIZE"


def test_saved_allocation_cannot_be_borrowed_with_escaping_target_attribute(tmp_path):
    analysis, routine, _ = case(tmp_path, "allocate(work(-3:n,7:8))\ndo i=-3,n\nwork(i,7)=a(i,7)\nenddo",
                                declarations="real(8),allocatable,target,save::work(:,:)")
    with pytest.raises(CompilationError, match="escape"):
        extracted(analysis, routine)


def test_foreign_node_or_mutated_source_role_cannot_borrow_original_proof(tmp_path):
    analysis, routine, path = case(tmp_path, "do i=-3,n\na(i,7)=1\nenddo")
    node, preceding, following = selected(routine)
    foreign_analysis = SourceEffects([path])
    foreign, _, _ = selected(foreign_analysis.routines[routine.qualified])
    with pytest.raises(CompilationError, match="original execution"):
        extract_region(analysis, routine, foreign, preceding=preceding, following=following)
    original = str(routine.execution)
    routine.execution.content = list(routine.execution.content) + [node]
    with pytest.raises(CompilationError, match="original source-backed routine"):
        extract_region(analysis, routine, node)
    assert str(routine.execution) != original
    path.write_text(path.read_text() + "! changed after parsing\n")
    with pytest.raises(CompilationError, match="source changed"):
        extract_region(analysis, routine, node)


def test_saved_guard_requires_original_procedure_local_storage(tmp_path):
    analysis, routine, _ = case(tmp_path, "do i=-3,n\na(i,7)=1\nenddo")
    binding = routine.scope.bindings["a"]
    with pytest.raises(CompilationError, match="saved array storage"):
        allocation_guard(analysis, routine, binding, ())


@pytest.mark.parametrize(("clauses", "accepted"), [("private(t)", True), ("shared(t)", False)])
def test_complete_joined_openmp_region_preserves_private_scalar_proof(tmp_path, clauses, accepted):
    analysis, routine, _ = case(tmp_path, f"""k=0
!$omp parallel default(shared) {clauses}
!$omp do
do i=-3,n
t=a(i,7)*2
a(i,7)=t+1
enddo
!$omp end do nowait
!$omp end parallel""", declarations="real(8)::t")
    group = original_joined_group(routine)
    if accepted:
        region = extract_region(analysis, routine, group)
        assert region.completion["has_openmp_in_closure"]
        assert "!$omp" not in region.source.lower()
        assert set(region.private_scalars) == {"i", "t"}
    else:
        with pytest.raises(CompilationError, match="PRIVATE"):
            extract_region(analysis, routine, group)


def test_openmp_opening_directives_in_specification_cannot_be_borrowed_as_serial_loop(tmp_path):
    analysis, routine, _ = case(tmp_path, """!$omp parallel default(shared)
!$omp do
do i=-3,n
a(i,7)=1
enddo
!$omp end do
!$omp end parallel""")
    with pytest.raises(CompilationError, match="OpenMP"):
        extracted(analysis, routine)


def test_comments_before_attached_joined_group_do_not_create_a_second_operation(tmp_path):
    analysis, routine, _ = case(tmp_path, """k=0
! ordinary documentation before the numerical region
!
!$omp parallel default(shared)
!$omp do
do i=-3,n
a(i,7)=a(i,7)+1
enddo
!$omp end do
!$omp end parallel""")
    original = tuple(_children(routine.execution))
    region = extract_region(analysis, routine, original[1:], preceding=original[:1])
    assert region.completion["has_openmp_in_closure"]
    assert region.written_resources == {"argument::a"}
    # Removing comments must not authorize a real operation outside the team.
    with pytest.raises(CompilationError, match="one counted DO or a complete joined team"):
        extract_region(analysis, routine, original)


@pytest.mark.parametrize("joined", [False, True])
def test_module_threadprivate_input_requires_original_serial_participation(tmp_path, joined):
    path = tmp_path / "threadprivate.f90"
    begin = "!$omp parallel default(shared)\n!$omp do\n" if joined else ""
    end = "!$omp end do\n!$omp end parallel\n" if joined else ""
    path.write_text(f"""module thread_state
real(8)::hidden(-3:4,7:8)
!$omp threadprivate(hidden)
end module
module inline_case
use thread_state,only:hidden
implicit none
contains
subroutine step(a,n)
real(8),intent(inout)::a(-3:,7:)
integer,intent(in)::n
integer::i,k
k=0
{begin}do i=-3,n
a(i,7)=a(i,7)+hidden(i,7)
enddo
{end}end subroutine
end module
""")
    analysis = SourceEffects([path])
    routine = analysis.routines["inline_case::step"]
    if joined:
        group = original_joined_group(routine)
        with pytest.raises(CompilationError, match="THREADPRIVATE"):
            extract_region(analysis, routine, group)
    else:
        assert "thread_state::hidden" in {item.resource for item in extracted(analysis, routine).parameters}


def test_extracted_loop_matches_original_native_fortran_complete_arrays_and_empty_shapes(tmp_path):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    analysis, routine, original = case(tmp_path, """do j=lbound(a,2)+1,ubound(a,2)-1
do i=lbound(a,1)+1,ubound(a,1)-1
a(i,j)=a(i,j)+real(i+17*j,8)+real(size(a)+size(a,1)+size(a,2)+lbound(a,1)+ubound(a,2),8) &
         +real(minimum+2147483647+1,8)
enddo
enddo""", declarations="integer,parameter::minimum=-2147483647-1")
    region = extracted(analysis, routine)
    assert [item.name for item in region.parameters] == ["a", "fort_region_lb_a_1", "fort_region_lb_a_2"]
    synthetic = region.write(tmp_path / "borrowed")
    module, procedure = region.entry.split("::")
    driver = tmp_path / "compare.f90"
    driver.write_text(f"""program compare
use inline_case,only:original=>step
use {module},only:borrowed=>{procedure}
implicit none
real(8),allocatable::actual(:,:),expected(:,:)
integer,parameter::xs(5)=[0,0,5,3,9],ys(5)=[0,5,0,4,7]
integer::shape,nx,ny,i,j,lower1,lower2
do shape=1,5
nx=xs(shape)
ny=ys(shape)
allocate(actual(-8:nx-9,-11:ny-12),expected(-8:nx-9,-11:ny-12))
do j=-11,ny-12
do i=-8,nx-9
actual(i,j)=real(i*7+j*3,8)*0.25d0
enddo
enddo
expected=actual
call original(expected,nx,.false.)
! These are the original procedure's dummy-descriptor bounds, independent of
! the caller allocation's different lower bounds and the synthetic LB1 dummy.
lower1=merge(-3,1,nx>0)
lower2=merge(7,1,ny>0)
call borrowed(actual,lower1,lower2)
if(any(actual/=expected)) error stop 'complete array or halo mismatch'
deallocate(actual,expected)
enddo
print *,'INLINE_NATIVE_FIELDS_OK'
end program
""")
    binary = tmp_path / "compare"
    result = subprocess.run([fortran, "-std=f2018", "-fcheck=all", "-ftrapv", str(original), str(synthetic),
                             str(driver), "-o", str(binary)], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "INLINE_NATIVE_FIELDS_OK" in result.stdout
