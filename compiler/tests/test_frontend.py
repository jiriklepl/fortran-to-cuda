"""Semantic frontend checks independent of dependence analysis and emission."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from compiler.analysis import build_execution_plan
from compiler.frontend import lower_file
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    CompilationError,
    Literal,
    Loop,
    Reference,
    ScalarType,
    Size,
    Unary,
)


def source_file(tmp_path: Path, routines: str, *, annotated: bool = True) -> Path:
    path = tmp_path / "kernels.f90"
    path.write_text(
        ("! kernels\n" if annotated else "! other\n")
        + "module example\nimplicit none\ninteger, parameter :: knd = kind(1.0d0)\ncontains\n"
        + routines
        + "\nend module example\n"
    )
    return path


def one_routine(specification: str, execution: str, arguments: str = "a,n") -> str:
    return f"! kernel\nsubroutine entry({arguments})\n{specification}\n{execution}\nend subroutine entry"


def test_signature_order_and_case_insensitive_names(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        one_routine(
            "integer, intent(in) :: N\nreal(knd), intent(out) :: A(:)\ninteger :: I",
            "do i = 1, n\n a(I) = 3.0_knd\nend do",
            "A,N",
        ),
    )
    function = lower_file(path, "ENTRY")
    assert [symbol.name for symbol in function.parameters] == ["A", "N"]
    assert function.module == "example"
    assert [symbol.id for symbol in function.symbols] == list(range(len(function.symbols)))
    loop = function.body.statements[0]
    assert isinstance(loop, Loop)
    assignment = loop.body.statements[0]
    assert isinstance(assignment, Assignment)
    assert assignment.target.symbol is function.parameters[0]
    assert assignment.location.path == str(path)
    assert assignment.location.line == 12
    with pytest.raises(FrozenInstanceError):
        function.name = "changed"  # type: ignore[misc]


def test_repeated_inline_calls_have_distinct_locals_and_shared_actuals(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        """! kernel
subroutine helper(out, inp, n)
real(knd), intent(inout) :: out(:)
real(knd), intent(in) :: inp(:)
integer, intent(in) :: n
integer :: i
real(knd) :: t
do i = 1, n
 t = inp(i) + 1.0_knd
 out(i) = t
end do
end subroutine helper
! kernel
subroutine entry(a,n)
real(knd), intent(inout) :: a(:)
integer, intent(in) :: n
integer :: i
call helper(a,a,n)
call HELPER(a,a,N)
end subroutine entry""",
    )
    function = lower_file(path, "entry")
    first, second = function.body.statements
    assert isinstance(first, Loop)
    assert isinstance(second, Loop)
    assert first.iterator != second.iterator
    assert len({symbol.id for symbol in function.symbols}) == len(function.symbols)
    first_t, first_out = first.body.statements
    second_t = second.body.statements[0]
    assert isinstance(first_t, Assignment)
    assert isinstance(second_t, Assignment)
    assert first_t.target.symbol != second_t.target.symbol
    assert isinstance(first_t.value, Binary)
    assert isinstance(first_t.value.left, ArrayAccess)
    assert first_t.value.left.symbol is first_out.target.symbol
    assert "entry at" in first_t.location.call_stack[0]
    assert "helper" in first_t.location.call_stack[0]
    assert first_t.location.call_stack != second_t.location.call_stack


@pytest.mark.parametrize("offset", [0, -1, 1])
def test_inlined_array_aliases_participate_in_dependence_analysis(tmp_path: Path, offset: int) -> None:
    subscript = "i" if offset == 0 else f"i{offset:+}"
    path = source_file(
        tmp_path,
        f"""! kernel
subroutine helper(out,inp,n)
real(knd), intent(inout) :: out(:)
real(knd), intent(in) :: inp(:)
integer, intent(in) :: n
integer :: i
do i=2,n-1
 out(i)=inp({subscript})+1.0_knd
enddo
end subroutine helper
! kernel
subroutine entry(a,n)
real(knd), intent(inout) :: a(:)
integer, intent(in) :: n
call helper(a,a,n)
end subroutine entry""",
    )
    function = lower_file(path, "entry")
    if offset == 0:
        plan = build_execution_plan(function)
        assert len(plan.regions) == 1
        target, value = plan.regions[0].assignments[0].target, plan.regions[0].assignments[0].value
        assert isinstance(value, Binary)
        assert isinstance(value.left, ArrayAccess)
        assert value.left.symbol is target.symbol
        return
    with pytest.raises(CompilationError, match="RAW|WAR") as failure:
        build_execution_plan(function)
    assert failure.value.location.call_stack
    assert "helper" in str(failure.value)
    assert "inlined through entry at" in str(failure.value)
    assert "witness" in str(failure.value)


def test_ordered_host_assignments_and_loops(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        one_routine(
            "real(knd), intent(inout) :: a(:)\ninteger, intent(in) :: n\ninteger :: i\nreal(knd) :: t",
            "do i=1,n\n a(i)=1.0_knd\nenddo\nt=2.0_knd\ndo i=1,n\n a(i)=a(i)*t\nenddo\nt=3.0_knd",
        ),
    )
    function = lower_file(path, "entry")
    assert [type(node) for node in function.body.statements] == [Loop, Assignment, Loop, Assignment]


def test_size_dimensions_parentheses_and_literal_normalization(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        one_routine(
            "real(knd), intent(out), dimension(:,:) :: a\ninteger, intent(in) :: n\ninteger :: i,j",
            "j = size(a,2)\ndo i=1,size(a,1),+1\n a(i,j)= (-(1.0d0 + 2.0_knd)) / (+3.0_knd)\nenddo",
        ),
    )
    function = lower_file(path, "entry")
    assignment, loop = function.body.statements
    assert isinstance(assignment.value, Size)
    assert assignment.value.dimension == 2
    assert isinstance(loop, Loop)
    assert isinstance(loop.upper, Size)
    value = loop.body.statements[0].value
    assert isinstance(value, Binary)
    assert value.operator == "/"
    assert isinstance(value.left, Unary)
    assert isinstance(value.left.operand, Binary)
    assert value.left.operand.left == Literal("1.0e0", ScalarType.REAL)


def test_unit_stride_and_comments_inside_loop(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        one_routine(
            "real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n\ninteger :: i",
            "! before loop\ndo i=1,n,1\n! inside loop\n a(i)=1.0_knd\n! after assignment\nenddo",
        ),
    )
    function = lower_file(path, "entry")
    assert isinstance(function.body.statements[0], Loop)
    assert len(function.body.statements[0].body.statements) == 1


def test_real_literal_precision_is_preserved(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        one_routine(
            "real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n",
            "a(1)=0.1\na(2)=0.1e0\na(3)=0.1d0\na(4)=0.1_knd\na(5)=(0.1+0.2)*0.3_knd",
        ),
    )
    function = lower_file(path, "entry")
    literals = [statement.value for statement in function.body.statements[:4]]
    assert all(isinstance(value, Literal) for value in literals)
    assert [value.dtype for value in literals] == [
        ScalarType.REAL32,
        ScalarType.REAL32,
        ScalarType.REAL,
        ScalarType.REAL,
    ]
    expression = function.body.statements[4].value
    assert isinstance(expression, Binary)
    assert isinstance(expression.left, Binary)
    assert expression.left.left.dtype == ScalarType.REAL32
    assert expression.left.right.dtype == ScalarType.REAL32
    assert expression.right.dtype == ScalarType.REAL


@pytest.mark.parametrize(
    ("specification", "execution", "message"),
    [
        ("real(knd), intent(out) :: a(:)\ninteger, intent(inout) :: n", "", "writable scalar"),
        ("real, intent(out) :: a(:)\ninteger, intent(in) :: n", "", "REAL\\(knd\\)"),
        ("real(knd), intent(out) :: a(:)\ninteger :: n", "", "requires INTENT"),
        ("real(knd), pointer, intent(out) :: a(:)\ninteger, intent(in) :: n", "", "POINTER"),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n\nreal(knd) :: b(3)", "", "assumed shape"),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n\nreal(knd) :: b(:)", "", "local arrays"),
        ("real(knd), intent(out) :: a(:,:,:,:)\ninteger, intent(in) :: n", "", "rank 1-3"),
        ("real(knd), intent(out) :: a(0:)\ninteger, intent(in) :: n", "", "assumed shape"),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n\nreal(knd), save :: b", "", "SAVE"),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n\nreal(knd) :: b=1.0_knd", "", "initialized"),
        (
            "real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n\ninteger :: i",
            "do i=1,n,2\n a(i)=1.0_knd\nenddo",
            "unit stride",
        ),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "if(n>0) a(1)=1.0_knd", "If_Stmt"),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "a(1)=sin(1.0_knd)", "unsupported intrinsic"),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "a(1)=2.0_knd**3", "unsupported expression"),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "a(1)=2_knd", "integer literal kinds"),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "a(1)=a(1:2)", "unsupported expression"),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "a(1)=size(a,n)", "literal dimension"),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "a(1)=size(a,2)", "outside rank"),
        ("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "a(1)=missing", "undeclared"),
        ("real(knd), intent(in) :: a(:)\ninteger, intent(in) :: n", "a(1)=1.0_knd", "cannot write INTENT"),
        (
            "real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n\ninteger :: i",
            "do i=1,n\n i=2\nenddo",
            "active loop iterator",
        ),
    ],
)
def test_unsupported_constructs_have_source_locations(
    tmp_path: Path, specification: str, execution: str, message: str
) -> None:
    path = source_file(tmp_path, one_routine(specification, execution))
    with pytest.raises(CompilationError, match=message) as failure:
        lower_file(path, "entry")
    assert failure.value.location is not None
    assert failure.value.location.path == str(path)
    assert failure.value.location.line > 1


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ("a(:),n", "whole variables"),
        ("a,n+1", "whole variables"),
        ("a,n=n", "whole variables"),
        ("n,n", "type or rank mismatch"),
        ("a", "wrong number"),
    ],
)
def test_call_argument_contract(tmp_path: Path, arguments: str, message: str) -> None:
    path = source_file(
        tmp_path,
        "! kernel\nsubroutine helper(b,n)\nreal(knd), intent(out) :: b(:)\ninteger, intent(in) :: n\nend subroutine\n"
        + one_routine("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", f"call helper({arguments})"),
    )
    with pytest.raises(CompilationError, match=message):
        lower_file(path, "entry")


def test_readonly_formal_cannot_be_written_through_actual(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        """! kernel
subroutine helper(b,n)
real(knd), intent(in) :: b(:)
integer, intent(in) :: n
b(1)=1.0_knd
end subroutine
"""
        + one_routine("real(knd), intent(inout) :: a(:)\ninteger, intent(in) :: n", "call helper(a,n)"),
    )
    with pytest.raises(CompilationError, match="cannot write INTENT") as failure:
        lower_file(path, "entry")
    assert "helper" in str(failure.value)
    assert failure.value.location.call_stack


def test_readonly_actual_cannot_be_passed_to_writable_formal(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        "! kernel\nsubroutine helper(b,n)\nreal(knd), intent(out) :: b(:)\ninteger, intent(in) :: n\nend subroutine\n"
        + one_routine("real(knd), intent(in) :: a(:)\ninteger, intent(in) :: n", "call helper(a,n)"),
    )
    with pytest.raises(CompilationError, match="read-only variable"):
        lower_file(path, "entry")


def test_recursive_inline_cycle(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        one_routine("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "call helper(a,n)")
        + "\n! kernel\nsubroutine helper(b,n)\nreal(knd), intent(out) :: b(:)\ninteger, intent(in) :: n\ncall entry(b,n)\nend subroutine helper",
    )
    with pytest.raises(CompilationError, match="recursive kernel call") as failure:
        lower_file(path, "entry")
    assert failure.value.location.call_stack


def test_unused_unsupported_routine_does_not_change_entry(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        "! kernel\nsubroutine unused(a)\nlogical, intent(out) :: a\na=.true.\nend subroutine\n"
        + one_routine("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "a(1)=1.0_knd"),
    )
    function = lower_file(path, "entry")
    assert len(function.body.statements) == 1


def test_unsupported_called_declaration_has_call_provenance(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        "! kernel\nsubroutine helper(a,n)\nreal(knd), intent(out) :: a(:)\ninteger, intent(inout) :: n\nend subroutine\n"
        + one_routine("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "call helper(a,n)"),
    )
    with pytest.raises(CompilationError, match="writable scalar") as failure:
        lower_file(path, "entry")
    assert failure.value.location.call_stack
    assert "entry at" in str(failure.value)


def test_unknown_unannotated_call_is_diagnosed(tmp_path: Path) -> None:
    path = source_file(
        tmp_path,
        "subroutine helper\nend subroutine\n"
        + one_routine("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", "call helper"),
    )
    with pytest.raises(CompilationError, match="unannotated or unknown"):
        lower_file(path, "entry")


def test_annotation_contract(tmp_path: Path) -> None:
    path = source_file(
        tmp_path, one_routine("real(knd), intent(out) :: a(:)\ninteger, intent(in) :: n", ""), annotated=False
    )
    with pytest.raises(CompilationError, match="begin with"):
        lower_file(path, "entry")
    path = source_file(tmp_path, "subroutine entry\nend subroutine entry")
    with pytest.raises(CompilationError, match="not found"):
        lower_file(path, "entry")


@pytest.mark.parametrize("component", ["u", "v", "w"])
def test_repository_stencils_lower_to_four_ordered_loop_regions(component: str) -> None:
    root = Path(__file__).resolve().parents[2]
    function = lower_file(root / "fortran-stencils" / f"elmm_cd{component}.f90", f"CD{component.upper()}")
    assert sum(isinstance(statement, Loop) for statement in function.body.statements) == 4
    for statement in function.body.statements:
        if isinstance(statement, Loop):
            assert statement.location.call_stack
        else:
            assert isinstance(statement.target, Reference)
