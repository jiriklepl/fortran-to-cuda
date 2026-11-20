"""Array assignments outside loops are written as the loop nests they stand for."""
import pytest

from compiler.resident.array_statements import lower_array_statements

HEAD = """module work
  implicit none
  integer :: n
  real(8), allocatable :: total(:,:), other(:,:)
  real(8), pointer :: view(:,:)
contains
  subroutine step(a, b, w, list)
    real(8), intent(inout) :: a(:,:)
    real(8), dimension(0:,0:), intent(in) :: b
    real(8), intent(in) :: w
    integer, intent(in) :: list(:)
    integer :: j
"""
TAIL = """  end subroutine
end module
"""


def lowered(statement):
    text, records = lower_array_statements(HEAD + "    " + statement + "\n" + TAIL, ["work::step"])
    body = text.split("    integer :: j\n", 1)[1].split("  end subroutine", 1)[0]
    return body, records


def test_a_whole_array_statement_becomes_a_nest_over_the_assigned_array():
    body, records = lowered("a = a + b * w")
    assert records == [{"procedure": "work::step", "line": 13, "lowered": True, "tag": "fort_arr1_",
                        "arrays": ["a", "b"], "text": "a = a + b * w"}]
    # One loop per dimension, the last outermost; the other array is read at the same position of its own bounds.
    assert body == (
        "    integer :: fort_arr1_1, fort_arr1_2\n"
        "    do fort_arr1_2 = lbound(a,2), ubound(a,2)\n"
        "      do fort_arr1_1 = lbound(a,1), ubound(a,1)\n"
        "        a(fort_arr1_1, fort_arr1_2) = a(fort_arr1_1, fort_arr1_2) + b(fort_arr1_1 - (lbound(a,1)) + &\n"
        "          (lbound(b,1)), fort_arr1_2 - (lbound(a,2)) + (lbound(b,2))) * w\n"
        "      end do\n"
        "    end do\n")


def test_sections_run_over_their_own_bounds_and_scalar_subscripts_stay():
    body, _records = lowered("a(2:n, j) = sqrt(b(1:n-1, j)) + a(2:n, j) - max(b(0, 0), w)")
    assert body == (
        "    integer :: fort_arr1_1\n"
        "    do fort_arr1_1 = 2, n\n"
        "      a(fort_arr1_1, j) = SQRT(b(fort_arr1_1 - (2) + (1), j)) + a(fort_arr1_1, j) - MAX(b(0, 0), w)\n"
        "    end do\n")
    # A step is the loop's step; a section with another step is read at the matching element.
    body, _records = lowered("a(n:1:-1, 1) = b(1:2*n:2, 1)")
    assert "do fort_arr1_1 = n, 1, - 1\n" in body
    assert "a(fort_arr1_1, 1) = b((1) + ((fort_arr1_1 - (n)) / (- 1)) * (2), 1)\n" in body
    # A scalar on the right is assigned to every element.
    body, _records = lowered("total = 0")
    assert "do fort_arr1_2 = lbound(total,2), ubound(total,2)" in body
    assert "total(fort_arr1_1, fort_arr1_2) = 0\n" in body


def test_a_one_line_if_keeps_its_condition_around_the_nest():
    body, _records = lowered("if (w > 0) a(:, 1) = b(:, 1)")
    assert body == (
        "    integer :: fort_arr1_1\n"
        "    if (w > 0) then\n"
        "      do fort_arr1_1 = lbound(a,1), ubound(a,1)\n"
        "        a(fort_arr1_1, 1) = b(fort_arr1_1 - (lbound(a,1)) + (lbound(b,1)), 1)\n"
        "      end do\n"
        "    end if\n")


@pytest.mark.parametrize(("statement", "reason"), [
    # The statement evaluates its right-hand side first; a nest would read what it has just assigned.
    ("a(2:n, 1) = a(1:n-1, 1)", "it reads other elements of a than it assigns"),
    ("total = other", "it may give total a new shape"),
    ("a = view", "view is a pointer and may share storage with another array"),
    ("a = scale * b", "scale is not declared in this file with all its attributes"),
    ("a(:, 1) = a(list, 1)", "an array is used as a subscript: list"),
    ("a = spread(w, 1, n)", "the intrinsic spread is not applied element by element"),
    ("a(:, 1) = b(:, 1); j = 1", "it is labelled or shares its line with another statement"),
])
def test_a_statement_that_a_nest_would_not_repeat_exactly_stays(statement, reason):
    body, records = lowered(statement)
    assert body == "    " + statement + "\n"
    assert records == [{"procedure": "work::step", "line": 13, "reason": reason}]


def test_working_storage_of_the_procedure_is_not_looked_at():
    # A small local array belongs to the host statements around it.
    source = HEAD.replace("    integer :: j\n", "    integer :: j\n    real(8) :: flux(6)\n") + "    flux = 0\n" + TAIL
    assert lower_array_statements(source, ["work::step"]) == (source, [])
    # One that outlives the call is storage like any other.
    saved = source.replace("real(8) :: flux(6)", "real(8), save :: flux(6)")
    text, records = lower_array_statements(saved, ["work::step"])
    assert "flux(fort_arr1_1) = 0" in text
    assert records[0]["lowered"]


def test_other_statements_and_other_procedures_are_left_alone():
    # A scalar assignment, an element, an assignment inside a loop (the front end's), a field of a type.
    source = HEAD + "    j = 1\n    a(1, 1) = w\n    do j = 1, n\n      a(:, j) = b(:, j)\n    end do\n" + TAIL
    assert lower_array_statements(source, ["work::step"]) == (source, [])
    source = HEAD + "    a = 0\n" + TAIL
    assert lower_array_statements(source, ["work::other", "elsewhere::*"]) == (source, [])
