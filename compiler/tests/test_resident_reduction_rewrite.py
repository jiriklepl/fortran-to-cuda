"""Loop nests that only accumulate scalars are rewritten into nests that store partial results."""
from pathlib import Path

import pytest

from compiler.resident.configure import configure_sources
from compiler.resident.directives import expand_directives
from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.resident.reductions import DIRECT, FINAL, SLOTS, _rename, lower_reductions
from compiler.scopes.resident_splice import ResidentDomain
from compiler.scopes.segments import license_line, parse_license
from compiler.tests.resident_application_support import fortran_compiler

DOMAIN = ResidentDomain("my_state", "flow", "check")

SOURCE = """module measures
  implicit none
  integer, parameter :: knd = kind(1.0d0)
  real(knd), allocatable :: u(:,:,:), mass(:)
  integer :: nx, ny, nz
  real(knd) :: dx
contains
  subroutine limits(fastest, nan, cells)
    real(knd), intent(out) :: fastest
    logical, intent(out) :: nan
    integer, intent(out) :: cells
    integer :: i, j, k
    real(knd) :: p
    fastest = 0
    nan = .false.
    cells = 0
    do k = 1, nz
      do j = 1, ny
        do i = 1, nx, 2
          p = abs(u(i,j,k)) / dx   ! speed over spacing
          fastest = max(fastest, p)
          if (p /= p) nan = .true.
          if (p > 1) cells = cells + 1
        end do
      end do
    end do
  end subroutine limits

  real(knd) function total() result(amount)
    integer :: q
    amount = 0
    @LICENSE@
    do q = 1, size(mass)
      amount = amount + mass(q)
    end do
  end function total

  subroutine plain(w)
    real(knd), intent(inout) :: w(:)
    integer :: q
    do q = 1, size(w)
      w(q) = 2 * w(q)
    end do
  end subroutine plain
end module measures
"""
LICENSED = SOURCE.replace("@LICENSE@", "!$fort reduction(+: amount)")
UNLICENSED = SOURCE.replace("    @LICENSE@\n", "")


def test_an_openmp_reduction_clause_becomes_a_statement_about_the_order():
    assert parse_license("parallel do private(i, J) reduction(max:Top) reduction(+ : a,b) schedule(static)") == (
        {"top": "max", "a": "+", "b": "+"}, {"i", "j"})
    assert license_line("parallel do private(i,j) reduction(max:top) reduction(+:a,b)") == (
        "!$fort reduction(max: top) reduction(+: a, b) private(i, j)")
    # A directive without a reduction clause says nothing about any order.
    assert license_line("parallel do private(i,j) schedule(guided,5)") is None
    # FIRSTPRIVATE and LASTPRIVATE variables keep their values across the loop's boundaries: they are not private.
    assert parse_license("parallel do firstprivate(a) lastprivate(b) private(c)") == ({}, {"c"})


def test_a_renamed_variable_keeps_its_name_in_character_literals_comments_and_fields():
    assert _rename(["  print *, 'i = ', i, \"i\", a%i ! i\n", "  x = 'it''s i' // trim(i)\n"], {"i": "k"}) == [
        "  print *, 'i = ', k, \"i\", a%i ! i\n", "  x = 'it''s i' // trim(k)\n"]


def test_a_nest_that_only_accumulates_gets_partial_results_a_fold_and_its_original():
    text, records = lower_reductions(LICENSED, ["measures::*"])
    limits, total = records
    assert (limits["procedure"], limits["line"], limits["variables"]) == (
        "measures::limits", 17, ["cells", "fastest", "nan"])
    assert limits["tag"] == "fort_red1_"
    assert total["tag"] == "fort_red2_"
    body = text[text.index("subroutine limits"):text.index("end subroutine limits")]
    # The procedure declares what the rewritten nests need after its own declarations, and asks a
    # directive whether the values are on the device: nothing else in it names the domain.
    assert body.index("real(knd) :: p\n") < body.index("logical, parameter :: fort_red1_compiled = .true.")
    assert "    fort_red1_now = .false.\n    !$fort active(fort_red1_now)\n" in body
    assert "flow" not in body
    assert f"REAL(KIND = knd), save, target :: fort_red1_p1({SLOTS}), fort_red1_q1({FINAL})" in body
    # A logical is accumulated in one byte and stored as an integer; an integer count stays one.
    assert "logical(1) :: fort_red1_a2\n" in body
    assert "integer, save, target :: fort_red1_p2(" in body
    # A short nest stores FINAL partial results directly; a long one stores SLOTS and folds them once more.
    short, long = body.index(f"if (fort_red1_total <= {DIRECT}) then"), body.index(f"fort_red1_used = min({SLOTS}, fort_red1_total)")
    assert short < body.index(f"do fort_red1_t = fort_red1_slot, fort_red1_total, {FINAL}\n") < long
    assert long < body.index(f"do fort_red1_t = fort_red1_slot, fort_red1_total, {SLOTS}\n")
    assert long < body.index(f"do fort_red1_t = fort_red1_slot, fort_red1_used, {FINAL}\n")
    assert "fort_red1_a1 = max(fort_red1_a1, fort_red1_p1(fort_red1_t))\n" in body
    # Each slot takes every stride-th iteration, with the loop variables computed from its position.
    assert "fort_red1_v_i = fort_red1_l3 + mod(fort_red1_o, fort_red1_n3) * fort_red1_s3\n" in body
    assert "fort_red1_n3 = max(0, ((nx) - fort_red1_l3 + fort_red1_s3) / fort_red1_s3)\n" in body
    assert "fort_red1_s3 = 2\n" in body
    # The body is the original one, on copies of what it assigns; its comment stays.
    assert body.count(
        "          fort_red1_v_p = abs(u(fort_red1_v_i,fort_red1_v_j,fort_red1_v_k)) / dx   ! speed over spacing\n"
        "          fort_red1_a1 = max(fort_red1_a1, fort_red1_v_p)\n"
        "          if (fort_red1_v_p /= fort_red1_v_p) fort_red1_a2 = .true.\n"
        "          if (fort_red1_v_p > 1) fort_red1_a3 = fort_red1_a3 + 1\n") == 2
    # A maximum starts from minus infinity, so that it is the same maximum whatever the values are.
    assert "fort_red1_i1 = fort_red_ieee_value(fastest, fort_red_negative_inf)\n" in body
    # The host folds what came back and leaves the loop variables as a completed DO would.
    fold = body.index("!$fort observe(fort_red1_q1, fort_red1_q2, fort_red1_q3) if(fort_red1_on)")
    assert fold < body.index("fastest = max(fastest, fort_red1_q1(fort_red1_slot))")
    assert "if (fort_red1_q2(fort_red1_slot) /= 0) nan = .true.\n" in body
    assert "cells = cells + fort_red1_q3(fort_red1_slot)\n" in body
    assert "      i = fort_red1_l3 + fort_red1_n3 * fort_red1_s3\n" in body[fold:]
    # The original nest follows for the other case, marked as host code that needs the values back.
    rest = body[body.index("    else\n", fold):]
    assert rest.startswith("    else\n      !$fort publish if(fort_red1_wide /= 0)\n    do k = 1, nz\n      !$fort host\n")
    assert "          fastest = max(fastest, p)\n" in rest


def test_a_single_loop_and_a_function_result_take_the_short_form():
    text, records = lower_reductions(LICENSED, ["measures::total"])
    record, = records
    assert (record["procedure"], record["variables"]) == ("measures::total", ["amount"])
    body = text[text.index("function total"):text.index("end function total")]
    assert "fort_red1_v_q = fort_red1_l1 + (fort_red1_t - 1) * fort_red1_s1\n" in body
    assert "fort_red1_a1 = fort_red1_a1 + mass(fort_red1_v_q)\n" in body
    assert "amount = amount + fort_red1_q1(fort_red1_slot)\n" in body
    # The line that stated the order is used up, and other procedures are left alone.
    assert "!$fort reduction" not in text
    assert text[text.index("subroutine limits"):text.index("end subroutine limits")] == LICENSED[
        LICENSED.index("subroutine limits"):LICENSED.index("end subroutine limits")]
    # What is left expands like any source with residency directives.
    expanded, directives = expand_directives(text, DOMAIN)
    assert [item["directive"] for item in directives] == ["active", "observe", "host", "publish", "host"]
    assert "fort_red1_now = flow%active" in expanded
    assert "!$fort" not in expanded


def test_nests_that_follow_each_other_bring_their_values_back_together():
    twice = LICENSED.replace("""    do q = 1, size(mass)
      amount = amount + mass(q)
    end do
""", """    do q = 1, size(mass)
      amount = amount + mass(q)
    end do
    ! and once more, backwards
    !$fort reduction(+: amount)
    do q = size(mass), 1, -1
      amount = amount + mass(q)
    end do
""")
    text, records = lower_reductions(twice, ["measures::total"])
    assert [record["tag"] for record in records] == ["fort_red1_", "fort_red2_"]
    body = text[text.index("function total"):text.index("end function total")]
    # Both nests are started before the host asks for either result, and it asks once.
    assert body.count("!$fort observe(") == 1
    observe = body.index("!$fort observe(fort_red1_q1, fort_red2_q1) if(fort_red1_on .or. fort_red2_on)")
    assert body.index("fort_red2_a1 = fort_red2_a1 + mass(fort_red2_v_q)") < observe
    first, second = (body.index(f"amount = amount + fort_red{number}_q1(fort_red{number}_slot)") for number in (1, 2))
    assert observe < first < body.index("    ! and once more, backwards\n") < second
    # A nest that reads what the one before it accumulates waits for that result.
    chained = twice.replace("do q = size(mass), 1, -1\n      amount = amount + mass(q)",
                            "do q = size(mass), 1, -1\n      heavy = max(heavy, amount * mass(q))").replace(
        "    integer :: q\n    amount = 0", "    integer :: q\n    real(knd) :: heavy\n    amount = 0")
    text, records = lower_reductions(chained, ["measures::total"])
    assert [record["variables"] for record in records] == [["amount"], ["heavy"]]
    assert text.count("!$fort observe(") == 2


def test_a_sum_of_reals_is_rewritten_only_where_the_order_is_free():
    text, records = lower_reductions(UNLICENSED, ["measures::total", "measures::plain"])
    assert text == UNLICENSED
    record, = records
    assert record["procedure"] == "measures::total"
    assert "tag" not in record
    assert "the sum amount depends on the order of its terms" in record["reason"]
    # A nest that accumulates nothing is not mentioned at all.
    assert lower_reductions(UNLICENSED, ["measures::plain"]) == (UNLICENSED, [])


@pytest.mark.parametrize(("old", "new", "reason"), [
    ("          if (p > 1) cells = cells + 1\n", "          if (p > 1) cells = cells + 1\n          u(i,j,k) = 0\n",
     "an assignment to an array element or a field"),
    ("          if (p > 1) cells = cells + 1\n", "          if (p > 1) cells = cells + 1\n          call note(p)\n",
     "a statement other than assignments, IF and DO: Call_Stmt"),
    ("  end subroutine limits", "    fastest = fastest + p\n  end subroutine limits",
     "p is named again after the nest and no private clause names it"),
    ("    logical, intent(out) :: nan\n", "    logical(1), intent(out) :: nan\n",
     "nan has a type its accumulation is not rewritten for"),
    ("        do i = 1, nx, 2\n          p = ", "        do i = 1, nx, 2\n          !$fort publish\n          p = ",
     "a directive inside the nest"),
])
def test_what_the_rewrite_cannot_keep_exact_stays_as_it_is(old, new, reason):
    assert old in UNLICENSED
    changed = UNLICENSED.replace(old, new)
    text, records = lower_reductions(changed, ["measures::limits"])
    record, = records
    assert text == changed
    assert "tag" not in record
    assert reason in record["reason"]


def test_a_private_clause_stands_in_for_what_the_procedure_cannot_show():
    # The temporaries of an internal procedure's nest belong to its host; only a clause says they are free.
    hosted = UNLICENSED.replace("""    integer :: i, j, k
    real(knd) :: p
    fastest = 0
    nan = .false.
    cells = 0
    do k = 1, nz""", """    integer :: i, j, k
    real(knd) :: p
    fastest = 0
    nan = .false.
    cells = 0
    call search
  contains
    subroutine search
    @LICENSE@
    do k = 1, nz""").replace("    end do\n  end subroutine limits", "    end do\n    end subroutine search\n  end subroutine limits")
    text, records = lower_reductions(hosted.replace("    @LICENSE@\n", ""), ["measures::limits::search"])
    assert "p is not local to the procedure and no private clause names it" in records[0]["reason"]
    text, records = lower_reductions(hosted.replace("@LICENSE@", "!$fort reduction(max: fastest) private(i, j, k, p)"),
                                     ["measures::limits::search"])
    assert records[0]["variables"] == ["cells", "fastest", "nan"]
    # The internal procedure declares the copies with the types its host gave the variables.
    search = text[text.index("subroutine search"):text.index("end subroutine search")]
    assert "REAL(KIND = knd) :: fort_red1_v_p\n" in search
    # A clause that names another operator than the nest uses is a mistake to report, not to follow.
    text, records = lower_reductions(hosted.replace("@LICENSE@", "!$fort reduction(min: fastest) private(i, j, k, p)"),
                                     ["measures::limits::search"])
    assert "fastest accumulates with max, not with the stated min" in records[0]["reason"]


OPENMP = """module sums
  implicit none
  real(8), allocatable :: mass(:)
contains
  real(8) function total() result(amount)
    integer :: q
    amount = 0
    !$omp parallel do private(q) &
    !$omp& reduction(+: amount)
    do q = 1, size(mass)
      amount = amount + mass(q)
    end do
    !$omp end parallel do
  end function total

  real(8) function odd() result(amount)
    integer :: q
    amount = 0
    !$omp parallel do reduction(+: amount)
    do q = 1, size(mass)
      amount = amount + erf(mass(q))
    end do
  end function odd
end module sums
"""


def test_the_preparation_carries_an_openmp_clause_to_the_rewrite_and_reports_what_compiled(tmp_path):
    fortran = fortran_compiler()
    if fortran is None:
        pytest.skip("gfortran is required")
    source = tmp_path / "sums.f90"
    source.write_text(OPENMP)
    record = configure_sources([source], ["sums::total"], tmp_path / "configured", fortran=fortran)
    text = Path(record["sources"][0]["prepared"]["path"]).read_text()
    # The directive of the selected function is gone, its clause stays; the other function is untouched.
    assert "    !$fort reduction(+: amount) private(q)\n\n    do q = 1, size(mass)" in text
    assert text.count("!$omp") == 1
    assert [row["procedure"] for row in record["sources"][0]["serialized_directives"]] == ["sums::total"] * 3

    project = ResidentProject(sources=(source,), selections=(Selection("sums", ("sums::*",)),))
    prepared = prepare(project, tmp_path / "prepared", fortran=fortran)
    compiled, kept = prepared["reductions"]
    assert (compiled["procedure"], compiled["variables"], compiled["compiled"]) == ("sums::total", ["amount"], True)
    # ERF is not among the functions a compiled nest may call, so that loop runs on the host as before.
    assert (kept["procedure"], kept["compiled"]) == ("sums::odd", False)
    assert "erf" in kept["reason"].lower()
    summary, = prepared["selections"]
    assert (summary["entries"], summary["nests"], summary["rejected"]) == (2, 3, [])
    text = (tmp_path / "prepared/sources/sums.f90").read_text()
    assert "logical, parameter :: fort_red1_compiled = .true.\n" in text
    assert "logical, parameter :: fort_red2_compiled = .false.\n" in text
    odd = text[text.index("function odd"):text.index("end function odd")]
    # Its rewritten nests are left as plain text behind the switch; the original follows a publication.
    assert "fort_device_run" not in odd
    publish = odd.index("if (resident%active .and. (fort_red2_wide /= 0)) call resident%publish()")
    assert publish < odd.index("amount = amount + erf(mass(q))")
