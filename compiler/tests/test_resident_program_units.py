"""Procedures outside modules are selected under prefixes of their own."""
import pytest

from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.resident.reductions import lower_reductions
from compiler.tests.resident_application_support import fortran_compiler, relax_project

pytestmark = pytest.mark.skipif(fortran_compiler() is None, reason="GNU Fortran is required for preprocessing")


def test_an_external_procedure_is_prepared_like_a_module_procedure(tmp_path):
    source = tmp_path / "legacy.f90"
    source.write_text("""subroutine relax(a, b, n, total)
integer, intent(in) :: n
real(8), intent(inout) :: a(n)
real(8), intent(in) :: b(n)
real(8), intent(out) :: total
integer :: i
!$omp parallel do
do i = 2, n - 1
  a(i) = 0.5d0 * (b(i-1) + b(i+1))
end do
!$omp end parallel do
total = 0
!$omp parallel do reduction(+: total)
do i = 1, n
  total = total + a(i)
end do
!$omp end parallel do
end subroutine
""")
    project = ResidentProject((source,), (Selection("all", ("$external::relax",), minimum_nests=4),))
    prepared = prepare(project, tmp_path / "prepared", fortran=fortran_compiler())
    # The stencil, and the three nests of the sum; the directives are gone as in a module procedure.
    assert prepared["selections"] == [{"label": "all", "entries": 3, "nests": 4, "rejected": []}]
    text = (tmp_path / "prepared/sources/legacy.f90").read_text()
    assert "!$omp" not in text
    assert text.count("call resident_require(fort_device_run(") == 3
    assert [(row["procedure"], row["compiled"]) for row in prepared["reductions"]] == [("$external::relax", True)]


def test_the_passes_find_external_procedures_and_named_main_programs():
    source = """program drive
  implicit none
  real(8) :: a(10), total
  integer :: i
  a = 1
  total = 0
  !$fort reduction(+: total)
  do i = 1, 10
    total = total + a(i)
  end do
  print *, total
contains
  subroutine inside()
    do i = 1, 10
      total = max(total, a(i))
    end do
  end subroutine
end program
subroutine apart(a, n, top)
  integer, intent(in) :: n
  real(8), intent(in) :: a(n)
  real(8), intent(inout) :: top
  integer :: i
  do i = 1, n
    top = max(top, a(i))
  end do
end subroutine
"""
    def found(*patterns):
        return [record["procedure"] for record in lower_reductions(source, patterns)[1]]

    assert found("$program::drive") == ["$program::drive"]
    assert found("$program::drive::inside") == ["$program::drive::inside"]
    assert found("$external::*") == ["$external::apart"]
    assert found("drive", "apart", "other::apart") == []


def test_a_main_program_with_its_time_loop_is_prepared(tmp_path):
    prepared = prepare(relax_project(), tmp_path / "prepared", fortran=fortran_compiler())
    # Three initial assignments in one entry, the sweep, the three nests of each maximum (in the
    # program and in the external procedure), the copy and the scaling.
    assert prepared["selections"] == [{"label": "all", "entries": 8, "nests": 12, "rejected": []}]
    assert [(row["procedure"], row["compiled"]) for row in prepared["reductions"]] == [
        ("$external::damp", True), ("$program::relax", True)]
    assert [(row["procedure"], row["lowered"]) for row in prepared["array_statements"]] == [
        ("$external::damp", True), *[("$program::relax", True)] * 4]
    # The directives before the output and the deallocation answer for the host statements there.
    assert prepared["audit"] == []
    text = (tmp_path / "prepared/sources/relax.f90").read_text()
    sweeps = text[text.index("do sweep = 1, sweeps"):text.index("open(newunit=unit")]
    # The host runs the sweep loop; the nests in its body are entries.
    assert sweeps.count("call resident_require(fort_device_run(") >= 3
    assert "call damp(u, n, top)" in sweeps
    # An external procedure's dummies get no TARGET: its callers have no interface to learn it from.
    damp = text[text.index("subroutine damp"):]
    assert "TARGET" not in damp.split("integer :: i, j")[0]
    assert "ALLOCATABLE, TARGET :: u" in text


def test_an_accumulation_inside_a_loop_is_rewritten_only_where_its_order_is_stated():
    source = """module sums
contains
subroutine rows(a, b, n, total)
real(8), intent(in) :: a(:,:)
real(8), intent(inout) :: b(:)
integer, intent(in) :: n
integer, intent(out) :: total
integer :: i, j, count
total = 0
do j = 1, n
  count = 0
  do i = 1, n
    if (a(i,j) > 0) count = count + 1
  end do
  b(j) = count
  {marker}
  do i = 1, n
    if (a(i,j) < 0) total = total + 1
  end do
end do
end subroutine
end module
"""
    # Without a word from the source the outer loop may be a nest whose iterations each count for themselves.
    text, records = lower_reductions(source.format(marker=""), ["sums::rows"])
    assert text == source.format(marker="")
    assert not any("tag" in record for record in records)
    text, records = lower_reductions(source.format(marker="!$fort reduction(+: total)"), ["sums::rows"])
    assert [record["variables"] for record in records if "tag" in record] == [["total"]]
    # The count of a row stays the statement it was.
    assert "if (a(i,j) > 0) count = count + 1" in text
    assert "total = total + fort_red1_q1(fort_red1_slot)" in text

