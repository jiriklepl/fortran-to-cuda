"""Calls to a small procedure are replaced by its statements, so that its nests join the caller's."""
import pytest

from compiler.resident.inlining import inline_calls
from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.tests.resident_application_support import fortran_compiler

SOURCE = """module halos
  implicit none
  integer, parameter :: knd = kind(1.0d0)
  integer :: unx, vnx
contains
  subroutine edge(a, component)
    real(knd), intent(inout) :: a(-1:)
    integer, intent(in) :: component
    integer :: i, n

    if (component==1) then
      n = unx
    else
      n = vnx
    end if
    ! what the field's own cells hold, doubled
    do i = 1, n
      a(i) = 2 * a(i)
    end do
  end subroutine edge

  subroutine both(u, v, w)
    real(knd), intent(inout) :: u(-1:), v(-1:), w(0:)
    integer :: i
    call edge(u, 1)
    call edge(v, 2)   ! the other component
    do i = 1, vnx
      w(i) = u(i) + v(i)
    end do
  end subroutine both
end module halos
"""


def inlined(source=SOURCE, callers=("halos::both",)):
    return inline_calls(source, callers, ["halos::edge"])


def test_each_call_becomes_the_statements_its_arguments_select():
    text, records = inlined()
    assert records == [{"caller": "halos::both", "callee": "halos::edge", "line": line, "inlined": True}
                       for line in (25, 26)]
    body = text[text.index("subroutine both"):text.index("end subroutine both")]
    # The constant argument decides the branch on it, and the one value the branch assigned is put in
    # its place: nothing but the loop is left of each call, on the caller's array.
    assert ("    ! what the field's own cells hold, doubled\n"
            "    do fort_inl1_i = 1, unx\n"
            "      u(fort_inl1_i) = 2 * u(fort_inl1_i)\n"
            "    end do\n") in body
    assert "    do fort_inl2_i = 1, vnx\n      v(fort_inl2_i) = 2 * v(fort_inl2_i)\n" in body
    assert "call edge" not in body
    assert "component" not in body
    assert "fort_inl1_n" not in body
    # The callee's loop variable gets a name of its own per call, declared after the caller's declarations.
    assert body.index("    integer :: i\n") < body.index("    INTEGER :: fort_inl1_i\n") < body.index(
        "    INTEGER :: fort_inl2_i\n") < body.index("    ! what the field's own cells hold")
    # The callee itself, and procedures that were not asked for, stay as they are.
    assert text[:text.index("subroutine both")] == SOURCE[:SOURCE.index("subroutine both")]
    assert inlined(callers=("halos::edge",)) == (SOURCE, [])


@pytest.mark.parametrize(("old", "new", "reason"), [
    ("u(-1:), v(-1:), w(0:)", "u(0:), v(-1:), w(0:)", "u is not a whole array declared with the bounds of a"),
    ("    call edge(u, 1)\n", "    call edge(u(:), 1)\n", "u(:) is not a whole array declared with the bounds of a"),
    ("    call edge(u, 1)\n", "    call edge(u, component=1)\n", "the arguments are not all given, by position"),
    ("    integer, intent(in) :: component\n", "    integer :: component\n",
     "its scalar argument component is not INTENT(IN)"),
    ("    integer :: i, n\n", "    integer :: i, n\n    real(knd) :: work(4)\n", "its local work is more than a plain scalar"),
    ("    do i = 1, n\n      a(i)", "    if (n < 1) return\n    do i = 1, n\n      a(i)", "it has a RETURN statement"),
])
def test_a_call_that_text_substitution_would_not_reproduce_stays_a_call(old, new, reason):
    assert old in SOURCE
    changed = SOURCE.replace(old, new)
    text, records = inlined(changed)
    assert reason in records[0]["reason"]
    assert "call edge(u" in text


def test_an_argument_the_statements_change_is_not_substituted():
    # n is assigned in the callee; a caller variable of that name passed for a scalar would change meaning.
    changed = SOURCE.replace("    integer, intent(in) :: component\n    integer :: i, n\n",
                             "    integer, intent(in) :: component\n    integer :: i, n\n").replace(
        "    integer :: i\n    call edge(u, 1)", "    integer :: i, n\n    n = 1\n    call edge(u, n)")
    text, records = inlined(changed)
    assert "the statements change what the argument n names" in records[0]["reason"]
    assert "call edge(u, n)" in text
    assert records[1] == {
        "caller": "halos::both", "callee": "halos::edge", "line": 27, "inlined": True}


def test_the_nests_of_inlined_calls_share_the_callers_entry(tmp_path):
    fortran = fortran_compiler()
    if fortran is None:
        pytest.skip("gfortran is required")
    source = tmp_path / "halos.f90"
    source.write_text(SOURCE)

    def summary(name, **options):
        project = ResidentProject(sources=(source,), selections=(
            Selection("all", ("halos::both", "halos::edge"), **options),))
        prepared = prepare(project, tmp_path / name, fortran=fortran)
        row, = prepared["selections"]
        return prepared, (row["entries"], row["nests"], row["rejected"])

    # As calls, the callee's nest and the caller's are entries of their own procedures.
    assert summary("calls")[1] == (2, 2, [])
    # In place of the calls, the two copied nests and the caller's own form one entry.
    prepared, counts = summary("inlined", inline=("halos::edge",))
    assert [item["inlined"] for item in prepared["inlined"]] == [True, True]
    assert counts == (2, 4, [])
    text = (tmp_path / "inlined/sources/halos.f90").read_text()
    both = text[text.index("subroutine both"):text.index("end subroutine both")]
    assert both.count("fort_device_run(") == 1
    assert "call edge" not in both
