"""Procedures that a loop nest calls are written in a form the nest can take in, and are inlined into it."""
from pathlib import Path

import pytest

from compiler.resident.helpers import prepare_helpers
from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.tests.resident_application_support import fortran_compiler

pytestmark = pytest.mark.skipif(fortran_compiler() is None, reason="GNU Fortran is required for preprocessing")

# A wall law in the style of an application's: a point loop over a list, and procedures that are not declared
# PURE, leave out optional results, return early and iterate to convergence with a cap.
WALLS = """module walls
implicit none
integer, parameter :: knd = 8
real(knd) :: viscosity = 1e-5_knd
integer :: calls = 0
type point
  integer :: xi
  real(knd) :: distx, disty, distz, ustar
end type
type(point), allocatable :: points(:)
real(knd), allocatable :: visc(:)
contains
subroutine viscs(u)
real(knd), intent(in) :: u(:)
integer :: i, xi
real(knd) :: dist(3), vel(3), wall
do i = 1, size(points)
  xi = points(i)%xi
  dist = [points(i)%distx, points(i)%disty, points(i)%distz]
  vel = [u(xi), 0.5_knd * u(xi), 0._knd]
  call flat_visc(wall, points(i)%ustar, dist, vel)
  visc(i) = wall
end do
end subroutine

subroutine flat_visc(wall, ustar, distvect, uvect)
real(knd), intent(out) :: wall
real(knd), intent(inout) :: ustar
real(knd), intent(in) :: distvect(3), uvect(3)
real(knd) :: vel, dist
if (viscosity <= 0) then
  ustar = 0
  wall = 0
  return
end if
call flat_stress(ustar, distvect, uvect, vel, dist)
if (vel > 0) then
  wall = ustar**2 * dist / vel
else
  wall = viscosity
end if
end subroutine

subroutine flat_stress(ustar, distvect, uvect, tan_vel, distance, tan_vect)
real(knd), intent(inout) :: ustar
real(knd), intent(in) :: distvect(3), uvect(3)
real(knd), optional, intent(out) :: tan_vel, distance, tan_vect(3)
real(knd) :: vel, dist
dist = norm2(distvect)
vel = norm2(uvect)
if (vel /= 0) call flat_ustar(ustar, vel, dist)
if (present(tan_vel)) tan_vel = vel
if (present(distance)) distance = dist
if (present(tan_vect)) tan_vect = uvect
end subroutine

subroutine flat_ustar(ustar, vel, dist)
real(knd), intent(inout) :: ustar
real(knd), intent(in) :: vel, dist
real(knd) :: previous
integer :: i
i = 0
do
  i = i + 1
  previous = ustar
  ustar = vel / (log(dist * previous / viscosity) / 0.4_knd + 5.5_knd)
  if (abs(ustar - previous) / abs(ustar) < 1e-4_knd) exit
  if (i >= 50) then
    ustar = 0
    exit
  end if
end do
end subroutine

subroutine counted(wall)
real(knd), intent(out) :: wall
calls = calls + 1
wall = 0
end subroutine
end module
"""


def test_the_procedures_a_nest_calls_are_written_so_that_they_can_be_inlined():
    text, records = prepare_helpers(WALLS, ["walls::viscs"])
    # The call that leaves out tan_vect calls a version without it, in which present() was decided.
    assert {"caller": "walls::flat_visc", "callee": "walls::flat_stress", "absent": ["tan_vect"],
            "version": "flat_stress_fort_1"}.items() <= next(item for item in records if "version" in item).items()
    version = text[text.index("subroutine flat_stress_fort_1("):]
    version = version[:version.index("end subroutine")]
    assert "tan_vect" not in version.split("\n", 1)[1]
    assert "present" not in version.lower()
    assert "tan_vel = vel" in version
    assert "distance = dist" in version
    assert "call flat_stress_fort_1(ustar, distvect, uvect, vel, dist)" in text
    # The early RETURN ends a branch; what followed it is the ELSE.
    visc = text[text.index("subroutine flat_visc("):text.index("end subroutine", text.index("subroutine flat_visc("))]
    assert "return" not in visc.lower()
    assert visc.index("wall = 0") < visc.index("else") < visc.index("call flat_stress_fort_1(")
    # The iteration to convergence counts to its cap; each EXIT clears the flag that guards what follows.
    ustar = text[text.index("subroutine flat_ustar("):]
    assert "do fort_trip_1 = 1, 50" in ustar
    assert "exit" not in ustar.lower().split("end subroutine")[0]
    assert ustar.index("fort_go_1 = .false.") < ustar.index("if (fort_go_1) then", ustar.index("fort_go_1 = .false."))
    assert next(item for item in records if "loop" in item)["procedure"] == "walls::flat_ustar"
    # The selected procedure and what it does not reach keep their statements.
    assert text[:text.index("subroutine flat_visc(")] == WALLS[:WALLS.index("subroutine flat_visc(")]
    assert text.endswith(WALLS[WALLS.index("subroutine counted("):])


def test_a_point_loop_that_calls_them_is_an_entry(tmp_path):
    source = tmp_path / "walls.f90"
    source.write_text(WALLS)
    prepared = prepare(ResidentProject((source,), (Selection("all", ("walls::viscs",), minimum_nests=1),)),
                       tmp_path / "prepared", fortran=fortran_compiler())
    summary, = prepared["selections"]
    assert (summary["entries"], summary["nests"], summary["rejected"]) == (1, 1, [])
    assert sorted((item.get("procedure") or item["caller"], next(key for key in ("version", "returns", "loop")
                                                                  if key in item))
                  for item in prepared["helpers"]) == [
        ("walls::flat_ustar", "loop"), ("walls::flat_visc", "returns"), ("walls::flat_visc", "version")]
    text = Path(prepared["sources"][0]["resident"]).read_text()
    assert text.count("call resident_require(fort_device_run(") == 1


def test_a_procedure_that_changes_a_module_variable_is_not_inlined(tmp_path):
    source = tmp_path / "walls.f90"
    source.write_text(WALLS.replace("  call flat_visc(wall, points(i)%ustar, dist, vel)\n",
                                    "  call flat_visc(wall, points(i)%ustar, dist, vel)\n  call counted(wall)\n"))
    prepared = prepare(ResidentProject((source,), (Selection("all", ("walls::viscs",)),)),
                       tmp_path / "prepared", fortran=fortran_compiler())
    rejection, = prepared["selections"][0]["rejected"]
    assert rejection["reason"] == "inline numerical helper changes or keeps more than its own arguments and locals"


def test_a_return_or_an_exit_that_cannot_be_taken_out_stays():
    text = """module m
contains
subroutine top(a)
real(8), intent(inout) :: a(:)
integer :: i
do i = 1, size(a)
call inner(a(i))
end do
end subroutine
subroutine inner(x)
real(8), intent(inout) :: x
integer :: k
do k = 1, 3
if (x > 1) return
x = 2 * x
end do
do
x = x - 1
if (x < 0) exit
end do
end subroutine
end module
"""
    rewritten, records = prepare_helpers(text, ["m::top"])
    assert rewritten == text
    assert records == [{"procedure": "m::inner", "reason": "a RETURN inside a loop or another construct cannot be taken out"}]
