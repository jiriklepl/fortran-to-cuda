"""A wall law in an application's own style compiles from its source: lists, pointers, cases, vectors, privates."""
from pathlib import Path

import pytest

from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.tests.resident_application_support import fortran_compiler

pytestmark = pytest.mark.skipif(fortran_compiler() is None, reason="GNU Fortran is required for preprocessing")

LAW = """module law
implicit none
integer, parameter :: knd = 8
real(knd) :: viscosity = 1e-5_knd, dxmin = 0.1_knd, dzmin = 0.05_knd
logical :: buoyant = .false.
type point
  integer :: xi, zk
  real(knd) :: z0, ustar, flux
  logical :: fixed
end type
type(point), allocatable, target :: points(:)
real(knd), allocatable :: visc(:)
contains
subroutine fluxes(u, w, component, direction)
real(knd), dimension(-2:), intent(in) :: u, w
integer, intent(in) :: component, direction
type(point), pointer :: p
integer :: i, xi
real(knd) :: vel(3), tan_vect(3), mag, drec(6), wall
drec = [1/dxmin, 1/dxmin, 1/dxmin, 1/dxmin, 1/dzmin, 1/dzmin]
!$omp parallel do private(i, xi, p, vel, tan_vect, mag, wall)
do i = 1, size(points)
  p => points(i)
  xi = p%xi
  vel = local_velocity(u, w, component, xi)
  if (p%fixed) then
    continue
  else if (p%z0 > 0 .and. buoyant) then
    call rough(p%ustar, p%z0, vel, tan_vect = tan_vect)
  else
    call smooth(p%ustar, vel, [0._knd, 0._knd, 0._knd], tan_vect = tan_vect)
  end if
  mag = norm2(tan_vect)
  if (mag > 1e-8_knd) then
    p%flux = (tan_vect(component) / mag) * p%ustar**2 * drec(direction)
  else
    p%flux = 0
  end if
  wall = mag * 2_knd
  if (.not. p%fixed) visc(i) = max(wall, 1e-5)
end do
!$omp end parallel do
end subroutine

pure function local_velocity(u, w, component, xi) result(vel)
real(knd) :: vel(3)
real(knd), dimension(-2:), intent(in) :: u, w
integer, intent(in) :: component, xi
select case (component)
  case (1)
    vel(1) = u(xi)
    vel(2) = 0
    vel(3) = (w(xi) + w(xi + 1)) / 2
  case default
    vel(1) = (u(xi) + u(xi - 1)) / 2
    vel(2) = 0
    vel(3) = w(xi)
end select
end function

subroutine smooth(ustar, uvect, walluvect, tan_vel, tan_vect)
real(knd), intent(inout) :: ustar
real(knd), intent(in) :: uvect(3), walluvect(3)
real(knd), optional, intent(out) :: tan_vel, tan_vect(3)
real(knd) :: vect(3), vel
vect = uvect - walluvect
vel = norm2(vect)
if (vel > 0) ustar = sqrt(viscosity * vel / 0.01_knd)
if (present(tan_vel)) tan_vel = vel
if (present(tan_vect)) tan_vect = vect
end subroutine

subroutine rough(ustar, z0, uvect, tan_vect)
real(knd), intent(inout) :: ustar
real(knd), intent(in) :: z0, uvect(3)
real(knd), intent(out) :: tan_vect(3)
ustar = norm2(uvect) * 0.41_knd / log(1 / z0) * (1 + z0) ** (1 / 4._knd)
tan_vect = uvect
end subroutine
end module
"""


def test_the_flux_loop_of_a_wall_law_is_an_entry(tmp_path):
    source = tmp_path / "law.f90"
    source.write_text(LAW)
    prepared = prepare(ResidentProject((source,), (Selection("all", ("law::fluxes",), minimum_nests=1),)),
                       tmp_path / "prepared", fortran=fortran_compiler())
    summary, = prepared["selections"]
    assert (summary["entries"], summary["nests"], summary["rejected"]) == (1, 1, [])
    # The pointer to the element is the element, the optional result has a version without it.
    kinds = {next(key for key in ("pointer", "cases", "version", "returns", "loop") if key in item)
             for item in prepared["helpers"]}
    assert {"pointer", "cases", "version"} <= kinds
    text = Path(prepared["sources"][0]["resident"]).read_text()
    # The OpenMP directive is gone; what it said of the loop's scalars stays at the DO statement.
    assert "!$omp" not in text
    assert "do i = 1, size(points)  !$fort private(i, mag, p, tan_vect, vel, wall, xi)" in text
    # The table of reciprocals goes to the entry element by element.
    call = text[text.index("call resident_require(fort_device_run("):]
    assert all(f"drec({index})" in call for index in range(1, 7))
    # A LOGICAL scalar of the module goes in the entry's own kind.
    assert "logical(buoyant, c_bool)" in call
