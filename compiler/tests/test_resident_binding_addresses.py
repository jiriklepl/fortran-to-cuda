"""Binding an array takes the array's own address, whatever the compiler can see about its contiguity."""
import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

# Two levels of assumed-shape dummies without CONTIGUOUS, as a procedure that forwards its arrays has them.
CALLER = """module caller
  use iso_c_binding
  use fort_resident_domain
  implicit none
  type(resident_domain) :: domain
contains
  subroutine inner(u, kinds)
    real(c_double), dimension(-2:,-2:,-2:), intent(inout), target :: u
    integer(c_int), dimension(-2:,-2:,-2:), intent(in), target :: kinds
    integer(c_int64_t) :: handles(2)
    handles(1) = domain%bind_real64(u, lbound(u), .true., origin='caller::u')
    handles(2) = domain%bind_integer32(kinds, lbound(kinds), .true., origin='caller::kinds')
    call domain%observe([domain%held(u), domain%held(kinds)])
    call domain%release_real(u)
  end subroutine
  subroutine outer(u, kinds)
    real(c_double), dimension(-2:,-2:,-2:), intent(inout), target :: u
    integer(c_int), dimension(-2:,-2:,-2:), intent(in), target :: kinds
    call inner(u, kinds)
  end subroutine
end module
"""


@pytest.mark.parametrize("compiler_name", ["gfortran-11", "gfortran-12", "gfortran-13", "gfortran-14", "gfortran-15",
                                           "gfortran-16", "gfortran"])
def test_no_packed_copy_is_made_of_an_array_that_is_handed_to_the_domain(tmp_path, compiler_name):
    # For an actual argument that it cannot see to be contiguous, a compiler hands a CONTIGUOUS dummy the
    # result of a pack call. gfortran 13 to 16 return the array itself from that call when it is
    # contiguous; with gfortran 11 an application stopped in the filter of ELMM, which forwards its
    # arrays this way: the domain had registered the address of a copy. The domain's procedures therefore
    # take application arrays through dummies that leave nothing to pack.
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"{compiler_name} unavailable")
    (tmp_path / "caller.f90").write_text(CALLER)
    for source in (RUNTIME / "scoped_memory.f90", RUNTIME / "resident_domain.f90"):
        subprocess.run([compiler, "-c", "-O3", "-fopenmp", "-ffree-line-length-none", str(source)],
                       cwd=tmp_path, check=True, capture_output=True, text=True)
    compiled = subprocess.run([compiler, "-c", "-O3", "-fopenmp", "-fdump-tree-original=/dev/stdout", "caller.f90"],
                              cwd=tmp_path, check=True, capture_output=True, text=True)
    assert "inner" in compiled.stdout
    assert "_gfortran_internal_pack" not in compiled.stdout
    assert "internal_unpack" not in compiled.stdout


# The same forwarding with stand-ins for the binding procedures, declared as the domain declares them,
# so that it links without the runtime and says at run time whose address arrived.
FORWARDING = """module forwarding
  use iso_c_binding
  implicit none
  type(c_ptr) :: real_address, integer_address
contains
  subroutine inner(u, kinds)
    real(c_double), dimension(-2:,-2:,-2:), intent(inout), target :: u
    integer(c_int), dimension(-2:,-2:,-2:), intent(in), target :: kinds
    call bind_real(u)
    call bind_integer(kinds)
  end subroutine
  subroutine outer(u, kinds)
    real(c_double), dimension(-2:,-2:,-2:), intent(inout), target :: u
    integer(c_int), dimension(-2:,-2:,-2:), intent(in), target :: kinds
    call inner(u, kinds)
  end subroutine
  subroutine bind_real(values)
    {real}
    if (.not. c_associated(c_loc(values), real_address)) error stop 'a copy of the real array'
  end subroutine
  subroutine bind_integer(values)
    {integer}
    if (.not. c_associated(c_loc(values), integer_address)) error stop 'a copy of the integer array'
  end subroutine
end module
program test
  use forwarding
  real(c_double), allocatable, target :: u(:,:,:)
  integer(c_int), allocatable, target :: kinds(:,:,:)
  allocate(u(16,7,7), kinds(16,7,7))
  real_address = c_loc(u)
  integer_address = c_loc(kinds)
  call outer(u, kinds)
end program
"""


@pytest.mark.parametrize("compiler_name", ["gfortran-11", "gfortran-12", "gfortran-13", "gfortran-14", "gfortran-15",
                                           "gfortran-16"])
def test_a_forwarded_array_is_bound_by_its_own_address(tmp_path, compiler_name):
    compiler = shutil.which(compiler_name)
    if compiler is None:
        pytest.skip(f"{compiler_name} unavailable")
    declarations = [line.strip() for line in (RUNTIME / "resident_domain.f90").read_text().splitlines()
                    if line.strip().endswith(":: values(..)") and "pointer" not in line]
    real, integer = (next(line for line in declarations if line.startswith(kind))
                     for kind in ("real(c_double)", "integer(c_int)"))
    # gfortran 11 and 12 hand a CONTIGUOUS dummy a copy here; later versions the array itself.
    assert not [line for line in declarations if "contiguous" in line.lower()]
    (tmp_path / "forwarding.f90").write_text(FORWARDING.format(real=real, integer=integer))
    subprocess.run([compiler, "-O3", "-fopenmp", "forwarding.f90", "-o", "forwarding"], cwd=tmp_path, check=True,
                   capture_output=True, text=True)
    result = subprocess.run([str(tmp_path / "forwarding")], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
