"""A procedure's temporary arrays get storage the domain lends, instead of an allocation in every call."""
from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.tests.resident_application_support import fortran_compiler

SOURCE = """module smooth
integer :: n
contains
subroutine twice(a)
real(8), contiguous, intent(inout) :: a(:,:,:)
real(8), allocatable :: old(:,:,:){more}
integer :: i, j, k
allocate(old(0:n+1, n, n))
do k = 1, n
do j = 1, n
do i = 0, n+1
old(i,j,k) = a(i+1,j,k)
enddo
enddo
enddo
do k = 1, n
do j = 1, n
do i = 1, n
a(i+1,j,k) = 0.5d0 * (old(i-1,j,k) + old(i+1,j,k))
enddo
enddo
enddo
{use}
deallocate(old)
end subroutine
end module
"""


def prepared(tmp_path, *, use="", more=""):
    tmp_path.mkdir(exist_ok=True)
    source = tmp_path / "smooth.f90"
    source.write_text(SOURCE.format(use=use, more=more))
    result = prepare(ResidentProject((source,), (Selection("all", ("smooth::twice",)),)), tmp_path / "prepared",
                     fortran=fortran_compiler())
    return result, (tmp_path / "prepared/sources/smooth.f90").read_text()


def test_an_array_allocated_and_deallocated_in_each_call_is_lent_by_the_domain(tmp_path):
    result, text = prepared(tmp_path)
    assert result["temporaries"] == [{"procedure": "smooth::twice", "name": "old"}]
    # A pointer takes what the domain lends; the two statements ask the domain.
    assert "REAL(KIND = 8), POINTER, CONTIGUOUS :: old(:, :, :)" in text
    assert "allocate(old" not in text
    work = text[text.index("call resident%work("):]
    assert "[integer :: 0, 1, 1], &\n" in work
    assert "[integer :: n + 1, n, n], old)" in work
    assert work.index("call resident_require(fort_device_run(") < work.index("call resident%work_done(old)")
    # Neither statement is host code that needs the array handed over or released.
    assert result["audit"] == []
    assert "release_real" not in text


def test_an_array_used_in_a_way_a_pointer_would_change_stays_an_allocation(tmp_path):
    for number, (use, more) in enumerate((
            # Whether it is allocated is asked, the whole array is written out, it is handed to unknown code.
            ("if (allocated(old)) n = n", ""), ("print *, old", ""), ("call other(old)", ""),
            # Another array shares its statement and is no temporary of this kind.
            ("", ", line(:)"))):
        source = SOURCE if more == "" else SOURCE.replace("allocate(old(0:n+1, n, n))",
                                                          "allocate(old(0:n+1, n, n), line(n))").replace(
                                                              "deallocate(old)", "deallocate(old, line)")
        directory = tmp_path / str(number)
        directory.mkdir()
        path = directory / "smooth.f90"
        path.write_text(source.format(use=use, more=more))
        result = prepare(ResidentProject((path,), (Selection("all", ("smooth::twice",)),)), directory / "prepared",
                         fortran=fortran_compiler())
        assert result["temporaries"] == [], use or more
        assert "allocate(old" in (directory / "prepared/sources/smooth.f90").read_text()
