"""Host procedures that use module arrays without being handed them, called from selected procedures."""
from pathlib import Path

import pytest

from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.tests.resident_application_support import fortran_compiler

pytestmark = pytest.mark.skipif(fortran_compiler() is None, reason="GNU Fortran is required for preprocessing")

SOURCE = """module state
implicit none
private
public :: u, n, relax, show, deeper, tune, peek, drop
integer :: n
real(8), allocatable :: u(:), hidden(:)
contains
subroutine relax()
integer :: i
do i = 1, n
hidden(i) = 0.5d0 * (hidden(i) + u(i))
enddo
end subroutine
subroutine show(total)
real(8), intent(out) :: total
total = sum(u)
end subroutine
subroutine deeper(total)
real(8), intent(out) :: total
call show(total)
end subroutine
subroutine tune()
u(1) = 0
end subroutine
subroutine peek(total)
real(8), intent(out) :: total
total = hidden(1)
end subroutine
subroutine drop()
deallocate(u)
end subroutine
end module
module work
use state, only: w => u, n, relax, show, deeper, tune, peek, drop
implicit none
contains
subroutine step(a, total)
real(8), contiguous, intent(inout) :: a(:)
real(8), intent(out) :: total
real(8) :: u(3)
integer :: i
do i = 1, n
w(i) = w(i) + a(i)
enddo
call relax()
call show(total)
call deeper(total)
call tune()
call peek(total)
call cpu_time(total)
u = total
a(1) = u(1)
end subroutine
subroutine finish()
call drop()
end subroutine
end module
"""


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    directory = tmp_path_factory.mktemp("module-arrays")
    source = directory / "state.f90"
    source.write_text(SOURCE)
    project = ResidentProject((source,), (Selection("all", ("work::step", "work::finish", "state::relax")),))
    record = prepare(project, directory / "prepared", fortran=fortran_compiler())
    text = Path(record["sources"][0]["resident"]).read_text()
    # The procedures of the second module, where the calls are.
    return record, text[text.index("module work"):]


def test_a_call_gets_the_module_arrays_its_host_procedure_uses(prepared):
    record, text = prepared
    found = {item["text"]: (item["arrays"], item["access"]) for item in record["audit"]}
    # show reads u, which the entry of step keeps on the device under the name w; deeper only calls show.
    assert found["CALL show(total)"] == (["u"], "arrays")
    assert found["CALL deeper(total)"] == (["u"], "arrays")
    # The local array u of step hides the module's, so the block gets the module's under a name of its own.
    show = text[text.index("call relax()"):text.index("call show(total)")]
    assert show.index("use state, only: fort_state_u => u") < show.index(
        "fort_host1 = allocated(fort_state_u)") < show.index(
            "call resident%observe([ &\n          resident%held(fort_state_u)])")
    assert "host_modify" not in show
    # A library procedure sees no array it is not handed, and a selected procedure sees to its own.
    assert "CALL cpu_time(total)" not in found
    assert "CALL relax" not in found


def test_a_module_array_the_host_procedure_changes_belongs_to_the_host_for_the_call(prepared):
    record, text = prepared
    found = {item["text"]: (item["arrays"], item["access"]) for item in record["audit"]}
    assert found["CALL tune"] == (["u"], "arrays")
    tune = text[text.index("call deeper(total)"):text.index("call peek(total)")]
    assert tune.index("call resident%host_modify([ &\n          resident%held(fort_state_u)])") < tune.index(
        "call tune()") < tune.index("call resident%host_done([ &\n          resident%held(fort_state_u)])")


def test_an_array_that_cannot_be_named_at_the_call_brings_every_value_back(prepared):
    record, text = prepared
    found = {item["text"]: (item["arrays"], item["access"]) for item in record["audit"]}
    # hidden is private to its module: the call cannot ask for it alone.
    assert found["CALL peek(total)"] == ([], "publish")
    peek = text[text.index("call tune()"):text.index("call peek(total)")]
    assert "if (resident%active) call resident%publish()" in peek


def test_a_host_procedure_that_deallocates_a_module_array_ends_the_stay_on_the_device(prepared):
    record, text = prepared
    found = {item["text"]: (item["arrays"], item["access"]) for item in record["audit"]}
    assert found["CALL drop"] == (["u"], "leave")
    finish = text[text.index("subroutine finish"):]
    assert finish.index("if (resident%active) call resident%end_epoch()") < finish.index("call drop()")


def test_an_entry_binds_a_renamed_module_array_under_the_name_its_nest_uses(prepared):
    _record, text = prepared
    step = text[text.index("subroutine step"):text.index("call relax()")]
    # In step the module's u is called w, and u is a local array that the nest does not touch.
    assert "resident%bind_real64(w,lbound(w),.true., &\n          origin='work::step::u')" in step
    assert "bind_real64(u," not in step
    assert "lbound(w,1)" in step


def test_host_code_that_uses_a_list_whose_fields_are_kept_gets_every_value_back(tmp_path):
    source = tmp_path / "walls.f90"
    source.write_text("""module walls
type point
integer :: xi
real(8) :: flux
end type
type(point), allocatable :: points(:)
integer :: n
contains
subroutine relax(u, total)
real(8), contiguous, intent(inout) :: u(:)
real(8), intent(out) :: total
integer :: i
do i = 1, size(points)
points(i)%flux = u(points(i)%xi)
enddo
n = n + 1
total = points(1)%flux
call show(total)
end subroutine
subroutine show(total)
real(8), intent(inout) :: total
total = total + points(2)%flux
end subroutine
end module
""")
    record = prepare(ResidentProject((source,), (Selection("all", ("walls::relax",)),)), tmp_path / "prepared",
                     fortran=fortran_compiler())
    assert [(item["text"], item["arrays"]) for item in record["audit"]] == [
        ("total = points(1) % flux", ["points"]), ("CALL show(total)", ["points"])]
    text = Path(record["sources"][0]["resident"]).read_text()
    # The device has the newest values of the field; they come back to the elements with a publication.
    tail = text[text.index("n = n + 1"):text.index("subroutine show")]
    assert tail.count("if (resident%active) call resident%publish()") == 2
    assert tail.index("publish()") < tail.index("total = points(1)%flux") < tail.rindex("publish()") \
        < tail.index("call show(total)")
