"""Fields of arrays of derived type as entry resources, and writes through index values."""
from pathlib import Path

import pytest

from compiler.ir import CompilationError
from compiler.scopes.resident_entries import (
    compile_resident_entries,
    compile_resident_entry,
    export_resident_entries,
    pass_scalars_by_value,
)
from compiler.scopes.resident_splice import ResidentDomain, entry_call_block

SOURCE = """module walls
implicit none
type point
  integer :: xi, yj
  real(8) :: flux
  real(8) :: history(3)
  logical :: active
end type
type(point), allocatable :: points(:), others(:)
type(point), allocatable :: grid(:,:)
integer, allocatable :: cell(:)
integer :: n
contains
subroutine correct(u)
real(8), contiguous, intent(inout) :: u(0:,0:)
integer :: i
do i = 1, size(points)
  u(points(i)%xi, points(i)%yj) = u(points(i)%xi, points(i)%yj) + points(i)%flux
end do
end subroutine
subroutine relax(u)
real(8), contiguous, intent(in) :: u(0:,0:)
integer :: i
do i = 1, size(points)
  points(i)%flux = 0.5d0*points(i)%flux + u(points(i)%xi, points(i)%yj)
end do
end subroutine
subroutine two_lists(u)
real(8), contiguous, intent(inout) :: u(0:,0:)
integer :: i
do i = 1, size(points)
  points(i)%flux = u(points(i)%xi, points(i)%yj)
end do
do i = 1, size(others)
  others(i)%flux = u(others(i)%xi, others(i)%yj)
end do
do i = 1, n
  others(i)%flux = 2*others(i)%flux
end do
end subroutine
subroutine by_index(a, v)
real(8), contiguous, intent(inout) :: a(:)
real(8), contiguous, intent(in) :: v(:)
integer :: i
do i = 1, n
  a(cell(i)) = a(cell(i)) + v(i)
end do
end subroutine
subroutine shifted(a)
real(8), contiguous, intent(inout) :: a(:)
integer :: i
do i = 1, n
  a(cell(i)) = a(cell(i) + 1)
end do
end subroutine
subroutine pair(a, v)
real(8), contiguous, intent(inout) :: a(:)
real(8), contiguous, intent(in) :: v(:)
integer :: i
do i = 1, n
  a(cell(i)) = a(cell(i)) + v(i)
end do
do i = 1, n
  a(cell(i) + 1) = a(cell(i) + 1) - v(i)
end do
end subroutine
subroutine mixed(a)
real(8), contiguous, intent(inout) :: a(:)
integer :: i
do i = 1, n
  a(cell(i)) = a(i)
end do
end subroutine
subroutine computed(a)
real(8), contiguous, intent(inout) :: a(:)
integer :: i, k
do i = 1, n
  k = cell(i)
  a(k) = a(k) + 1
end do
end subroutine
subroutine unsupported(u)
real(8), contiguous, intent(inout) :: u(0:,0:)
integer :: i
do i = 1, n
  u(i, 0) = grid(i, 1)%flux
end do
do i = 1, n
  u(i, 0) = points(i)%history(2)
end do
do i = 1, n
  if (points(i)%active) u(i, 0) = 0
end do
end subroutine
end module
"""


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    directory = tmp_path_factory.mktemp("structures")
    source = directory / "walls.f90"
    source.write_text(SOURCE)
    return directory, export_resident_entries([source], None, directory / "exported")


def entries(exported, procedure):
    return [entry for entry in exported[1]["entries"] if entry["procedure"] == "walls::" + procedure]


def test_each_field_is_one_array_resource_with_the_shape_of_its_array(exported):
    entry, = entries(exported, "relax")
    resources = {item["name"]: item for item in entry["resources"]}
    assert set(resources) == {"u", "points%flux", "points%xi", "points%yj"}
    assert (resources["points%flux"]["type"], resources["points%flux"]["kind"], resources["points%flux"]["rank"]) == (
        "real", 8, 1)
    assert (resources["points%xi"]["type"], resources["points%xi"]["kind"]) == ("integer", 4)
    assert all("element_field" in resources[name]["attributes"] for name in resources if name != "u")
    assert entry["written_resources"] == ["walls::points%flux"]
    assert entry["structures"] == [{"name": "points", "resource": "walls::points", "counted": True}]
    # In the entry the fields are plain arrays, and SIZE of the list is the extent of one of them.
    text = Path(entry["input_path"]).read_text()
    assert "intent(inout) :: fort_region_field_1(:)" in text
    assert "DO i = 1, SIZE(fort_region_field_1)" in text
    assert "points" not in text
    # Reading through index values is independent across iterations; the write is by the loop variable.
    assert entry["injective"] == []


def test_nests_over_different_lists_get_entries_of_their_own(exported):
    first, second = entries(exported, "two_lists")
    assert [item["structures"][0]["name"] for item in (first, second)] == ["points", "others"]
    assert (len(first["source_loop_spans"]), len(second["source_loop_spans"])) == (1, 2)
    # Every nest of the first entry runs once per element, so an empty list leaves it nothing to do.
    assert first["structures"][0]["counted"] is True
    # The second entry has a nest bounded by n: it is not known to do nothing when its list is empty.
    assert second["structures"][0]["counted"] is False


def test_fields_the_entry_cannot_represent_are_rejected_with_their_reason(exported):
    reasons = [item["reason"] for item in exported[1]["rejected"] if item["procedure"] == "walls::unsupported"]
    assert len(reasons) == 3
    assert "one element of a rank-one array of derived type" in reasons[0]
    assert "needs one scalar variable object: points(i) % history(2)" in reasons[1]
    assert "scalar REAL or default INTEGER component" in reasons[2]


def test_a_write_through_indices_names_what_the_caller_must_check(exported):
    scatter, = entries(exported, "correct")
    assert scatter["injective"] == [{
        "resource": "argument::u", "array": "u", "loops": ["DO i = 1, SIZE(points)"],
        "indices": ["points(i) % xi", "points(i) % yj"],
        "index_resources": ["walls::points%xi", "walls::points%yj"]}]
    plain, = entries(exported, "by_index")
    assert plain["injective"] == [{"resource": "argument::a", "array": "a", "loops": ["DO i = 1, n"],
                                   "indices": ["cell(i)"], "index_resources": ["walls::cell"]}]
    # Every subscript tuple of the array is named, within a nest and across the nests of an entry.
    assert [item["indices"] for item in entries(exported, "shifted")[0]["injective"]] == [["cell(i)"], ["cell(i) + 1"]]
    both, = entries(exported, "pair")
    assert [(item["loops"], item["indices"]) for item in both["injective"]] == [
        (["DO i = 1, n"], ["cell(i)"]), (["DO i = 1, n"], ["cell(i) + 1"])]
    # A tuple that does not go through index values, or a subscript the nest computes itself, cannot be checked.
    assert entries(exported, "mixed")[0]["injective"] == []
    assert entries(exported, "computed")[0]["injective"] == []


def test_the_check_is_what_lets_the_scatter_run_on_the_device(exported, tmp_path):
    directory, export = exported
    valued = pass_scalars_by_value(export, tmp_path / "values", resources=("walls::n",))
    by_procedure = {entry["procedure"].split("::")[1]: entry for entry in valued["entries"]
                    if entry["procedure"] != "walls::two_lists"}
    plain = by_procedure["by_index"]
    with pytest.raises(CompilationError, match="Cannot parallelize"):
        compile_resident_entry(plain["input_path"], plain["entry"], tmp_path / "unchecked")
    report = compile_resident_entry(plain["input_path"], plain["entry"], tmp_path / "checked", injective=["a"])
    assert all(region["gpu_available"] for region in report["scoped"]["region_execution"])
    # The assumption covers every tuple that goes through index values, so a second, shifted tuple is fine.
    shifted = by_procedure["shifted"]
    report = compile_resident_entry(shifted["input_path"], shifted["entry"], tmp_path / "shifted", injective=["a"])
    assert all(region["gpu_available"] for region in report["scoped"]["region_execution"])
    # It also makes two nests over the same iterations independent of each other: they become one region.
    pair = by_procedure["pair"]
    report = compile_resident_entry(pair["input_path"], pair["entry"], tmp_path / "pair", injective=["a"])
    assert len(report["scoped"]["region_execution"]) == 1
    # It says nothing about a subscript that is not an index value.
    mixed = by_procedure["mixed"]
    with pytest.raises(CompilationError, match="Cannot parallelize"):
        compile_resident_entry(mixed["input_path"], mixed["entry"], tmp_path / "mixed", injective=["a"])
    # Compiling a whole export passes the assumption exactly where a requirement was recorded, so a
    # nest whose subscripts the caller cannot check stays rejected.
    package = compile_resident_entries(valued, tmp_path / "entries")
    compiled = {item["region"]["procedure"] for item in package["entries"]}
    assert {"walls::correct", "walls::relax", "walls::by_index", "walls::shifted", "walls::pair"} <= compiled
    rejected = {item["procedure"] for item in package["rejected"]}
    assert {"walls::mixed", "walls::computed"} <= rejected


def test_the_call_block_binds_fields_and_checks_before_it_runs(exported, tmp_path):
    entry, = entries(exported, "correct")
    package = compile_resident_entries({**exported[1], "entries": [entry], "rejected": []}, tmp_path / "entries")
    item, = package["entries"]
    block = entry_call_block(item["region"], item["entry"], ResidentDomain("state", "flow", "require"))
    assert "use fort_scoped_memory, only: FORT_SCOPE_GPU, FORT_SCOPE_INTEGER32, FORT_SCOPE_REAL64" in block
    assert ("flow%bind_field(c_loc(points(lbound(points,1))%flux), &\n"
            "          int(storage_size(points)/8,c_size_t),size(points,kind=c_size_t),FORT_SCOPE_REAL64, &\n"
            "          .false.,origin='walls::correct::points%flux')") in block
    # The fields' lower bounds are those of the list.
    assert block.count("lbound(points,1), &") + block.count("lbound(points,1)))") == 3
    order = [block.index(text) for text in (
        "      if (size(points) > 0) then\n", "      fort_resident = flow%active\n",
        "flow%bind_field(", "if (any(fort_checked /= [flow%generation, flow%stamp])) then",
        "DO i = 1, SIZE(points)", "fort_resident = fort_distinct", "call require(fort_device_run(",
        "if (flow%active) call flow%publish()", "@ORIGINAL@")]
    assert order == sorted(order)
    assert block.rstrip().endswith("      endif\n      endif\n    end block")


def test_all_tuples_of_an_array_are_checked_in_one_mask(exported, tmp_path):
    entry, = entries(exported, "pair")
    package = compile_resident_entries({**exported[1], "entries": [entry], "rejected": []}, tmp_path / "entries")
    item, = package["entries"]
    block = entry_call_block(item["region"], item["entry"], ResidentDomain("state", "flow", "require"))
    # An element that the two nests both name must be found, so their tuples share the mask.
    assert block.count(" allocate(fort_seen_") == 1
    assert block.count("deallocate(fort_seen_1)") == 1
    first = block.index("else if (fort_seen_1(cell(i))) then")
    second = block.index("else if (fort_seen_1(cell(i) + 1)) then")
    assert block.index("allocate(fort_seen_1(") < first < second < block.index("deallocate(fort_seen_1)")
