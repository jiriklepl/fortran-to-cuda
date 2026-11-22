"""A whole application is prepared for residency from a short description, without a device."""
import json
import re
import subprocess
from pathlib import Path

import pytest

from compiler.resident.build import ENTRY_SOURCES
from compiler.resident.project import ResidentProject, Selection, prepare
from compiler.tests.resident_application_support import BOX, UNITS, box_project, fortran_compiler

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"

pytestmark = pytest.mark.skipif(fortran_compiler() is None, reason="GNU Fortran is required for preprocessing")


@pytest.fixture(scope="module")
def box(tmp_path_factory):
    directory = tmp_path_factory.mktemp("box") / "prepared"
    return directory, prepare(box_project(), directory, fortran=fortran_compiler())


def test_the_box_is_prepared_from_its_description(box):
    directory, prepared = box
    assert json.loads((directory / "prepared.json").read_text()) == prepared
    assert [(row["label"], row["entries"], row["nests"], row["rejected"]) for row in prepared["selections"]] == [
        ("step", 2, 2, []), ("halo", 1, 3, []), ("sources", 5, 5, []), ("diagnostics", 4, 6, [])]
    assert len(prepared["entries"]) == 12
    assert all((Path(entry) / name).is_file() for entry in prepared["entries"] for name in ENTRY_SOURCES)
    assert prepared["domain"] == {"module": "fort_resident_application", "object": "resident", "default": True}
    # Four in the sources, and five that each of the three rewritten accumulating loops brings.
    assert prepared["directives"] == 19
    # The one array assignment outside a loop, in store, was written as its nest.
    assert [(Path(row["source"]).name, row["procedure"], row.get("lowered")) for row in prepared["array_statements"]] == [
        ("stepper.f90", "stepper::store", True)]
    # Its selected procedures hold nothing but loop nests, so no host statement touches a bound array.
    assert prepared["audit"] == []
    sources = {Path(row["original"]).name: row for row in prepared["sources"]}
    assert list(sources) == [name + ".f90" for name in UNITS]
    # A source nothing touched is compiled from the original file; the others from prepared copies.
    assert sources["params.f90"] == {"original": str(BOX / "params.f90"), "resident": str(BOX / "params.f90"),
                                     "changed": False}
    assert all(row["changed"] and Path(row["resident"]).parent == directory / "sources"
               for name, row in sources.items() if name != "params.f90")
    assert all((BOX / (name + ".f90")).read_text().count("resident%") == 0 for name in UNITS)


def test_entries_are_called_while_resident_and_the_original_loops_run_otherwise(box):
    directory, _prepared = box
    stepper = (directory / "sources/stepper.f90").read_text()
    store = stepper[stepper.index("subroutine store(a, b)"):stepper.index("end subroutine store")]
    assert "if (resident%active) then" in store
    assert "fort_handle_a=resident%bind_real64(a,lbound(a),.true., &\n          origin='stepper::store::a')" in store
    assert "call resident_require(fort_device_run(" in store
    # store is one array assignment in the source; the nest it stands for is what the entry replaces.
    assert "a(1:nx,1:ny,1:nz) = b(1:nx,1:ny,1:nz)" not in store
    assert (store.index("      else\n") < store.index("a(fort_arr1_1, fort_arr1_2, fort_arr1_3) = b(fort_arr1_1, ")
            < store.index("      endif\n"))
    # The selected procedures run from one serial caller, so their directives are gone; the precision is fixed.
    assert "!$omp" not in stepper
    assert "#ifdef" not in (directory / "sources/fields.f90").read_text()
    # The halo copies of a direction meet in a box of one cell. Their entry was compiled for the other
    # sizes and says when it was called with that one; the original copies run then.
    boundaries = (directory / "sources/boundaries.f90").read_text()
    assert "fort_resident = fort_status /= FORT_SCOPE_DECLINED" in boundaries


def test_directives_in_the_source_place_the_epoch_and_the_host_reads(box):
    directory, _prepared = box
    main = (directory / "sources/main.f90").read_text()
    # The application states no budget: the domain takes what the device has free.
    enter = main.index("if (min(nx, ny, nz) >= 2) call resident%begin_epoch(int(0, c_int))")
    loop = main.index("do step = 1, steps")
    leave = main.index("if (resident%active) call resident%end_epoch()")
    write = main.index("write(unit) u")
    close = main.index("call resident%close()")
    assert enter < loop < leave < write < close < main.index("deallocate(u, unew)")
    assert "!$fort" not in main
    diagnostics = (directory / "sources/diagnostics.f90").read_text()
    observe = diagnostics.index("call resident%observe(fort_observed)")
    assert "fort_observed(1) = resident%bind(a, lbound(a), .true., origin='diagnostics::a')" in diagnostics[:observe]
    # One field of the source list is brought back through the domain's copy of it.
    field = diagnostics.index("if (size(points) > 0) call resident%observe_field(c_loc(points(lbound(points,1))%seen))")
    assert observe < field < diagnostics.index("energy = energy + a(i,j,k)*a(i,j,k)")
    # The energy loop was not selected: it stays host code and keeps no entry call.
    report = diagnostics[diagnostics.index("subroutine report"):diagnostics.index("end subroutine report")]
    assert "fort_device_run" not in report


def test_loops_that_only_accumulate_scalars_run_as_two_nests_and_a_host_fold(box):
    directory, prepared = box
    assert prepared["reductions"] == [
        {"procedure": "sources::range", "line": 55, "variables": ["every", "lowest"],
         "source": str(BOX / "sources.f90"), "compiled": True},
        {"procedure": "diagnostics::summary", "line": 38, "variables": ["broken", "warm", "weight"],
         "source": str(BOX / "diagnostics.f90"), "compiled": True},
        {"procedure": "diagnostics::peak", "line": 61, "variables": ["peak"],
         "source": str(BOX / "diagnostics.f90"), "compiled": True}]
    diagnostics = (directory / "sources/diagnostics.f90").read_text()
    summary = diagnostics[diagnostics.index("subroutine summary"):diagnostics.index("end subroutine summary")]
    # One entry stores the partial results of a short loop; another stores and folds those of a long
    # one. Either way only the folded values come back to the host.
    assert summary.count("call resident_require(fort_device_run(") == 2
    call = summary.rindex("call resident_require(fort_device_run(")
    observe = summary.index("call resident%observe(fort_observed)")
    fold = summary.index("warm = warm + fort_red1_q1(fort_red1_slot)")
    assert call < observe < fold < summary.index("if (fort_red1_q3(fort_red1_slot) /= 0) broken = .true.")
    # The original loop is still there for the time the field is on the host.
    assert summary.index("\n    else\n", fold) < summary.index("            warm = warm + 1", fold)
    assert "!$omp" not in summary
    assert "!$fort" not in diagnostics
    # A function's loop is rewritten the same way, with its result as the accumulated scalar.
    peak = diagnostics[diagnostics.index("function peak"):diagnostics.index("end function peak")]
    assert "peak = max(peak, fort_red2_q1(fort_red2_slot))" in peak
    assert "fort_red2_i1 = fort_red_ieee_value(peak, fort_red_negative_inf)" in peak


def test_the_prepared_fortran_compiles_against_the_runtime_modules(box, tmp_path):
    _directory, prepared = box
    fortran = fortran_compiler()
    units = [RUNTIME / "scoped_memory.f90", RUNTIME / "resident_domain.f90", RUNTIME / "resident_application.f90",
             *(Path(entry) / "shared_interface.f90" for entry in prepared["entries"]),
             *(Path(row["resident"]) for row in prepared["sources"])]
    for index, unit in enumerate(units):
        result = subprocess.run([fortran, "-c", "-cpp", "-DDOUBLE", "-fopenmp", "-ffree-line-length-none",
                                 "-Wall", "-Werror=implicit-interface", str(unit), "-o", f"unit{index}.o"],
                                cwd=tmp_path, capture_output=True, text=True)
        assert result.returncode == 0, unit.name + "\n" + result.stderr


PAIR = """module pair
integer :: n
contains
subroutine both(a, b)
real(8), contiguous, intent(inout) :: a(:), b(:)
integer :: i
do i = 1, n
a(i) = a(i) + 1
enddo
{between}
do i = 1, n
b(i) = 2*b(i)
enddo
n = size(a)
do i = 2, n
{inside}
b(i) = b(i-1) + a(i)
enddo
end subroutine
end module
"""


def prepare_pair(tmp_path, *, between="", inside="", **selection):
    tmp_path.mkdir(exist_ok=True)
    source = tmp_path / "pair.f90"
    source.write_text(PAIR.format(between=between, inside=inside))
    project = ResidentProject(sources=(source,), selections=(Selection("all", ("pair::both",), **selection),))
    prepared = prepare(project, tmp_path / "prepared", fortran=fortran_compiler())
    return prepared, Path(prepared["sources"][0]["resident"]).read_text()


def test_adjacent_nests_share_an_entry_and_a_rejected_nest_gets_its_arrays_first(tmp_path):
    # The assignment to n ends the group, so the loop-carried nest is compiled, and rejected, on its own.
    prepared, text = prepare_pair(tmp_path)
    summary, = prepared["selections"]
    assert (summary["entries"], summary["nests"]) == (1, 2)
    rejection, = summary["rejected"]
    assert "Cannot parallelize" in rejection["reason"]
    assert text.count("call resident_require(fort_device_run(") == 1
    # The loop-carried nest stays host code. Where it runs at all, the array it reads is brought up to
    # date, the one it changes is handed to the host for as long as the nest takes; nothing else moves.
    tail = text[text.index("call resident_require(fort_device_run("):]
    begin = tail.index("fort_host = resident%active .and. ((n) >= (2))")
    observe = tail.index("call resident%observe([ &\n          resident%held(a)])")
    modify = tail.index("call resident%host_modify([ &\n          resident%held(b)])")
    nest = tail.index("b(i) = b(i-1) + a(i)")
    done = tail.index("if (fort_host) call resident%host_done([ &\n          resident%held(b)])")
    assert begin < observe < modify < nest < done
    assert "publish" not in text


def test_a_directive_between_two_nests_separates_their_entries_and_stays_in_place(tmp_path):
    prepared, text = prepare_pair(tmp_path, between="!$fort publish if(n > 100)")
    summary, = prepared["selections"]
    assert (summary["entries"], summary["nests"]) == (2, 2)
    first = text.index("call resident_require(fort_device_run(")
    between = text.index("if (resident%active .and. (n > 100)) call resident%publish()")
    assert first < between < text.index("call resident_require(fort_device_run(", between)


def test_a_nest_with_a_directive_inside_is_left_to_the_host(tmp_path):
    prepared, text = prepare_pair(tmp_path, inside="!$fort observe(a)")
    summary, = prepared["selections"]
    # It is neither exported nor counted as a rejection: its own directive keeps it coherent.
    assert (summary["entries"], summary["nests"], summary["rejected"]) == (1, 2, [])
    loop = text.index("do i = 2, n")
    assert text.index("call resident%observe(fort_observed)", loop) < text.index("b(i) = b(i-1) + a(i)", loop)
    assert "call resident%publish()" not in text


def test_rejected_nests_may_be_declared_unreachable_and_coverage_is_enforced(tmp_path):
    prepared, text = prepare_pair(tmp_path / "unreachable", native_rejections=False)
    assert len(prepared["selections"][0]["rejected"]) == 1
    assert "call resident%publish()" not in text
    with pytest.raises(ValueError, match="compiled 2 loop nests; at least 3 are required"):
        prepare_pair(tmp_path / "floor", minimum_nests=3)


def test_a_description_that_cannot_be_followed_is_refused(tmp_path):
    source = tmp_path / "pair.f90"
    source.write_text(PAIR.format(between="", inside=""))
    selection = Selection("all", ("pair::both",))
    with pytest.raises(ValueError, match="distinct labels"):
        prepare(ResidentProject((source,), (selection, selection)), tmp_path / "labels", fortran=fortran_compiler())
    with pytest.raises(ValueError, match="not project sources: other.f90"):
        prepare(ResidentProject((source,), (selection,), rewrites={"other.f90": ("tile_scratch",)}),
                tmp_path / "rewrite", fortran=fortran_compiler())
    with pytest.raises(ValueError, match="rewrite found nothing to change"):
        prepare(ResidentProject((source,), (selection,), rewrites={"pair.f90": ("tile_scratch",)}),
                tmp_path / "nothing", fortran=fortran_compiler())


def test_an_edit_can_add_directives_to_sources_the_application_does_not_own(tmp_path):
    source = tmp_path / "pair.f90"
    source.write_text(PAIR.format(between="", inside=""))

    def edit(texts):
        (path, text), = texts.items()
        assert path == source
        return {path: text.replace("do i = 2, n\n", "!$fort observe(a, b)\ndo i = 2, n\n"),
                "extra.f90": "module extra\ncontains\nsubroutine stop_here()\n!$fort leave\nend subroutine\nend module\n"}

    project = ResidentProject((source,), (Selection("all", ("pair::both",), native_rejections=False),), edit=edit)
    prepared = prepare(project, tmp_path / "prepared", fortran=fortran_compiler())
    text = Path(prepared["sources"][0]["resident"]).read_text()
    assert prepared["directives"] == 2
    assert "fort_observed(2) = resident%bind(b, lbound(b), .true., origin='pair::b')" in text
    assert source.read_text() == PAIR.format(between="", inside="")
    # A source the edit adds is written next to the others, with its directives expanded.
    extra = prepared["sources"][1]
    assert extra["original"] is None
    assert Path(extra["resident"]).name == "extra.f90"
    assert "call resident%end_epoch()" in Path(extra["resident"]).read_text()
    with pytest.raises(ValueError, match="plain file name"):
        prepare(ResidentProject((source,), project.selections, edit=lambda texts: {"sub/extra.f90": ""}),
                tmp_path / "nested", fortran=fortran_compiler())


AUDITED = """module audited
integer :: n
contains
subroutine step(a, b)
real(8), contiguous, intent(inout) :: a(:), b(:)
integer :: i
do i = 1, n
a(i) = a(i) + 1
enddo
{before}
b(1) = a(n)
call finish(a)
n = n + 1
end subroutine
subroutine finish(a)
real(8), contiguous, intent(inout) :: a(:)
integer :: i
do i = 1, n
a(i) = 2*a(i)
enddo
end subroutine
subroutine untouched(c)
real(8), contiguous, intent(inout) :: c(:)
c(1) = 0
end subroutine
end module
"""


def audited(tmp_path, procedures, *, before="", **options):
    tmp_path.mkdir(exist_ok=True)
    source = tmp_path / "audited.f90"
    source.write_text(AUDITED.format(before=before))
    project = ResidentProject((source,), (Selection("all", procedures),), **options)
    return prepare(project, tmp_path / "prepared", fortran=fortran_compiler())["audit"]


def test_the_audit_reports_host_statements_that_touch_arrays_an_entry_binds(tmp_path):
    first, second = audited(tmp_path, ("audited::step",))
    # The element copy reads a on the host right after an entry updated it on the device. No entry of
    # this procedure uses b, but b is a dummy: its array may be one that an entry elsewhere keeps there.
    assert (first["procedure"], first["arrays"], first["text"], first["access"]) == (
        "audited::step", ["a", "b"], "b(1) = a(n)", "arrays")
    # The array then goes to a procedure that is not selected and so runs on the host.
    assert (second["arrays"], second["calls"], second["text"]) == (["a"], ["audited::finish"], "CALL finish(a)")
    # The line is that of the preprocessed source the entries were exported from.
    assert Path(first["source"]).read_text().splitlines()[first["first_line"] - 1] == "b(1) = a(n)"
    assert first["id"] != second["id"]
    # Statements that name no array are not reported. A selected procedure without a nest of its own
    # still answers for what it does to the arrays it is handed: its callers hand them over as they are.
    everything = audited(tmp_path / "all", ("audited::*",))
    assert not any("n = n + 1" in item["text"] for item in everything)
    assert [(item["text"], item["arrays"]) for item in everything if item["procedure"] == "audited::untouched"] == [
        ("c(1) = 0", ["c"])]


def test_a_directive_or_a_selected_callee_answers_a_finding(tmp_path):
    # With finish selected, the call hands the array to code that manages it; the copy remains.
    finding, = audited(tmp_path / "callee", ("audited::step", "audited::finish"))
    assert finding["text"] == "b(1) = a(n)"
    # A directive directly before the statement says the host values are current there.
    assert audited(tmp_path / "directive", ("audited::step", "audited::finish"), before="!$fort observe(a)") == []


def test_a_wrapper_that_only_passes_arrays_on_counts_as_what_it_calls(tmp_path):
    wrapped = AUDITED.replace("call finish(a)", "call twice(a)").replace("subroutine untouched(c)", """subroutine twice(a)
real(8), contiguous, intent(inout) :: a(:)
call finish(a)
call finish(a)
end subroutine
subroutine untouched(c)""")

    def findings(directory, procedures, source):
        directory.mkdir()
        (directory / "audited.f90").write_text(source.format(before=""))
        project = ResidentProject((directory / "audited.f90",), (Selection("all", procedures),))
        return [item["text"] for item in prepare(project, directory / "prepared", fortran=fortran_compiler())["audit"]]

    # The wrapper is host code, but everything it does with the array is done by a selected procedure.
    assert findings(tmp_path / "selected", ("audited::step", "audited::finish"), wrapped) == ["b(1) = a(n)"]
    # With finish on the host as well, the array ends up in host loops: the call is reported.
    assert findings(tmp_path / "host", ("audited::step",), wrapped) == ["b(1) = a(n)", "CALL twice(a)"]
    # A wrapper that also reads the array itself is no wrapper.
    reading = wrapped.replace("call finish(a)\ncall finish(a)", "call finish(a)\nn = int(a(1))")
    assert findings(tmp_path / "reading", ("audited::step", "audited::finish"), reading) == [
        "b(1) = a(n)", "CALL twice(a)"]
    # Passing a section hands over other storage than the array an entry binds: that is a use here.
    section = AUDITED.replace("call finish(a)", "call finish(a(2:n))")
    assert findings(tmp_path / "section", ("audited::step", "audited::finish"), section) == [
        "b(1) = a(n)", "CALL finish(a(2 : n))"]


def test_host_statements_get_their_access_generated(tmp_path):
    finding, = audited(tmp_path / "report", ("audited::step", "audited::finish"))
    assert finding["access"] == "arrays"
    text = (tmp_path / "report/prepared/sources/audited.f90").read_text()
    # The copy reads a and changes b: a is brought up to date and stays on the device, b belongs to the
    # host while the statement runs. Whether the domain holds either is asked at that moment.
    access = text[text.index("fort_host = resident%active"):text.index("call finish(a)")]
    assert "if (fort_host) call resident%observe([ &\n          resident%held(a)])" in access
    assert access.index("observe") < access.index("call resident%host_modify([ &\n          resident%held(b)])") \
        < access.index("b(1) = a(n)") < access.index("call resident%host_done([ &\n          resident%held(b)])")
    # A directive in front of the statement says the source sees to it: then nothing is generated.
    assert audited(tmp_path / "directive", ("audited::step", "audited::finish"), before="!$fort observe(a)") == []
    assert "fort_host" not in (tmp_path / "directive/prepared/sources/audited.f90").read_text()


def test_a_call_into_host_code_hands_over_what_the_callee_may_change(tmp_path):
    first, second = audited(tmp_path, ("audited::step",))
    assert (first["access"], second["access"], second["calls"]) == ("arrays", "arrays", ["audited::finish"])
    text = (tmp_path / "prepared/sources/audited.f90").read_text()
    call = text[text.index("b(1) = a(n)"):text.index("n = n + 1")]
    call = call[call.index("end block"):]
    # finish takes its array INTENT(INOUT): the host owns it for the call, and the device copy is stale after.
    assert call.index("call resident%host_modify([ &") < call.index("call finish(a)") < call.index(
        "if (fort_host) call resident%host_done([ &")


def test_an_array_assignment_whose_nest_does_not_compile_is_the_host_statement_it_was(tmp_path):
    source = tmp_path / "sums.f90"
    source.write_text("""module sums
integer :: n
integer(8) :: wide
contains
subroutine step(a, b)
real(8), contiguous, intent(inout) :: a(:), b(:)
integer :: i
do i = 1, n
a(i) = a(i) + 1
enddo
b = a
a = a * wide
end subroutine
end module
""")
    prepared = prepare(ResidentProject((source,), (Selection("all", ("sums::step",), minimum_nests=2),)),
                       tmp_path / "prepared", fortran=fortran_compiler())
    copy, scaled = prepared["array_statements"]
    # Lines are those of the preprocessed source, as in the audit.
    assert copy["lowered"]
    assert scaled["line"] == copy["line"] + 1
    # A 64-bit integer is not a type the entries take, so the second nest stays on the host.
    assert scaled["lowered"] is False
    assert "wide" in scaled["reason"]
    # It is not listed as a rejected nest but as the statement, for the audit too, and gets the statement's
    # access: its one array is read and changed.
    assert prepared["selections"] == [{"label": "all", "entries": 2, "nests": 2, "rejected": []}]
    finding, = prepared["audit"]
    assert (finding["arrays"], finding["text"], finding["access"]) == (["a"], "a = a * wide", "arrays")
    text = (tmp_path / "prepared/sources/sums.f90").read_text()
    tail = text[text.index("call resident%host_modify([ &"):]
    assert tail.index("resident%held(a)") < tail.index("a(fort_arr2_1) = a(fort_arr2_1) * wide") \
        < tail.index("call resident%host_done([ &")


def test_fields_of_an_array_of_derived_type_are_bound_through_the_domain(box):
    directory, _prepared = box
    text = (directory / "sources/sources.f90").read_text()
    sense = text[text.index("subroutine sense(a)"):text.index("end subroutine sense")]
    # Each field the nest uses is one resource; the one it writes is marked so, and the array needs TARGET.
    assert ("resident%bind_field(c_loc(points(lbound(points,1))%seen), &\n"
            "          int(storage_size(points)/8,c_size_t),size(points,kind=c_size_t),FORT_SCOPE_REAL64, &\n"
            "          .true.,origin='sources::sense::points%seen')") in sense
    assert "FORT_SCOPE_INTEGER32, &\n          .false.,origin='sources::sense::points%xi')" in sense
    assert "TYPE(source), ALLOCATABLE, TARGET :: points(:)" in text
    # Every nest runs once per source, so an empty list skips the block, also when not resident.
    assert sense.index("      if (size(points) > 0) then\n") < sense.index("fort_resident = resident%active")
    # Reading through indices needs no check; only the scatter in inject has one.
    assert "fort_distinct" not in sense


def test_a_write_through_indices_is_checked_once_per_publication_and_falls_back(box):
    directory, _prepared = box
    text = (directory / "sources/sources.f90").read_text()
    inject = text[text.index("subroutine inject(a)"):text.index("end subroutine inject")]
    check = inject.index("if (any(fort_checked /= [resident%generation, resident%stamp])) then")
    # The index fields are made current on the host, then every source must name a cell of its own.
    observe = inject.index("call resident%observe([", check)
    loop = inject.index("DO q = 1, SIZE(points)", observe)
    seen = inject.index("else if (fort_seen_1(points(q) % xi, points(q) % yj, points(q) % zk)) then", loop)
    assert check < observe < loop < seen < inject.index("fort_resident = fort_distinct")
    assert "any([points(q) % xi, points(q) % yj, points(q) % zk] < lbound(a))" in inject
    # Sources that share a cell keep the original loop, after the device values are published.
    call = inject.index("call resident_require(fort_device_run(")
    publish = inject.index("if (resident%active) call resident%publish()")
    assert call < publish < inject.index("a(points(q)%xi, points(q)%yj, points(q)%zk) = ", publish)


def test_a_call_into_host_code_that_reaches_entries_ends_the_stay_on_the_device(tmp_path):
    source = tmp_path / "mixed.f90"
    source.write_text("""module mixed
integer :: n
contains
subroutine step(a)
real(8), contiguous, intent(inout) :: a(:)
call double(a)
call both(a)
end subroutine
subroutine double(a)
real(8), contiguous, intent(inout) :: a(:)
integer :: i
do i = 1, n
a(i) = 2*a(i)
enddo
end subroutine
subroutine both(a)
real(8), contiguous, intent(inout) :: a(:)
call double(a)
if (a(1) > 0) a(1) = 0
end subroutine
end module
""")
    project = ResidentProject((source,), (Selection("all", ("mixed::step", "mixed::double")),))
    prepared = prepare(project, tmp_path / "prepared", fortran=fortran_compiler())
    # both is host code that touches the array itself and also calls a selected procedure with it: giving
    # it the array would take it from the entry inside. The values come home before the call instead.
    finding, = prepared["audit"]
    assert (finding["text"], finding["access"]) == ("CALL both(a)", "leave")
    text = (tmp_path / "prepared/sources/mixed.f90").read_text()
    step = text[text.index("subroutine step(a)"):text.index("subroutine double(a)")]
    assert step.index("call double(a)") < step.index("if (resident%active) call resident%end_epoch()") \
        < step.index("call both(a)")
    assert "host_modify" not in step



def test_generated_access_asks_whether_an_array_exists_and_knows_a_component_from_an_array(tmp_path):
    source = tmp_path / "sums.f90"
    source.write_text("""module sums
type record
real(8) :: total
end type
integer :: n
real(8), allocatable :: total(:)
type(record) :: kept
contains
subroutine step(a, with)
real(8), contiguous, intent(inout) :: a(:)
logical, intent(in) :: with
integer :: i
if (with) then
do i = 1, n
total(i) = total(i) + a(i)
enddo
end if
kept%total = a(1)
n = n + int(total(1))
end subroutine
end module
""")
    prepared = prepare(ResidentProject((source,), (Selection("all", ("sums::step",)),)), tmp_path / "prepared",
                       fortran=fortran_compiler())
    component, array = prepared["audit"]
    # ELMM has both: a module array and a component of a derived type with the same name.
    assert (component["text"], component["arrays"]) == ("kept % total = a(1)", ["a"])
    assert (array["text"], array["arrays"]) == ("n = n + INT(total(1))", ["total"])
    text = (tmp_path / "prepared/sources/sums.f90").read_text()
    access = text[text.index("kept%total = a(1)"):text.index("n = n + int(total(1))")]
    # The array is allocated only in runs that use it; binding it otherwise would stop the run.
    assert "logical :: fort_host, fort_host1\n" in access
    assert "fort_host1 = allocated(total)\n" in access
    assert "if (fort_host1) call resident%observe([ &\n" in access


def test_host_statements_on_one_line_share_their_access(tmp_path):
    source = tmp_path / "pair.f90"
    source.write_text("""module pair
integer :: n
contains
subroutine both(a, b)
real(8), contiguous, intent(inout) :: a(:), b(:)
integer :: i
do i = 1, n
a(i) = a(i) + b(i)
enddo
a(1) = b(n); b(1) = a(n)
end subroutine
end module
""")
    prepared = prepare(ResidentProject((source,), (Selection("all", ("pair::both",)),)), tmp_path / "prepared",
                       fortran=fortran_compiler())
    assert [item["text"] for item in prepared["audit"]] == ["a(1) = b(n)", "b(1) = a(n)"]
    text = (tmp_path / "prepared/sources/pair.f90").read_text()
    # One block around the line: each array is read by one statement and changed by the other.
    assert text.count("a(1) = b(n); b(1) = a(n)") == 1
    line = text[text.index("fort_host = resident%active"):]
    assert line.index("call resident%host_modify([ &\n          resident%held(a), &\n          resident%held(b)])") \
        < line.index("a(1) = b(n); b(1) = a(n)") < line.index("call resident%host_done([ &")
    assert "observe" not in line[:line.index("call resident%host_done([ &")]


MARCH = """module march
integer :: n, steps
contains
subroutine run(a, b, c)
real(8), contiguous, intent(inout) :: a(:,:), b(:,:), c(:,:,:)
real(8) :: rate
integer :: step, i, j
rate = 0.1d0
do step = 1, steps
do j = 1, n
do i = 1, n
if (b(i,j) > 0) a(i,j) = a(i,j) + (b(i-1,j) + b(i+1,j) - 2 * a(i,j)) / rate
enddo
enddo
enddo
{more}
end subroutine
end module
"""


def marched(tmp_path, more=""):
    tmp_path.mkdir(exist_ok=True)
    source = tmp_path / "march.f90"
    source.write_text(MARCH.format(more=more))
    prepared = prepare(ResidentProject((source,), (Selection("all", ("march::run",)),)), tmp_path / "prepared",
                       fortran=fortran_compiler())
    return prepared, (tmp_path / "prepared/sources/march.f90").read_text()


def test_the_host_runs_a_loop_that_repeats_one_nest(tmp_path):
    prepared, text = marched(tmp_path)
    summary, = prepared["selections"]
    # The steps are no dimension of the nest: each one assigns the same elements. The host counts them
    # and the nest in the body is the entry.
    assert (summary["entries"], summary["nests"], summary["rejected"]) == (1, 1, [])
    assert [(item["first_line"], item["last_line"]) for item in summary["descended"]] == [(11, 17)]
    loop = text[text.index("do step = 1, steps"):]
    assert loop.index("do step = 1, steps") < loop.index("call resident_require(fort_device_run(") < loop.index(
        "\nenddo\n\nend subroutine")
    assert "publish" not in text


def test_the_host_runs_a_sweep_through_planes_that_each_use_the_one_before(tmp_path):
    prepared, text = marched(tmp_path, more="""do step = 2, steps
do j = 1, n
do i = 1, n
c(i,j,step) = c(i,j,step-1) + a(i,j)
enddo
enddo
enddo""")
    summary, = prepared["selections"]
    # The planes cannot be computed side by side, but each plane can: the host walks through them.
    assert (summary["entries"], summary["nests"], summary["rejected"]) == (2, 2, [])
    # Next to each other the two loops are one entry at first; it does not compile, and each is looked at
    # on its own.
    assert [(item["first_line"], item["last_line"]) for item in summary["descended"]] == [
        (11, 24), (11, 17), (18, 24)]
    sweep = text[text.index("do step = 2, steps"):]
    assert sweep.index("call resident_require(fort_device_run(") < sweep.index("\nenddo\n")
    assert "publish" not in text


def test_a_loop_is_rejected_as_a_whole_when_nothing_inside_it_compiles(tmp_path):
    # The plane depends on the row before as well: neither the sweep nor what it holds can be an entry. The
    # host gets the arrays once for the whole loop, not once per plane.
    prepared, text = marched(tmp_path, more="""do step = 2, steps
do j = 2, n
do i = 2, n
c(i,j,step) = c(i-1,j-1,step-1) + c(i-1,j,step) + c(i,j-1,step)
enddo
enddo
enddo""")
    summary, = prepared["selections"]
    assert (summary["entries"], summary["nests"]) == (1, 1)
    whole, = summary["rejected"]
    assert (whole["first_line"], whole["last_line"]) == (18, 24)
    assert len(whole["inside"]) >= 1
    assert all("Cannot parallelize" in reason for reason in whole["inside"])
    assert [(item["first_line"], item["last_line"]) for item in summary["descended"]] == [(11, 24), (11, 17)]
    assert text.count("call resident_require(fort_device_run(") == 1


def test_a_scalar_only_a_branch_reads_is_read_at_the_call_when_it_is_always_there(tmp_path):
    # Only a branch of the nest divides by rate. Read at the call, the scalar may be undefined where no
    # iteration takes the branch, but a variable that is always there can be loaded all the same: nothing
    # uses the value. That holds for a local, wherever it is assigned, and for a module variable.
    plain = MARCH.format(more="")
    module_scalar = plain.replace("real(8) :: rate\n", "").replace("rate = 0.1d0\n", "").replace(
        "integer :: n, steps\n", "integer :: n, steps\nreal(8) :: rate\n")

    def rejected(name, text):
        directory = tmp_path / name
        directory.mkdir()
        source = directory / "march.f90"
        source.write_text(text)
        prepared = prepare(ResidentProject((source,), (Selection("all", ("march::run",)),)), directory / "prepared",
                           fortran=fortran_compiler())
        return [reason for item in prepared["selections"][0]["rejected"] for reason in item["inside"]]

    assert rejected("local", plain) == []
    assert rejected("branch", plain.replace("rate = 0.1d0", "if (n > 3) rate = 0.1d0")) == []
    assert rejected("module", module_scalar) == []
    # An allocatable scalar may not be there at all: it is read only where the original statements read it.
    assert rejected("allocatable", module_scalar.replace("real(8) :: rate", "real(8), allocatable :: rate")) == [
        "Resident GPU entry contains native-only regions: CUDA value capture would evaluate a protected scalar"]


LEVELS = """module levels
integer :: n, steps
contains
subroutine run(u)
real(8), contiguous, intent(inout) :: u(0:,0:,:)
integer :: step, i, j, old, new
old = 1
new = 2
do step = 1, steps
do j = 1, n
do i = 1, n
u(i,j,new) = 0.25d0 * (u(i-1,j,old) + u(i+1,j,old) + u(i,j-1,old) + u(i,j+1,old))
enddo
enddo
old = new
new = 3 - old
enddo
end subroutine
end module
"""


def test_a_nest_that_is_independent_for_some_values_of_its_scalars_checks_them_when_it_is_called(tmp_path):
    source = tmp_path / "levels.f90"
    source.write_text(LEVELS)
    prepared = prepare(ResidentProject((source,), (Selection("all", ("levels::run",)),)), tmp_path / "prepared",
                       fortran=fortran_compiler())
    summary, = prepared["selections"]
    # The nest reads one level of the array and writes another: independent unless the two are the same.
    assert (summary["entries"], summary["nests"], summary["rejected"]) == (1, 1, [])
    entry, = prepared["entries"]
    condition = (Path(entry) / "shared_entry.cu").read_text().split("return FORT_SCOPE_DECLINED;")[0].rsplit("if (!", 1)[1]
    assert re.search(r"\(1 \* fort_v\d+_new\)\) \+ \(-1 \* fort_v\d+_old\)\) >= 0\)", condition)
    text = (tmp_path / "prepared/sources/levels.f90").read_text()
    call = text[text.index("fort_status = fort_device_run("):text.index("old = new")]
    # Called with equal levels the entry does nothing and says so; the original statements run instead.
    assert call.index("fort_resident = fort_status /= FORT_SCOPE_DECLINED") < call.index(
        "call resident_require(fort_status)") < call.index("if (.not. fort_resident) then") < call.index(
            "if (resident%active) call resident%publish()") < call.index("u(i,j,new) = 0.25d0")


def test_a_condition_that_only_says_a_loop_is_too_short_is_no_condition(tmp_path):
    source = tmp_path / "levels.f90"
    source.write_text(LEVELS.replace("u(i,j,new) = 0.25d0 * (u(i-1,j,old)", "u(i,j,old) = 0.25d0 * (u(i-1,j,old)"))
    prepared = prepare(ResidentProject((source,), (Selection("all", ("levels::run",)),)), tmp_path / "prepared",
                       fortran=fortran_compiler())
    summary, = prepared["selections"]
    # In place the nest conflicts with itself whenever it has two iterations: that stays a rejection.
    assert summary["entries"] == 0
    assert "Cannot parallelize" in "".join(item["reason"] + "".join(item.get("inside", ())) for item in summary["rejected"])


def test_an_entry_takes_an_assumed_shape_array_only_when_it_is_contiguous(tmp_path):
    source = tmp_path / "strides.f90"
    source.write_text("""module strides
integer :: n
contains
subroutine add(u, w)
real(8), intent(inout) :: u(:)
real(8), contiguous, intent(in) :: w(:)
integer :: i
do i = 1, n
u(i) = u(i) + w(i)
enddo
end subroutine
end module
""")
    prepare(ResidentProject((source,), (Selection("all", ("strides::add",), minimum_nests=1),)), tmp_path / "prepared",
            fortran=fortran_compiler())
    text = (tmp_path / "prepared/sources/strides.f90").read_text()
    # u may be every second element of its caller's array: bound by its first element and its extent, it
    # would be taken for the elements next to each other. w says that it is not such a section.
    assert "fort_resident = resident%active .and. is_contiguous(u)\n" in text
    assert "is_contiguous(w)" not in text
    call = text[text.index("fort_resident = resident%active"):]
    assert call.index("call resident_require(fort_device_run(") < call.index("      else\n") < call.index(
        "if (resident%active) call resident%publish()") < call.index("u(i) = u(i) + w(i)")


FRAMES = """module frames
integer :: n
type, abstract :: frame
  integer :: number = 0
contains
  procedure :: save => frame_save
  procedure(fill_interface), deferred :: fill
end type
abstract interface
  subroutine fill_interface(self, a)
    import :: frame
    class(frame), intent(inout) :: self
    real(8), contiguous, intent(in) :: a(:)
  end subroutine
end interface
type, extends(frame) :: plane
  real(8) :: kept = 0
contains
  procedure :: fill => plane_fill
end type
type(plane) :: planes(2)
contains
subroutine step(a, due)
real(8), contiguous, intent(inout) :: a(:)
logical, intent(in) :: due
integer :: i
do i = 1, n
a(i) = a(i) + 1
enddo
call save_all(a, due)
end subroutine
subroutine save_all(a, due)
real(8), contiguous, intent(in) :: a(:)
logical, intent(in) :: due
integer :: i
do i = 1, size(planes)
call planes(i)%save(a, due)
enddo
end subroutine
subroutine frame_save(self, a, due)
class(frame), intent(inout) :: self
real(8), contiguous, intent(in) :: a(:)
logical, intent(in) :: due
if (due) then
self%number = self%number + 1
call self%fill(a)
end if
end subroutine
subroutine plane_fill(self, a)
class(plane), intent(inout) :: self
real(8), contiguous, intent(in) :: a(:)
self%kept = a(1)
end subroutine
end module
"""


def test_a_call_through_a_type_bound_procedure_is_what_the_bound_procedures_are(tmp_path):
    source = tmp_path / "frames.f90"
    source.write_text(FRAMES)
    prepared = prepare(ResidentProject((source,), (Selection("all", (
        "frames::step", "frames::save_all", "frames::frame_save")),)), tmp_path / "prepared", fortran=fortran_compiler())
    # save is bound to a selected procedure: the calls through it hand nothing over, in every step. fill is
    # deferred; the one procedure an extending type binds under that name reads the array and is host code.
    finding, = prepared["audit"]
    assert (finding["procedure"], finding["text"], finding["calls"], finding["access"]) == (
        "frames::frame_save", "CALL self % fill(a)", ["frames::plane_fill"], "arrays")
    text = (tmp_path / "prepared/sources/frames.f90").read_text()
    save = text[text.index("subroutine frame_save"):text.index("subroutine plane_fill")]
    # The array comes back where a frame is due and stays on the device: the bound procedure only reads it.
    assert save.index("if (due) then") < save.index("call resident%observe([ &\n          resident%held(a)])") \
        < save.index("call self%fill(a)")
    assert "host_modify" not in text
    assert "fort_host" not in text[text.index("subroutine save_all"):text.index("subroutine frame_save")]
