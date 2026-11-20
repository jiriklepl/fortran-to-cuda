"""A whole application is prepared for residency from a short description, without a device."""
import json
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
    assert store.index("      else\n") < store.index("          a(i,j,k) = b(i,j,k)") < store.index("      endif\n")
    # The selected procedures run from one serial caller, so their directives are gone; the precision is fixed.
    assert "!$omp" not in stepper
    assert "#ifdef" not in (directory / "sources/fields.f90").read_text()
    # The halo copies are guarded, and the guard is checked before the entry is used.
    boundaries = (directory / "sources/boundaries.f90").read_text()
    assert "if (.not. (nx >= 2 .and. ny >= 2 .and. nz >= 2)) error stop" in boundaries


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


def test_adjacent_nests_share_an_entry_and_a_rejected_nest_publishes_first(tmp_path):
    # The assignment to n ends the group, so the loop-carried nest is compiled, and rejected, on its own.
    prepared, text = prepare_pair(tmp_path)
    summary, = prepared["selections"]
    assert (summary["entries"], summary["nests"]) == (1, 2)
    rejection, = summary["rejected"]
    assert "Cannot parallelize" in rejection["reason"]
    assert text.count("call resident_require(fort_device_run(") == 1
    # The loop-carried nest stays host code and gets the device values first.
    publish = text.index("if (resident%active) call resident%publish()")
    assert text.index("call resident_require(fort_device_run(") < publish < text.index("b(i) = b(i-1) + a(i)", publish)


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
    # The element copy reads a on the host right after an entry updated it on the device.
    assert (first["procedure"], first["arrays"], first["text"], first["known"]) == (
        "audited::step", ["a"], "b(1) = a(n)", False)
    # The array then goes to a procedure that is not selected and so runs on the host.
    assert (second["arrays"], second["calls"], second["text"]) == (["a"], ["audited::finish"], "CALL finish(a)")
    # The line is that of the preprocessed source the entries were exported from.
    assert Path(first["source"]).read_text().splitlines()[first["first_line"] - 1] == "b(1) = a(n)"
    assert first["id"] != second["id"]
    # Statements that name no bound array, and procedures without entries, are not reported.
    assert not any("n = n + 1" in item["text"] or item["procedure"] == "audited::untouched"
                   for item in audited(tmp_path / "all", ("audited::*",)))


def test_a_directive_or_a_selected_callee_answers_a_finding(tmp_path):
    # With finish selected, the call hands the array to code that manages it; the copy remains.
    finding, = audited(tmp_path / "callee", ("audited::step", "audited::finish"))
    assert finding["text"] == "b(1) = a(n)"
    # A directive directly before the statement says the host values are current there.
    assert audited(tmp_path / "directive", ("audited::step", "audited::finish"), before="!$fort observe(a)") == []


def test_known_findings_are_accepted_and_strict_mode_refuses_new_ones(tmp_path):
    finding, = audited(tmp_path / "report", ("audited::step", "audited::finish"))
    accepted, = audited(tmp_path / "known", ("audited::step", "audited::finish"),
                        known_host_accesses=frozenset({finding["id"]}), strict_audit=True)
    assert accepted["known"]
    with pytest.raises(ValueError, match=finding["id"] + r" audited::step line \d+: b\(1\) = a\(n\)"):
        audited(tmp_path / "strict", ("audited::step", "audited::finish"), strict_audit=True)
    assert not (tmp_path / "strict/prepared/sources").exists()


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
