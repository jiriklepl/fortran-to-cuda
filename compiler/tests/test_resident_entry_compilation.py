"""Exported resident entries are compiled in process, exactly as the command line compiles them."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.ir import CompilationError
from compiler.scopes.resident_entries import (
    compile_resident_entries,
    compile_resident_entry,
    export_resident_entries,
    pass_scalars_by_value,
)

ROOT = Path(__file__).resolve().parents[2]

SOURCE = """module grid
integer :: nx
real(8) :: dx
end module
module fields
use grid
contains
subroutine step(a,b)
real(8),intent(inout)::a(-1:),b(-1:)
integer::i
do i=1,nx
a(i)=a(i)+dx
enddo
end subroutine
subroutine carried(b)
real(8),intent(inout)::b(-1:)
integer::i
do i=2,nx
b(i)=b(i-1)+1
enddo
end subroutine
end module
"""


def export(tmp_path, procedures=("fields::step", "fields::carried")):
    source = tmp_path / "input.f90"
    source.write_text(SOURCE)
    result = export_resident_entries([source], list(procedures), tmp_path / "exported")
    assert not result["rejected"]
    return result


def test_supported_entries_are_written_and_a_rejected_span_stays_with_the_caller(tmp_path):
    exported = export(tmp_path)
    record = compile_resident_entries(exported, tmp_path / "entries")
    assert json.loads((tmp_path / "entries/device-entries.json").read_text()) == json.loads(json.dumps(record))
    item, = record["entries"]
    assert item["region"]["procedure"] == "fields::step"
    directory = Path(item["directory"])
    assert directory.name == item["region"]["source_identity"][:16]
    for name in ("shared_entry.cu", "shared_interface.f90", "runtime.hpp", "scoped_runtime.cu", "scoped_regions.hpp"):
        assert (directory / name).is_file()
    report = json.loads((directory / "compiler-report.json").read_text())
    assert report["supported"]
    assert report["scoped"] == json.loads(json.dumps(item["entry"]))
    assert all(region["gpu_available"] for region in item["entry"]["region_execution"])
    assert item["entry"]["argument_order"][:2] == ["context", "mode"]
    # The loop-carried nest is the compiler's rejection, with the span the caller keeps native.
    rejection, = record["rejected"]
    original = next(entry for entry in exported["entries"] if entry["procedure"] == "fields::carried")
    assert (rejection["procedure"], rejection["first_line"], rejection["last_line"]) == (
        "fields::carried", original["first_line"], original["last_line"])
    assert "Cannot parallelize" in rejection["reason"]
    assert not (tmp_path / "entries" / original["source_identity"][:16]).exists()


def test_in_process_compilation_writes_what_the_command_line_writes(tmp_path):
    entry, = export(tmp_path, ("fields::step",))["entries"]
    report = compile_resident_entry(entry["input_path"], entry["entry"], tmp_path / "direct")
    command = [sys.executable, "-m", "compiler", "--input", entry["input_path"], "--kernel", entry["entry"],
               "--opt-level", "1", "--fallback", "error", "--json", "--output-dir", str(tmp_path / "cli"),
               "--common-header", "runtime.hpp", "--host-threads", "4"]
    result = subprocess.run(command, cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
                            capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == json.loads(json.dumps(report))
    written = sorted(path.name for path in (tmp_path / "cli").iterdir())
    assert sorted(path.name for path in (tmp_path / "direct").iterdir()) == sorted([*written, "compiler-report.json"])
    for name in written:
        assert (tmp_path / "direct" / name).read_bytes() == (tmp_path / "cli" / name).read_bytes(), name


def test_an_entry_needs_a_positive_thread_budget(tmp_path):
    entry, = export(tmp_path, ("fields::step",))["entries"]
    with pytest.raises(CompilationError, match="thread budget"):
        compile_resident_entry(entry["input_path"], entry["entry"], tmp_path / "invalid", host_threads=0)
    assert not (tmp_path / "invalid").exists()


def test_lower_bounds_and_scalars_that_are_always_there_become_value_arguments(tmp_path):
    exported = export(tmp_path, ("fields::step",))
    rewritten = pass_scalars_by_value(exported, tmp_path / "values")
    entry, = rewritten["entries"]
    original, = exported["entries"]
    assert entry["value_inputs"] == ["fort_region_lb_a_1", "dx", "nx"]
    assert Path(entry["input_path"]).parent == tmp_path / "values"
    assert entry["original_numerical_input"]["path"] == original["input_path"]
    text = Path(entry["input_path"]).read_text()
    assert "integer, value, intent(in) :: fort_region_lb_a_1" in text
    assert "integer, value, intent(in) :: nx" in text
    assert "real(8), value, intent(in) :: dx" in text
    # Applying the pass to its own output changes nothing.
    again, = pass_scalars_by_value(rewritten, tmp_path / "again")["entries"]
    assert Path(again["input_path"]).read_text() == text
    # The rewritten entry still compiles and takes the scalars by value.
    report = compile_resident_entry(entry["input_path"], entry["entry"], tmp_path / "entry")
    passing = {item["name"]: item["passing"] for item in report["scoped"]["scalar_parameters"]}
    assert passing == {"fort_region_lb_a_1": "value", "dx": "value", "nx": "value"}


LOWERED = """module stencil
integer :: n
contains
pure real(8) function centred(i, a, h)
integer, intent(in) :: i
real(8), contiguous, intent(in) :: a(-2:)
real(8), intent(in) :: h
centred = (a(i+1) - a(i-1)) / (2*h)
end function
pure real(8) function total(a)
real(8), contiguous, intent(in) :: a(-2:)
total = sum(a)
end function
subroutine slope(u, du, h)
real(8), contiguous, intent(in) :: u(0:)
real(8), contiguous, intent(inout) :: du(0:)
real(8), intent(in) :: h
integer :: i
do i = 1, n
du(i) = centred(i - 2, u, h)
enddo
end subroutine
subroutine scaled(u, du)
real(8), contiguous, intent(in) :: u(0:)
real(8), contiguous, intent(inout) :: du(0:)
integer :: i
do i = 1, n
du(i) = total(u)
enddo
end subroutine
end module
"""


def test_a_helper_whose_array_argument_has_a_lower_bound_is_copied_from_position_one(tmp_path):
    source = tmp_path / "stencil.f90"
    source.write_text(LOWERED)
    exported = export_resident_entries([source], ["stencil::slope"], tmp_path / "exported")
    assert not exported["rejected"]
    entry, = exported["entries"]
    text = Path(entry["input_path"]).read_text()
    helper = text[text.index("contains", text.index("contains") + 1):]
    # The copied helper declares the array from one and shifts every element by the original lower bound.
    assert "a(:)" in helper.replace(" ", "").lower()
    assert "-2:" not in helper.replace(" ", "")
    compact = helper.replace(" ", "")
    assert "a((i+1)-(-2)+1)" in compact
    assert "a((i-1)-(-2)+1)" in compact
    report = compile_resident_entry(entry["input_path"], entry["entry"], tmp_path / "entry")
    assert report["supported"]
    # A helper that uses such an array as a whole is not copied: its bounds would differ.
    rejection, = export_resident_entries([source], ["stencil::scaled"], tmp_path / "whole")["rejected"]
    assert "require element references" in rejection["reason"]
