"""Checked native access builders use full descriptors and live TARGET boxes."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.ir import CompilationError
from compiler.scopes.access import build_native_access, build_native_accesses
from compiler.tests.test_native_sections import analyze


def test_native_plan_and_execution_share_stable_checked_access_description(tmp_path):
    _, sections = analyze(tmp_path, "a(-1:1,:)=b(2:4,:)")
    accesses = build_native_accesses(sections, {"argument::a": "a_handle", "argument::b": "b_handle"}, "fort_native")
    output, input_ = accesses
    assert output.handle == "a_handle"
    assert input_.handle == "b_handle"
    source = "\n".join(output.prepare)
    assert "fort_scope_layout_get(fort_context, a_handle," in source
    assert "_origin(1) = -2_c_int64_t" in source
    assert "_extents(1) == 0_c_size_t) fort_native_0_origin(1) = 1_c_int64_t" in source
    assert "-1_c_int64_t - fort_native_0_origin(1)" in source
    assert "%read_count" not in source
    assert "%write_count" in source
    assert "%overwrite_count" in source
    assert "int(fort_native_0_extents(2), c_int64_t)" in source
    assert "FORT_SCOPE_BOUNDARY" in source
    assert "host_begin" not in source
    assert "plan_add" not in source
    assert "lower_bounds" not in source
    assert all(len(line) <= 132 for line in (*output.specification, *output.prepare))
    assert any("target" in line for line in output.specification)


def test_native_access_mapping_rejects_read_only_alias_ambiguity(tmp_path):
    _, sections = analyze(tmp_path, "i=int(sum(a(-1:1,:))+sum(b(2:4,:)))")
    assert sections.available, sections.reason
    with pytest.raises(CompilationError, match="aliases require a proved common physical mapping"):
        build_native_accesses(sections, {"argument::a": "same_handle", "argument::b": "same_handle"}, "fort_native")
    with pytest.raises(CompilationError, match="buffer mapping is unavailable"):
        build_native_accesses(sections, {}, "fort_native")


@pytest.mark.parametrize(("body", "declaration", "shape", "expected", "status"), [
    ("a(-1:1,:)=7", "real(8),intent(inout)::a(-2:,:)", [5, 3], [1, 4, 0, 3], 0),
    ("a(:,:)=7", "real(8),intent(inout)::a(-2:,:)", [0, 3], [], 0),
    ("a(3:2,:)=7", "real(8),intent(inout)::a(:,:)", [2, 3], [], 0),
    ("a(0,:)=7", "real(8),intent(inout)::a(:,:)", [2, 0], [], 5),
    ("a(1:4,:)=7", "real(8),intent(inout)::a(:,:)", [2, 3], [], 5),
    ("a(lbound(a,1):ubound(a,1),:)=7", "real(8),intent(inout)::a(-2:,:)", [0, 3], [], 0),
    ("a(lbound(a,1):ubound(a,1),:)=7", "real(8),intent(inout)::a(-2:,:)", [5, 3], [0, 5, 0, 3], 0),
    ("a(1:size(a,1),:)=7", "real(8),intent(inout)::a(:,:)", [5, 3], [0, 5, 0, 3], 0),
    ("a(1:size(a,1),:)=7", "real(8),intent(inout)::a(:,:)", [2147483648, 3], [], 5),
])
def test_generated_native_coordinates_against_real_fortran_descriptors(tmp_path, body, declaration, shape, expected, status):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    _, sections = analyze(tmp_path, body, declaration=declaration)
    assert sections.available, sections.reason
    resource, = sections.resources
    code = build_native_access(resource, "1_c_int64_t", "fort_native", context="1_c_int64_t", status="result")
    # A tiny descriptor-only stub exposes the same public layout ABI. No CUDA,
    # runtime linkage, payload allocation, source execution or timing is needed.
    interface = Path(__file__).resolve().parents[1] / "runtime" / "scoped_memory.f90"
    stub = """function layout_get(context,buffer,layout) bind(C,name='fort_scope_layout_get') result(status)
use iso_c_binding
use fort_scoped_memory,only:fort_scope_layout
integer(c_int64_t),value::context,buffer
type(fort_scope_layout),intent(out)::layout
integer(c_int)::status
integer(c_size_t),target,save::extents(2)
EXTENTS_ASSIGNMENT
layout=fort_scope_layout()
layout%rank=2
layout%extents=c_loc(extents)
status=0
end function
""".replace("EXTENTS_ASSIGNMENT", "extents=[" + ",".join(f"{value}_c_size_t" for value in shape) + "]")
    checked = "\n".join(["module check_access", "use iso_c_binding", "use fort_scoped_memory", "implicit none", "contains",
                          "subroutine prepare(result,counts,coordinates)", "integer(c_int),intent(out)::result",
                          "integer(c_size_t),intent(out)::counts(2),coordinates(4)", *code.specification,
                          "result=0", "counts=0", "coordinates=0", *code.prepare,
                          f"counts=[{code.access_name}%write_count,{code.access_name}%overwrite_count]"])
    if expected:
        checked += "\ncoordinates=[fort_native_w_lows(1,1),fort_native_w_highs(1,1),fort_native_w_lows(2,1),fort_native_w_highs(2,1)]"
    checked += "\nend subroutine\nend module\n"
    driver = """program caller
use check_access
implicit none
integer(c_int)::status
integer(c_size_t)::counts(2),coords(4)
call prepare(status,counts,coords)
STATUS_ASSERTION
COUNT_ASSERTION
COORDINATE_ASSERTION
print *,'ACCESS_OK'
end program
""".replace("STATUS_ASSERTION", f"if(status/={status}) error stop 'status disagreement'")
    driver = driver.replace("COUNT_ASSERTION", "if(any(counts/=1)) error stop 'section count disagreement'" if expected
                            else "if(any(counts/=0)) error stop 'empty or boundary section count disagreement'")
    driver = driver.replace("COORDINATE_ASSERTION", "if(any(coords/=[" + ",".join(map(str, expected))
                            + "])) error stop 'coordinate disagreement'" if expected else "")
    source = tmp_path / "check.f90"
    source.write_text(stub + checked + driver)
    binary = tmp_path / "check"
    result = subprocess.run([fortran, "-std=f2018", "-fcheck=all", str(interface), str(source), "-o", str(binary)],
                            cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    executed = subprocess.run([str(binary)], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "ACCESS_OK" in executed.stdout
