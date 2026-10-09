"""Native effects project through child descriptors onto one canonical root."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.frontend.native_sections import NativeSections
from compiler.ir import CompilationError
from compiler.scopes.access import build_native_access, build_native_view_accesses
from compiler.scopes.views import forget_view
from compiler.tests.test_native_sections import analyze


def test_exact_native_view_rectangles_keep_original_formal_bounds_and_root_origins(tmp_path):
    _, sections = analyze(tmp_path, "a(-1:1,:)=b(2:4,:)")
    handles = {"argument::a": "ha", "argument::b": "hb"}
    views = {"argument::a": "va", "argument::b": "vb"}
    output, input_ = build_native_view_accesses(sections, views, handles, "fort_native")
    body = "\n".join(output.prepare)
    assert output.handle == "ha"
    assert input_.handle == "hb"
    assert body.index("if (hb == ha)") < body.index("va%buffer /= ha")
    assert "fort_status = FORT_SCOPE_ALIAS" in body
    assert "if (va%buffer /= ha)" in body
    assert body.index("va%buffer /= ha") < body.index("fort_scope_view_get_v1")
    assert "fort_scope_view_get_v1(fort_context, va," in body
    assert "_origin(1) = -2_c_int64_t" in body
    assert "_layout%extents" in body
    assert "_layout%origins" in body
    assert "fort_native_0_physical_origin(1)" in body
    assert "%lower_bounds" not in body
    assert all("READ_ALL" not in line and "WRITE_ALL" not in line for line in output.prepare)
    assert any("fort_scope_view_layout_v1" in line for line in output.specification)
    assert all(len(line) <= 132 for item in (output, input_) for line in (*item.specification, *item.prepare))


@pytest.mark.parametrize("missing", ["view", "handle"])
def test_missing_native_view_mapping_is_a_boundary(tmp_path, missing):
    _, sections = analyze(tmp_path, "a(-1:1,:)=7")
    views = {} if missing == "view" else {"argument::a": "va"}
    handles = {} if missing == "handle" else {"argument::a": "ha"}
    with pytest.raises(CompilationError, match="native view mapping is unavailable"):
        build_native_view_accesses(sections, views, handles, "fort_native")


def test_unknown_or_aliased_native_views_cannot_use_whole_root_fallback(tmp_path):
    with pytest.raises(CompilationError, match="unknown effects"):
        build_native_view_accesses(NativeSections(False, "unknown effects"), {}, {}, "fort_native")
    _, sections = analyze(tmp_path, "a(-1:1,:)=b(2:4,:)")
    with pytest.raises(CompilationError, match="aliases require a proved common physical mapping"):
        build_native_view_accesses(sections, {"argument::a": "va", "argument::b": "vb"},
                                   {"argument::a": "same", "argument::b": "same"}, "fort_native")


def test_query_and_execution_keep_same_checked_effects_with_context_specific_failure_paths(tmp_path):
    _, sections = analyze(tmp_path, "a(-1:1,:)=7")
    kwargs = {"views": {"argument::a": "va"}, "handles": {"argument::a": "ha"}, "prefix": "fort_native"}
    query, = build_native_view_accesses(sections, **kwargs, on_error=("exit fort_query",))
    execution, = build_native_view_accesses(sections, **kwargs, on_error=("error stop 'no replay'",))
    assert query.specification == execution.specification
    assert query.prepare != execution.prepare
    assert tuple(line.replace("exit fort_query", "ON_ERROR") for line in query.prepare) == tuple(
        line.replace("error stop 'no replay'", "ON_ERROR") for line in execution.prepare)
    assert query.access_name == execution.access_name
    assert query.handle == execution.handle


@pytest.mark.parametrize(("query", "function"), [
    (True, "fort_scope_plan_forget_sections_v1"),
    (False, "fort_scope_forget_sections_v1"),
])
def test_native_partial_out_discards_only_the_actual_view(query, function):
    body = "\n".join(forget_view("va", 2, query=query, on_error=("return",)))
    assert "fort_scope_view_get_v1(fort_context, va," in body
    assert function + "(fort_context, va%buffer," in body
    assert "fort_discard_origin + fort_discard_extent" in body
    assert "if (all(fort_discard_extent > 0_c_size_t))" in body
    assert "forget_definition" not in body


@pytest.mark.parametrize(("body", "declaration", "shape", "origins", "expected", "status", "view_buffer", "stub_status"), [
    ("a(-1:0,2:3)=7", "real(8),intent(inout)::a(-2:,:)", [3, 4], [2, 1], [3, 5, 2, 4], 0, 1, 0),
    ("a(:,:)=7", "real(8),intent(inout)::a(-2:,:)", [3, 4], [2, 1], [2, 5, 1, 5], 0, 1, 0),
    ("a(lbound(a,1):ubound(a,1),:)=7", "real(8),intent(inout)::a(-2:,:)",
     [3, 4], [2, 1], [2, 5, 1, 5], 0, 1, 0),
    ("a(:,:)=7", "real(8),intent(inout)::a(-2:,:)", [0, 4], [0, 1], [], 0, 1, 0),
    ("a(0,:)=7", "real(8),intent(inout)::a(:,:)", [2, 0], [2, 0], [], 5, 1, 0),
    ("a(-1:2,:)=7", "real(8),intent(inout)::a(-2:,:)", [3, 4], [2, 1], [], 5, 1, 0),
    ("a(:,:)=7", "real(8),intent(inout)::a(:,:)", [3, 4], [2, 1], [], 5, 2, 0),
    ("a(:,:)=7", "real(8),intent(inout)::a(:,:)", [3, 4], [2, 1], [], 2, 1, 2),
])
def test_generated_native_view_coordinates_against_public_fortran_descriptor(
        tmp_path, body, declaration, shape, origins, expected, status, view_buffer, stub_status):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    _, sections = analyze(tmp_path, body, declaration=declaration)
    assert sections.available, sections.reason
    resource, = sections.resources
    code = build_native_access(resource, "1_c_int64_t", "fort_native", context="1_c_int64_t",
                               status="result", view="borrowed")
    shape_text = ",".join(str(value) + "_c_size_t" for value in shape)
    origin_text = ",".join(str(value) + "_c_size_t" for value in origins)
    stub = f"""function view_get(context,view,layout) bind(C,name='fort_scope_view_get_v1') result(status)
use iso_c_binding
use fort_scoped_memory
integer(c_int64_t),value::context
type(fort_scope_view_v1),intent(in)::view
type(fort_scope_view_layout_v1),intent(out)::layout
integer(c_int)::status
integer(c_size_t),target,save::extents(2),origins(2)
extents=[{shape_text}]
origins=[{origin_text}]
layout=fort_scope_view_layout_v1()
layout%root%rank=2
layout%extents=c_loc(extents)
layout%origins=c_loc(origins)
status={stub_status}
end function
"""
    checked = "\n".join(["module check_access", "use iso_c_binding", "use fort_scoped_memory", "implicit none", "contains",
                          "subroutine prepare(result,counts,coordinates)", "integer(c_int),intent(out)::result",
                          "integer(c_size_t),intent(out)::counts(2),coordinates(4)",
                          "type(fort_scope_view_v1)::borrowed", *code.specification,
                          "result=0", "counts=0", "coordinates=0", f"borrowed%buffer={view_buffer}_c_int64_t",
                          *code.prepare, f"counts=[{code.access_name}%write_count,{code.access_name}%overwrite_count]"])
    if expected:
        checked += ("\ncoordinates=[fort_native_w_lows(1,1),fort_native_w_highs(1,1),"
                    "fort_native_w_lows(2,1),fort_native_w_highs(2,1)]")
    checked += "\nend subroutine\nend module\n"
    driver = f"""program caller
use check_access
implicit none
integer(c_int)::status
integer(c_size_t)::counts(2),coords(4)
call prepare(status,counts,coords)
if(status/={status}) error stop 'status disagreement'
if(any(counts/={1 if expected else 0})) error stop 'count disagreement'
"""
    if expected:
        driver += "if(any(coords/=[" + ",".join(map(str, expected)) + "])) error stop 'physical coordinate disagreement'\n"
    driver += "print *,'NATIVE_VIEW_ACCESS_OK'\nend program\n"
    interface = Path(__file__).resolve().parents[1] / "runtime" / "scoped_memory.f90"
    source = tmp_path / "check.f90"
    source.write_text(stub + checked + driver)
    binary = tmp_path / "check"
    compiled = subprocess.run([fortran, "-std=f2018", "-fcheck=all", str(interface), str(source), "-o", str(binary)],
                              cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run([str(binary)], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "NATIVE_VIEW_ACCESS_OK" in executed.stdout


def test_nested_distinct_formal_expressions_cannot_hide_equal_root_tokens(tmp_path):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    _, sections = analyze(tmp_path, "a(-1:1,:)=b(2:4,:)")
    codes = build_native_view_accesses(sections, {"argument::a": "va", "argument::b": "vb"},
                                       {"argument::a": "va%buffer", "argument::b": "vb%buffer"},
                                       "fort_native", context="1_c_int64_t", status="result")
    source = tmp_path / "alias.f90"
    source.write_text("""function view_get(context,view,layout) bind(C,name='fort_scope_view_get_v1') result(status)
use iso_c_binding
use fort_scoped_memory
integer(c_int64_t),value::context
type(fort_scope_view_v1),intent(in)::view
type(fort_scope_view_layout_v1),intent(out)::layout
integer(c_int)::status
error stop 'alias must be rejected before metadata preparation'
end function
module check_alias
use iso_c_binding
use fort_scoped_memory
implicit none
contains
subroutine prepare(result)
integer(c_int),intent(out)::result
type(fort_scope_view_v1)::va,vb
""" + "\n".join(line for code in codes for line in code.specification) + "\n" + """result=0
va%buffer=1_c_int64_t
vb%buffer=1_c_int64_t
""" + "\n".join(line for code in codes for line in code.prepare) + "\n" + """end subroutine
end module
program caller
use check_alias
implicit none
integer(c_int)::status
call prepare(status)
if(status/=FORT_SCOPE_ALIAS) error stop 'runtime root aliases were admitted'
print *,'NATIVE_VIEW_ALIAS_OK'
end program
""")
    interface = Path(__file__).resolve().parents[1] / "runtime" / "scoped_memory.f90"
    binary = tmp_path / "alias"
    compiled = subprocess.run([fortran, "-std=f2018", "-fcheck=all", str(interface), str(source), "-o", str(binary)],
                              cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run([str(binary)], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "NATIVE_VIEW_ALIAS_OK" in executed.stdout
