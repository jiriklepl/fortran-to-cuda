"""Native boundary effects describe exact physical sections, including holes."""

from __future__ import annotations

import pytest
import shutil
import subprocess
from pathlib import Path

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.access import build_native_access
from compiler.tests.test_native_sections import analyze


def test_runtime_affine_bounds_have_checked_source_dependencies(tmp_path):
    _, sections = analyze(tmp_path, "a(1:n-1,:)=a(1:n-1,:)+7")
    assert sections.available, sections.reason
    resource, = sections.resources
    upper = resource.writes[0].axes[0].upper
    assert upper.operator == "-"
    assert upper.children[0].resource == "argument::n"
    with pytest.raises(CompilationError, match="safely reached scalar mapping"):
        build_native_access(resource, "handle", "fort_exact")
    code = build_native_access(resource, "handle", "fort_exact", parameters={"argument::n": "original_n"})
    body = "\n".join(code.prepare)
    assert "int(original_n, c_int64_t)" in body
    assert "- 1_c_int64_t)" in body
    assert "2147483647_c_int64_t" in body
    assert all(len(line) <= 132 for line in (*code.specification, *code.prepare))


def test_unit_loop_faces_remain_separate_with_original_zero_trip_guard(tmp_path):
    _, sections = analyze(tmp_path, "do i=1,n\na(-2,i)=a(-2,i)+1\na(2,i)=4\nenddo")
    assert sections.available, sections.reason
    resource, = sections.resources
    assert len(resource.writes) == 2
    assert len(resource.reads) == 1
    assert [box.axes[0].lower.value for box in resource.writes] == [-2, 2]
    assert all(box.axes[0].point and not box.axes[1].point for box in resource.writes)
    assert all(len(box.iterations) == 1 for box in resource.writes)
    code = build_native_access(resource, "handle", "fort_exact", parameters={"argument::n": "original_n"})
    body = "\n".join(code.prepare)
    guard = body.index("if (1_c_int64_t > fort_exact_bound")
    coordinate = body.index("fort_exact_lower(1)")
    assert guard < coordinate
    assert body.index("if (.not. fort_exact_empty) then", guard) < coordinate


@pytest.mark.parametrize("body,points", [
    ("a(-2:2:2,:)=7", [-2, 0, 2]),
    ("a(2:-2:-2,:)=7", [2, 0, -2]),
    ("do i=-2,2,2\na(i,:)=7\nenddo", [-2, 0, 2]),
])
def test_constant_strides_are_bounded_unions_never_bounding_overwrites(tmp_path, body, points):
    _, sections = analyze(tmp_path, body)
    assert sections.available, sections.reason
    resource, = sections.resources
    assert len(resource.overwrites) == len(points)
    assert resource.writes == resource.overwrites
    assert [box.axes[0].lower.value for box in resource.overwrites] == points
    assert all(box.axes[0].point for box in resource.overwrites)
    assert not any(box.axes[0].lower.value == -2 and box.axes[0].upper.value == 2
                   for box in resource.overwrites)


@pytest.mark.parametrize("body,reason", [
    ("a(-2:64:2,:)=7", "rectangle budget"),
    ("n=n-1\na(n,:)=7", "bound changes"),
    ("do i=1,n\na(i,i)=7\nenddo", "rectangular iterator"),
    ("do i=1,n\na(i:i+1,:)=7\nenddo", "moving array sections"),
    ("a(n*n,:)=7", "affine scalar arithmetic"),
])
def test_unproved_or_expensive_unions_remain_unknown(tmp_path, body, reason):
    _, sections = analyze(tmp_path, body)
    assert not sections.available
    assert reason in sections.reason
    assert sections.resources == ()


def test_original_allocation_descriptor_supplies_hidden_negative_bounds(tmp_path):
    path = tmp_path / "renamed.f90"
    path.write_text("module fields\nreal(8),allocatable::scratch(:,:)\ncontains\n"
                    "subroutine touch(n)\ninteger,intent(in)::n\ninteger::i\n"
                    "do i=1,n\nscratch(-3,i)=7\nenddo\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    analysis.authorize_stable_module_allocatables({"fields::scratch"})
    sections = analysis.native_sections("fields::touch")
    assert sections.available, sections.reason
    resource, = sections.resources
    assert resource.descriptor_lower_bounds
    assert resource.lower_bounds == (None, None)
    code = build_native_access(resource, "handle", "fort_exact", parameters={"argument::n": "original_n"})
    body = "\n".join(code.prepare)
    assert "%lower_bounds" in body
    assert body.index("c_associated(fort_exact_layout%lower_bounds)") < body.index("c_f_pointer(fort_exact_layout%lower_bounds")
    assert "fort_exact_origin(1) = fort_exact_descriptor_origin(1)" in body
    assert "_extents(1) == 0_c_size_t) fort_exact_origin(1) = 1_c_int64_t" in body


def test_empty_literal_stride_union_has_no_written_or_defined_holes(tmp_path):
    _, sections = analyze(tmp_path, "a(2:-2:2,:)=7")
    assert sections.available, sections.reason
    resource, = sections.resources
    assert resource.writes == resource.overwrites == ()


@pytest.mark.parametrize("body,n,expected,status", [
    ("a(1:n-1,:)=7", 3, [(3, 5, 0, 3)], 0),
    ("do i=1,n\na(-100,i)=7\nenddo", 0, [], 0),
    ("do i=1,n\na(-2,i)=7\nenddo", 3, [(0, 1, 0, 3)], 0),
    ("do i=n,1,-1\na(-2,i)=7\nenddo", 3, [(0, 1, 0, 3)], 0),
    ("a(n*2,:)=7", 2147483647, [], 5),
    ("a(-2:2:2,:)=7", 1, [(0, 1, 0, 3), (2, 3, 0, 3), (4, 5, 0, 3)], 0),
])
def test_generated_affine_coordinates_against_public_layout(tmp_path, body, n, expected, status):
    """Exercise generated checks, zero-trip guards and exact hole coverage."""
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    _, sections = analyze(tmp_path, body)
    assert sections.available, sections.reason
    resource, = sections.resources
    code = build_native_access(resource, "1_c_int64_t", "fort_exact", context="1_c_int64_t", status="result",
                               parameters={"argument::n": "original_n"})
    interface = Path(__file__).resolve().parents[1] / "runtime" / "scoped_memory.f90"
    stub = """function layout_get(context,buffer,layout) bind(C,name='fort_scope_layout_get') result(status)
use iso_c_binding
use fort_scoped_memory,only:fort_scope_layout
integer(c_int64_t),value::context,buffer
type(fort_scope_layout),intent(out)::layout
integer(c_int)::status
integer(c_size_t),target,save::extents(2)=[5,3]
integer(c_int64_t),target,save::lowers(2)=[-2,1]
layout=fort_scope_layout()
layout%rank=2
layout%extents=c_loc(extents)
layout%lower_bounds=c_loc(lowers)
status=0
end function
"""
    checked = "\n".join(["module check_access", "use iso_c_binding", "use fort_scoped_memory", "implicit none", "contains",
        "subroutine prepare(original_n,result,count,coordinates)", "integer(c_int64_t),intent(in)::original_n",
        "integer(c_int),intent(out)::result", "integer(c_size_t),intent(out)::count,coordinates(4,32)",
        *code.specification, "result=0", "count=0", "coordinates=0", *code.prepare,
        f"count={code.access_name}%write_count"])
    if expected:
        checked += "\ncoordinates(1,1:count)=fort_exact_w_lows(1,1:count)\ncoordinates(2,1:count)=fort_exact_w_highs(1,1:count)"
        checked += "\ncoordinates(3,1:count)=fort_exact_w_lows(2,1:count)\ncoordinates(4,1:count)=fort_exact_w_highs(2,1:count)"
    checked += "\nend subroutine\nend module\n"
    assertions = [f"if(status/={status}) error stop 'status disagreement'", f"if(count/={len(expected)}) error stop 'count disagreement'"]
    assertions += [f"if(any(coords(:,{index})/=[" + ",".join(map(str, values)) + "])) error stop 'coordinate disagreement'"
                   for index, values in enumerate(expected, 1)]
    driver = "\n".join(["program caller", "use check_access", "implicit none", "integer(c_int)::status",
        "integer(c_size_t)::count,coords(4,32)", f"call prepare({n}_c_int64_t,status,count,coords)", *assertions,
        "print *,'AFFINE_ACCESS_OK'", "end program"])
    source = tmp_path / "check.f90"
    source.write_text(stub + checked + driver)
    binary = tmp_path / "check"
    built = subprocess.run([fortran, "-std=f2018", "-fcheck=all", str(interface), str(source), "-o", str(binary)],
                           cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(binary)], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "AFFINE_ACCESS_OK" in ran.stdout
