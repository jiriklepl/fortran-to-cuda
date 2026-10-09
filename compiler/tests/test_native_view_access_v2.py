"""Rank-reduced native views project retained and fixed physical axes exactly."""

from __future__ import annotations

import shutil
import subprocess
import re
from pathlib import Path

import pytest

from compiler.scopes.access import build_native_access
from compiler.tests.test_native_sections import analyze


def test_rank_reduced_mapping_keeps_formal_bounds_and_full_root_sections(tmp_path):
    _, sections = analyze(tmp_path, "a(-2:-1,2:4)=7")
    resource, = sections.resources
    code = build_native_access(resource, "ha", "fort_plane", view="va")
    specification, body = "\n".join(code.specification), "\n".join(code.prepare)
    flattened_body = " ".join(body.replace("&", "").split())
    assert "fort_scope_view_layout_v2" in specification
    assert "fort_scope_view_get_v2(fort_context, va," in body
    assert "_w_lows(15,1)" in specification
    assert "_root%rank" not in body
    assert "fort_plane_layout%root%rank > 15" in body
    assert "do fort_plane_root_axis = 1, int(fort_plane_layout%root%rank)" in body
    assert "fort_plane_root_axis = int(fort_plane_physical_axes(1)) + 1" in body
    assert "fort_plane_physical_origin(fort_plane_root_axis) + 1_c_size_t" in flattened_body
    # Each output/must-write box has its own empty guard; checking the last
    # guard against the first address would compare different rectangles.
    blocks, checked = [], 0
    for line in re.sub(r"\s*&\n\s*&\s*", " ", body).splitlines():
        line = line.strip()
        if line.startswith("if (") and line.endswith(" then"):
            blocks.append(line)
        elif line == "endif":
            blocks.pop()
        elif "fort_plane_physical_origin(fort_plane_root_axis) + 1_c_size_t" in line:
            assert "if (.not. fort_plane_empty) then" in blocks
            checked += 1
    assert checked > 0
    assert "fort_plane_origin(1) = -2_c_int64_t" in body
    assert all(len(line) <= 132 for line in (*code.specification, *code.prepare))


def test_explicit_v1_view_access_remains_available(tmp_path):
    _, sections = analyze(tmp_path, "a(-2,:)=7")
    resource, = sections.resources
    code = build_native_access(resource, "ha", "fort_plane", view="va", view_abi=1)
    assert "fort_scope_view_get_v1" in "\n".join(code.prepare)
    assert any("fort_scope_view_layout_v1" in line for line in code.specification)
    assert not any("physical_axes" in line for line in code.prepare)


@pytest.mark.parametrize("axes,body,extents,expected", [
    ([2, 0], "a(-2:-1,2:4)=7", [3, 4], [2, 5, 2, 3, 3, 5]),
    ([0, 2], "a(-2:-1,2:4)=7", [3, 4], [1, 3, 2, 3, 4, 7]),
    ([2, 0], "a(:,:)=7", [0, 4], []),
])
def test_generated_rank_reduced_coordinates_against_public_fortran_layout(tmp_path, axes, body, extents, expected):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    _, sections = analyze(tmp_path, body)
    resource, = sections.resources
    code = build_native_access(resource, "1_c_int64_t", "fort_plane", context="1_c_int64_t", view="view", status="result")
    interface = Path(__file__).resolve().parents[1] / "runtime" / "scoped_memory.f90"
    stub = """function view_get(context,view,layout) bind(C,name='fort_scope_view_get_v2') result(status)
use iso_c_binding
use fort_scoped_memory,only:fort_scope_view_v2,fort_scope_view_layout_v2
integer(c_int64_t),value::context
type(fort_scope_view_v2),intent(in)::view
type(fort_scope_view_layout_v2),intent(out)::layout
integer(c_int)::status
integer(c_size_t),target,save::origins(3)=[1,2,3],root_extents(3)=[6,7,8]
integer(c_size_t),target,save::extents(2)=EXTENTS
integer(c_int32_t),target,save::axes(2)=AXES
layout=fort_scope_view_layout_v2()
layout%root%rank=3
layout%root%extents=c_loc(root_extents)
layout%rank=2
layout%origins=c_loc(origins)
layout%extents=c_loc(extents)
layout%axes=c_loc(axes)
status=0
end function
""".replace("EXTENTS", "[" + ",".join(map(str, extents)) + "]").replace("AXES", "[" + ",".join(map(str, axes)) + "]")
    checked = "\n".join(["module check_access", "use iso_c_binding", "use fort_scoped_memory", "implicit none", "contains",
        "subroutine prepare(result,count,coordinates)", "integer(c_int),intent(out)::result",
        "integer(c_size_t),intent(out)::count,coordinates(6)", "type(fort_scope_view_v2)::view",
        *code.specification, "result=0", "count=0", "coordinates=0", "view%buffer=1", *code.prepare,
        f"count={code.access_name}%write_count"])
    if expected:
        checked += "\ncoordinates=[fort_plane_w_lows(1,1),fort_plane_w_highs(1,1),fort_plane_w_lows(2,1), &"
        checked += "\nfort_plane_w_highs(2,1),fort_plane_w_lows(3,1),fort_plane_w_highs(3,1)]"
    checked += "\nend subroutine\nend module\n"
    assertions = ["if(status/=0) error stop 'status disagreement'", f"if(count/={int(bool(expected))}) error stop 'count disagreement'"]
    if expected:
        assertions += ["if(any(coords/=[" + ",".join(map(str, expected)) + "])) error stop 'coordinate disagreement'"]
    driver = "\n".join(["program caller", "use check_access", "implicit none", "integer(c_int)::status",
        "integer(c_size_t)::count,coords(6)", "call prepare(status,count,coords)", *assertions,
        "print *,'PLANE_ACCESS_OK'", "end program"])
    source, binary = tmp_path / "check.f90", tmp_path / "check"
    source.write_text(stub + checked + driver)
    built = subprocess.run([fortran, "-std=f2018", "-fcheck=all", str(interface), str(source), "-o", str(binary)],
                           cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stdout + built.stderr
    ran = subprocess.run([str(binary)], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert "PLANE_ACCESS_OK" in ran.stdout
