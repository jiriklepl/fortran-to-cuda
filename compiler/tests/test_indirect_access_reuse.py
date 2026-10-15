"""Identical native footprints reuse storage without weakening inspection."""

import shutil
import subprocess
from dataclasses import is_dataclass
from pathlib import Path

import pytest

from compiler.frontend.immutable_graph import freeze_graph
from compiler.scopes.indirect_access import build_indirect_accesses
from compiler.tests.test_indirect_native_sections import compile_inspector, fixture


def emitted(sections):
    code, = build_indirect_accesses(
        sections, {"argument::a": "1_c_int64_t"}, "fort_inspect",
        resource_names={"argument::markers": "markers"},
        context="1_c_int64_t", status="result")
    return code, "\n".join(code.prepare)


@pytest.mark.parametrize("sealed", [False, True])
def test_identical_ordered_footprints_share_retained_target_storage(tmp_path, sealed):
    _, sections = fixture(tmp_path)
    if sealed:
        freeze_graph(sections, own_if=is_dataclass,
                     borrowed=(*sections.metadata, *sections.scalars))
    code, source = emitted(sections)
    assert f"{code.access_name}%writes = {code.access_name}%reads" in source
    assert "do fort_inspect_0_r_i" in source
    assert "do fort_inspect_0_w_i" not in source
    assert "fort_inspect_0_w_sections" not in "\n".join(code.specification)
    assert "%overwrite_count" not in source
    assert source.index("fort_inspect_0_scan_steps >") < source.index("%writes =")


@pytest.mark.parametrize("body", [
    "do q=1,size(markers)\n"
    "a(markers(q)%row,markers(q)%column)=a(markers(q)%row+1,markers(q)%column)\nenddo",
    # The unordered point sets match, but exact-union insertion order differs.
    "do q=1,size(markers)\n"
    "a(markers(q)%row,markers(q)%column)=a(markers(q)%row+1,markers(q)%column)\n"
    "a(markers(q)%row+1,markers(q)%column)=a(markers(q)%row,markers(q)%column)\nenddo",
])
def test_different_expressions_or_reference_order_keep_separate_unions(tmp_path, body):
    _, sections = fixture(tmp_path, body)
    assert sections.available, sections.reason
    code, source = emitted(sections)
    assert f"{code.access_name}%writes = {code.access_name}%reads" not in source
    assert "do fort_inspect_0_r_i" in source
    assert "do fort_inspect_0_w_i" in source
    status, boxes, counts = compile_inspector(tmp_path, [(-2, 0)], body=body)
    assert status == 0
    assert boxes
    assert counts["READS"] == counts["WRITES"] == 1


def test_equivalent_final_index_keeps_original_intermediate_overflow_guard(tmp_path):
    body = ("do q=1,size(markers)\n"
            "a((markers(q)%row-2147483647)+2147483647,markers(q)%column)="
            "a(markers(q)%row,markers(q)%column)\nenddo")
    # Final row -2 is in bounds, but the original subtraction is outside the
    # supported default-INTEGER ABI. Algebraic footprint equality is unsafe.
    status, boxes, counts = compile_inspector(tmp_path, [(-2, 0)], body=body)
    assert status == 5
    assert boxes == []
    assert counts == {}


@pytest.mark.parametrize(("limit", "expected"), [(3, 5), (4, 0)])
@pytest.mark.parametrize("nested_empty", [False, True])
def test_reused_write_scan_still_consumes_its_full_logical_budget(
        tmp_path, monkeypatch, limit, expected, nested_empty):
    import compiler.scopes.indirect_access as lowering
    monkeypatch.setattr(lowering, "INSPECTION_STEP_LIMIT", limit)
    body = None
    extra = ""
    points = [(-2, 0), (-2, 0)]
    if nested_empty:
        points = []
        extra = "integer::r"
        body = ("do q=1,2\ndo r=1,0\n"
                "a(markers(q)%row,markers(q)%column)="
                "a(markers(q)%row,markers(q)%column)+markers(q)%value\nenddo\nenddo")
    status, boxes, counts = compile_inspector(tmp_path, points, body=body, extra=extra)
    assert status == expected
    if expected:
        assert boxes == []
        assert counts == {}
    else:
        expected_count = 0 if nested_empty else 1
        assert counts == {"READS": expected_count, "WRITES": expected_count}


@pytest.mark.parametrize(("limit", "expected"), [(1, 5), (2, 0)])
def test_unlooped_reused_access_still_counts_twice(tmp_path, monkeypatch, limit, expected):
    import compiler.scopes.indirect_access as lowering
    monkeypatch.setattr(lowering, "INSPECTION_STEP_LIMIT", limit)
    body = "a(markers(1)%row,markers(1)%column)=a(markers(1)%row,markers(1)%column)+markers(1)%value"
    status, _, counts = compile_inspector(tmp_path, [(-2, 0)], body=body)
    assert status == expected
    assert counts == ({} if expected else {"READS": 1, "WRITES": 1})


def test_shared_partial_write_correction_matches_original_complete_field_once(tmp_path):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    _, sections = fixture(tmp_path)
    code, _ = emitted(sections)
    interface = Path(__file__).resolve().parents[1] / "runtime" / "scoped_memory.f90"
    generated = "\n".join([
        "function layout_get(context,buffer,layout) bind(C,name='fort_scope_layout_get') result(status)",
        "use iso_c_binding", "use fort_scoped_memory,only:fort_scope_layout",
        "integer(c_int64_t),value::context,buffer", "type(fort_scope_layout),intent(out)::layout",
        "integer(c_int)::status", "integer(c_size_t),target,save::extents(2)=[5_c_size_t,5_c_size_t]",
        "layout=fort_scope_layout()", "layout%rank=2", "layout%extents=c_loc(extents)",
        "status=0", "end function", "module inspected", "use iso_c_binding",
        "use fort_scoped_memory", "use geometry", "implicit none", "contains",
        "subroutine checked_adjust(a,markers,result)", "real(8),intent(inout)::a(-2:,0:)",
        "type(marker_t),intent(in)::markers(:)", "integer(c_int),intent(out)::result",
        *code.specification, "result=0", *code.prepare,
        "call adjust(a,markers,size(markers))", "end subroutine", "end module",
        "program caller", "use inspected", "implicit none",
        "real(8)::a(-2:2,0:4),expected(-2:2,0:4)", "type(marker_t)::markers(3)",
        "integer(c_int)::result", "integer::i",
        "a=reshape([(real(i,8),i=1,25)],[5,5])", "expected=a",
        "markers(1)=marker_t(-2_8,0_8,3.0_8)",
        "markers(2)=marker_t(-2_8,0_8,5.0_8)",
        "markers(3)=marker_t(2_8,4_8,-2.0_8)",
        "call adjust(expected,markers,size(markers))", "call checked_adjust(a,markers,result)",
        "if(result/=FORT_SCOPE_OK) error stop 'unexpected inspection rejection'",
        "if(any(a/=expected)) error stop 'correction differs or was repeated'",
        "if(a(-2,0)/=9.0_8.or.a(2,4)/=23.0_8) error stop 'correction missing'",
        "print *, 'FIELDS_OK'", "end program",
    ])
    source = tmp_path / "correction.f90"
    source.write_text(generated)
    binary = tmp_path / "correction"
    result = subprocess.run(
        [fortran, "-std=f2018", "-fcheck=all", str(interface), str(tmp_path / "renamed.f90"),
         str(source), "-o", str(binary)],
        cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(binary)], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "FIELDS_OK" in result.stdout
