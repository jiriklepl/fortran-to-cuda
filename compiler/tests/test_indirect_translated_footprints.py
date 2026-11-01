"""Invocation-local exact translated footprints preserve reached source checks."""

from pathlib import Path
import shutil
import subprocess

import pytest

from compiler.frontend.indirect_sections import analyze_indirect_sections
from compiler.frontend.source_effects import SourceEffects
from compiler.scopes.indirect_access import build_indirect_accesses
from compiler.tests.test_indirect_native_sections import compile_inspector, fixture


def body(second="markers(q)%row+1", *, column="markers(q)%column", loop="do q=1,size(markers)"):
    return (loop + "\n"
            "a(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+1\n"
            f"a({second},{column})=a({second},{column})+1\nenddo")


def covered(boxes):
    return {(row - 2, col) for xlo, ylo, xhi, yhi in boxes
            for row in range(xlo, xhi) for col in range(ylo, yhi)}


@pytest.mark.parametrize("points", [[], [(-2, 0)], [(-2, 0)] * 7,
                                   [(-2, 0), (-2, 2), (0, 1), (-1, 3)],
                                   [(-2, col) for col in range(5)]])
def test_adjacent_translations_reuse_one_scan_and_preserve_exact_holes(tmp_path, points):
    _, sections = fixture(tmp_path, body())
    code, = build_indirect_accesses(sections, {"argument::a": "1_c_int64_t"}, "fort_inspect",
                                   resource_names={"argument::markers": "markers"})
    emitted = "\n".join(code.prepare)
    assert "_cache_lows" in emitted
    assert "do fort_inspect_0_r_i0_0" in emitted
    assert "do fort_inspect_0_r_i1_0" not in emitted
    assert "%writes = fort_inspect_0_access%reads" in emitted
    status, boxes, _ = compile_inspector(tmp_path, points, body=body())
    assert status == 0
    assert covered(boxes) == set(points) | {(x + 1, y) for x, y in points}


@pytest.mark.parametrize("second", [
    "(markers(q)%row-2147483647)+2147483647",
    "(markers(q)%row+2147483647)-2147483647",
])
def test_reused_final_coordinate_keeps_every_original_intermediate(tmp_path, second):
    point = (-2, 0) if "row-" in second else (1, 0)
    status, boxes, counts = compile_inspector(tmp_path, [point], body=body(second))
    assert status == 5 and not boxes and not counts


@pytest.mark.parametrize("points,shape", [([(2, 0)], (5, 5)),
                                          ([(2**63 - 1, 0)], (5, 5)),
                                          ([(-2 + 2 * i, 2 * i) for i in range(33)], (70, 70))])
def test_translated_bounds_source_integer_and_capacity_fail_before_execution(tmp_path, points, shape):
    status, boxes, counts = compile_inspector(tmp_path, points, shape=shape, body=body())
    assert status == 5 and not boxes and not counts


@pytest.mark.parametrize("limit,expected", [(7, 5), (8, 0)])
def test_reused_nested_empty_traversals_charge_all_directions(tmp_path, monkeypatch, limit, expected):
    import compiler.scopes.indirect_access as lowering
    monkeypatch.setattr(lowering, "INSPECTION_STEP_LIMIT", limit)
    nested = body(loop="do q=1,2\ndo r=1,0") + "\nenddo"
    status, boxes, counts = compile_inspector(tmp_path, [], body=nested, extra="integer::r")
    assert status == expected and not boxes
    assert counts == ({} if expected else {"READS": 0, "WRITES": 0})


@pytest.mark.parametrize("limit,expected", [(3, 5), (4, 0)])
def test_reused_unlooped_references_keep_full_step_charge(tmp_path, monkeypatch, limit, expected):
    import compiler.scopes.indirect_access as lowering
    monkeypatch.setattr(lowering, "INSPECTION_STEP_LIMIT", limit)
    unlooped = body().split("\n", 1)[1].rsplit("\n", 1)[0].replace("markers(q)", "markers(1)")
    status, boxes, counts = compile_inspector(tmp_path, [(-2, 0)], body=unlooped)
    assert status == expected
    assert covered(boxes) == (set() if expected else {(-2, 0), (-1, 0)})
    assert counts == ({} if expected else {"READS": 1, "WRITES": 1})


def test_negative_metadata_bounds_and_reverse_traversal_are_retained(tmp_path):
    points = [(-2, 0), (-2, 2), (0, 1)]
    reverse = body(loop="do q=ubound(markers,1),lbound(markers,1),-1")
    status, boxes, _ = compile_inspector(tmp_path, points, body=reverse, metadata_lower=-3)
    assert status == 0
    assert covered(boxes) == set(points) | {(x + 1, y) for x, y in points}


def test_unsupported_coordinate_recipe_keeps_original_scans(tmp_path):
    _, sections = fixture(tmp_path, body("markers(q)%row*2"))
    code, = build_indirect_accesses(sections, {"argument::a": "1_c_int64_t"}, "fort_inspect",
                                   resource_names={"argument::markers": "markers"})
    source = "\n".join(code.prepare)
    assert "_cache_lows" not in source
    assert "do fort_inspect_0_r_i0_0" in source and "do fort_inspect_0_r_i1_0" in source
    status, boxes, _ = compile_inspector(tmp_path, [(1, 0)], body=body("markers(q)%row*2"))
    assert status == 0 and covered(boxes) == {(1, 0), (2, 0)}


@pytest.mark.parametrize("widened", [True, False])
def test_changed_conservative_admission_never_changes_exact_coverage(tmp_path, monkeypatch, widened):
    import compiler.scopes.indirect_access as lowering
    monkeypatch.setattr(lowering, "RECTANGLE_LIMIT", 4)
    if widened:
        points = [(1, 1), (-1, 3), (0, 2), (-2, 3), (-1, 0)]
        selected = body()
    else:
        points = [(-1, 3), (-2, 2), (1, 1), (-1, 1), (-1, 0), (-2, 1)]
        selected = body("markers(q)%row", column="markers(q)%column+1")
    # Exact old reference-major insertion is retained as a test control. It is
    # no longer a required policy for the bounded representation's admission.
    groups = lowering._translated_groups
    monkeypatch.setattr(lowering, "_translated_groups", lambda refs: [[x] for g in groups(refs) for x in g])
    old = tmp_path / "original"; old.mkdir()
    old_status, old_boxes, _ = compile_inspector(old, points, body=selected)
    monkeypatch.setattr(lowering, "_translated_groups", groups)
    new = tmp_path / "translated"; new.mkdir()
    status, boxes, _ = compile_inspector(new, points, body=selected)
    assert (old_status, status) == ((5, 0) if widened else (0, 5))
    expected = set(points) | ({(x + 1, y) for x, y in points} if widened
                              else {(x, y + 1) for x, y in points})
    assert covered(boxes if widened else old_boxes) == expected


def test_metadata_changes_and_distinct_origins_recompute_before_native_work(tmp_path):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    original = tmp_path / "original.f90"
    original.write_text("""module original
implicit none
type marker_t
 integer(8)::row,column
end type
contains
subroutine adjust(a,b,markers)
real(8),intent(inout)::a(-2:,0:),b(1:,-3:)
type(marker_t),intent(in)::markers(:)
integer::q
do q=1,size(markers)
a(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+1
a(markers(q)%row+1,markers(q)%column)=a(markers(q)%row+1,markers(q)%column)+1
b(markers(q)%row+3,markers(q)%column-3)=b(markers(q)%row+3,markers(q)%column-3)+2
b(markers(q)%row+4,markers(q)%column-3)=b(markers(q)%row+4,markers(q)%column-3)+2
enddo
end subroutine
end module
""")
    analysis = SourceEffects([original]); routine = analysis.routines["original::adjust"]
    sections = analyze_indirect_sections(analysis, routine.qualified, routine.execution.content)
    assert sections.available, sections.reason
    codes = build_indirect_accesses(sections, {"argument::a": "1_c_int64_t", "argument::b": "2_c_int64_t"},
                                    "fort_inspect", resource_names={"argument::markers": "markers"},
                                    context="1_c_int64_t", status="result")
    interface = Path(__file__).resolve().parents[1] / "runtime" / "scoped_memory.f90"
    lines = ["function layout_get(context,buffer,layout) bind(C,name='fort_scope_layout_get') result(status)",
             "use iso_c_binding", "use fort_scoped_memory,only:fort_scope_layout",
             "integer(c_int64_t),value::context,buffer", "type(fort_scope_layout),intent(out)::layout",
             "integer(c_int)::status", "integer(c_size_t),target,save::extents(2)=[5_c_size_t,5_c_size_t]",
             "layout=fort_scope_layout()", "layout%rank=2", "layout%extents=c_loc(extents)",
             "status=0", "end function", "module inspected", "use iso_c_binding",
             "use fort_scoped_memory", "use original", "implicit none", "contains",
             "subroutine checked(a,b,markers,result)", "real(8),intent(inout)::a(-2:,0:),b(1:,-3:)",
             "type(marker_t),intent(in)::markers(:)", "integer(c_int),intent(out)::result",
             "type(fort_scope_section),pointer::boxes(:)", "integer(c_size_t),pointer::lo(:),hi(:)",
             *[s for code in codes for s in code.specification], "result=0",
             *[s for code in codes for s in code.prepare]]
    for code in codes:
        access = code.access_name
        lines += [f"if ({access}%read_count/=1.or.{access}%write_count/=1)error stop 'wrong count'",
                  f"call c_f_pointer({access}%writes,boxes,[1])",
                  "call c_f_pointer(boxes(1)%lower,lo,[2])", "call c_f_pointer(boxes(1)%upper,hi,[2])",
                  "if(any(lo/=[markers(1)%row+2,markers(1)%column]))error stop 'stale or wrong origin'",
                  "if(any(hi/=[markers(1)%row+4,markers(1)%column+1]))error stop 'overcoverage'"]
    lines += ["call adjust(a,b,markers)", "end subroutine", "end module", "program caller", "use inspected",
              "implicit none", "real(8)::a(-2:2,0:4),b(1:5,-3:1),ea(-2:2,0:4),eb(1:5,-3:1)",
              "type(marker_t)::markers(1)", "integer(c_int)::result", "a=0;b=0;ea=0;eb=0",
              "markers(1)=marker_t(-2_8,0_8)", "call adjust(ea,eb,markers)", "call checked(a,b,markers,result)",
              "if(result/=0)error stop 'first inspection rejected'", "markers(1)=marker_t(0_8,3_8)",
              "call adjust(ea,eb,markers)", "call checked(a,b,markers,result)",
              "if(result/=0)error stop 'changed metadata rejected'",
              "if(any(a/=ea).or.any(b/=eb))error stop 'native work missing or replayed'",
              "print *,'FIELDS_AND_ORIGINS_OK'", "end program"]
    source = tmp_path / "checked.f90"; source.write_text("\n".join(lines))
    binary = tmp_path / "checked"
    compiled = subprocess.run([fortran, "-std=f2018", "-fcheck=all", "-ffunction-sections", "-Wl,--gc-sections",
                               str(interface), str(original), str(source), "-o", str(binary)],
                              cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run([str(binary)], cwd=tmp_path, text=True, capture_output=True, timeout=60)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "FIELDS_AND_ORIGINS_OK" in executed.stdout


def test_different_loop_domains_do_not_share_a_reached_footprint(tmp_path):
    selected = ("do q=1,1\na(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+1\nenddo\n"
                "do q=2,2\na(markers(q)%row+1,markers(q)%column)=a(markers(q)%row+1,markers(q)%column)+1\nenddo")
    _, sections = fixture(tmp_path, selected)
    code, = build_indirect_accesses(sections, {"argument::a": "1_c_int64_t"}, "fort_inspect",
                                   resource_names={"argument::markers": "markers"})
    assert "_cache_lows" not in "\n".join(code.prepare)
    status, boxes, _ = compile_inspector(tmp_path, [(-2, 0), (0, 3)], body=selected)
    assert status == 0 and covered(boxes) == {(-2, 0), (1, 3)}


def test_original_minval_maxval_names_do_not_capture_generated_extrema(tmp_path, monkeypatch):
    from dataclasses import replace
    from compiler.tests import test_indirect_native_sections as helper
    original = helper.build_indirect_accesses

    def emit(*args, **kwargs):
        codes = original(*args, **kwargs)
        return tuple(replace(code, specification=(*code.specification,
                     "integer(c_int64_t), external :: minval, maxval")) for code in codes)

    # These original procedure names have no implementation and may never be
    # called by inspection. An accidental generated MINVAL/MAXVAL call would
    # resolve as external and fail this actual Fortran compile/link.
    monkeypatch.setattr(helper, "build_indirect_accesses", emit)
    status, boxes, _ = helper.compile_inspector(tmp_path, [(-2, 0), (0, 3)], body=body())
    assert status == 0 and covered(boxes) == {(-2, 0), (-1, 0), (0, 3), (1, 3)}
