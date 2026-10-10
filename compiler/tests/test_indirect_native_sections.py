"""Indirect native hooks inspect original metadata without inventing volumes."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.frontend.indirect_sections import analyze_indirect_sections
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.indirect_access import build_indirect_accesses


def fixture(tmp_path, body=None, *, metadata="type(marker_t),intent(in)::markers(:)", extra="", fields=None):
    body = body or "do q=1,size(markers)\na(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+markers(q)%value\nenddo"
    fields = fields or "integer(8)::row,column\nreal(8)::value"
    source = tmp_path / "renamed.f90"
    source.write_text("module geometry\nimplicit none\ntype marker_t\n" + fields + "\nend type\ncontains\n"
                      "subroutine adjust(a,markers,n)\nreal(8),intent(inout)::a(-2:,0:)\n" + metadata
                      + "\ninteger,intent(in)::n\ninteger::q\n" + extra + "\n" + body
                      + "\nend subroutine\nend module\n")
    analysis = SourceEffects([source])
    routine = analysis.routines["geometry::adjust"]
    sections = analyze_indirect_sections(analysis, routine.qualified, routine.execution.content)
    return analysis, sections


def test_original_metadata_accesses_have_separate_typed_native_effects(tmp_path):
    analysis, sections = fixture(tmp_path)
    assert sections.available, sections.reason
    resource, = sections.resources
    assert resource.lower_bounds == (-2, 0)
    assert resource.reads == resource.writes == resource.overwrites == ()
    assert [item.action for item in sections.references] == ["read", "write"]
    assert {item.member for reference in sections.references for item in reference.indices} == {"row", "column"}
    record = sections.public()
    assert record["metadata_reservations"] == ["argument::markers"]
    assert record["rectangle_limit"] == 32
    assert record["inspection_step_limit_per_resource"] == 1_048_576
    assert record["proves_scatter_independence"] is False
    assert "original reached native boundary" in record["preparation"]
    assert json.loads(json.dumps(record)) == record
    assert not analysis.native_sections("geometry::adjust").available


@pytest.mark.parametrize(("body", "extra", "metadata", "reason"), [
    ("do q=1,n\na(markers(q)%row,markers(q)%column)=1\nenddo\nn=2", "", "type(marker_t),intent(in)::markers(:)",
     "scalar changes"),
    ("do q=1,size(markers)\nmarkers(q)%row=2\na(markers(q)%row,markers(q)%column)=1\nenddo", "",
     "type(marker_t),intent(inout)::markers(:)", "metadata changes"),
    ("do q=1,size(markers)\na(markers(q)%row,markers(q)%column)=helper()\nenddo", "",
     "type(marker_t),intent(in)::markers(:)", "function effects are unsupported"),
    ("do q=1,size(markers)\na(int(markers(q)%value),markers(q)%column)=1\nenddo", "",
     "type(marker_t),intent(in)::markers(:)", "whole-array descriptor"),
    ("do q=1,size(markers)\na(markers(q)%row,:)=1\nenddo", "", "type(marker_t),intent(in)::markers(:)", "scalar point"),
    ("do q=1,size(markers)\nif(n>0)a(markers(q)%row,markers(q)%column)=1\nenddo", "",
     "type(marker_t),intent(in)::markers(:)", "assignment-only"),
    ("do q=1,size(markers)\na(markers(q)%row,markers(q)%column)=1\nenddo", "",
     "type(marker_t),pointer::markers(:)", "lifetime is uncertain"),
    ("do q=1,size(markers)\na(markers(q)%row,markers(q)%column)=1\nenddo", "",
     "type(marker_t),optional,intent(in)::markers(:)", "lifetime is uncertain"),
])
def test_unproved_index_sources_retain_conservative_native_effects(tmp_path, body, extra, metadata, reason):
    _, sections = fixture(tmp_path, body, extra=extra, metadata=metadata)
    assert not sections.available
    assert reason in sections.reason
    assert not sections.resources


def test_inspector_cannot_authorize_foreign_or_stale_source(tmp_path):
    analysis, sections = fixture(tmp_path)
    assert sections.available
    from fparser.two.Fortran2003 import Assignment_Stmt
    forged = analyze_indirect_sections(analysis, "geometry::adjust", Assignment_Stmt("a(1,1)=2"))
    assert not forged.available
    assert "original source authority" in forged.reason
    analysis.inputs.verify()
    path = tmp_path / "renamed.f90"
    path.write_text(path.read_text().replace("+markers(q)%value", "-markers(q)%value"))
    stale = analyze_indirect_sections(analysis, "geometry::adjust", analysis.routines["geometry::adjust"].execution.content)
    assert not stale.available
    assert "changed" in stale.reason


def test_native_worksharing_inspection_requires_its_exact_original_team_token(tmp_path):
    from dataclasses import replace
    from fparser.two import Fortran2003 as F
    from fparser.two.utils import walk

    body = ("!$omp parallel private(q)\n!$omp do\ndo q=1,size(markers)\n"
            "a(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+markers(q)%value\n"
            "enddo\n!$omp end do\n!$omp end parallel")
    analysis, unavailable = fixture(tmp_path, body)
    assert not unavailable.available
    assert "completion proof" in unavailable.reason
    routine = analysis.routines["geometry::adjust"]
    loops = tuple(walk(routine.execution, F.Block_Nonlabel_Do_Construct))
    native = analysis.worksharing_native_completion(routine.qualified, routine.execution.content, loops)
    sections = analyze_indirect_sections(analysis, routine.qualified, loops, completion=native)
    assert sections.available, sections.reason
    numerical = analysis.worksharing_completion(routine.qualified, routine.execution.content, loops)
    assert not analyze_indirect_sections(analysis, routine.qualified, loops, completion=numerical).available
    assert not analyze_indirect_sections(analysis, routine.qualified, loops, completion=replace(native)).available
    assert not analyze_indirect_sections(analysis, routine.qualified, loops).available


def test_private_state_from_other_worksharing_units_cannot_be_inspected_by_master(tmp_path):
    from fparser.two import Fortran2003 as F
    from fparser.two.utils import walk
    body = ("!$omp parallel private(q,j)\n!$omp do\ndo j=1,n\ncontinue\nenddo\n!$omp end do\n"
            "!$omp do\ndo q=1,size(markers)\na(markers(q)%row,j)=1\nenddo\n!$omp end do\n!$omp end parallel")
    analysis, _ = fixture(tmp_path, body, extra="integer::j")
    routine = analysis.routines["geometry::adjust"]
    loops = tuple(walk(routine.execution, F.Block_Nonlabel_Do_Construct))
    native = analysis.worksharing_native_completion(routine.qualified, routine.execution.content, (loops[1],))
    sections = analyze_indirect_sections(analysis, routine.qualified, (loops[1],), completion=native)
    assert not sections.available
    assert "uniform original-team scalar" in sections.reason


def test_multiple_native_operands_keep_metadata_offsets_and_original_resource_identity(tmp_path):
    source = tmp_path / "renamed-chain.f90"
    source.write_text("""module coordinates
implicit none
type cell_t
 integer::first,second,third
 real(8)::value
end type
contains
subroutine correct(out,input,coeff,sites)
real(8),intent(inout)::out(-2:,0:,-1:)
real(8),intent(in)::input(-2:,0:,-1:),coeff(-2:,0:,-1:)
type(cell_t),intent(in)::sites(:)
integer::index
do index=1,size(sites)
out(sites(index)%first,sites(index)%second,sites(index)%third)= &
out(sites(index)%first,sites(index)%second,sites(index)%third)+ &
coeff((sites(index)%first-1)+1,sites(index)%second,sites(index)%third)* &
(input((sites(index)%first-1)+1,sites(index)%second,sites(index)%third)- &
input(sites(index)%first-1,sites(index)%second,sites(index)%third))+sites(index)%value
enddo
end subroutine
end module
""")
    analysis = SourceEffects([source])
    routine = analysis.routines["coordinates::correct"]
    sections = analyze_indirect_sections(analysis, routine.qualified, routine.execution.content)
    assert sections.available, sections.reason
    assert {item.resource for item in sections.resources} == {"argument::out", "argument::input", "argument::coeff"}
    assert all(item.lower_bounds == (-2, 0, -1) for item in sections.resources)
    assert len([item for item in sections.references if item.resource == "argument::input"]) == 2
    assert {item.resource for item in sections.references if item.action == "write"} == {"argument::out"}
    codes = build_indirect_accesses(sections, {item.resource: str(index + 1) + "_c_int64_t"
                                              for index, item in enumerate(sections.resources)},
                                    "fort_native", resource_names={"argument::sites": "sites"})
    assert len(codes) == 3
    with pytest.raises(CompilationError, match="aliases require"):
        build_indirect_accesses(sections, {item.resource: "same_handle" for item in sections.resources},
                                "fort_native", resource_names={"argument::sites": "sites"})


@pytest.mark.parametrize("allocation_change", [False, True])
def test_original_module_metadata_allocation_uses_segment_local_lifetime_proof(tmp_path, allocation_change):
    source = tmp_path / "allocated-metadata.f90"
    source.write_text("""module storage
type marker_t
 integer::row,column
 real(8)::value
end type
type(marker_t),allocatable::markers(:)
contains
subroutine adjust(a)
real(8),intent(inout)::a(-2:,0:)
integer::q
do q=1,size(markers)
a(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+markers(q)%value
enddo
""" + ("deallocate(markers)\n" if allocation_change else "") + "end subroutine\nend module\n")
    analysis = SourceEffects([source])
    routine = analysis.routines["storage::adjust"]
    sections = analyze_indirect_sections(analysis, routine.qualified, routine.execution.content)
    if allocation_change:
        assert not sections.available
        assert "assignment-only" in sections.reason
    else:
        assert sections.available, sections.reason
        assert [item.root for item in sections.metadata] == ["storage::markers"]
        code, = build_indirect_accesses(sections, {"argument::a": "1_c_int64_t"}, "fort_inspect",
                                        resource_names={"storage::markers": "markers"})
        prepare = "\n".join(code.prepare)
        assert prepare.index(".not. allocated(markers)") < prepare.index("size(markers")
        assert "descriptor stable throughout the selected" in sections.public()["metadata_lifetime"]


def test_inspector_rejects_missing_original_mapping_and_canonical_aliases(tmp_path):
    _, sections = fixture(tmp_path)
    with pytest.raises(CompilationError, match="metadata/descriptor mapping"):
        build_indirect_accesses(sections, {"argument::a": "handle"}, "fort_inspect", resource_names={})
    with pytest.raises(CompilationError, match="buffer mapping"):
        build_indirect_accesses(sections, {}, "fort_inspect", resource_names={"argument::markers": "markers"})


def test_generated_inspector_has_no_per_point_runtime_calls_or_derived_type_layout(tmp_path):
    _, sections = fixture(tmp_path)
    code, = build_indirect_accesses(sections, {"argument::a": "1_c_int64_t"}, "fort_inspect",
                                    resource_names={"argument::markers": "markers"})
    source = "\n".join(code.prepare)
    assert source.count("fort_scope_layout_get(") == 1
    assert "markers(" in source and ")%row" in source
    assert "c_sizeof" not in source and "c_loc(markers" not in source
    assert "%overwrite_count" not in source
    assert "%read_count" in source and "%write_count" in source
    assert all(len(line) <= 132 for line in (*code.specification, *code.prepare))


def compile_inspector(tmp_path, points, *, shape=(5, 5), body=None, metadata_lower=1,
                      source_metadata_lower=None, extra=""):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("Fortran compiler unavailable")
    if source_metadata_lower is None:
        source_metadata_lower = metadata_lower
    metadata = f"type(marker_t),intent(in)::markers({source_metadata_lower}:)"
    _, sections = fixture(tmp_path, body, metadata=metadata, extra=extra)
    assert sections.available, sections.reason
    code, = build_indirect_accesses(sections, {"argument::a": "1_c_int64_t"}, "fort_inspect",
                                    resource_names={"argument::markers": "markers"},
                                    context="1_c_int64_t", status="result")
    interface = Path(__file__).resolve().parents[1] / "runtime" / "scoped_memory.f90"
    stub = f"""function layout_get(context,buffer,layout) bind(C,name='fort_scope_layout_get') result(status)
use iso_c_binding
use fort_scoped_memory,only:fort_scope_layout
integer(c_int64_t),value::context,buffer
type(fort_scope_layout),intent(out)::layout
integer(c_int)::status
integer(c_size_t),target,save::extents(2)=[{shape[0]}_c_size_t,{shape[1]}_c_size_t]
layout=fort_scope_layout()
layout%rank=2
layout%extents=c_loc(extents)
status=0
end function
"""
    generated = "\n".join([
        "module checked", "use iso_c_binding", "use fort_scoped_memory", "implicit none",
        "type marker_t", "integer(8)::row,column", "real(8)::value", "end type", "contains",
        "subroutine inspect(markers,result)", metadata, "integer(c_int),intent(out)::result",
        *code.specification, "result=0", *code.prepare,
        f"print *, 'READS', {code.access_name}%read_count", f"print *, 'WRITES', {code.access_name}%write_count",
        "do fort_inspect_0_item=1,fort_inspect_0_w_total",
        "print *, 'BOX', fort_inspect_0_w_lows(:,fort_inspect_0_item), fort_inspect_0_w_highs(:,fort_inspect_0_item)",
        "enddo", "end subroutine", "end module", "program caller", "use checked", "implicit none",
        f"type(marker_t)::markers({metadata_lower}:{metadata_lower + len(points) - 1})", "integer(c_int)::result",
        *[f"markers({metadata_lower + index})%row={point[0]}_8\nmarkers({metadata_lower + index})%column={point[1]}_8"
          for index, point in enumerate(points)], "call inspect(markers,result)", "print *,'STATUS',result", "end program"])
    source = tmp_path / "check.f90"
    source.write_text(stub + generated)
    binary = tmp_path / "check"
    compiled = subprocess.run([fortran, "-std=f2018", "-fcheck=all", str(interface), str(source), "-o", str(binary)],
                              cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    executed = subprocess.run([str(binary)], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    rows = [line.split() for line in executed.stdout.splitlines()]
    status = next(int(row[1]) for row in rows if row[0] == "STATUS")
    boxes = [tuple(map(int, row[1:])) for row in rows if row[0] == "BOX"]
    counts = {row[0]: int(row[1]) for row in rows if row[0] in {"READS", "WRITES"}}
    return status, boxes, counts


@pytest.mark.parametrize("duplicates", [False, True])
def test_runtime_inspection_preserves_opposite_faces_and_complete_exact_union(tmp_path, duplicates):
    points = [(row, col) for row in (-2, 2) for col in range(5)]
    if duplicates:
        points *= 3
    status, boxes, counts = compile_inspector(tmp_path, points)
    assert status == 0
    assert set(boxes) == {(0, 0, 1, 5), (4, 0, 5, 5)}
    assert counts == {"READS": 2, "WRITES": 2}


def test_runtime_inspection_retains_holes_in_a_nonrectangular_union(tmp_path):
    points = [(-2, 0), (-2, 1), (-1, 0)]
    status, boxes, counts = compile_inspector(tmp_path, points)
    assert status == 0
    covered = {(row - 2, col) for xlo, ylo, xhi, yhi in boxes
               for row in range(xlo, xhi) for col in range(ylo, yhi)}
    assert covered == set(points)
    assert counts == {"READS": 2, "WRITES": 2}


@pytest.mark.parametrize(("points", "shape"), [
    ([(3, 0)], (5, 5)),
    ([(-3, 0)], (5, 5)),
    ([(2**63 - 1, 0)], (5, 5)),
    ([(-2 + 2 * index, 2 * index) for index in range(33)], (70, 70)),
    ([(-2, 0)], (2**31, 5)),
])
def test_runtime_invalid_or_overbudget_inspection_closes_before_native_work(tmp_path, points, shape):
    status, boxes, counts = compile_inspector(tmp_path, points, shape=shape)
    assert status == 5
    assert not boxes and not counts


def test_empty_metadata_preserves_no_payload_reads_and_zero_sections(tmp_path):
    status, boxes, counts = compile_inspector(tmp_path, [])
    assert status == 0
    assert not boxes
    assert counts == {"READS": 0, "WRITES": 0}


def test_scan_budget_closes_repeated_points_before_unbounded_metadata_work(tmp_path, monkeypatch):
    import compiler.scopes.indirect_access as lowering
    monkeypatch.setattr(lowering, "INSPECTION_STEP_LIMIT", 3)
    # Repetition fits one rectangle, so the work cap is independent of the
    # rectangle cap. The fourth metadata iteration closes before its fields.
    status, boxes, counts = compile_inspector(tmp_path, [(-2, 0)] * 4)
    assert status == 5
    assert not boxes and not counts


def test_scan_budget_also_counts_outer_iterations_with_an_empty_inner_loop(tmp_path, monkeypatch):
    import compiler.scopes.indirect_access as lowering
    monkeypatch.setattr(lowering, "INSPECTION_STEP_LIMIT", 3)
    body = ("do q=1,5\ndo r=1,0\n"
            "a(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+markers(q)%value\n"
            "enddo\nenddo")
    status, boxes, counts = compile_inspector(tmp_path, [], body=body, extra="integer::r")
    assert status == 5
    assert not boxes and not counts


def test_original_negative_metadata_bounds_are_checked_and_preserved(tmp_path):
    body = ("do q=lbound(markers,1),ubound(markers,1)\n"
            "a(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+markers(q)%value\nenddo")
    status, boxes, _ = compile_inspector(tmp_path, [(-2, 0), (-2, 1)], body=body, metadata_lower=-3)
    assert status == 0
    assert boxes == [(0, 0, 1, 2)]


def test_negative_stride_inspection_uses_original_coordinates_without_modifying_iterator(tmp_path):
    body = ("do q=size(markers),1,-1\n"
            "a(markers(q)%row,markers(q)%column)=a(markers(q)%row,markers(q)%column)+markers(q)%value\nenddo")
    status, boxes, _ = compile_inspector(tmp_path, [(-2, 0), (-2, 1), (-2, 2)], body=body)
    assert status == 0
    assert boxes == [(0, 0, 1, 3)]
