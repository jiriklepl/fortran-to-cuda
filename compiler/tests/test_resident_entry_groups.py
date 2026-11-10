"""Resident extraction shares interfaces without crossing source control/effects."""
from pathlib import Path

import pytest
from fparser.two.utils import walk

from compiler.frontend.lowering import lower_file
from compiler.frontend.source_effects import SourceEffects, _kind
from compiler.ir import CompilationError
from compiler.scopes.regions import extract_region
from compiler.scopes.resident_entries import export_resident_entries
from compiler.transforms.fusion import optimize_function


def export(tmp_path, body):
    source = tmp_path / "input.f90"
    source.write_text("""module grouped
contains
subroutine step(a,b,n,flag)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
""" + body + """
end subroutine
subroutine observe(n)
integer,intent(in)::n
if(n<0) print *,n
end subroutine
end module
""")
    result = export_resident_entries([source], ["grouped::step"], tmp_path / "entries")
    assert not result["rejected"]
    return result


FIRST = "do i=1,n\na(i)=a(i)+1\nenddo\n"
SECOND = "do i=2,n\nb(i)=a(i-1)\nenddo\n"


def test_adjacent_nests_share_entry_but_keep_cross_point_dependency(tmp_path):
    result = export(tmp_path, FIRST + "! ordinary source comment\n" + SECOND)
    entry, = result["entries"]
    assert len(entry["source_loop_spans"]) == 2
    function = lower_file(Path(entry["input_path"]), entry["entry"])
    _, plan = optimize_function(function)
    assert len(plan.regions) == 2


def test_grouping_exposes_legal_same_coordinate_fusion(tmp_path):
    result = export(tmp_path, FIRST + "do i=1,n\nb(i)=a(i)\nenddo\n")
    entry, = result["entries"]
    function = lower_file(Path(entry["input_path"]), entry["entry"])
    _, plan = optimize_function(function)
    assert len(plan.regions) == 1
    assert any("fusion" in text and "applied" in text for text in plan.reports)


def test_branch_alternatives_and_native_calls_separate_groups(tmp_path):
    result = export(tmp_path, "if(flag) then\n" + FIRST + SECOND + "else\n" + FIRST
                    + "endif\ncall observe(n)\n" + FIRST + SECOND)
    assert [len(entry["source_loop_spans"]) for entry in result["entries"]] == [2, 1, 2]
    for entry in result["entries"]:
        assert "call observe" not in Path(entry["input_path"]).read_text().lower()


def test_scalar_definition_ends_group(tmp_path):
    result = export(tmp_path, FIRST + "i=n\n" + SECOND)
    assert [len(entry["source_loop_spans"]) for entry in result["entries"]] == [1, 1]


def test_many_nests_have_no_work_unit_count_cutoff(tmp_path):
    result = export(tmp_path, FIRST * 40)
    entry, = result["entries"]
    assert len(entry["source_loop_spans"]) == 40


@pytest.mark.parametrize("body", [FIRST + "call observe(n)\n" + SECOND,
                                  FIRST + "i=n\n" + SECOND,
                                  "if(flag) then\n" + FIRST + "else\n" + SECOND + "endif\n"])
def test_direct_extraction_cannot_skip_effect_or_branch_boundaries(tmp_path, body):
    export(tmp_path, body)
    analysis = SourceEffects([tmp_path / "input.f90"])
    routine = analysis.routines["grouped::step"]
    loops = tuple(item for item in walk(routine.execution) if _kind(item) == "Block_Nonlabel_Do_Construct")
    with pytest.raises(CompilationError, match="consecutive original sibling loops"):
        extract_region(analysis, routine, loops)
