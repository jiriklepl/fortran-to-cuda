"""Read-only counted nests retain exact incoming reads, not definition credit."""
from copy import copy

import pytest

from compiler.frontend.reached_read_requirements import native_read_sections, prove_native_read_requirements
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "incoming::inspect"
SOURCE = """module incoming
implicit none
contains
subroutine inspect(a,n,total)
real(8),intent(in)::a(-3:)
integer,intent(in)::n
real(8),intent(inout)::total
integer::i
do i=-3,n
 total=total+a(i)
enddo
end subroutine
end module
"""


def issue(tmp_path, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines[ENTRY]
    selection = tuple(routine.execution.content)
    proof = prove_native_read_requirements(analysis, ENTRY, selection)
    return analysis, selection, proof


def test_counted_read_geometry_keeps_empty_iteration_guard_and_exact_bounds(tmp_path):
    analysis, selection, proof = issue(tmp_path)
    assert proof is not None
    sections = native_read_sections(analysis, ENTRY, selection, proof)
    resource, = sections.resources
    assert resource.resource == "argument::a"
    assert not resource.writes
    assert not resource.overwrites
    read, = resource.reads
    axis, = read.axes
    assert axis.lower.value == -3
    assert axis.upper.resource == "argument::n"
    assert read.iterations
    assert proof._include_entry is False


def test_perfect_nested_read_only_scalar_reduction(tmp_path):
    source = (SOURCE.replace("a(-3:)", "a(-3:,2:)")
              .replace("integer::i", "integer::i,j")
              .replace(" total=total+a(i)", "do j=2,n\n total=total+a(i,j)\nenddo"))
    analysis, selection, proof = issue(tmp_path, source)
    assert proof is not None
    resource, = native_read_sections(analysis, ENTRY, selection, proof).resources
    read, = resource.reads
    assert len(read.axes) == len(read.iterations) == 2


@pytest.mark.parametrize("body", [
    "a(i)=a(i)+1", "a(i)=a(i-1)", "if(i>0)total=total+a(i)",
    "call observe(a(i),total)", "total=merge(a(i),0.0_8,i>0)",
    "total=total+a(i)\na(i)=0", "do while(total>0)\ntotal=total+a(i)\nenddo",
])
def test_conditional_calls_and_payload_writes_do_not_gain_entry_reads(tmp_path, body):
    source = SOURCE.replace(" total=total+a(i)", body).replace("intent(in)::a", "intent(inout)::a")
    _analysis, _selection, proof = issue(tmp_path, source)
    assert proof is None


def test_unknown_payload_selector_fails_complete_original_effect_validation(tmp_path):
    source = SOURCE.replace(" total=total+a(i)", "total=total+unknown(a(i))")
    with pytest.raises(CompilationError, match="complete original ordered effects"):
        issue(tmp_path, source)


def test_mutated_loop_bound_does_not_grant_incoming_rectangles(tmp_path):
    source = SOURCE.replace("integer,intent(in)::n", "integer,intent(inout)::n")
    source = source.replace("real(8),intent(in)::a", "integer,intent(in)::a")
    source = source.replace(" total=total+a(i)", "n=n+a(i)")
    with pytest.raises(CompilationError):
        issue(tmp_path, source)


@pytest.mark.parametrize("mutation", ["copy", "foreign_analysis", "foreign_selection"])
def test_counted_read_token_cannot_be_rebound(tmp_path, mutation):
    analysis, selection, proof = issue(tmp_path)
    if mutation == "copy":
        proof = copy(proof)
    elif mutation == "foreign_analysis":
        analysis = SourceEffects([tmp_path / "source.f90"])
    else:
        selection = (copy(selection[0]),)
    with pytest.raises(CompilationError):
        native_read_sections(analysis, ENTRY, selection, proof)
