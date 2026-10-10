"""One reached proof need not repeatedly authenticate the same source tree."""

from copy import copy
from types import SimpleNamespace

from fparser.two import Fortran2003 as F
from fparser.two.utils import walk
import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.segments import fragment


def fixture(tmp_path):
    path = tmp_path / "authority.f90"
    path.write_text("""module authority
contains
subroutine apply(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
a(1)=a(1)+1
! original first comment
! original second comment
!$omp parallel private(i)
!$omp do
do i=1,n
a(i)=a(i)+1
enddo
!$omp end do
!$omp do
do i=1,n
a(i)=a(i)+2
enddo
!$omp end do
!$omp end parallel
end subroutine
end module
""")
    return path, SourceEffects([path])


def test_original_comment_selection_uses_one_authentication_pass(tmp_path, monkeypatch):
    path, analysis = fixture(tmp_path)
    routine = analysis.routines["authority::apply"]
    comments = [node for node in walk(routine.execution) if type(node).__name__ == "Comment"
                and not str(node).lstrip().lower().startswith("!$omp")]
    assert len(comments) == 2
    assignment = walk(routine.execution, F.Assignment_Stmt)[0]
    calls = []
    original = analysis._require_original

    def counted(requested):
        calls.append(requested)
        return original(requested)

    monkeypatch.setattr(analysis, "_require_original", counted)
    graph, identities = analysis._selected_source(routine.qualified, (*comments, assignment))
    assert identities == (graph.node_id(assignment),)
    assert calls == [routine.qualified]
    independent = SourceEffects([path])
    foreign_comment = next(node for node in walk(independent.routines[routine.qualified].execution)
                           if type(node).__name__ == "Comment"
                           and str(node) == str(comments[0]))
    with pytest.raises(CompilationError, match="comment lacks original source authority"):
        analysis._selected_source(routine.qualified, (foreign_comment, assignment))


def test_segment_projection_authenticates_once_and_keeps_original_identity_checks(tmp_path, monkeypatch):
    _path, analysis = fixture(tmp_path)
    routine = analysis.routines["authority::apply"]
    assignment = walk(routine.execution, F.Assignment_Stmt)[0]
    graph, identities = analysis._selected_source(routine.qualified, (assignment,))
    calls = []
    original = analysis._require_original

    def counted(requested):
        calls.append(requested)
        return original(requested)

    monkeypatch.setattr(analysis, "_require_original", counted)
    projected, projection, _guards = analysis._segment_projection(routine.qualified, identities)
    assert calls == [routine.qualified]
    assert projection is projected.routines[routine.qualified]
    assert projection.execution.content == [assignment]
    assert graph.source_nodes(identities[0]) == (assignment,)
    assert analysis.routines[routine.qualified] is routine


@pytest.mark.parametrize("change", ["source", "execution", "span"])
def test_reached_proof_rechecks_authority_on_each_call(tmp_path, change):
    path, analysis = fixture(tmp_path)
    routine = analysis.routines["authority::apply"]
    assignment = walk(routine.execution, F.Assignment_Stmt)[0]
    _graph, identities = analysis._selected_source(routine.qualified, (assignment,))
    analysis._segment_projection(routine.qualified, identities)
    if change == "source":
        path.write_text(path.read_text().replace("a(1)=a(1)+1", "a(1)=a(1)+2"))
    elif change == "execution":
        routine.execution = copy(routine.execution)
    else:
        assignment.item.span = (1, 1)
    with pytest.raises(CompilationError, match="(changed|original.*authority)"):
        analysis._segment_projection(routine.qualified, identities)


def test_native_fragment_issues_copied_authority_once_after_capture_facts(tmp_path, monkeypatch):
    _path, analysis = fixture(tmp_path)
    routine = analysis.routines["authority::apply"]
    original_nodes = tuple(routine.execution.content)
    # Select the exact joined group, excluding the preceding scalar assignment.
    first = next(index for index, node in enumerate(original_nodes)
                 if str(node).lstrip().lower().startswith("!$omp parallel"))
    joined = original_nodes[first:]
    loops = tuple(walk(joined, F.Block_Nonlabel_Do_Construct))
    proof = analysis.worksharing_native_completion(routine.qualified, joined, (loops[0],))
    builder = SimpleNamespace(analysis=analysis, entry=routine,
                              inline=SimpleNamespace(original_selection=lambda nodes: tuple(nodes)),
                              config=SimpleNamespace(scope_execution="reached"))
    issuances = []
    original = SourceEffects.worksharing_native_completion

    def counted(self, requested, group, selected):
        issuances.append(self)
        return original(self, requested, group, selected)

    monkeypatch.setattr(SourceEffects, "worksharing_native_completion", counted)
    operation = fragment(builder, (loops[0],), kind="original native worksharing",
                         native_metadata=True, completion=proof)
    assert len(issuances) == 1
    assert issuances[0] is not analysis
    assert operation.summary["complete"]
    copied = operation.summary["native_completion"]
    assert copied["native_effects_authority"]
    assert copied["structured_identity"] != proof.structured_identity
    assert operation.sections.available, operation.sections.reason
