"""A complete original join owns nested directives, never their payload effects."""

from copy import copy

import pytest

from compiler.frontend import resource_reachability as rr
from compiler.frontend.native_completion import prove_graph_joined_completion
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


ENTRY = "nested_work::step"


def fixture(tmp_path, *, nowait=False, replacement=None, sealed=True):
    loop = "!$omp do\ndo i=1,n\nb(i)=a(i)+1.0_8\nenddo\n!$omp end do"
    if nowait:
        loop += " nowait"
    source = ("module nested_work\nimplicit none\ncontains\n"
              "subroutine step(a,b,n,flag)\nreal(8),intent(in)::a(:)\n"
              "real(8),intent(inout)::b(:)\ninteger,intent(in)::n\n"
              "logical,intent(in)::flag\ninteger::i\ncontinue\n"
              "!$omp parallel private(i)\nif(flag)then\n" + loop + "\nelse\n"
              + loop.replace("+1.0_8", "-1.0_8") + "\nendif\n!$omp end parallel\n"
              "end subroutine\nend module\n")
    if replacement:
        source = source.replace(*replacement)
    path = tmp_path / "nested.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines[ENTRY]
    groups = [g for g in grouped_original_sequence(routine.execution.content, {})
              if isinstance(g, NativeGroupCandidate)]
    selected = groups[0].original_nodes if groups else tuple(routine.execution.content[1:])
    return analysis, selected


@pytest.mark.parametrize("nowait", [False, True])
def test_checked_conditional_worksharing_keeps_both_payload_branches(tmp_path, nowait):
    analysis, selected = fixture(tmp_path, nowait=nowait)
    ordinary = analysis.joined_completion(ENTRY, selected)
    reached = rr.prove_resource_reachability(analysis, ENTRY, selected, ())
    assert not reached.barriers
    completion = prove_graph_joined_completion(analysis, ENTRY, selected, reached.native_graph)
    assert completion._ordinary_join is ordinary
    assert reached.native_graph.templates[0]._source_budget_receipts[0][2] is ordinary
    payload = [(a.storage.binding.name, a.kind) for a in reached.accesses]
    assert payload.count(("a", "read")) == 2
    assert payload.count(("b", "write")) == 2
    public = reached.native_graph.public()
    assert not public["execution_authority"]
    assert not public["motion_authority"]
    assert not public["internal_cut_authority"]
    # Original-host admission has its own clause/effect rules; the payload
    # completion proof must not erase those independently reported blockers.
    if nowait:
        assert reached.native_graph.host_blockers


@pytest.mark.parametrize("replacement", [
    ("!$omp do\n", "!$omp task\n!$omp do\n"),
    ("!$omp end parallel", "!$omp end parallel nowait"),
    ("!$omp end parallel", ""),
    ("if(flag)then", "if(i>0)then"),
    ("b(i)=a(i)+1.0_8", "call unknown(a,b)"),
])
def test_unproved_nested_source_keeps_completion_boundary(tmp_path, replacement):
    analysis, selected = fixture(tmp_path, replacement=replacement)
    graph = rr.prove_native_effect_graph(analysis, ENTRY, selected)
    with pytest.raises(CompilationError):
        prove_graph_joined_completion(analysis, ENTRY, selected, graph)


@pytest.mark.parametrize("mutation", ["registry", "source"])
def test_nested_ownership_revalidates_exact_enclosing_join(tmp_path, mutation):
    # Standalone analyses recheck live files; sealed sessions retain immutable
    # captured source and check live files at the publication boundary.
    analysis, selected = fixture(tmp_path, sealed=mutation != "source")
    graph = rr.prove_native_effect_graph(analysis, ENTRY, selected)
    assert graph.completion_eligible
    ordinary = graph.templates[0]._source_budget_receipts[0][2]
    if mutation == "registry":
        analysis._joined_completions[ordinary.identity] = copy(ordinary)
    else:
        path = tmp_path / "nested.f90"
        path.write_text(path.read_text().replace("+1.0_8", "+2.0_8"))
    with pytest.raises(CompilationError):
        graph.validate(analysis, ENTRY, selected)


def test_copied_ordinary_token_cannot_own_nested_directives(tmp_path, monkeypatch):
    analysis, selected = fixture(tmp_path)
    ordinary = analysis.joined_completion(ENTRY, selected)
    monkeypatch.setattr(analysis, "joined_completion", lambda *_: copy(ordinary))
    graph = rr.prove_native_effect_graph(analysis, ENTRY, selected)
    assert not graph.completion_eligible
    with pytest.raises(CompilationError):
        prove_graph_joined_completion(analysis, ENTRY, selected, graph)
