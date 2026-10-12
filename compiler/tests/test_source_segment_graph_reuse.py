"""Local graph reuse retains fresh public and token-callback authority."""

from copy import copy, deepcopy

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError

ENTRY = "segment_graph::advance"
DEMANDS = ("segment_summary", "native_sections_for_nodes")


def fixture(tmp_path, *, descriptor=False, team=False):
    path = tmp_path / "segments.f90"
    declaration = "real(8)," + ("allocatable," if descriptor else "")
    shape = "(:)" if descriptor else "(-2:)"
    first = "1" if descriptor else "-2"
    body = f"do i={first},n\nb(i)=a(i)+1.0_8\nend do\n"
    if team:
        body = "!$omp parallel private(i)\n!$omp do\n" + body + "!$omp end do\n!$omp end parallel\n"
    tail = "call move_alloc(b,temporary)\n" if descriptor else "b(-2)=b(-2)+2\n"
    path.write_text("module segment_graph\ninteger,parameter::marker=3\ncontains\n"
        "subroutine advance(a,b,n)\n" + declaration + "intent(in)::a" + shape + "\n"
        + declaration + "intent(inout)::b" + shape + "\ninteger,intent(in)::n\ninteger::i\n"
        "real(8),allocatable::temporary(:)\ncontinue\n" + body + tail
        + "end subroutine\nsubroutine unrelated()\ninteger::scratch\nscratch=marker\n"
        "end subroutine\nend module\n")
    analysis = SourceEffects([path], summary_cache=SummaryCache())
    loop, = walk(analysis.routines[ENTRY].execution, F.Block_Nonlabel_Do_Construct)
    return path, analysis, (loop,)


def record(analysis, demand, selected, **kwargs):
    result = getattr(analysis, demand)(ENTRY, selected, **kwargs)
    if demand == "segment_summary":
        assert result["complete"], result["reasons"]
        assert result["summary_role"] == "reached_source_segment"
        return result
    assert result.available, result.reason
    assert {resource.resource for resource in result.resources} == {"argument::a", "argument::b"}
    return result.public()


@pytest.mark.parametrize("demand", DEMANDS)
def test_fresh_segment_records_match_independent_projection_authentication(tmp_path, monkeypatch, request, demand):
    _path, analysis, selected = fixture(tmp_path)
    # Prime independent child/descriptor proof registries equally before
    # measuring the two segment paths on the same original analysis.
    record(analysis, demand, selected)
    analysis._segments.clear()
    original_structure = SourceEffects.structure
    original_projection = SourceEffects._segment_projection
    calls = []

    def structure(current, requested):
        if current is analysis:
            calls.append(requested)
        return original_structure(current, requested)

    def independent_projection(current, *args, **kwargs):
        kwargs["_graph"] = None
        return original_projection(current, *args, **kwargs)

    monkeypatch.setattr(SourceEffects, "structure", structure)
    with monkeypatch.context() as previous:
        previous.setattr(SourceEffects, "_segment_projection", independent_projection)
        expected = record(analysis, demand, selected)
    previous_calls = tuple(calls)
    assert len(previous_calls) >= 2
    assert set(previous_calls) == {ENTRY}
    analysis._segments.clear()
    calls.clear()
    actual = record(analysis, demand, selected)
    assert calls == [ENTRY] * (len(previous_calls) - 1)
    request.node.user_properties.extend((
        ("previous_root_structure_demands", len(previous_calls)),
        ("forwarded_root_structure_demands", len(calls))))
    assert actual == expected
    # The next public demand still checks original authority, including a
    # cached segment record, and returns an independent public document.
    saved = deepcopy(actual)
    actual.clear()
    calls.clear()
    assert record(analysis, demand, selected) == saved
    assert calls == [ENTRY]


@pytest.mark.parametrize("demand", DEMANDS)
@pytest.mark.parametrize("change", ["file", "unselected_ast", "binding", "span"])
def test_next_public_demand_rejects_mutated_original_authority(tmp_path, demand, change):
    path, analysis, selected = fixture(tmp_path)
    record(analysis, demand, selected)
    if change == "file":
        path.write_text(path.read_text() + "! changed original input\n")
    elif change == "unselected_ast":
        statement, = walk(analysis.routines["segment_graph::unrelated"].execution, F.Assignment_Stmt)
        statement.items = (*statement.items[:2], F.Int_Literal_Constant("4"))
    elif change == "binding":
        analysis.routines[ENTRY].scope.bindings["a"].kind = 4
    else:
        selected[0].content[0].item.span = (1, 1)
    with pytest.raises(CompilationError, match="(authority|source changed)"):
        getattr(analysis, demand)(ENTRY, selected)


@pytest.mark.parametrize("demand", DEMANDS)
def test_equal_looking_copied_selection_never_borrows_graph_authority(tmp_path, demand):
    _path, analysis, selected = fixture(tmp_path)
    record(analysis, demand, selected)
    with pytest.raises(CompilationError, match="original source authority"):
        getattr(analysis, demand)(ENTRY, (copy(selected[0]),))


def mutate_module(analysis):
    declaration = next(node for node in walk(analysis.modules["segment_graph"].node, F.Type_Declaration_Stmt)
                       if "marker" in str(node).lower())
    literal, = walk(declaration, F.Int_Literal_Constant)
    literal.items = ("4", literal.items[1])


@pytest.mark.parametrize("demand", DEMANDS)
def test_descriptor_callback_mutation_requires_new_projection_authentication(tmp_path, monkeypatch, demand):
    _path, analysis, selected = fixture(tmp_path, descriptor=True)
    _graph, identities = analysis._selected_source(ENTRY, selected)
    proof = prove_descriptor_window(analysis, ENTRY, identities, ("argument::a", "argument::b"))
    original_validate = type(proof).validate
    original_projection = SourceEffects._segment_projection
    projections = []
    changed = False

    def validate(current, *args, **kwargs):
        nonlocal changed
        result = original_validate(current, *args, **kwargs)
        if current is proof and not changed:
            changed = True
            mutate_module(analysis)
        return result

    def projection(current, *args, **kwargs):
        if current is analysis:
            projections.append(kwargs.get("_graph"))
        return original_projection(current, *args, **kwargs)

    monkeypatch.setattr(type(proof), "validate", validate)
    monkeypatch.setattr(SourceEffects, "_segment_projection", projection)
    with pytest.raises(CompilationError, match="original source-backed module authority"):
        getattr(analysis, demand)(ENTRY, selected, descriptor_window=proof)
    assert changed
    assert projections == [None]
    assert not analysis._segments


def test_completion_callback_mutation_requires_new_projection_authentication(tmp_path, monkeypatch):
    _path, analysis, selected = fixture(tmp_path, team=True)
    nodes = tuple(node for node in analysis.routines[ENTRY].execution.content
                  if type(node).__name__ not in {"Continue_Stmt", "Assignment_Stmt"})
    proof = analysis.worksharing_native_completion(ENTRY, nodes, selected)
    original_validate = type(proof).validate
    original_projection = SourceEffects._segment_projection
    projections = []

    def validate(current, *args, **kwargs):
        result = original_validate(current, *args, **kwargs)
        if current is proof:
            mutate_module(analysis)
        return result

    def projection(current, *args, **kwargs):
        if current is analysis:
            projections.append(kwargs.get("_graph"))
        return original_projection(current, *args, **kwargs)

    monkeypatch.setattr(type(proof), "validate", validate)
    monkeypatch.setattr(SourceEffects, "_segment_projection", projection)
    with pytest.raises(CompilationError, match="original source-backed module authority"):
        analysis.native_sections_for_nodes(ENTRY, selected, completion=proof)
    assert projections == [None]
