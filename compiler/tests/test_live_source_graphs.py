"""Active source contracts survive bounded metadata-cache eviction."""

import gc
from weakref import ref

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def fixture(tmp_path, *, sealed=False):
    path = tmp_path / "graphs.f90"
    path.write_text("module work\ncontains\n" + "\n".join(
        f"subroutine step{i}(a)\nreal(8),intent(inout)::a(:)\n"
        "a(1)=a(1)+1\nend subroutine" for i in range(5)) + "\nend module\n")
    analysis = SourceEffects([path], procedures=2)
    if sealed:
        analysis.seal_session()
    return path, analysis


def evict(analysis):
    for i in range(1, 5):
        analysis.structure(f"work::step{i}")
    assert len(analysis._structures) == 2


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("demand", ["structure", "selection", "descriptor"])
def test_active_graph_remains_exact_original_after_eviction(tmp_path, sealed, demand):
    _path, analysis = fixture(tmp_path, sealed=sealed)
    original = analysis.structure("work::step0")
    assignment, = walk(analysis.routines["work::step0"].execution, F.Assignment_Stmt)
    evict(analysis)
    if demand == "selection":
        graph, selected = analysis._selected_source("work::step0", (assignment,))
        assert selected == (original.node_id(assignment),)
    else:
        if demand == "descriptor":
            assert analysis.descriptor_stability("work::step0")["complete"]
        graph = analysis.structure("work::step0")
    assert graph is original
    assert graph.source_nodes(graph.node_id(assignment)) == (assignment,)
    assert len(analysis._structures) <= analysis.procedure_limit


def test_unreferenced_evicted_graph_does_not_retain_parser_graph(tmp_path):
    _path, analysis = fixture(tmp_path)
    graph = analysis.structure("work::step0")
    weak = ref(graph)
    del graph
    evict(analysis)
    gc.collect()
    assert weak() is None


def test_changed_authority_does_not_reuse_live_graph(tmp_path):
    _path, analysis = fixture(tmp_path)
    original = analysis.structure("work::step0")
    evict(analysis)
    analysis.operation_limit += 1
    current = analysis.structure("work::step0")
    assert current is not original
    assert current.authority_identity != original.authority_identity


@pytest.mark.parametrize("sealed", [False, True])
def test_evicted_live_graph_still_requires_current_source(tmp_path, sealed):
    path, analysis = fixture(tmp_path, sealed=sealed)
    original = analysis.structure("work::step0")
    evict(analysis)
    if sealed:
        # A sealed session owns immutable captured source. Replacing its
        # namespace must still fail before consulting a retained graph.
        original_registry = analysis.routines
        replacement = dict(original_registry)
        with pytest.raises(CompilationError, match="original source registry is immutable"):
            analysis.routines = replacement
        assert analysis.routines is original_registry
        assert analysis.structure(original.procedure) is original
        evict(analysis)
        # Bypass only for this fault injection; the retained graph must reject it.
        object.__setattr__(analysis, "routines", replacement)
    else:
        path.write_text(path.read_text().replace("a(1)+1", "a(1)+2"))
    with pytest.raises(CompilationError, match="(source|inputs|registry).*(changed|authority)"):
        analysis.structure(original.procedure)


def test_private_segment_does_not_borrow_source_graph_identity(tmp_path):
    _path, analysis = fixture(tmp_path, sealed=True)
    original = analysis.structure("work::step0")
    assignment, = walk(analysis.routines["work::step0"].execution, F.Assignment_Stmt)
    selected = (original.node_id(assignment),)
    projected, _routine, _guard = analysis._segment_projection("work::step0", selected)
    # This payload-only segment is not the original procedure. A live graph
    # in the source analysis must not give it original procedure authority.
    with pytest.raises(CompilationError, match="original source-backed procedure authority"):
        projected.structure("work::step0")
    assert not projected._structures
    assert analysis.structure("work::step0") is original
