"""Lifetime proofs scan complete original source beyond optimizer windows."""

from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_unit_resolution import ENTRY, prove
from compiler.tests.test_descriptor_unit_resolution import fixture as bounded_fixture


def fixture(tmp_path, *args, **kwargs):
    """Use the same source-sized original skeleton policy as Graph ScopeBuilder."""
    bounded, _, _, paths = bounded_fixture(tmp_path, *args, **kwargs)
    analysis = SourceEffects(paths, operations=bounded.operation_limit, procedures=bounded.procedure_limit)
    analysis.source_graph_operation_limit = {
        name: max(analysis.operation_limit,
                  sum(getattr(node, "item", None) is not None for node in walk(routine.execution)))
        for name, routine in analysis.routines.items()}
    originals = tuple(node for node in analysis.routines[ENTRY].execution.content
                      if type(node).__name__ != "Comment")
    graph = analysis.structure(ENTRY)
    assert graph.available
    return analysis, tuple(graph.node_id(node) for node in originals), originals, paths



@pytest.mark.parametrize("count", [257, 300])
def test_complete_original_selection_exceeds_numerical_search_window(tmp_path, count):
    body = "\n".join("a(1)=a(1)+1.0_8" for _ in range(count))
    analysis, selected, _, paths = fixture(tmp_path, body)
    before = paths[-1].read_bytes()
    record = prove(analysis, selected).public()
    assert len(record["selected_node_ids"]) == count
    assert record["source_operations"] == count
    assert record["resolution_validation"]["units"] == count
    assert record["resolution_validation"]["per_unit_lookup_limit"] == 256
    assert record["schema_version"] == 7
    assert record["resolution_validation"]["aggregate_source_scan_limit"] == record["source_scan_accounting"]["limit"]
    assert record["source_scan_accounting"]["limit"] > 65536
    assert record["entry_definition_authority"] is False
    assert record["execution_authorized"] is False
    assert paths[-1].read_bytes() == before


def test_complete_child_and_all_repeated_call_sites_are_counted(tmp_path):
    child = ("subroutine leaf(a)\nreal(8),intent(inout)::a(:)\n"
             + "\n".join("a(1)=a(1)+1.0_8" for _ in range(270)) + "\nend subroutine")
    analysis, selected, _, _ = fixture(tmp_path, "\n".join("call leaf(a)" for _ in range(280)), children=child)
    record = prove(analysis, selected).public()
    assert len(record["selected_node_ids"]) == 280
    assert record["source_operations"] == 550
    assert set(record["source_call_closure"]) == {"application::leaf"}
    assert record["source_traversal_entries"] < 65536


@pytest.mark.parametrize("tail", ["deallocate(a)", "call absent(a)"])
def test_no_unsafe_original_child_tail_is_truncated_after_256_operations(tmp_path, tail):
    child = ("subroutine leaf(a)\nreal(8),allocatable,intent(inout)::a(:)\n"
             + "\n".join("a(1)=a(1)+1.0_8" for _ in range(270)) + "\n" + tail + "\nend subroutine")
    analysis, selected, _, _ = fixture(tmp_path, "call leaf(a)", children=child)
    with pytest.raises(CompilationError, match="unknown selected storage"):
        prove(analysis, selected)
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("mutation", ["source", "binding", "selection"])
def test_large_selection_cannot_reuse_substituted_original_authority(tmp_path, mutation):
    analysis, selected, _, paths = fixture(tmp_path, "\n".join("a(1)=a(1)+1.0_8" for _ in range(270)))
    proof = prove(analysis, selected)
    if mutation == "source":
        paths[-1].write_text(paths[-1].read_text() + "\n! changed original\n")
    elif mutation == "binding":
        routine = analysis.routines[ENTRY]
        routine.scope.bindings["a"] = copy(routine.scope.bindings["a"])
    else:
        selected = (*selected[:-1], "foreign::node")
    with pytest.raises(CompilationError):
        proof.validate(analysis, ENTRY, selected)


def test_many_roots_are_bounded_by_original_binding_inventory(tmp_path):
    names = [f"a{i}" for i in range(257)]
    path = tmp_path / "roots.f90"
    path.write_text("module roots\ncontains\nsubroutine advance(" + ",".join(names)
                    + ")\nreal(8),allocatable,intent(inout)::" + ",".join(name + "(:)" for name in names)
                    + "\na0(1)=a0(1)+1.0_8\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    entry = "roots::advance"
    original, = analysis.routines[entry].execution.content
    selected = (analysis.structure(entry).node_id(original),)
    roots = tuple("argument::" + name for name in names)
    proof = prove_descriptor_window(analysis, entry, selected, roots)
    assert len(proof.public()["resources"]) == 257
    assert proof.public()["entry_definition_authority"] is False
    with pytest.raises(CompilationError, match="bounded unique original formal roots"):
        prove_descriptor_window(analysis, entry, selected, (*roots, "argument::absent"))


def test_large_selection_cannot_hide_one_oversized_resolution_unit(tmp_path):
    count = 260
    declarations = "real(8)::" + ",".join(f"v{i}" for i in range(count))
    body = "\n".join("a(1)=a(1)+1.0_8" for _ in range(270))
    body += "\na(1)=" + "+".join(f"v{i}" for i in range(count))
    analysis, selected, _, _ = fixture(tmp_path, body, declarations=declarations)
    with pytest.raises(CompilationError, match="resolution budget exhausted"):
        prove(analysis, selected)
    assert analysis.operation_limit == 256
    assert not getattr(analysis, "_descriptor_window_proofs", {})
