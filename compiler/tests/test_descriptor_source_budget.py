"""Defensive proof work follows only exact reached original syntax."""
from copy import copy

import pytest
from fparser.two.utils import walk

from compiler.frontend.descriptor_budget import DescriptorSourceBudget
from compiler.frontend.descriptor_windows import _Check, prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

ENTRY = "budget::advance"


def fixture(tmp_path, count=1, *, tail="", extra=""):
    path = tmp_path / "budget.f90"
    path.write_text("module budget\ncontains\nsubroutine advance(a)\n"
                    "real(8),allocatable,intent(inout)::a(:)\n"
                    + "\n".join("a(1)=a(1)+1.0_8" for _ in range(count)) + "\n" + tail
                    + "\nend subroutine\n" + extra + "\nend module\n")
    analysis = SourceEffects([path])
    analysis.source_graph_operation_limit = {
        name: max(analysis.operation_limit,
                  sum(getattr(node, "item", None) is not None for node in walk(routine.execution)))
        for name, routine in analysis.routines.items()}
    routine = analysis.routines[ENTRY]
    graph = analysis.structure(ENTRY)
    originals = tuple(node for node in routine.execution.content if type(node).__name__ != "Comment")
    selected = tuple(graph.node_id(node) for node in originals[:count])
    return path, analysis, routine, graph, selected


def test_small_original_input_retains_default_limit_without_virtual_tick_counts(tmp_path):
    _path, analysis, routine, graph, selected = fixture(tmp_path)
    budget = DescriptorSourceBudget(analysis)
    budget.reached(routine, graph, selected)
    budget.validate()
    assert budget.limit == 65536
    public = budget.public()
    item, = public["reached"]
    assert item["procedure"] == ENTRY
    assert item["selected_node_ids"] == list(selected)
    assert item["original_graph_operation_units"] == 1
    assert item["original_declaration_units"] == 2
    assert item["allowance"] == 8 * item["unique_original_parser_nodes"] + 256 * 3
    assert public["execution_authorized"] is False
    assert public["entry_definition_authority"] is False
    assert public["payload_access_authorized"] is False


def test_large_reached_original_source_has_an_explicit_linear_allowance(tmp_path):
    _path, analysis, routine, graph, selected = fixture(tmp_path, 300)
    budget = DescriptorSourceBudget(analysis)
    budget.reached(routine, graph, selected)
    budget.validate()
    item, = budget.public()["reached"]
    assert item["original_graph_operation_units"] == 300
    assert budget.limit == item["allowance"]
    assert budget.limit > 65536
    assert budget.limit == 8 * item["unique_original_parser_nodes"] + 256 * 302


def test_repeated_exact_original_routine_cannot_mint_another_allowance(tmp_path):
    _path, analysis, routine, graph, selected = fixture(tmp_path, 300)
    budget = DescriptorSourceBudget(analysis)
    budget.reached(routine, graph, selected)
    original = budget.public()
    for _ in range(150):
        budget.reached(routine, graph, selected)
    assert budget.public() == original
    assert len(budget.public()["reached"]) == 1


def test_unselected_tail_and_unreached_large_procedure_supply_no_credit(tmp_path):
    extra = ("subroutine unused(x)\nreal(8)::x\n" + "\n".join("x=x+1" for _ in range(300))
             + "\nend subroutine")
    tail = "\n".join("a(1)=a(1)*2" for _ in range(300))
    _path, analysis, routine, graph, selected = fixture(tmp_path, tail=tail, extra=extra)
    budget = DescriptorSourceBudget(analysis)
    budget.reached(routine, graph, selected)
    assert budget.limit == 65536
    item, = budget.public()["reached"]
    assert item["original_graph_operation_units"] == 1
    assert item["unique_original_parser_nodes"] < 40
    assert item["procedure"] == ENTRY
    assert item["selected_node_ids"] == list(selected)


@pytest.mark.parametrize("mutation", ["foreign_routine", "foreign_graph", "foreign_selection", "policy", "source"])
def test_budget_cannot_retag_forged_or_changed_original_authority(tmp_path, mutation):
    path, analysis, routine, graph, selected = fixture(tmp_path, 300)
    budget = DescriptorSourceBudget(analysis)
    budget.reached(routine, graph, selected)
    if mutation == "foreign_routine":
        routine = copy(routine)
    elif mutation == "foreign_graph":
        graph = copy(graph)
    elif mutation == "foreign_selection":
        selected = (*selected[:-1], "foreign::node")
    elif mutation == "policy":
        analysis.operation_limit += 1
    else:
        path.write_text(path.read_text() + "\n! source changed\n")
    with pytest.raises(CompilationError):
        budget.reached(routine, graph, selected)


def test_original_routine_cannot_retag_a_different_selection_for_more_credit(tmp_path):
    _path, analysis, routine, graph, selected = fixture(tmp_path, 300)
    budget = DescriptorSourceBudget(analysis)
    budget.reached(routine, graph, selected[:1])
    with pytest.raises(CompilationError, match="retag"):
        budget.reached(routine, graph, selected)
    assert budget.limit == 65536


def test_public_counts_and_manual_accounting_cannot_supply_an_original_allowance(tmp_path):
    _path, analysis, routine, graph, selected = fixture(tmp_path)
    budget = DescriptorSourceBudget(analysis)
    budget.reached(routine, graph, selected)
    with pytest.raises(CompilationError, match="original routine"):
        budget.reached(budget.public(), graph, selected)
    budget._limit += 1
    with pytest.raises(CompilationError, match="accounting"):
        budget.validate()


def test_extended_proof_publishes_exact_input_formula_but_no_execution_authority(tmp_path):
    _path, analysis, _routine, _graph, selected = fixture(tmp_path, 300)
    proof = prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))
    record = proof.public()
    assert record["schema_version"] == 7
    accounting = record["source_scan_accounting"]
    assert accounting["formula"] == "max(65536, sum(8*S + per_unit_lookup_limit*U))"
    assert accounting["limit"] == max(65536, sum(row["allowance"] for row in accounting["reached"]))
    assert accounting["limit"] == record["resolution_validation"]["aggregate_source_scan_limit"]
    assert record["source_operations"] == 300
    assert accounting["repeated_callee_credit"] is False
    assert accounting["unrelated_source_credit"] is False
    assert record["entry_definition_authority"] is False
    assert record["execution_authorized"] is False
    assert proof.validate(analysis, ENTRY, selected) is proof


def test_repeated_charged_work_cannot_reset_or_exceed_an_original_allowance(tmp_path):
    _path, analysis, routine, graph, selected = fixture(tmp_path)
    check = _Check(analysis)
    check.source_budget.reached(routine, graph, selected)
    limit = check.source_budget.limit
    for _ in range(limit):
        check.tick()
    check.source_budget.reached(routine, graph, selected)
    assert check.nodes == limit
    assert check.source_budget.limit == limit
    with pytest.raises(CompilationError, match="original-source traversal budget"):
        check.tick()
    assert check.nodes == limit + 1


@pytest.mark.parametrize("tail", ["deallocate(a)", "call absent(a)"])
def test_extended_source_bound_never_skips_an_unsafe_original_tail(tmp_path, tail):
    _path, analysis, _routine, _graph, selected = fixture(tmp_path, 300, tail=tail)
    # Include the original unsafe tail in the demanded contiguous selection.
    graph = analysis.structure(ENTRY)
    routine = analysis.routines[ENTRY]
    selected = tuple(graph.node_id(node) for node in routine.execution.content
                     if type(node).__name__ != "Comment")
    with pytest.raises(CompilationError, match="unknown selected storage"):
        prove_descriptor_window(analysis, ENTRY, selected, ("argument::a",))
    assert not getattr(analysis, "_descriptor_window_proofs", {})

