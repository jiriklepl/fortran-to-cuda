"""Default closure capacity follows complete original source, not work count."""

import pytest
from fparser.two.utils import Base, walk

from compiler.frontend import source_effects
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError

ENTRY = "inventory::run"
ROOT = "argument::a"


def consume(analysis, selected, consumer):
    if consumer == "window":
        return prove_descriptor_window(analysis, ENTRY, selected, (ROOT,))
    return prove_storage_mutation_effects(analysis, ENTRY)


def tree(tmp_path, *, count=70, tail="a(1)=1", sealed=False, **budgets):
    path = tmp_path / "inventory.f90"
    leaves = [f"subroutine leaf{i}(a)\nreal(8),allocatable,intent(inout)::a(:)\n"
              f"{tail if i == count - 1 else 'a(1)=1'}\nend subroutine\n"
              for i in range(count)]
    path.write_text("module inventory\nreal(8),allocatable::hidden(:)\ncontains\n"
                    "subroutine run(a)\nreal(8),allocatable,intent(inout)::a(:)\n"
                    + "\n".join(f"call leaf{i}(a)" for i in range(count))
                    + "\nend subroutine\n" + "".join(leaves) + "end module\n")
    analysis = SourceEffects([path], **budgets)
    if sealed:
        analysis.seal_session()
    routine = analysis.routines[ENTRY]
    _graph, selected = analysis._selected_source(ENTRY, tuple(routine.execution.content))
    return analysis, selected


@pytest.mark.parametrize("sealed", [False, True])
def test_default_scans_all_70_unique_callees_in_both_descriptor_consumers(tmp_path, sealed):
    analysis, selected = tree(tmp_path, sealed=sealed)
    window = prove_descriptor_window(analysis, ENTRY, selected, (ROOT,)).public()
    assert analysis.procedure_limit > 71
    assert len(window["source_call_closure"]) == 70
    assert "inventory::leaf69" in window["source_call_closure"]
    assert window["resolution_validation"]["procedure_limit"] == analysis.procedure_limit
    assert window["execution_authorized"] is False
    storage = prove_storage_mutation_effects(analysis, ENTRY)
    assert storage.preserves_descriptors()
    assert {item["procedure"] for item in storage.public()["closure"]} == {
        ENTRY, *(f"inventory::leaf{i}" for i in range(70))}
    assert storage.public()["payload_access_authorized"] is False


@pytest.mark.parametrize("tail", ["deallocate(hidden)", "call absent(a)"])
def test_mutation_and_unknown_call_after_old_64_boundary_are_never_truncated(tmp_path, tail):
    analysis, selected = tree(tmp_path, tail=tail, sealed=True)
    storage = prove_storage_mutation_effects(analysis, ENTRY)
    record = storage.public()
    assert len(record["closure"]) == 71
    assert not storage.preserves_descriptors()
    assert any(item["procedure"] == "inventory::leaf69" for item in record["closure"])
    assert record["mutations"] or record["frontiers"]
    with pytest.raises(CompilationError, match="dealloc|unsupported|unavailable|unknown"):
        prove_descriptor_window(analysis, ENTRY, selected, (ROOT,))
    assert not getattr(analysis, "_descriptor_window_proofs", {})


@pytest.mark.parametrize("consumer", ["window", "storage"])
def test_recursion_after_70_shallow_callees_still_rejects(tmp_path, consumer):
    analysis, selected = tree(tmp_path, tail="call run(a)", sealed=True)
    with pytest.raises(CompilationError, match="recurs"):
        consume(analysis, selected, consumer)


@pytest.mark.parametrize("consumer", ["window", "storage", "summary"])
def test_explicit_procedure_limit_remains_an_independent_diagnostic_cap(tmp_path, consumer):
    analysis, selected = tree(tmp_path, count=3, procedures=3)
    assert analysis.procedure_limit == 3
    if consumer == "summary":
        record = analysis.report(ENTRY)
        assert record["budgets"]["procedures"] == 3
        assert not record["complete"]
        assert len(record["procedures"]) == 3
        assert any("callee effects incomplete: inventory::leaf2" in item["reasons"]
                   for item in record["procedures"])
    else:
        with pytest.raises(CompilationError, match="closure"):
            consume(analysis, selected, consumer)


def test_ordinary_effect_closure_uses_same_inventory_sized_default(tmp_path):
    analysis, _selected = tree(tmp_path, operations=4096)
    report = analysis.report(ENTRY)
    assert report["complete"]
    assert len(report["procedures"]) == 71
    assert report["budgets"]["procedures"] == analysis.procedure_limit
    assert report["budgets"]["operations"] == 4096
    authority = analysis._summary_authority()
    assert authority["source_closure_bound_version"] == source_effects.SOURCE_CLOSURE_BOUND_VERSION == 1
    assert authority["budgets"]["procedures"] == analysis.procedure_limit


@pytest.mark.parametrize("program_roots", [(), ("$program::main",)])
def test_inventory_counts_external_functions_and_programs_without_granting_execution(tmp_path, program_roots):
    path = tmp_path / "units.f90"
    path.write_text("subroutine ext\nend subroutine\n"
                    "real function external_value(x)\nreal::x\nexternal_value=x+1\nend function\n"
                    "program main\ninteger::i\ni=1\ncall ext\nend program\n")
    analysis = SourceEffects([path], program_roots=program_roots)
    expected = len(walk(analysis.inputs.parse(path.resolve()), Base))
    assert analysis.procedure_limit == expected
    assert analysis.procedure_limit > len(analysis._source_syntax._nodes)
    assert "$external::ext" in analysis.routines
    assert "$external::external_value" in analysis.functions
    assert ("$program::main" in analysis.routines) == bool(program_roots)
    if program_roots:
        assert analysis.routines["$program::main"].source_kind == "program"
        assert not analysis.report("$program::main")["complete"]


@pytest.mark.parametrize(("count", "operations"), [(20, 16)])
def test_distinct_original_function_requests_are_not_one_global_lookup_budget(tmp_path, count, operations):
    path = tmp_path / "requests.f90"
    path.write_text("module requests\ncontains\nsubroutine run(a)\n"
                    "real(8),allocatable,intent(inout)::a(:)\n"
                    + "\n".join(f"a(1)=value({i}.0_8)" for i in range(count))
                    + "\nend subroutine\nreal(8) function value(x)\nreal(8),intent(in)::x\n"
                    "value=x+1.0_8\nend function\nend module\n")
    analysis = SourceEffects([path], operations=operations)
    # A graph's source size and one semantic lookup's work cap are separate.
    analysis.source_graph_operation_limit = {"requests::run": 512}
    analysis.seal_session()
    routine = analysis.routines["requests::run"]
    _graph, selected = analysis._selected_source(routine.qualified, tuple(routine.execution.content))
    public = prove_descriptor_window(analysis, routine.qualified, selected, (ROOT,)).public()
    requirements = [item for name, rows in public.items() if name in {
        "native_source_function_requirements", "native_descriptor_source_function_requirements",
        "native_storage_function_requirements"} for item in rows]
    assert len(requirements) == count
    assert public["resolution_validation"]["per_unit_lookup_limit"] == operations
    assert len(selected) == count
    assert public["execution_authorized"] is False


def test_default_depth_eight_is_still_independent_of_source_inventory(tmp_path):
    path = tmp_path / "depth.f90"
    path.write_text("module depth\ncontains\n" + "".join(
        f"subroutine level{i}(a)\nreal(8),allocatable,intent(inout)::a(:)\n"
        + (f"call level{i+1}(a)" if i < 8 else "a(1)=1")
        + "\nend subroutine\n" for i in range(9)) + "end module\n")
    analysis = SourceEffects([path])
    assert analysis.depth_limit == 8
    assert analysis.procedure_limit > 9
    with pytest.raises(CompilationError, match="closure"):
        prove_storage_mutation_effects(analysis, "depth::level0")
    routine = analysis.routines["depth::level0"]
    _graph, selected = analysis._selected_source(routine.qualified, tuple(routine.execution.content))
    with pytest.raises(CompilationError, match="closure budget"):
        prove_descriptor_window(analysis, routine.qualified, selected, (ROOT,))


def test_empty_source_inventory_cannot_create_a_zero_budget():
    analysis = SourceEffects([])
    assert analysis.procedure_limit == 1
    assert not analysis.routines
