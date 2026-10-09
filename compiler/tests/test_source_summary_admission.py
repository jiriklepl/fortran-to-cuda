"""A valid cache digest does not establish a structurally valid source proof."""

import copy

import pytest
from fparser.two import Fortran2003 as F

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_source_summary_reuse import ENTRY, graph, records, seed_closure


@pytest.mark.parametrize("damage", [
    "missing_definition_changes", "bad_arguments", "bad_persistent_state", "bad_descriptor_requirements",
    "bad_completion", "bad_sections", "bad_composition", "bad_ordered_effects", "bad_cloneable",
    "bad_operation", "bad_mapping", "bad_mapping_descriptor", "bad_call_definition_events", "bad_original_arguments",
])
def test_restamped_malformed_public_summary_is_a_cache_miss(tmp_path, damage):
    paths = graph(tmp_path)
    original = SourceEffects(paths).report(ENTRY)
    damaged = copy.deepcopy(original)
    summaries = records(damaged)
    leaf = summaries["operators::adjust"]
    call = summaries[ENTRY]["operations"][0]
    if damage == "missing_definition_changes":
        del leaf["definition_changes"]
    elif damage == "bad_arguments":
        leaf["arguments"] = "not a list"
    elif damage == "bad_persistent_state":
        leaf["persistent_state"] = {}
    elif damage == "bad_descriptor_requirements":
        leaf["descriptor_requirements"] = [None]
    elif damage == "bad_completion":
        leaf["native_completion"]["available"] = "true"
    elif damage == "bad_sections":
        leaf["native_sections"] = []
    elif damage == "bad_composition":
        leaf["effect_composition"]["available"] = False
    elif damage == "bad_ordered_effects":
        leaf["ordered_effects"] = {}
    elif damage == "bad_cloneable":
        leaf["cloneable"] = "true"
    elif damage == "bad_operation":
        leaf["operations"][0]["kind"] = "unrecognized_effect"
    elif damage == "bad_mapping":
        call["resource_mappings"] = [None]
    elif damage == "bad_mapping_descriptor":
        call["resource_mappings"][0]["formal_descriptor"]["rank"] = "1"
    elif damage == "bad_call_definition_events":
        call["definition_events"] = "missing events"
    else:
        call["original_arguments"] = [None]
    analysis = SourceEffects(paths)
    # Store a real JSON closure with updated child, summary and payload digests.
    # Generic storage accepts it; source-proof admission must still reject it.
    seed_closure(analysis, damaged)
    report = analysis.report(ENTRY)
    assert report["complete"], report
    assert report["summary_cache"]["rejected"] >= 1
    assert records(report)[ENTRY]["summary_identity"] == records(original)[ENTRY]["summary_identity"]
    assert records(report)["operators::adjust"]["definition_changes"] == []


def test_valid_complete_public_summaries_remain_reusable_after_structural_admission(tmp_path):
    paths = graph(tmp_path)
    first = SourceEffects(paths).report(ENTRY)
    second = SourceEffects(paths).report(ENTRY)
    assert first["complete"]
    assert second["complete"]
    assert second["summary_cache"]["memory_hits"] >= 1
    assert second["summary_cache"]["rejected"] == 0
    assert records(second)[ENTRY]["summary_identity"] == records(first)[ENTRY]["summary_identity"]


@pytest.mark.parametrize("query", ["native_sections", "span", "empty_span"])
def test_direct_public_queries_verify_source_before_reusing_proofs(tmp_path, query):
    path = tmp_path / "source.f90"
    path.write_text("module source\ncontains\nsubroutine update(a)\nreal(8),intent(inout)::a(:)\na(1)=7\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    analysis.native_sections("source::update")
    analysis.summarize_span(["source::update"])
    path.write_text(path.read_text().replace("a(1)=7", "a(2)=9"))
    run = ((lambda: analysis.native_sections("source::update")) if query == "native_sections" else
           (lambda: analysis.summarize_span([] if query == "empty_span" else ["source::update"])))
    with pytest.raises(CompilationError, match="source changed"):
        run()


def test_typed_sections_recompute_private_execution_projection_without_summary_query(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text("module source\ncontains\nsubroutine update(a)\nreal(8),intent(inout)::a(:)\na(1)=7\nend subroutine\nend module\n")
    analysis = SourceEffects([path], procedures=1)
    original = analysis.native_sections("source::update")
    assert original.resources[0].writes[0].axes[0].lower.value == 1
    routine = analysis.routines["source::update"]
    routine.execution = copy.copy(routine.execution)
    routine.execution.content = [F.Assignment_Stmt("a(2)=9")]
    projected = analysis.native_sections("source::update")
    assert projected is not original
    assert projected.resources[0].writes[0].axes[0].lower.value == 2
    assert analysis.native_sections("source::update") is projected
    assert len(analysis._native_sections) == 1


def test_in_place_execution_projection_cannot_borrow_original_source_summary(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text("module source\ncontains\nsubroutine update(a)\nreal(8),intent(inout)::a(:)\na(1)=7\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    original = analysis.summarize("source::update")
    original_sections = analysis.native_sections("source::update")
    analysis.routines["source::update"].execution.content[:] = [F.Assignment_Stmt("a(2)=9")]
    projected = analysis.summarize("source::update")
    assert projected["summary_role"] == "private_projection"
    assert projected["summary_identity"] != original["summary_identity"]
    sections = analysis.native_sections("source::update")
    assert sections is not original_sections
    assert sections.resources[0].writes[0].axes[0].lower.value == 2
    assert analysis.native_sections("source::update") is sections


def test_typed_sections_recompute_original_bound_roles_without_summary_query(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text("module source\ncontains\nsubroutine update(a)\nreal(8),intent(inout)::a(-2:)\na(-1)=7\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    original = analysis.native_sections("source::update")
    binding = analysis.routines["source::update"].scope.bindings["a"]
    binding.lower_bound_nodes = (F.Level_2_Unary_Expr("-3"),)
    projected = analysis.native_sections("source::update")
    assert projected is not original
    assert original.resources[0].lower_bounds == (-2,)
    assert projected.resources[0].lower_bounds == (-3,)
