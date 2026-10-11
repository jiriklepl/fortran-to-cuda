"""The public reached path discovers exact windows without caller proof input."""

from copy import deepcopy
from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.scopes.window_discovery import MAX_ATTEMPTS, discover_reached_window
from compiler.tests.test_reached_window_owner import ENTRY, ROOTS, SOURCE
from compiler.tests.test_source_scopes import FACT


def discovery_fixture(directory, source=SOURCE, *, policy="sections", captures=None, device_budget=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "stage.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": captures if captures is not None else {root: deepcopy(FACT) for root in ROOTS}}
    if device_budget is not None:
        facts["device_budget_bytes"] = device_budget
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy=policy, scope_execution="reached"))
    return path, builder


def assert_original_declines(builder, expected):
    graph = builder.analysis.structure(ENTRY)
    found = []
    for attempt in builder.reached_window_discovery["attempts"]:
        declines = attempt.get("source_declines", [])
        assert len(declines) <= builder.analysis.operation_limit
        for point in declines:
            original, = graph.source_nodes(point["node_id"])
            assert point["phase"] in {"prefix_disable", "active_close"}
            assert point["span"] == list(graph.nodes[point["node_id"]].span)
            assert point["original_source_kind"] == type(original).__name__
            assert point["reason"] in attempt["reason"]
            if expected in point["reason"] or expected == point["original_source_kind"]:
                found.append(point)
    assert found


def test_public_reached_path_discovers_window_and_keeps_original_prefix_and_tail(tmp_path):
    path, builder = discovery_fixture(tmp_path)
    assert not getattr(builder.analysis, "_reached_window_proofs", {})
    facts = deepcopy(builder.facts)
    outputs, report = builder.run()
    discovery = report["reached_window_discovery"]
    assert discovery["selected"]
    assert len(discovery["attempts"]) == 1
    assert discovery["attempts"][0]["accepted"]
    cut, = discovery["attempts"][0]["planned_closures"]["close_points"]
    assert cut["original_source_kind"] == "Call_Stmt"
    assert "move_alloc" in str(builder.analysis.structure(ENTRY).source_nodes(cut["node_id"])[0]).lower()
    assert discovery["attempts"][-1]["numerical_invocations"] == 2
    scope, = report["scopes"]
    assert scope["reached_window"]["automatic_discovery"]
    assert scope["reached_window"]["proof_identity"] == discovery["proof_identity"]
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.index("call native_prefix(a)") < text.index("do stage=1,iterations")
    assert text.index("fort_scope_close(") < text.index("call move_alloc(a,temporary)")
    assert text.count("call native_prefix(a)") == text.count("call move_alloc(a,temporary)") == 1
    assert path.read_text() == SOURCE
    assert builder.facts == facts
    assert builder.inline.builder is builder
    assert builder.region_owners[ENTRY] is builder.inline
    assert builder.validate_reached_window() is not None


def test_discovery_reuses_numerical_leaf_for_two_original_calls(tmp_path):
    helper = """subroutine update(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=a(i)+real(i+4,8)
enddo
end subroutine
"""
    source = SOURCE.replace("subroutine step(a,b,n,iterations)", helper + "subroutine step(a,b,n,iterations)")
    first = source.index("do i=-2,n-3", source.index("subroutine step("))
    last = source.index("call move_alloc(a,temporary)", first)
    source = source[:first] + "call update(a,b,n)\ncall update(a,b,n)\n" + source[last:]
    _path, builder = discovery_fixture(tmp_path, source)
    _outputs, report = builder.run()
    assert report["reached_window_discovery"]["selected"]
    scope, = report["scopes"]
    assert scope["gpu_leaves"] == ["renamed_stage::update#region1"]
    coordinator, = scope["module_coordinators"]
    assert coordinator["procedure"] == "renamed_stage::update"
    assert len(coordinator["calls"]) == 2
    assert len({site["mapped_call_node_id"] for site in coordinator["calls"]}) == 2
    assert coordinator["structured_summary_identity"] == coordinator["mapped_reached_plan"]["plan"]["structured_identity"]
    assert report["reached_window_discovery"]["attempts"][-1]["numerical_invocations"] == 2
    assert len([item for item in report["numerical_decisions"] if item["supported"]]) == 1


def test_unknown_array_dependency_cannot_be_hidden_by_omitting_its_capture(tmp_path):
    source = SOURCE.replace("real(8),allocatable::temporary(:)", "real(8),allocatable::temporary(:)\nreal(8)::c(8)")
    source = source.replace("b(i)=a(i)+real(i+4,8)", "b(i)=a(i)+c(i+3)")
    _path, builder = discovery_fixture(tmp_path, source)
    assert not discover_reached_window(builder)
    assert not builder.outputs
    assert not builder.edits
    assert not builder.scopes
    assert any("untracked array" in attempt["reason"] for attempt in builder.reached_window_discovery["attempts"])
    assert_original_declines(builder, "untracked array")


def test_module_allocatable_facts_do_not_substitute_for_selected_lifetime_proof(tmp_path):
    source = SOURCE.replace("real(8)::prefix_total=0.0_8", "real(8)::prefix_total=0.0_8\nreal(8),allocatable::hidden(:)")
    source = source.replace("b(i)=a(i)+real(i+4,8)", "b(i)=a(i)+hidden(i)")
    source = source.replace("a(i)=2.0_8*b(i)", "deallocate(hidden)\nallocate(hidden(8))\na(i)=2.0_8*b(i)")
    captures = {root: deepcopy(FACT) for root in (*ROOTS, "renamed_stage::hidden")}
    _path, builder = discovery_fixture(tmp_path, source, captures=captures)
    assert not discover_reached_window(builder)
    assert not builder.outputs
    assert not builder.edits
    assert not builder.scopes
    assert any("Deallocate_Stmt" in attempt["reason"]
               for attempt in builder.reached_window_discovery["attempts"])
    assert_original_declines(builder, "Deallocate_Stmt")
    assert any(attempt.get("source_declines") and "needs two admitted numerical" in attempt["reason"]
               for attempt in builder.reached_window_discovery["attempts"])


def test_bounded_rejected_search_has_no_generated_artifacts_or_registered_tokens(tmp_path):
    source = SOURCE.replace("do stage=1,iterations\n", "do stage=1,iterations\n" + "call unknown_work()\n" * 20)
    _path, builder = discovery_fixture(tmp_path, source)
    assert not discover_reached_window(builder)
    record = builder.reached_window_discovery
    assert len(record["attempts"]) == MAX_ATTEMPTS
    assert "budget exhausted" in record["reason"]
    for name in ("outputs", "edits", "clones", "generated", "queries", "numerical_ir", "batch_chains",
                 "scopes", "boundaries", "resolved_calls", "coordinators", "coordinator_reasons"):
        assert not getattr(builder, name), name
    assert not builder.inline.regions
    assert not getattr(builder.analysis, "_reached_window_proofs", {})
    assert not getattr(builder.analysis, "_descriptor_window_proofs", {})
    assert not builder.runtime_provenance.records


def test_auto_without_calibration_stays_native_without_window_instrumentation(tmp_path):
    path, builder = discovery_fixture(tmp_path, policy="auto")
    outputs, report = builder.run()
    assert not report["reached_window_discovery"]["selected"]
    assert any("automatic GPU alternative" in attempt["reason"]
               for attempt in report["reached_window_discovery"]["attempts"])
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text == SOURCE
    assert "fort_window_spent_" not in text
    assert all(scope["automatic_preflight"]["contexts_created"] == 0 for scope in report["scopes"])


def test_whole_entry_stable_formals_keep_existing_owner_path(tmp_path):
    source = SOURCE.replace("call move_alloc(a,temporary)\ncall move_alloc(temporary,a)\n", "")
    _path, builder = discovery_fixture(tmp_path, source)
    _outputs, report = builder.run()
    assert "reached_window_discovery" not in report
    assert all("reached_window" not in scope for scope in report["scopes"])


def test_failed_search_preserves_existing_fallback_artifacts(tmp_path, monkeypatch):
    source = SOURCE.replace("a(i)=2.0_8*b(i)", "call opaque_element(a(i),b(i))")
    _path, actual = discovery_fixture(tmp_path, source)
    # Parse the same original source and supplied facts afresh. A second path
    # would intentionally change artifact identities and invalidate comparison.
    control = ScopeBuilder([actual.entry.scope.path], ENTRY, facts=deepcopy(actual.facts),
                           options=actual.options, config=actual.config)
    actual_outputs, actual_report = actual.run()
    monkeypatch.setattr("compiler.scopes.window_discovery.discover_reached_window", lambda builder: False)
    control_outputs, control_report = control.run()
    assert not actual_report["reached_window_discovery"]["selected"]
    # Diagnostics intentionally expose the rejected search; generated code,
    # variants, source patches and runtime provenance must otherwise match.
    assert {name: value for name, value in actual_outputs.items() if name != "scope-manifest.json"} == {
        name: value for name, value in control_outputs.items() if name != "scope-manifest.json"}
    for field in ("scopes", "boundaries", "source_edits", "runtime_provenance", "implementation_variants",
                  "resolved_calls", "inline_numerical_regions", "child_numerical_regions"):
        assert actual_report[field] == control_report[field], field


def test_excluded_tail_observes_final_iterator_and_prevents_two_kernel_discovery(tmp_path):
    source = SOURCE.replace("tail_visits=tail_visits+1", "tail_visits=tail_visits+i")
    _path, builder = discovery_fixture(tmp_path, source)
    assert not discover_reached_window(builder)
    assert not builder.scopes
    assert not builder.edits
    assert any("two admitted numerical" in attempt["reason"]
               for attempt in builder.reached_window_discovery["attempts"])


def test_next_backedge_prefix_observes_final_iterator_without_suffix_kill(tmp_path):
    source = SOURCE.replace("do stage=1,iterations\n", "do stage=1,iterations\nprefix_visits=prefix_visits+i\n")
    # The input is intentionally an invocation fact for the original scalar;
    # no numerical decision may remove its next-iteration observable value.
    source = source.replace("call native_prefix(a)\n", "call native_prefix(a)\ni=7\n")
    _path, builder = discovery_fixture(tmp_path, source)
    assert not discover_reached_window(builder)
    assert not builder.scopes
    assert not builder.edits


def test_window_preparation_keeps_excluded_fixed_private_array_consumer(tmp_path):
    source = SOURCE.replace("real(8),allocatable::temporary(:)",
                            "real(8),allocatable::temporary(:)\nreal(8)::scratch(2)")
    source = source.replace("b(i)=a(i)+real(i+4,8)",
                            "scratch(1)=a(i)\nscratch(2)=real(i+4,8)\nb(i)=sum(scratch)")
    source = source.replace("tail_visits=tail_visits+1", "prefix_total=sum(scratch)\ntail_visits=tail_visits+1")
    _path, builder = discovery_fixture(tmp_path, source)
    outer = next(node for node in builder.entry.execution.content
                 if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    first = next(node for node in outer.content if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    builder.inline.prepare((first,))
    assert not builder.inline.regions
    assert any("private array is live after" in boundary["reason"] for boundary in builder.boundaries)
