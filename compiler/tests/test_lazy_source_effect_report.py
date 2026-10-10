"""Reached diagnostics never expand a whole call closure just to print it."""

from fparser.two import Fortran2003 as F
from fparser.two.utils import walk
import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError


def fixture(tmp_path, *, oversized=True):
    path = tmp_path / "renamed.f90"
    path.write_text("""module diagnostics
implicit none
contains
subroutine leaf(a)
real(8),intent(inout)::a(:)
a=2*a
end subroutine
subroutine root(a,flag)
real(8),intent(inout)::a(:)
logical,intent(in)::flag
if(flag) then
call distant(a)
call unavailable(a)
endif
call leaf(a)
end subroutine
subroutine distant(a)
real(8),intent(inout)::a(:)
""" + "a=a+1\n" * (130 if oversized else 1) + "end subroutine\nend module\n")
    return path, SourceEffects([path])


def test_lazy_report_authenticates_local_source_without_materializing_any_call(tmp_path, monkeypatch):
    _path, analysis = fixture(tmp_path)
    structure = analysis.structure("diagnostics::root")
    before = analysis._summary_cache.stats
    def forbid(*_args, **_kwargs):
        raise AssertionError("a lazy diagnostic must not request any effect closure")
    monkeypatch.setattr(analysis, "summarize", forbid)
    report = analysis.report("root", materialize=False)
    assert report["schema_version"] == 2
    assert report["entry"] == "diagnostics::root"
    assert report["complete"] is False
    assert report["closure_complete_available"] is False
    assert report["closure_materialization"] == "not_requested"
    assert report["procedures"] == []
    assert report["summarized_operations"] == 0
    assert report["materialized_rejections"] == []
    assert report["structured_effects"] == structure.public()
    assert "unrequested whole-entry closure" in report["compatibility"]
    nodes = report["structured_effects"]["nodes"]
    assert any(node["kind"] == "branch" for node in nodes)
    assert any(node.get("procedure") == "diagnostics::distant" for node in nodes)
    assert any(node["kind"] == "boundary" and "unavailable" in str(node) for node in nodes)
    assert analysis._summary_cache.stats == before
    assert not analysis._closures and not analysis.summaries


def test_lazy_report_retains_previously_proved_leaves_and_reached_segments_only(tmp_path, monkeypatch):
    _path, analysis = fixture(tmp_path)
    leaf = analysis.summarize("diagnostics::leaf")
    calls = walk(analysis.routines["diagnostics::root"].execution, F.Call_Stmt)
    segment = analysis.segment_summary("diagnostics::root", (calls[-1],))
    assert leaf["complete"] and segment["complete"]
    assert "diagnostics::root" not in analysis._closures
    def forbid(*_args, **_kwargs):
        raise AssertionError("report must only publish existing proofs")
    monkeypatch.setattr(analysis, "summarize", forbid)
    report = analysis.report("diagnostics::root", materialize=False)
    assert report["procedures"] == [leaf, segment]
    assert {row["summary_role"] for row in report["procedures"]} == {"source", "reached_source_segment"}
    assert report["summarized_operations"] == len(leaf["operations"]) + len(segment["operations"])
    assert report["call_graph"]["scope"].startswith("already materialized complete proofs")
    assert len(report["call_graph"]["edges"]) == 1
    assert report["call_graph"]["edges"][0]["callee"] == "diagnostics::leaf"
    assert report["complete"] is False
    report["procedures"][0]["operations"].clear()
    assert analysis.report("diagnostics::root", materialize=False)["procedures"][0]["operations"]


def test_observed_incomplete_proofs_retain_rejection_reasons_without_becoming_complete_rows(tmp_path):
    _path, analysis = fixture(tmp_path)
    failed = analysis.summarize("diagnostics::distant")
    assert not failed["complete"]
    report = analysis.report("diagnostics::root", materialize=False)
    assert report["procedures"] == []
    rejection, = report["materialized_rejections"]
    assert rejection["procedure"] == "diagnostics::distant"
    assert rejection["summary_identity"] == failed["summary_identity"]
    assert rejection["reasons"] == failed["reasons"]
    assert any("budget" in reason for reason in rejection["reasons"])


def test_lazy_report_does_not_include_previously_completed_parent_closures(tmp_path):
    path, analysis = fixture(tmp_path, oversized=False)
    # Retain a complete caller proof to demonstrate that the diagnostic format
    # remains leaf/segment evidence rather than claiming the requested entry.
    path.write_text(path.read_text().replace("call unavailable(a)\n", ""))
    analysis = SourceEffects([path])
    assert analysis.summarize("diagnostics::root")["complete"]
    report = analysis.report("diagnostics::root", materialize=False)
    assert {row["procedure"] for row in report["procedures"]} == {"diagnostics::leaf", "diagnostics::distant"}
    assert not report["closure_complete_available"]
    assert not report["complete"]


def test_lazy_report_preserves_explicitly_requested_reduction_diagnostics(tmp_path, monkeypatch):
    path = tmp_path / "reduction.f90"
    path.write_text("""module reductions
contains
subroutine root(a,total)
real(8),intent(in)::a(:)
real(8),intent(out)::total
total=maxval(a)
end subroutine
end module
""")
    analysis = SourceEffects([path])
    request = analysis.reduction_candidates("reductions::root")
    assert request["candidate_count"] == 1
    def forbid(*_args, **_kwargs):
        raise AssertionError("no unsolicited closure or reduction proof")
    monkeypatch.setattr(analysis, "summarize", forbid)
    monkeypatch.setattr(analysis, "reduction_candidates", forbid)
    report = analysis.report("reductions::root", materialize=False)
    assert report["source_reductions"]["analysis"] == "requested"
    assert report["source_reductions"]["requests"] == [request]
    assert report["openmp_reductions"]["analysis"] == "not_requested"


def test_lazy_report_invalidates_existing_facts_when_contract_authority_changes(tmp_path):
    _path, analysis = fixture(tmp_path)
    leaf = analysis.summarize("diagnostics::leaf")
    segment_node = walk(analysis.routines["diagnostics::leaf"].execution, F.Assignment_Stmt)[0]
    segment = analysis.segment_summary("diagnostics::leaf", (segment_node,))
    initial = analysis.report("diagnostics::root", materialize=False)
    assert len(initial["procedures"]) == 2
    analysis.contracts["library::new_contract"] = {"version": "new source-bound identity"}
    changed = analysis.report("diagnostics::root", materialize=False)
    assert changed["procedures"] == []
    assert changed["structured_effects"]["analysis_identity"] not in {
        leaf["analysis_identity"], segment["analysis_identity"]}


def test_lazy_report_rejects_changed_original_source(tmp_path):
    path, analysis = fixture(tmp_path)
    analysis.structure("diagnostics::root")
    path.write_text(path.read_text().replace("a=2*a", "a=3*a"))
    with pytest.raises(CompilationError, match="changed"):
        analysis.report("diagnostics::root", materialize=False)


def test_default_reporting_retains_legacy_payload_and_cache_accounting(tmp_path):
    path, _unused = fixture(tmp_path, oversized=False)
    first = SourceEffects([path], summary_cache=tmp_path / "implicit-cache")
    second = SourceEffects([path], summary_cache=tmp_path / "explicit-cache")
    implicit = first.report("diagnostics::leaf")
    explicit = second.report("diagnostics::leaf", materialize=True)
    assert implicit == explicit
    assert implicit["schema_version"] == 1
    assert implicit["complete"]
    assert "closure_materialization" not in implicit
    assert first._summary_cache.stats == second._summary_cache.stats


def test_lazy_report_never_exports_private_projection_proofs(tmp_path):
    _path, analysis = fixture(tmp_path)
    graph, identities = analysis._selected_source("diagnostics::leaf",
        analysis.routines["diagnostics::leaf"].execution.content)
    projected, _routine, _guard = analysis._segment_projection("diagnostics::leaf", identities)
    assert projected.summarize("diagnostics::leaf")["complete"]
    report = projected.report("diagnostics::leaf", materialize=False)
    assert not report["structured_effects"]["available"]
    assert not report["procedures"]
    assert not report["closure_complete_available"]


def test_reached_owner_generation_does_not_request_original_whole_entry_closure(tmp_path, monkeypatch):
    from compiler.tests.test_lexical_source_owner import emit
    original = SourceEffects.summarize
    def reached_only(self, requested, *args, **kwargs):
        if requested == "local_owner::step" and self._summary_authority()["role"] == "source":
            raise AssertionError("reached ownership must not materialize its whole original call closure")
        return original(self, requested, *args, **kwargs)
    monkeypatch.setattr(SourceEffects, "summarize", reached_only)
    _path, outputs, manifest = emit(tmp_path)
    assert manifest["scope_count"] == 1
    assert manifest["inline_numerical_regions"]["dispatcher"]
    assert manifest["native_effects"]["closure_materialization"] == "not_requested"
    assert not manifest["native_effects"]["closure_complete_available"]
    assert manifest["scopes"][0]["boundaries"]
    assert all(item["path"] in outputs for item in manifest["build_sources"])
