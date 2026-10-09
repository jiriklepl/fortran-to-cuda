"""Reusable effects retain source, configuration and capture-proof authority."""

from __future__ import annotations

import copy
from hashlib import sha256

import pytest

import compiler.frontend.source_effects as source_effects
from compiler.frontend.source_effects import SOURCE_SUMMARY_VERSION, SourceEffects
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError

ENTRY = "runner::advance"


@pytest.fixture(autouse=True)
def _fresh_cache(monkeypatch):
    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def graph(tmp_path):
    leaf = write(tmp_path, "operators.f90", """module operators
contains
subroutine adjust(a)
real(8),intent(inout)::a(:)
a=a+1
end subroutine
subroutine unused(a)
real(8),intent(inout)::a(:)
a=4*a
end subroutine
end module
""")
    branches = write(tmp_path, "branches.f90", """module branches
use operators,only:renamed=>adjust
contains
subroutine left(a)
real(8),intent(inout)::a(:)
call renamed(a)
end subroutine
subroutine right(a)
real(8),intent(inout)::a(:)
call renamed(a)
end subroutine
end module
""")
    root = write(tmp_path, "runner.f90", """module runner
use branches,only:left,right
contains
subroutine advance(a)
real(8),intent(inout)::a(:)
call left(a)
call right(a)
end subroutine
end module
""")
    return [root, branches, leaf]


def records(report):
    return {item["procedure"]: item for item in report["procedures"]}


def configured(paths, tmp_path):
    entries = []
    for source in paths:
        prepared = write(tmp_path, "prepared_" + source.name, source.read_text())
        entries.append({"source": str(source), "path": str(prepared),
                        "sha256": sha256(prepared.read_bytes()).hexdigest(),
                        "line_map": list(range(1, len(prepared.read_text().splitlines()) + 1))})
    dependency = write(tmp_path, "configuration.inc", "integer,parameter::option=1\n")
    return {"schema_version": 1, "source_inputs": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
            "preserves_source_order": True, "configuration": {"defines": ["FIRST_CONFIGURATION"]},
            "dependencies": {str(dependency): sha256(dependency.read_bytes()).hexdigest()}, "entries": entries}


def test_independent_analyzers_reuse_complete_diamond_and_repeated_leaf(tmp_path):
    paths = graph(tmp_path)
    original = SourceEffects(paths).report(ENTRY)
    assert original["complete"], original
    assert original["summary_version"] == SOURCE_SUMMARY_VERSION
    assert original["summary_cache"]["hits"] == 0
    imported = SourceEffects(paths).report(ENTRY)
    assert imported["complete"], imported
    assert imported["summary_cache"]["memory_hits"] == 1
    assert imported["summarized_operations"] == original["summarized_operations"]
    assert [item["procedure"] for item in imported["procedures"]] == [item["procedure"] for item in original["procedures"]]
    for name, summary in records(imported).items():
        assert summary["summary_identity"] == records(original)[name]["summary_identity"]
        assert len(summary["summary_identity"]) == 64
        assert summary["analysis_identity"] == records(original)[name]["analysis_identity"]
        for operation in summary["operations"]:
            if operation["kind"] == "call":
                assert operation["summary_identity"] == records(imported)[operation["procedure"]]["summary_identity"]
    leaf = SourceEffects(paths).report("operators::adjust")
    assert leaf["summary_cache"]["memory_hits"] == 1
    assert [item["procedure"] for item in leaf["procedures"]] == ["operators::adjust"]


def test_explicit_disk_cache_reuses_across_independent_instances(tmp_path):
    paths = graph(tmp_path)
    directory = tmp_path / "summary_cache"
    first = SourceEffects(paths, summary_cache=directory).report(ENTRY)
    second = SourceEffects(paths, summary_cache=directory).report(ENTRY)
    assert first["complete"]
    assert second["complete"]
    assert second["summary_cache"]["disk_hits"] == 1
    assert second["summary_cache"]["memory_hits"] == 0
    assert records(second)[ENTRY]["summary_identity"] == records(first)[ENTRY]["summary_identity"]


def test_changed_source_invalidates_reuse_and_old_analysis_cannot_skip_verification(tmp_path):
    paths = graph(tmp_path)
    analysis = SourceEffects(paths)
    before = analysis.report(ENTRY)
    paths[-1].write_text(paths[-1].read_text().replace("a=a+1", "a=a*2"))
    after = SourceEffects(paths).report(ENTRY)
    assert after["complete"]
    assert after["summary_cache"]["hits"] == 0
    assert records(after)[ENTRY]["analysis_identity"] != records(before)[ENTRY]["analysis_identity"]
    assert records(after)["operators::adjust"]["summary_identity"] != records(before)["operators::adjust"]["summary_identity"]
    with pytest.raises(CompilationError, match="source changed"):
        analysis.report(ENTRY)


@pytest.mark.parametrize("change", ["configuration", "dependency", "prepared", "mapping"])
def test_configured_source_identity_includes_configuration_dependency_prepared_text_and_mapping(tmp_path, change):
    paths = graph(tmp_path)
    document = configured(paths, tmp_path)
    before = SourceEffects(paths, analysis_sources=document).report(ENTRY)
    changed = copy.deepcopy(document)
    if change == "configuration":
        changed["configuration"]["defines"] = ["SECOND_CONFIGURATION"]
    elif change == "dependency":
        dependency = tmp_path / "configuration.inc"
        dependency.write_text("integer,parameter::option=2\n")
        changed["dependencies"][str(dependency)] = sha256(dependency.read_bytes()).hexdigest()
    elif change == "prepared":
        prepared = tmp_path / "prepared_operators.f90"
        prepared.write_text(prepared.read_text().replace("a=a+1", "a=a+2"))
        changed["entries"][-1]["sha256"] = sha256(prepared.read_bytes()).hexdigest()
    else:
        changed["entries"][0]["line_map"][-1] = None
    after = SourceEffects(paths, analysis_sources=changed).report(ENTRY)
    assert after["complete"], after
    assert after["summary_cache"]["hits"] == 0
    assert records(after)[ENTRY]["analysis_identity"] != records(before)[ENTRY]["analysis_identity"]


def test_configured_document_is_snapshotted_before_later_caller_mutation(tmp_path):
    paths = graph(tmp_path)
    document = configured(paths, tmp_path)
    original = copy.deepcopy(document)
    analysis = SourceEffects(paths, analysis_sources=document)
    document["configuration"]["defines"].append("UNRELATED_CHANGE")
    document["entries"][0]["line_map"].clear()
    document["dependencies"].clear()
    first = analysis.report(ENTRY)
    second = SourceEffects(paths, analysis_sources=original).report(ENTRY)
    assert second["summary_cache"]["hits"] == 1
    assert first["analysis_sources"]["configuration"] == original["configuration"]
    assert records(first)[ENTRY]["analysis_identity"] == records(second)[ENTRY]["analysis_identity"]


def contract_source(tmp_path):
    return write(tmp_path, "contract.f90", """module contract_caller
use opaque_library,only:adjust
contains
subroutine advance(a)
real(8),intent(inout)::a(:)
call adjust(a)
end subroutine
end module
""")


def contract():
    return {"opaque_library::adjust": {"identity": "library-v1", "lifetime": "stable", "escapes": False,
            "ordering": "serial", "complete": True, "descriptor_changes": False,
            "effects": [{"kind": "write", "argument": 0, "section": "whole"}]}}


def test_contract_identity_changes_invalidate_shared_and_local_closures(tmp_path):
    path = contract_source(tmp_path)
    facts = contract()
    analysis = SourceEffects([path], contracts=facts)
    facts["opaque_library::adjust"]["identity"] = "caller-mutation"
    first = analysis.report("contract_caller::advance")
    same = SourceEffects([path], contracts=contract()).report("contract_caller::advance")
    assert first["complete"]
    assert same["summary_cache"]["hits"] == 1
    assert records(first)["contract_caller::advance"]["summary_identity"] == records(same)["contract_caller::advance"]["summary_identity"]
    analysis.contracts["opaque_library::adjust"]["identity"] = "library-v2"
    changed = analysis.report("contract_caller::advance")
    assert changed["complete"]
    assert records(changed)["contract_caller::advance"]["analysis_identity"] != records(first)["contract_caller::advance"]["analysis_identity"]
    new_facts = contract()
    new_facts["opaque_library::adjust"]["identity"] = "library-v3"
    different = SourceEffects([path], contracts=new_facts).report("contract_caller::advance")
    assert different["summary_cache"]["hits"] == 0


def test_capture_authorization_identity_prevents_borrowing_an_unproved_allocation(tmp_path):
    path = write(tmp_path, "storage.f90", """module storage
real(8),allocatable::scratch(:),other(:)
contains
subroutine inspect(a)
real(8),intent(inout)::a(:)
a(1)=scratch(1)
end subroutine
end module
""")
    authorized = SourceEffects([path])
    authorized.authorize_stable_module_allocatables({"storage::scratch"})
    first = authorized.report("storage::inspect")
    again = SourceEffects([path])
    again.authorize_stable_module_allocatables({"storage::scratch"})
    reused = again.report("storage::inspect")
    assert reused["complete"]
    assert reused["summary_cache"]["hits"] == 1
    unproved = SourceEffects([path]).report("storage::inspect")
    assert not unproved["complete"]
    assert unproved["summary_cache"]["hits"] == 0
    assert records(unproved)["storage::inspect"]["capture_lifetime_requirements"] == [
        {"resource": "storage::scratch", "authorized": False}]
    changed = SourceEffects([path])
    changed.authorize_stable_module_allocatables({"storage::scratch", "storage::other"})
    different = changed.report("storage::inspect")
    assert different["complete"]
    assert different["summary_cache"]["hits"] == 0
    assert records(first)["storage::inspect"]["analysis_identity"] != records(different)["storage::inspect"]["analysis_identity"]


def test_returned_report_mutation_does_not_modify_local_or_shared_authority(tmp_path):
    paths = graph(tmp_path)
    analysis = SourceEffects(paths)
    first = analysis.report(ENTRY)
    expected = records(first)[ENTRY]["summary_identity"]
    first["sources"].clear()
    first["budgets"]["depth"] = 1000
    first["procedures"][0]["complete"] = False
    first["procedures"][0]["operations"].clear()
    local = analysis.report(ENTRY)
    imported = SourceEffects(paths).report(ENTRY)
    assert local["complete"]
    assert imported["complete"]
    assert len(local["sources"]) == 3
    assert records(local)[ENTRY]["summary_identity"] == expected
    assert records(imported)[ENTRY]["summary_identity"] == expected
    assert local["budgets"]["depth"] == 8


def test_private_execution_and_intent_projection_cannot_borrow_whole_source_cache(tmp_path):
    path = write(tmp_path, "projection.f90", """module projection
contains
subroutine advance(a)
real(8),intent(out)::a(:)
a=1
a=a+1
end subroutine
end module
""")
    original = SourceEffects([path]).report("projection::advance")
    assert original["complete"]
    analysis = SourceEffects([path])
    whole = analysis.report("projection::advance")
    assert whole["summary_cache"]["hits"] == 1
    routine = analysis.routines["projection::advance"]
    routine.execution = copy.copy(routine.execution)
    routine.execution.content = list(routine.execution.content[1:])
    routine.scope.bindings["a"] = copy.copy(routine.scope.bindings["a"])
    routine.scope.bindings["a"].intent = None
    projected = analysis.report("projection::advance")
    summary = records(projected)["projection::advance"]
    assert projected["complete"], projected
    assert summary["summary_role"] == "private_projection"
    assert summary["definition_changes"] == []
    assert [operation["kind"] for operation in summary["operations"]] == ["read", "overwrite"]
    assert summary["analysis_identity"] != records(original)["projection::advance"]["analysis_identity"]
    assert projected["summary_cache"]["hits"] == whole["summary_cache"]["hits"]
    source_again = SourceEffects([path]).report("projection::advance")
    assert records(source_again)["projection::advance"]["summary_identity"] == records(original)["projection::advance"]["summary_identity"]


def seed_closure(analysis, report, *, extra=None):
    summaries = copy.deepcopy(records(report))
    if extra is not None:
        summaries[extra["procedure"]] = copy.deepcopy(extra)
    refreshed = set()

    def refresh(name):
        if name in refreshed:
            return
        summary = summaries[name]
        for operation in summary["operations"]:
            if operation["kind"] == "call":
                refresh(operation["procedure"])
                operation["summary_identity"] = summaries[operation["procedure"]]["summary_identity"]
        analysis._stamp_summary(summary)
        refreshed.add(name)

    for name in summaries:
        refresh(name)
    authority = analysis._summary_authority()
    analysis._summary_cache.store(authority, ENTRY, {"summaries": summaries, "order": list(summaries),
        "operations": sum(len(summary["operations"]) for summary in summaries.values())})


@pytest.mark.parametrize("budget", [{"depth": 2}, {"procedures": 3}, {"operations": 5}])
def test_cached_complete_closure_is_re_admitted_under_every_budget(tmp_path, budget):
    paths = graph(tmp_path)
    high = SourceEffects(paths).report(ENTRY)
    assert high["complete"]
    low = SourceEffects(paths, **budget)
    seed_closure(low, high)
    report = low.report(ENTRY)
    assert not report["complete"]
    assert report["summary_cache"]["rejected"] >= 1
    assert records(report)[ENTRY]["reasons"]


def test_cached_closure_cannot_import_unreachable_complete_procedures(tmp_path):
    paths = graph(tmp_path)
    original = SourceEffects(paths).report(ENTRY)
    unused = SourceEffects(paths).report("operators::unused")["procedures"][0]
    analysis = SourceEffects(paths)
    seed_closure(analysis, original, extra=unused)
    report = analysis.report(ENTRY)
    assert report["complete"]
    assert report["summary_cache"]["rejected"] >= 1
    assert "operators::unused" not in records(report)
