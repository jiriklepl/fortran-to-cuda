"""Independent diagnostics remain bounded and cannot authorize execution."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.preflight import preflight_scope, preflight_source_scopes
from compiler.scopes.source import ScopeBuilder

SOURCE = """module inspect_me
implicit none
contains
subroutine step(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
call opaque(a)
do i=1,n
a(i)=2*a(i)+1
enddo
call missing(a)
do i=1,n
a(i)=3*a(i)+2
enddo
end subroutine
end module
"""


def subject(tmp_path, source=SOURCE):
    path = tmp_path / "original.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": {"storage": "stable", "initialized": "whole",
                                         "escapes": False, "allocation_changes": False}}}
    builder = ScopeBuilder([path], "inspect_me::step", facts=facts, options=CompilerOptions(),
                           config=OffloadConfig("auto", scope_execution="reached"))
    return path, builder


def test_independent_failures_do_not_hide_later_numerical_or_cost_diagnostics(tmp_path):
    path, builder = subject(tmp_path)
    snapshot = preflight_scope(builder)
    report = snapshot.public()
    assert not report["execution_authorized"]
    assert not report["artifacts_emitted"]
    assert not report["truncated"]
    assert [row["kind"] for row in report["operations"]] == ["call", "loop", "call", "loop"]
    loops = [row for row in report["operations"] if row["kind"] == "loop"]
    assert all(row["numerical"]["status"] == "available" for row in loops)
    assert all(row["native"]["status"] == "available" for row in loops)
    assert all(row["cost"]["status"] == "unavailable" for row in loops)
    assert all("native Fortran compute calibration is missing" in row["cost"]["reasons"] for row in loops)
    assert any("opaque" in item["reason"] for item in report["blockers"])
    assert any("missing" in item["reason"] for item in report["blockers"])
    assert all(row["transfers"]["upload_bytes"] is None for row in report["operations"])
    assert report["scope_automatic_estimate"]["status"] == "not_requested"
    assert report["sources"][str(path)] == sha256(path.read_bytes()).hexdigest()
    assert not builder.outputs
    assert not builder.edits
    assert not builder.scopes
    assert not builder.generated
    report["operations"][0]["blockers"].clear()
    assert snapshot.public()["operations"][0]["blockers"]
    with pytest.raises(FrozenInstanceError):
        snapshot._json = "{}"


def test_branch_order_and_guards_preserved_without_evaluation(tmp_path):
    source = SOURCE.replace("call opaque(a)", "if(n>0)then").replace(
        "call missing(a)", "else").replace("end subroutine", "endif\nend subroutine")
    _path, builder = subject(tmp_path, source)
    report = preflight_scope(builder).public()
    assert [row["kind"] for row in report["operations"]] == ["condition", "loop", "loop"]
    assert not report["branch_predicates_evaluated"]
    assert report["operations"][1]["guard"] == ["n > 0"]
    assert report["operations"][2]["guard"] == [".not.(n > 0)"]
    assert report["operations"][1]["span"][0] < report["operations"][2]["span"][0]


@pytest.mark.parametrize(("opening", "closing"), [
    ("!$omp parallel do private(i)", "!$omp end parallel do"),
    ("!$omp parallel private(i)\n!$omp do", "!$omp end do\n!$omp end parallel"),
])
def test_whole_joined_team_keeps_completion_and_pricing_separate(tmp_path, opening, closing):
    source = SOURCE.replace("call opaque(a)", opening).replace("call missing(a)", closing)
    _path, builder = subject(tmp_path, source)
    report = preflight_scope(builder).public()
    team, loop = report["operations"]
    assert team["kind"] == "joined_team"
    assert team["completion"]["status"] == "available"
    assert team["numerical"]["status"] == "available"
    assert team["cost"]["status"] == "unavailable"
    assert team["cost"]["native_participation"] == "fork_join"
    assert team["span"][1] < loop["span"][0]


def test_local_callee_body_inspected_once_without_entry_effect_closure(tmp_path, monkeypatch):
    source = SOURCE.replace("call opaque(a)", "call child(a,n)").replace(
        "call missing(a)", "call child(a,n)").replace("end module", """subroutine child(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=a(i)+1
enddo
end subroutine
end module""")
    _path, builder = subject(tmp_path, source)
    original = type(builder.analysis).summarize

    def no_parent_closure(analysis, *args, **kwargs):
        assert analysis is not builder.analysis, "preflight requested the original complete closure"
        return original(analysis, *args, **kwargs)

    monkeypatch.setattr(type(builder.analysis), "summarize", no_parent_closure)
    report = preflight_scope(builder).public()
    assert len(report["procedures"]) == 2
    calls = [row for row in report["operations"] if row["kind"] == "call"]
    assert all(row["native"]["status"] == "not_requested" for row in calls)
    assert len(report["calls"]) == 2
    assert all(edge["body_inspection"] == "inspected" for edge in report["calls"])
    child, = [row for row in report["operations"] if row["procedure"] == "inspect_me::child"]
    assert child["resources"][0]["capture_facts"]["status"] == "not_requested"
    assert builder.analysis._closures == {}


def test_mutated_source_aborts_snapshot_instead_of_publishing_partial_diagnostics(tmp_path, monkeypatch):
    path, builder = subject(tmp_path)
    original = NumericalPreparationProvider.prepare

    def mutate(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        path.write_text(SOURCE + "! changed during diagnostics\n")
        return result

    monkeypatch.setattr(NumericalPreparationProvider, "prepare", mutate)
    with pytest.raises(CompilationError, match="changed|match"):
        preflight_scope(builder)


def test_shared_report_budget_is_explicit_and_does_not_raise_proof_limits(tmp_path):
    source = """module inspect_me
contains
subroutine step(a)
real(8)::a(:)
call child(a)
end subroutine
subroutine child(a)
real(8)::a(:)
call final(a)
end subroutine
subroutine final(a)
real(8)::a(:)
a(1)=1
end subroutine
end module
"""
    _path, builder = subject(tmp_path, source)
    builder.analysis.operation_limit = 2
    report = preflight_scope(builder).public()
    assert report["truncated"]
    assert report["limits"]["operations"] == 2
    assert len(report["operations"]) == 2
    assert report["calls"][-1]["body_inspection"] == "budget_limit"
    assert builder.analysis.operation_limit == 2


def test_cli_preflight_writes_no_output_directory(tmp_path):
    path, builder = subject(tmp_path)
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(builder.facts))
    destination = tmp_path / "must-not-be-created"
    result = subprocess.run([sys.executable, "-m", "compiler", "--form-scopes", "--scope-preflight",
        "--input", str(path), "--kernel", "inspect_me::step", "--scope-facts", str(facts_path),
        "--memory-model", "scoped", "--scope-execution", "reached", "--gpu-policy", "auto",
        "--output-dir", str(destination), "--json"], capture_output=True, text=True, check=True)
    report = json.loads(result.stdout)
    assert report["outputs"] == []
    assert not report["preflight"]["execution_authorized"]
    assert not destination.exists()


def test_source_hash_bound_facts_remain_mandatory(tmp_path):
    path, builder = subject(tmp_path)
    builder.facts["sources"][str(path)] = "0" * 64
    with pytest.raises(CompilationError, match="source hashes"):
        preflight_source_scopes([path], "inspect_me::step", facts=builder.facts,
                                options=builder.options, config=builder.config)


def test_progress_observer_gets_detached_provisional_rows_in_source_order(tmp_path):
    _path, builder = subject(tmp_path)
    events = []

    def observer(event):
        events.append(json.loads(json.dumps(event)))
        if "operation" in event:
            event["operation"]["span"].clear()
            event["operation"]["numerical"]["status"] = "forged_observation"

    report = preflight_scope(builder, observer=observer).public()
    assert events[0]["event"] == "procedure_started"
    complete = [event for event in events if event["event"] == "operation_complete"]
    starts = [event for event in events if event["event"] == "operation_started"]
    assert [event["operation"]["ordinal"] for event in complete] == [0, 1, 2, 3]
    assert len(starts) == len(complete)
    assert all(event["provisional"] for event in events)
    assert all(row["span"] for row in report["operations"])
    assert all(row["numerical"]["status"] != "forged_observation" for row in report["operations"])


def test_source_mutation_by_progress_observer_prevents_final_snapshot(tmp_path):
    path, builder = subject(tmp_path)

    def observer(event):
        if event["event"] == "operation_complete" and event["operation"]["ordinal"] == 3:
            path.write_text(SOURCE + "! callback changed input\n")

    with pytest.raises(CompilationError, match="changed|match"):
        preflight_scope(builder, observer=observer)
