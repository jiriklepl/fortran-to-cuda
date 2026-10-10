"""Shared numerical artifacts preserve each original participation contract."""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_compute_costs import profile_v2
from compiler.tests.test_source_scopes import FACT


def prepare(tmp_path, body, *, profile=None):
    source = """module renamed_dispatch
implicit none
contains
subroutine evaluate(a,b,n)
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:)
integer,intent(in)::n
integer::i
""" + body + "end subroutine\nend module\n"
    path = tmp_path / "renamed_dispatch.f90"
    path.write_text(source)
    builder = ScopeBuilder([path], "renamed_dispatch::evaluate", options=CompilerOptions(),
        config=OffloadConfig(policy="auto", profile=profile_v2() if profile is None else profile), facts={
            "schema_version": 1, "participation": "serial",
            "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
            "captures": {"argument::a": FACT, "argument::b": FACT}})
    inline = builder.inline
    inline.prepare(tuple(builder.entry.execution.content))
    return inline


@pytest.mark.parametrize("joined_first", [False, True])
def test_equal_serial_and_joined_work_keeps_distinct_native_models(tmp_path, joined_first):
    loop = "do i=1,n\na(i)=2*b(i)\nenddo\n"
    joined = "!$omp parallel do private(i)\n" + loop + "!$omp end parallel do\n"
    body = joined + loop + joined if joined_first else loop + joined + joined
    inline = prepare(tmp_path, body)
    serial, teams = [], []
    for procedure, extraction in inline.regions.items():
        original_joined = extraction.completion.get("has_openmp_in_closure", False)
        (teams if original_joined else serial).append(procedure)
        generated = inline.generated[procedure].scoped
        participation = "fork_join" if original_joined else "serial"
        assert generated["compute_estimates"]["native_participation"] == participation
        unit, = generated["planning"]["units"]
        assert unit["compute_model"]["native_fortran"]["backend_identity"] == "native_" + participation
    serial_procedure, = serial
    first_team, second_team = teams
    assert inline.entries[first_team] == inline.entries[second_team]
    assert inline.generated[first_team] is inline.generated[second_team]
    assert inline.entries[serial_procedure] != inline.entries[first_team]
    assert inline.generated[serial_procedure] is not inline.generated[first_team]


@pytest.mark.parametrize("runtime_first", [False, True])
def test_equal_static_and_runtime_scheduled_loops_do_not_share_cost_authority(tmp_path, runtime_first):
    loop = "do i=1,n\na(i)=2*b(i)\nenddo\n"
    static = "!$omp parallel do private(i) schedule(static)\n" + loop + "!$omp end parallel do\n"
    runtime = static.replace("schedule(static)", "schedule(runtime)")
    body = runtime + static + static if runtime_first else static + runtime + static
    inline = prepare(tmp_path, body)
    known, unknown = [], []
    for procedure, extraction in inline.regions.items():
        assert extraction.completion["available"]
        assert extraction.completion["has_openmp_in_closure"]
        original_runtime = "schedule(runtime)" in "\n".join(map(str, extraction.nodes)).lower()
        (unknown if original_runtime else known).append(procedure)
        generated = inline.generated[procedure].scoped
        unit, = generated["planning"]["units"]
        if original_runtime:
            assert generated["compute_estimates"]["native_participation"] == "fork_join_runtime"
            assert unit["compute_model"] is None
            assert not generated["automatic_estimate_available"]
            assert "runtime schedule validation is missing" in generated["automatic_reason"]
        else:
            assert generated["compute_estimates"]["native_participation"] == "fork_join"
            assert unit["compute_model"]["native_fortran"]["backend_identity"] == "native_fork_join"
    runtime_procedure, = unknown
    first_static, second_static = known
    assert inline.entries[first_static] == inline.entries[second_static]
    assert inline.entries[runtime_procedure] != inline.entries[first_static]


def test_accepted_runtime_evidence_exposes_per_query_requirement_and_range(tmp_path, monkeypatch):
    from compiler.tests.test_schedule_calibration import schedule_profile

    body = """!$omp parallel do private(i) schedule(runtime)
do i=1,n
a(i)=2*b(i)
enddo
!$omp end parallel do
"""
    inline = prepare(tmp_path, body, profile=schedule_profile(monkeypatch))
    generated, = inline.generated.values()
    public = generated.scoped
    assert public["compute_estimates"]["native_participation"] == "fork_join_runtime"
    unit, = public["planning"]["units"]
    assert unit["compute_model"]["runtime_schedule"] == {"kind": "static", "chunk": 0}
    assert unit["compute_model"]["item_range"] == [131072, 1048576]
    assert public["automatic_estimate_available"]
    source = next(text for name, text in generated.artifacts.items()
                  if name.endswith(".cu") and "record_compute(" in text)
    assert "omp_get_schedule" in source
