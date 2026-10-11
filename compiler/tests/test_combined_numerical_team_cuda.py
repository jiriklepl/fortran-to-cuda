"""Actual original teams coordinate prepared producer/native-face chains."""

import json
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.tests.test_combined_numerical_team import emit_combined_team_fixture
from compiler.tests.test_reached_numerical_effects_cuda import (
    build_reached_numerical_binaries,
    check_prepared_chain,
)
from compiler.tests.test_source_scopes import run

VOTE_PROBE = r"""#include <cfenv>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <omp.h>
extern "C" int __real_fort_scope_numerical_environment_supported();
extern "C" int __real_fort_scope_plan_reset_mode(std::int64_t, int);
extern "C" int __wrap_fort_scope_plan_reset_mode(std::int64_t context, int mode) {
    // Earlier serial segments also reset plans. Exercise this original team's
    // pre-execution failure path, rather than closing its owner beforehand.
    if (!std::getenv("FORT_TEST_PLAN_RESET_FAILURE") || omp_get_level() != 1)
        return __real_fort_scope_plan_reset_mode(context, mode);
    std::fprintf(stderr, "TEST_PLAN_RESET_FAILURE level=%d\n", omp_get_level());
    return 1; // FORT_SCOPE_ARGUMENT: no query may overwrite this failure.
}
extern "C" int __wrap_fort_scope_numerical_environment_supported() {
    if (!std::getenv("FORT_TEST_TEAM_VETO") || omp_get_level() != 1)
        return __real_fort_scope_numerical_environment_supported();
    const int thread = omp_get_thread_num(), saved = std::fegetround();
    const bool veto = thread == omp_get_num_threads()-1;
    if (saved < 0 || (veto && std::fesetround(FE_UPWARD))) std::abort();
    const int supported = __real_fort_scope_numerical_environment_supported();
    if (veto && std::fesetround(saved)) std::abort();
    std::fprintf(stderr, "TEST_ACTUAL_TEAM_VOTE thread=%d level=%d supported=%d\n", thread, omp_get_level(), supported);
    return supported;
}
"""


@pytest.fixture(scope="module", params=((4, False), (4, True), (8, False), (8, True)))
def combined_numerical_binaries(tmp_path_factory, request):
    precision, resource_fallback = request.param
    return build_reached_numerical_binaries(tmp_path_factory, precision, resource_fallback,
        emit_fixture=emit_combined_team_fixture, prefix="combined_numerical_")


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("path", ["eligible", "negative", "overflow", "empty", "late"])
def test_original_team_retains_resident_chain_without_replay(combined_numerical_binaries, disabled, path):
    check_prepared_chain(combined_numerical_binaries, disabled, path)


@pytest.mark.cuda
def test_wrong_original_team_budget_closes_before_native_once(combined_numerical_binaries):
    directory, candidate, native, precision, resource_fallback, _resources = combined_numerical_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "2", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    reference = run([str(native), "eligible"], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate), "eligible"], cwd=directory, env=environment)
    actual = (directory / "fields.bin").read_bytes()
    assert actual == expected
    assert "array temporary" not in (reference.stderr + result.stderr).lower()
    assert not any(line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED upload ", "FORT_SCOPED download "))
                   for line in result.stderr.splitlines())
    label = "wrong-team-budget"
    (directory / (label + "-native.bin")).write_bytes(expected)
    (directory / (label + "-candidate.bin")).write_bytes(actual)
    (directory / (label + ".stderr")).write_text(result.stderr)
    (directory / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "precision_bytes": precision, "host_threads": 2,
        "required_host_threads": 4, "resource_preexecution_fallback": resource_fallback,
        "complete_fields_halos_bounds_and_original_visits_equal": True,
        "cuda_launches_and_transfers": 0, "native_sha256": sha256(expected).hexdigest(),
        "candidate_sha256": sha256(actual).hexdigest(), "performance_samples": 0,
    }, indent=2, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def combined_vote_binary(combined_numerical_binaries):
    directory, candidate, *_rest = combined_numerical_binaries
    source, target = directory / "vote-probe.cpp", directory / "vote-probe.o"
    source.write_text(VOTE_PROBE)
    host = shutil.which("g++-14") or shutil.which("g++")
    assert host
    commands = json.loads((directory / "commands.json").read_text())
    compile_command = [host, "-O2", "-std=c++17", "-fopenmp", "-c", str(source), "-o", str(target)]
    run(compile_command, cwd=directory)
    original_link, = [item["argv"] for item in commands if item["role"] == "candidate-link"]
    wrapped = directory / "candidate-vote-probe"
    assert original_link[-2:] == ["-o", str(candidate)]
    link_command = [*original_link[:-2], str(target),
                    "-Wl,--wrap=fort_scope_numerical_environment_supported",
                    "-Wl,--wrap=fort_scope_plan_reset_mode", "-o", str(wrapped)]
    run(link_command, cwd=directory)
    (directory / "vote-probe-commands.json").write_text(json.dumps({
        "source_sha256": sha256(VOTE_PROBE.encode()).hexdigest(),
        "test_only_fault_injection": True, "original_candidate_unchanged": True,
        "commands": [compile_command, link_command],
    }, indent=2) + "\n")
    return wrapped


@pytest.mark.cuda
def test_one_actual_participant_vetoes_before_numerical_execution(
        combined_numerical_binaries, combined_vote_binary):
    directory, _candidate, native, precision, resource_fallback, _resources = combined_numerical_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1",
                   "FORT_TEST_TEAM_VETO": "1"}
    run([str(native), "eligible"], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(combined_vote_binary), "eligible"], cwd=directory, env=environment)
    actual = (directory / "fields.bin").read_bytes()
    assert actual == expected
    calls = re.findall(r"BEGIN_CALL (\d+)\n(.*?)END_CALL \1\n", result.stderr, re.DOTALL)
    assert [int(ordinal) for ordinal, _trace in calls] == list(range(1, 7))
    for _ordinal, trace in calls:
        votes = re.findall(r"TEST_ACTUAL_TEAM_VOTE thread=(\d+) level=(\d+) supported=(\d+)", trace)
        assert sorted(votes) == [("0", "1", "1"), ("1", "1", "1"), ("2", "1", "1"), ("3", "1", "0")]
    assert not any(line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED upload ", "FORT_SCOPED download "))
                   for line in result.stderr.splitlines())
    label = "actual-participant-veto"
    (directory / (label + "-native.bin")).write_bytes(expected)
    (directory / (label + "-candidate.bin")).write_bytes(actual)
    (directory / (label + ".stderr")).write_text(result.stderr)
    (directory / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "precision_bytes": precision, "resource_preexecution_fallback": resource_fallback,
        "test_only_linker_probe": "unsupported rounding during one actual original participant's check; restored before body",
        "sampled_or_additional_team": False, "actual_participant_votes_per_call": 4, "calls": 6,
        "complete_fields_halos_bounds_and_original_visits_equal": True, "cuda_launches_and_transfers": 0,
        "native_sha256": sha256(expected).hexdigest(), "candidate_sha256": sha256(actual).hexdigest(),
        "performance_samples": 0,
    }, indent=2, sort_keys=True) + "\n")


@pytest.mark.cuda
def test_plan_reset_failure_closes_before_original_body_once(
        combined_numerical_binaries, combined_vote_binary):
    directory, _candidate, native, precision, resource_fallback, _resources = combined_numerical_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1",
                   "FORT_TEST_PLAN_RESET_FAILURE": "1"}
    run([str(native), "eligible"], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(combined_vote_binary), "eligible"], cwd=directory, env=environment)
    actual = (directory / "fields.bin").read_bytes()
    assert actual == expected
    calls = re.findall(r"BEGIN_CALL (\d+)\n(.*?)END_CALL \1\n", result.stderr, re.DOTALL)
    assert [int(ordinal) for ordinal, _trace in calls] == list(range(1, 7))
    assert all(re.findall(r"TEST_PLAN_RESET_FAILURE level=(\d+)", trace) == ["1"] for _ordinal, trace in calls)
    assert not any(line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED upload ", "FORT_SCOPED download "))
                   for line in result.stderr.splitlines())
    label = "plan-reset-failure"
    (directory / (label + "-native.bin")).write_bytes(expected)
    (directory / (label + "-candidate.bin")).write_bytes(actual)
    (directory / (label + ".stderr")).write_text(result.stderr)
    (directory / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "precision_bytes": precision, "resource_preexecution_fallback": resource_fallback,
        "test_only_linker_probe": "preexecution plan reset returns FORT_SCOPE_ARGUMENT without changing context",
        "actual_original_team_level": 1, "reset_failures": 6,
        "complete_fields_halos_bounds_and_original_visits_equal": True, "cuda_launches_and_transfers": 0,
        "native_sha256": sha256(expected).hexdigest(), "candidate_sha256": sha256(actual).hexdigest(),
        "performance_samples": 0,
    }, indent=2, sort_keys=True) + "\n")
