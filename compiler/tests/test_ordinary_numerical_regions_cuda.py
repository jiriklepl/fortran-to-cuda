"""Ordinary lexical owners retain ordered whole-team numerical work."""

import json
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.tests.support.numerical_cuda import _transfers, build_reached_numerical_binaries
from compiler.tests.test_combined_numerical_team_cuda import VOTE_PROBE
from compiler.tests.test_ordinary_numerical_regions import emit_ordinary_numerical_fixture
from compiler.tests.test_source_scopes import run

ORDINARY_VOTE_PROBE = VOTE_PROBE.replace("#include <cfenv>", "#include <atomic>\n#include <cfenv>").replace(
    'if (!std::getenv("FORT_TEST_TEAM_VETO") || omp_get_level() != 1)',
    'if ((!std::getenv("FORT_TEST_TEAM_VETO") && !std::getenv("FORT_TEST_TEAM_VETO_LATE")) || omp_get_level() != 1)',
).replace(
    "const bool veto = thread == omp_get_num_threads()-1;",
    'static std::atomic<unsigned> votes{0};\n'
    '    // Only the last participant advances this count. The master also\n'
    '    // checks its environment while preparing a numerical query.\n'
    '    const bool veto = thread == omp_get_num_threads()-1 &&\n'
    '        (!std::getenv("FORT_TEST_TEAM_VETO_LATE") || votes.fetch_add(1) % 2 == 1);',
)


@pytest.fixture(scope="module", params=((4, False), (4, True), (8, False), (8, True)))
def ordinary_team_binaries(tmp_path_factory, request):
    precision, resource_fallback = request.param
    return build_reached_numerical_binaries(tmp_path_factory, precision, resource_fallback,
        emit_fixture=emit_ordinary_numerical_fixture, prefix="ordinary_team_", expected_gpu_leaves=2)


def _compare(binaries, *, path="eligible", disabled=False, threads=4, binary=None, extra=None, label=None):
    directory, candidate, native, precision, resource_fallback, resources = binaries
    environment = {**os.environ, "OMP_NUM_THREADS": str(threads), "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1",
                   **(extra or {})}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    reference = run([str(native), path], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(binary or candidate), path], cwd=directory, env=environment)
    actual = (directory / "fields.bin").read_bytes()
    label = label or path + ("-disabled" if disabled else "-cuda")
    (directory / (label + "-native.bin")).write_bytes(expected)
    (directory / (label + "-candidate.bin")).write_bytes(actual)
    (directory / (label + ".stderr")).write_text(result.stderr)
    assert actual == expected
    assert "array temporary" not in (reference.stderr + result.stderr).lower()
    calls = re.findall(r"BEGIN_CALL (\d+)\n(.*?)END_CALL \1\n", result.stderr, re.DOTALL)
    assert [int(ordinal) for ordinal, _trace in calls] == list(range(1, 7 if path == "eligible" else 3))
    (directory / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "precision_bytes": precision, "host_threads": threads,
        "resource_preexecution_fallback": resource_fallback,
        "complete_fields_halos_bounds_and_original_visits_equal": True,
        "native_sha256": sha256(expected).hexdigest(), "candidate_sha256": sha256(actual).hexdigest(),
        "launches": result.stderr.count("FORT_SCOPED launch "), "performance_samples": 0,
    }, indent=2, sort_keys=True) + "\n")
    return calls, resources, result.stderr


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("path", ["eligible", "empty"])
def test_ordinary_whole_groups_keep_order_and_complete_fields(ordinary_team_binaries, disabled, path):
    calls, resources, trace = _compare(ordinary_team_binaries, disabled=disabled, path=path)
    _directory, _candidate, _native, precision, resource_fallback, _resources = ordinary_team_binaries
    numerical = path == "eligible" and not disabled and not resource_fallback
    ids = {root: item["registration_identity"] for root, item in resources.items()}
    for ordinal, call_trace in calls:
        assert call_trace.count("FORT_SCOPED launch ") == (3 if numerical else 0)
        transfers = _transfers(call_trace)
        if not numerical:
            assert not any(transfers.values())
            continue
        contexts = re.findall(r"^FORT_SCOPED (?:launch|upload|download) .*?\bcontext=(\S+)", call_trace, re.MULTILINE)
        assert contexts
        assert len(set(contexts)) == 1
        cells = (4, 4, 11, 11, 4100, 4100)[int(ordinal) - 1]
        assert transfers["upload"][ids["argument::a"]] == cells * precision
        assert transfers["upload"][ids["argument::b"]] == (cells + 1) * precision
        assert transfers["upload"][ids["argument::c"]] == 0
        assert transfers["upload"][ids["quartz_numerics::coefficient"]] == 2 * precision
        assert sum(transfers["upload"].values()) == (2 * cells + 3) * precision
        assert transfers["download"][ids["argument::a"]] == 0
        assert transfers["download"][ids["argument::b"]] == cells * precision
        assert transfers["download"][ids["argument::c"]] == cells * precision
    assert trace.count("FORT_SCOPED launch ") == (18 if numerical else 0)


@pytest.fixture(scope="module")
def ordinary_vote_binary(ordinary_team_binaries):
    directory, candidate, *_rest = ordinary_team_binaries
    source, target = directory / "vote-probe.cpp", directory / "vote-probe.o"
    # The source probe restores the original rounding mode before the native body.
    source.write_text(ORDINARY_VOTE_PROBE)
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
        "source_sha256": sha256(ORDINARY_VOTE_PROBE.encode()).hexdigest(), "test_only_fault_injection": True,
        "commands": [compile_command, link_command],
    }, indent=2) + "\n")
    return wrapped


@pytest.mark.cuda
@pytest.mark.parametrize("failure", ["participant", "reset", "team-budget"])
def test_ordinary_preexecution_failure_runs_original_group_once(
        ordinary_team_binaries, ordinary_vote_binary, failure):
    extra = ({"FORT_TEST_TEAM_VETO": "1"} if failure == "participant" else
             {"FORT_TEST_PLAN_RESET_FAILURE": "1"} if failure == "reset" else {})
    calls, _resources, trace = _compare(ordinary_team_binaries, binary=ordinary_vote_binary,
        threads=2 if failure == "team-budget" else 4, extra=extra, label="fallback-" + failure)
    assert not any(line.startswith(("FORT_SCOPED launch ", "FORT_SCOPED upload ", "FORT_SCOPED download "))
                   for line in trace.splitlines())
    if failure == "participant":
        for _ordinal, call_trace in calls:
            votes = re.findall(r"TEST_ACTUAL_TEAM_VOTE thread=(\d+) level=(\d+) supported=(\d+)", call_trace)
            assert sorted(votes) == [("0", "1", "1"), ("1", "1", "1"), ("2", "1", "1"), ("3", "1", "0")]
    if failure == "reset":
        assert all(re.findall(r"TEST_PLAN_RESET_FAILURE level=(\d+)", call_trace) == ["1"]
                   for _ordinal, call_trace in calls)


@pytest.mark.cuda
def test_late_team_veto_publishes_gpu_prefix_without_replay(ordinary_team_binaries, ordinary_vote_binary):
    _directory, _candidate, _native, precision, resource_fallback, _resources = ordinary_team_binaries
    if resource_fallback:
        pytest.skip("late veto requires an executed GPU prefix; resource fallback has its own controls")
    calls, resources, trace = _compare(ordinary_team_binaries, binary=ordinary_vote_binary,
        extra={"FORT_TEST_TEAM_VETO_LATE": "1"}, label="late-participant-veto")
    ids = {root: item["registration_identity"] for root, item in resources.items()}
    for ordinal, call_trace in calls:
        votes = re.findall(r"TEST_ACTUAL_TEAM_VOTE thread=(\d+) level=(\d+) supported=(\d+)", call_trace)
        # Eight actual participant votes plus the first group's master query
        # recheck. Only the second actual last participant vetoes execution.
        assert len(votes) == 9
        assert [(level, supported) for thread, level, supported in votes if thread == "3"] == [("1", "1"), ("1", "0")]
        assert sum(supported == "0" for _thread, _level, supported in votes) == 1
        assert {level for _thread, level, _supported in votes} == {"1"}
        assert call_trace.count("FORT_SCOPED launch ") == 2
        transfers = _transfers(call_trace)
        cells = (4, 4, 11, 11, 4100, 4100)[int(ordinal) - 1]
        assert transfers["upload"][ids["argument::b"]] == cells * precision
        assert transfers["download"][ids["argument::b"]] == cells * precision
        assert transfers["download"][ids["argument::c"]] == 0
        assert transfers["upload"][ids["argument::c"]] == 0
    assert trace.count("FORT_SCOPED launch ") == 12
