"""Partial input coverage survives GPU selection and closes native suffixes."""

import json
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.tests.support.numerical_cuda import _transfers, build_reached_numerical_binaries
from compiler.tests.test_joined_deferred_definitions import emit_partial_joined_fixture
from compiler.tests.test_ordinary_numerical_regions_cuda import ORDINARY_VOTE_PROBE
from compiler.tests.test_source_scopes import run


def partial_driver(precision):
    return f"""program verify
use iso_fortran_env,only:error_unit
use quartz_owner,only:advance,prefix_visits,tail_visits
use quartz_numerics,only:bridge_visits,middle_visits
implicit none
real({precision}),allocatable::a(:),b(:),c(:)
integer::repeat,i,unit,tile
character(16)::path
call get_command_argument(1,path)
if(path/='eligible'.and.path/='negative'.and.path/='overflow') error stop 'unknown path'
tile=3
if(path=='negative') tile=-3
! The original sole positive tile and its next DO index fit INTEGER32.
if(path=='overflow') tile=2147483641
allocate(a(-11:1),b(-14:-2),c(-18:-6))
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do repeat=1,2
! Exterior sentinels are defined for full-field comparison only; their
! coverage is not admitted by the partial input capture fact.
do i=lbound(a,1),ubound(a,1)
a(i)=real(i+11+2*repeat,{precision})*0.125_{precision}
enddo
b=-77._{precision}
c=-91._{precision}
write(error_unit,'(a,i0)') 'BEGIN_CALL ',repeat
call advance(a,b,c,7,tile)
write(error_unit,'(a,i0)') 'END_CALL ',repeat
if(prefix_visits/=repeat.or.bridge_visits/=repeat.or.middle_visits/=repeat.or.tail_visits/=repeat) &
error stop 'original continuation replayed'
if(lbound(a,1)/=-11.or.ubound(a,1)/=1) error stop 'input bounds changed'
if(lbound(b,1)/=-14.or.ubound(b,1)/=-2) error stop 'producer bounds changed'
if(lbound(c,1)/=-18.or.ubound(c,1)/=-6) error stop 'consumer bounds changed'
if(b(-14)/=-77._{precision}.or.b(-2)/=-77._{precision}) error stop 'producer halo changed'
if(c(-18)/=-91._{precision}.or.c(-6)/=-91._{precision}) error stop 'consumer halo changed'
write(unit)repeat,tile,prefix_visits,bridge_visits,middle_visits,tail_visits
write(unit)lbound(a),ubound(a),lbound(b),ubound(b),lbound(c),ubound(c)
write(unit)a,b,c
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=[4, 8])
def partial_binaries(tmp_path_factory, request):
    return build_reached_numerical_binaries(tmp_path_factory, request.param, False,
        emit_fixture=emit_partial_joined_fixture, driver_source=partial_driver,
        prefix="joined_partial_", expected_gpu_leaves=3)


def compare_partial(binaries, path, disabled=False, *, binary=None, veto=False):
    directory, candidate, native, precision, _fallback, resources = binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    if veto:
        environment["FORT_TEST_TEAM_VETO_LATE"] = "1"
    reference = run([str(native), path], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(binary or candidate), path], cwd=directory, env=environment)
    actual = (directory / "fields.bin").read_bytes()
    label = path + ("-veto" if veto else "-disabled" if disabled else "-cuda")
    (directory / (label + "-native.bin")).write_bytes(expected)
    (directory / (label + "-candidate.bin")).write_bytes(actual)
    (directory / (label + ".stderr")).write_text(result.stderr)
    assert actual == expected
    assert "array temporary" not in (reference.stderr + result.stderr).lower()
    pattern = r"BEGIN_CALL (\d+)\n(.*?)END_CALL \1\n"
    calls = re.findall(pattern, result.stderr, re.DOTALL)
    assert [int(ordinal) for ordinal, _trace in calls] == [1, 2]
    event_prefixes = ("FORT_SCOPED launch ", "FORT_SCOPED upload ", "FORT_SCOPED download ")
    assert not any(line.startswith(event_prefixes)
                   for line in re.sub(pattern, "", result.stderr, flags=re.DOTALL).splitlines())
    ids = {root: item["registration_identity"] for root, item in resources.items()}
    eligible = path == "eligible" and not veto
    launches = 0 if disabled else 3 if eligible else 1
    for _ordinal, trace in calls:
        assert trace.count("FORT_SCOPED launch ") == launches
        transfers = _transfers(trace)
        if disabled:
            assert not any(transfers.values())
            continue
        assert trace.count("FORT_SCOPED create ") == 1
        assert trace.count("FORT_SCOPED close ") == 1
        close = trace.index("FORT_SCOPED close ")
        assert trace.rindex("FORT_SCOPED launch ") < close
        assert not any(line.startswith(event_prefixes) for line in trace[close:].splitlines())
        contexts = re.findall(r"^FORT_SCOPED (?:launch|upload|download|create|close) .*?\bcontext=(\S+)",
                              trace, re.MULTILINE)
        assert contexts
        assert len(set(contexts)) == 1
        assert dict(transfers["upload"]) == {
            ids["argument::a"]: 11 * precision,
            ids["argument::b"]: (22 if eligible else 11) * precision,
            ids["quartz_numerics::coefficient"]: (2 if eligible else 1) * precision}
        expected_down = {ids["argument::b"]: 11 * precision}
        if eligible:
            expected_down[ids["argument::c"]] = 11 * precision
        assert dict(transfers["download"]) == expected_down
        if not eligible:
            assert "uninitialized_read" in trace
    (directory / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "precision_bytes": precision, "path": path, "cuda_disabled": disabled,
        "actual_participant_veto": veto, "input_defined_physical_section": {"lower": [1], "upper": [12]},
        "complete_fields_halos_bounds_and_original_visits_equal": True,
        "native_sha256": sha256(expected).hexdigest(), "candidate_sha256": sha256(actual).hexdigest(),
        "launches_per_call": launches, "performance_samples": 0,
    }, indent=2, sort_keys=True) + "\n")
    return calls


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("path", ["eligible", "negative", "overflow"])
def test_partial_input_and_late_tile_rejection_preserve_original_suffix(partial_binaries, path, disabled):
    compare_partial(partial_binaries, path, disabled)


@pytest.fixture(scope="module")
def partial_vote_binary(partial_binaries):
    directory, candidate, *_rest = partial_binaries
    source, target = directory / "vote-probe.cpp", directory / "vote-probe.o"
    # Existing probe changes one actual participant's rounding only during
    # every second reached numerical preflight, restoring it before native work.
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
        "source_sha256": sha256(ORDINARY_VOTE_PROBE.encode()).hexdigest(),
        "test_only_fault_injection": True, "original_candidate_unchanged": True,
        "commands": [compile_command, link_command],
    }, indent=2) + "\n")
    return wrapped


@pytest.mark.cuda
def test_late_participant_veto_closes_before_native_partial_input_read(partial_binaries, partial_vote_binary):
    calls = compare_partial(partial_binaries, "eligible", binary=partial_vote_binary, veto=True)
    for _ordinal, trace in calls:
        vote_pattern = r"TEST_ACTUAL_TEAM_VOTE thread=(\d+) level=(\d+) supported=(\d+)"
        launch, close = trace.index("FORT_SCOPED launch "), trace.index("FORT_SCOPED close ")
        # JoinedTeam probes every participant once per reached numerical DO.
        # The veto closes ownership before the third DO can probe or execute.
        assert sorted(re.findall(vote_pattern, trace[:launch])) == [
            (str(thread), "1", "1") for thread in range(4)]
        assert sorted(re.findall(vote_pattern, trace[launch:close])) == [
            ("0", "1", "1"), ("1", "1", "1"), ("2", "1", "1"), ("3", "1", "0")]
        assert not re.findall(vote_pattern, trace[close:])
