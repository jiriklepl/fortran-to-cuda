"""A late original descriptor mismatch publishes the GPU prefix without replay."""

import json
import os
import re
from hashlib import sha256

import pytest

from compiler.tests.support.numerical_cuda import _transfers, build_reached_numerical_binaries
from compiler.tests.test_ordinary_numerical_regions import emit_ordinary_numerical_fixture
from compiler.tests.test_source_scopes import run


def emit_descriptor_fixture(directory, *, precision=8, device_budget=None):
    return emit_ordinary_numerical_fixture(directory, precision=precision, device_budget=device_budget,
        captures={"argument::c": {"expected_descriptor": {"lower_bounds": [-4], "extents": [13]}}})


def descriptor_driver(precision):
    return f"""program verify
use iso_fortran_env,only:error_unit
use quartz_owner,only:advance,prefix_visits,tail_visits
use quartz_numerics,only:bridge_visits,middle_visits
implicit none
real({precision}),allocatable::a(:),b(:),c(:)
integer::extra,repeat,i,unit
character(16)::path
call get_command_argument(1,path)
if(path/='matching'.and.path/='mismatch') error stop 'unknown path'
extra=0
if(path=='mismatch') extra=1
allocate(a(-11:1),b(-14:-2),c(-18:-6+extra))
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do repeat=1,2
do i=lbound(a,1),ubound(a,1)
a(i)=real(i+11+2*repeat,{precision})*0.125_{precision}
enddo
b=-77._{precision}
c=-91._{precision}
write(error_unit,'(a,i0)') 'BEGIN_CALL ',repeat
call advance(a,b,c,7,3)
write(error_unit,'(a,i0)') 'END_CALL ',repeat
if(prefix_visits/=repeat.or.bridge_visits/=repeat.or.middle_visits/=repeat.or.tail_visits/=repeat) &
error stop 'original continuation replayed'
if(lbound(a,1)/=-11.or.ubound(a,1)/=1) error stop 'input bounds changed'
if(lbound(b,1)/=-14.or.ubound(b,1)/=-2) error stop 'producer bounds changed'
if(lbound(c,1)/=-18.or.ubound(c,1)/=-6+extra) error stop 'consumer bounds changed'
if(b(-14)/=-77._{precision}.or.b(-2)/=-77._{precision}) error stop 'producer halo changed'
if(c(-18)/=-91._{precision}.or.c(-6)/=-91._{precision}) error stop 'consumer halo changed'
if(extra==1) then
if(c(-5)/=-91._{precision}) error stop 'unread consumer exterior changed'
endif
write(unit)repeat,extra,prefix_visits,bridge_visits,middle_visits,tail_visits
write(unit)lbound(a),ubound(a),lbound(b),ubound(b),lbound(c),ubound(c)
write(unit)a,b,c
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=[4, 8])
def descriptor_binaries(tmp_path_factory, request):
    return build_reached_numerical_binaries(tmp_path_factory, request.param, False,
        emit_fixture=emit_descriptor_fixture, driver_source=descriptor_driver,
        prefix="capture_descriptor_", expected_gpu_leaves=2)


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("path", ["matching", "mismatch"])
def test_original_descriptor_match_and_late_mismatch_preserve_complete_fields(descriptor_binaries, path, disabled):
    directory, candidate, native, precision, _resource_fallback, resources = descriptor_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    reference = run([str(native), path], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate), path], cwd=directory, env=environment)
    actual = (directory / "fields.bin").read_bytes()
    label = path + ("-disabled" if disabled else "-cuda")
    (directory / (label + "-native.bin")).write_bytes(expected)
    (directory / (label + "-candidate.bin")).write_bytes(actual)
    (directory / (label + ".stderr")).write_text(result.stderr)
    assert actual == expected
    assert "array temporary" not in (reference.stderr + result.stderr).lower()
    pattern = r"BEGIN_CALL (\d+)\n(.*?)END_CALL \1\n"
    calls = re.findall(pattern, result.stderr, re.DOTALL)
    assert [int(ordinal) for ordinal, _trace in calls] == [1, 2]
    event_prefixes = ("FORT_SCOPED launch ", "FORT_SCOPED upload ", "FORT_SCOPED download ",
                      "FORT_SCOPED create ", "FORT_SCOPED close ")
    assert not any(line.startswith(event_prefixes)
                   for line in re.sub(pattern, "", result.stderr, flags=re.DOTALL).splitlines())
    ids = {root: item["registration_identity"] for root, item in resources.items()}
    launches = 0 if disabled else 3 if path == "matching" else 2
    for _ordinal, trace in calls:
        events = [line for line in trace.splitlines() if line.startswith(event_prefixes)]
        assert trace.count("FORT_SCOPED launch ") == launches
        if disabled:
            assert not any(line.startswith(event_prefixes[:3]) for line in events)
            continue
        assert trace.count("FORT_SCOPED create ") == 1
        assert trace.count("FORT_SCOPED close ") == 1
        assert trace.rindex("FORT_SCOPED launch ") < trace.index("FORT_SCOPED close ")
        assert not any(line.startswith(event_prefixes[:3])
                       for line in trace[trace.index("FORT_SCOPED close "):].splitlines())
        contexts = [re.findall(r"\bcontext=(\S+)", event) for event in events]
        assert all(len(context) == 1 for context in contexts)
        assert len({context[0] for context in contexts}) == 1
        transfers = _transfers(trace)
        expected_uploads = {ids["argument::a"]: 11 * precision,
                            ids["argument::b"]: (12 if path == "matching" else 11) * precision,
                            ids["quartz_numerics::coefficient"]: 2 * precision}
        expected_downloads = {ids["argument::b"]: 11 * precision}
        if path == "matching":
            expected_downloads[ids["argument::c"]] = 11 * precision
        assert dict(transfers["upload"]) == expected_uploads
        assert dict(transfers["download"]) == expected_downloads
    assert result.stderr.count("FORT_SCOPED launch ") == 2 * launches
    (directory / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "precision_bytes": precision, "path": path, "cuda_disabled": disabled,
        "expected_descriptor": {"lower_bounds": [-4], "extents": [13]},
        "actual_consumer_extent": 13 if path == "matching" else 14,
        "consumer_initial_definition": "none", "complete_fields_halos_bounds_and_visits_equal": True,
        "native_sha256": sha256(expected).hexdigest(), "candidate_sha256": sha256(actual).hexdigest(),
        "launches_per_call": launches, "one_numerical_context_per_call": not disabled,
        "performance_samples": 0,
    }, indent=2, sort_keys=True) + "\n")
