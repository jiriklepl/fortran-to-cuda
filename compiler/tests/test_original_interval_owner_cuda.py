"""Repeated original intervals retain mixed-team corrections exactly once."""

import json
import os
import re

import pytest

from compiler.tests.support.numerical_cuda import _transfers, build_reached_numerical_binaries
from compiler.tests.test_interval_draft_cuda import DRIVER
from compiler.tests.test_original_interval_owner import emit_absorbing_interval, emit_repeated_interval
from compiler.tests.test_source_scopes import run


def emit(directory, *, precision, device_budget):
    assert precision == 8
    assert device_budget is None
    path = directory / "owner.f90"
    builder, _owner, outputs, report = emit_repeated_interval(path)
    return [path], outputs, report, builder


ABSORBING_DRIVER = """program verify
use iso_fortran_env,only:error_unit
use interval_fields
implicit none
integer::i,step,path,unit,ordinal,expected_bad
allocate(scratch(0:0))
scratch=0.d0
ordinal=0
expected_bad=0
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do path=0,2
do step=1,2
ordinal=ordinal+1
do i=-1,16
old(i)=real(i+7*step,8)*0.125d0
enddo
viscosity=-91.d0
star=-93.d0
out=-95.d0
write(error_unit,'(a,i0)')'BEGIN_INTERVAL ',ordinal
call advance(16,path/=0,path==1)
write(error_unit,'(a,i0)')'END_INTERVAL ',ordinal
if(path/=0) expected_bad=expected_bad+1
if(visits/=ordinal.or.bad_visits/=expected_bad) error stop 'native work replayed'
if(scratch(0)/=real(expected_bad,8)) error stop 'opaque native update replayed'
if(viscosity(-1)/=-91.d0.or.viscosity(16)/=-91.d0) error stop 'producer halo changed'
if(star(-1)/=-93.d0.or.star(16)/=-93.d0) error stop 'intermediate halo changed'
if(out(-1)/=-95.d0.or.out(16)/=-95.d0) error stop 'output halo changed'
write(unit)visits,bad_visits,lbound(old),ubound(old),old,viscosity,star,out,scratch
enddo
enddo
close(unit)
deallocate(scratch)
end program
"""


def emit_absorbing(directory, *, precision, device_budget):
    assert precision == 8
    assert device_budget is None
    path = directory / "owner.f90"
    builder, _owner, outputs, report = emit_absorbing_interval(path)
    return [path], outputs, report, builder


@pytest.mark.cuda
def test_absorbing_close_before_or_after_gpu_prefix_never_replays_or_reopens(tmp_path_factory):
    directory, candidate, native, _precision, _fallback, resources = build_reached_numerical_binaries(
        tmp_path_factory, 8, False, emit_fixture=emit_absorbing, prefix="absorbing_interval_",
        driver_source=lambda _precision: ABSORBING_DRIVER, expected_gpu_leaves=3, expected_capture_states={})
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"}
    environment.pop("FORT_RUNTIME_TRACE", None)
    run([str(native)], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    observed = run([str(candidate)], cwd=directory, env={**environment, "FORT_RUNTIME_TRACE": "1"})
    assert (directory / "fields.bin").read_bytes() == expected
    traces = re.findall(r"BEGIN_INTERVAL (\d+)\n(.*?)END_INTERVAL \1\n", observed.stderr, re.DOTALL)
    assert [number for number, _trace in traces] == list(map(str, range(1, 7)))
    ledger = []
    for (number, trace), launches in zip(traces, (3, 3, 0, 0, 1, 1), strict=True):
        assert trace.count("FORT_SCOPED launch ") == launches, trace
        assert trace.count("FORT_SCOPED close ") == int(launches > 0), trace
        assert len(set(re.findall(r"FORT_SCOPED .*?\bcontext=(\S+)", trace))) == int(launches > 0), trace
        traffic = _transfers(trace)
        assert bool(traffic["upload"]) == bool(launches)
        assert bool(traffic["download"]) == bool(launches)
        if launches == 1:
            viscosity = resources["interval_fields::viscosity"]["registration_identity"]
            assert dict(traffic["download"]) == {viscosity: 128}
        ledger.append({"invocation": int(number), "launches": launches,
                       "upload_bytes": dict(traffic["upload"]), "download_bytes": dict(traffic["download"])})
    (directory / "transfer-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (directory / "trace.log").write_text(observed.stderr)
    (directory / "native-fields.bin").write_bytes(expected)
    for overrides in ({"CUDA_VISIBLE_DEVICES": ""}, {"OMP_NUM_THREADS": "2"}):
        run([str(candidate)], cwd=directory, env={**environment, **overrides})
        assert (directory / "fields.bin").read_bytes() == expected


@pytest.mark.cuda
def test_each_original_iteration_owns_and_publishes_its_mixed_window(tmp_path_factory):
    driver = DRIVER.replace("if(visits/=step)", "if(visits/=2*step)")
    directory, candidate, native, _precision, _fallback, resources = build_reached_numerical_binaries(
        tmp_path_factory, 8, False, emit_fixture=emit, prefix="original_interval_",
        driver_source=lambda _precision: driver, expected_gpu_leaves=3, expected_capture_states={})
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"}
    environment.pop("FORT_RUNTIME_TRACE", None)
    run([str(native)], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    observed = run([str(candidate)], cwd=directory, env={**environment, "FORT_RUNTIME_TRACE": "1"})
    assert (directory / "fields.bin").read_bytes() == expected
    traces = re.findall(r"BEGIN_INTERVAL (\d+)\n(.*?)END_INTERVAL \1\n", observed.stderr, re.DOTALL)
    assert [number for number, _trace in traces] == ["1", "2"]
    ledger = []
    for number, trace in traces:
        assert trace.count("FORT_SCOPED launch ") == 6, trace
        assert trace.count("FORT_SCOPED close ") == 2, trace
        assert len(set(re.findall(r"FORT_SCOPED .*?\bcontext=(\S+)", trace))) == 2, trace
        traffic = _transfers(trace)
        assert sum(traffic["upload"].values()) > 0
        assert sum(traffic["download"].values()) > 0
        ledger.append({"invocation": int(number), "launches": 6, "closed_intervals": 2,
                       "upload_bytes": dict(traffic["upload"]), "download_bytes": dict(traffic["download"])})
    assert len(resources) == 4
    (directory / "transfer-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (directory / "trace.log").write_text(observed.stderr)
    (directory / "native-fields.bin").write_bytes(expected)
    for overrides in ({"CUDA_VISIBLE_DEVICES": ""}, {"OMP_NUM_THREADS": "2"}):
        run([str(candidate)], cwd=directory, env={**environment, **overrides})
        assert (directory / "fields.bin").read_bytes() == expected
