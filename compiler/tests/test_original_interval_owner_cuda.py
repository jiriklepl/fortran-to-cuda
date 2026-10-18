"""Repeated original intervals retain mixed-team corrections exactly once."""

import json
import os
import re

import pytest

from compiler.tests.support.numerical_cuda import _transfers, build_reached_numerical_binaries
from compiler.tests.test_interval_draft_cuda import DRIVER
from compiler.tests.test_original_interval_owner import emit_repeated_interval
from compiler.tests.test_source_scopes import run


def emit(directory, *, precision, device_budget):
    assert precision == 8
    assert device_budget is None
    path = directory / "owner.f90"
    builder, _owner, outputs, report = emit_repeated_interval(path)
    return [path], outputs, report, builder


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
