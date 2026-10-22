"""Original teams and exact native corrections share one cross-call owner."""

import json
import os
import re
from hashlib import sha256

import pytest

from compiler.tests.support.numerical_cuda import _transfers, build_reached_numerical_binaries
from compiler.tests.test_interval_draft import generate
from compiler.tests.test_interval_native_artifacts import cross_module_builder
from compiler.tests.test_source_scopes import run

DRIVER = """program verify
use iso_fortran_env,only:error_unit
use interval_fields
implicit none
integer::i,step,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do step=1,2
do i=-1,16
old(i)=real(i+7*step,8)*0.125d0
enddo
viscosity=-91.d0
star=-93.d0
out=-95.d0
write(error_unit,'(a,i0)')'BEGIN_INTERVAL ',step
call advance(16,step==2)
write(error_unit,'(a,i0)')'END_INTERVAL ',step
if(visits/=step) error stop 'native work replayed'
if(viscosity(-1)/=-91.d0.or.viscosity(16)/=-91.d0) error stop 'producer halo changed'
if(star(-1)/=-93.d0.or.star(16)/=-93.d0) error stop 'intermediate halo changed'
if(out(-1)/=-95.d0.or.out(16)/=-95.d0) error stop 'output halo changed'
write(unit)visits,lbound(old),ubound(old),old,viscosity,star,out
enddo
close(unit)
end program
"""


def emit(directory, *, precision, device_budget):
    assert precision == 8
    assert device_budget is None
    path, builder, outputs, report = generate(directory)
    assert report["scope_count"] == 1, report.get("lexical_owner_attempt")
    assert not report["scopes"][0]["boundaries"]
    return [path], outputs, report, builder


def emit_cross_module(directory, *, precision, device_budget):
    from compiler._legacy.config import LegacyOffloadConfig as OffloadConfig

    assert precision == 8
    assert device_budget is None
    sources, builder = cross_module_builder(directory, OffloadConfig(policy="sections", scope_execution="reached"))
    outputs, report = builder.run()
    assert report["scope_count"] == 1
    assert not report["scopes"][0]["boundaries"]
    return sources, outputs, report, builder


@pytest.mark.cuda
@pytest.mark.parametrize("emit_fixture", [emit, emit_cross_module], ids=["internal", "cross_module"])
def test_original_teams_share_storage_across_guarded_exact_native_correction(tmp_path_factory, emit_fixture):
    directory, candidate, native, _precision, _fallback, resources = build_reached_numerical_binaries(
        tmp_path_factory, 8, False, emit_fixture=emit_fixture, prefix="interval_draft_",
        driver_source=lambda _precision: DRIVER, expected_gpu_leaves=3, expected_capture_states={})
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"}
    environment.pop("FORT_RUNTIME_TRACE", None)
    run([str(native)], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    observed = run([str(candidate)], cwd=directory, env={**environment, "FORT_RUNTIME_TRACE": "1"})
    actual = (directory / "fields.bin").read_bytes()
    assert actual == expected
    traces = re.findall(r"BEGIN_INTERVAL (\d+)\n(.*?)END_INTERVAL \1\n", observed.stderr, re.DOTALL)
    assert [number for number, _trace in traces] == ["1", "2"]
    identities = {name.split("::")[-1]: item["registration_identity"] for name, item in resources.items()}
    ledger = []
    for number, trace in traces:
        assert trace.count("FORT_SCOPED launch ") == 3, trace
        assert trace.count("FORT_SCOPED close ") == 1, trace
        contexts = re.findall(r"FORT_SCOPED .*?\bcontext=(\S+)", trace)
        assert len(set(contexts)) == 1, trace
        traffic = _transfers(trace)
        assert dict(traffic["upload"]) == {
            identities["old"]: 128, identities["out"]: 128,
            **({identities["viscosity"]: 8} if number == "2" else {})}, trace
        assert dict(traffic["download"]) == {
            identities["viscosity"]: 128, identities["star"]: 128, identities["out"]: 128}, trace
        ledger.append({"invocation": int(number), "launches": 3,
                       "upload_bytes": dict(traffic["upload"]), "download_bytes": dict(traffic["download"])})
    (directory / "interval-transfer-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (directory / "interval-trace.log").write_text(observed.stderr)
    (directory / "native-fields.bin").write_bytes(expected)
    (directory / "resident-fields.bin").write_bytes(actual)
    for overrides in ({}, {"CUDA_VISIBLE_DEVICES": ""}, {"OMP_NUM_THREADS": "2"}):
        result = run([str(candidate)], cwd=directory, env={**environment, **overrides})
        assert (directory / "fields.bin").read_bytes() == expected
        assert "FORT_SCOPED " not in result.stderr
    (directory / "interval-result.json").write_text(json.dumps({
        "full_arrays_bounds_halos_and_native_visits": "bitwise_equal",
        "native_sha256": sha256(expected).hexdigest(), "resident_sha256": sha256(actual).hexdigest(),
        "matched_bytes": len(actual), "automatic_pricing_qualified": False,
        "disabled_device_and_wrong_team_fallback": "bitwise_equal"}, indent=2) + "\n")
