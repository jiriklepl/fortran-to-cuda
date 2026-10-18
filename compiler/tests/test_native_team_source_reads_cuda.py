"""Native indirect halo corrections retain current GPU sections and exact fallback."""

import json
import os
import re
from functools import partial
from hashlib import sha256

import pytest

from compiler.tests.support.numerical_cuda import _transfers, build_reached_numerical_binaries
from compiler.tests.test_native_team_source_reads import emit_native_team_fixture
from compiler.tests.test_source_scopes import run

DRIVER = """program verify
use iso_fortran_env,only:error_unit
use interval_fields
implicit none
integer::i,step,n,unit
real(8)::expected
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do step=1,3
n=8
if(step==3) n=16
do i=-1,16
old(i)=real(i+7*step,8)*0.125d0
enddo
viscosity=-91.d0
star=-93.d0
out=-95.d0
write(error_unit,'(a,i0)')'BEGIN_NATIVE_TEAM ',step
call advance(n,step==2)
write(error_unit,'(a,i0)')'END_NATIVE_TEAM ',step
if(visits/=2*step) error stop 'native prefix or remaining suffix replayed'
do i=0,n-1
expected=3.d0*old(i)
if(step==2.and.i==0) expected=CORRECTED
if(star(i)/=expected) error stop 'diffusion or native correction changed'
if(i>=8) expected=expected-95.d0
if(out(i)/=expected) error stop 'accumulation or entry state changed'
enddo
expected=-93.d0
if(step==2) expected=HALO
if(star(-1)/=expected) error stop 'native halo correction changed'
if(star(16)/=-93.d0) error stop 'untouched intermediate halo changed'
if(viscosity(-1)/=-91.d0.or.viscosity(16)/=-91.d0) error stop 'producer halo changed'
if(out(-1)/=-95.d0.or.out(16)/=-95.d0) error stop 'output halo changed'
if(n==8.and.any(out(8:16)/=-95.d0)) error stop 'unread outer sections changed'
write(unit)visits,lbound(old),ubound(old),lbound(viscosity),ubound(viscosity)
write(unit)lbound(star),ubound(star),lbound(out),ubound(out)
write(unit)old,viscosity,star,out
enddo
close(unit)
end program
"""


def emit(directory, *, precision, device_budget, read_modify_write):
    assert precision == 8
    assert device_budget is None
    path = directory / "native_team.f90"
    selected, owner, outputs, report = emit_native_team_fixture(path, read_modify_write=read_modify_write)
    assert owner.execution_plan.validate().numerical_invocations == 3
    return [path], outputs, report, selected


@pytest.mark.cuda
@pytest.mark.parametrize("read_modify_write", [True, False], ids=["resident_correction", "unsupported_entry_reads"])
def test_native_team_halo_reads_and_suffix_fallback_execute_once(tmp_path_factory, read_modify_write):
    driver = DRIVER.replace("CORRECTED", "4.d0*old(i)" if read_modify_write else "old(i)")
    driver = driver.replace("HALO", "-93.d0+old(-1)" if read_modify_write else "old(-1)")
    directory, candidate, native, _precision, _fallback, resources = build_reached_numerical_binaries(
        tmp_path_factory, 8, False, emit_fixture=partial(emit, read_modify_write=read_modify_write),
        prefix="native_team_source_reads_", driver_source=lambda _precision: driver, expected_gpu_leaves=3,
        expected_capture_states={"interval_fields::out": "none"})
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"}
    environment.pop("FORT_RUNTIME_TRACE", None)
    run([str(native)], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate)], cwd=directory, env={**environment, "FORT_RUNTIME_TRACE": "1"})
    observed = (directory / "fields.bin").read_bytes()
    assert observed == expected
    traces = re.findall(r"BEGIN_NATIVE_TEAM (\d+)\n(.*?)END_NATIVE_TEAM \1\n", result.stderr, re.DOTALL)
    assert [number for number, _ in traces] == ["1", "2", "3"]
    identities = {root.split("::")[-1]: item["registration_identity"] for root, item in resources.items()}
    ledger = []
    for number, trace in traces:
        nbytes = 128 if number == "3" else 64
        prefix_fallback = number == "2" and not read_modify_write
        launches = 2 if prefix_fallback else 3
        assert trace.count("FORT_SCOPED launch ") == launches, trace
        assert trace.count("FORT_SCOPED close ") == 1, trace
        transfers = _transfers(trace)
        uploads = {identities["old"]: nbytes}
        downloads = {identities["viscosity"]: nbytes, identities["star"]: nbytes}
        if not prefix_fallback:
            uploads[identities["out"]] = nbytes
            downloads[identities["out"]] = nbytes
            if number == "2":
                uploads[identities["star"]] = 8
        assert dict(transfers["upload"]) == uploads, trace
        assert dict(transfers["download"]) == downloads, trace
        ledger.append({"invocation": int(number), "launches": launches, "prefix_fallback": prefix_fallback,
                       "observed_upload_bytes": dict(transfers["upload"]), "predicted_upload_bytes": uploads,
                       "observed_download_bytes": dict(transfers["download"]), "predicted_download_bytes": downloads})
    for overrides in ({"CUDA_VISIBLE_DEVICES": ""}, {"OMP_NUM_THREADS": "2"}):
        fallback = run([str(candidate)], cwd=directory, env={**environment, **overrides, "FORT_RUNTIME_TRACE": "1"})
        assert (directory / "fields.bin").read_bytes() == expected
        if "CUDA_VISIBLE_DEVICES" in overrides:
            assert "FORT_SCOPED launch " not in fallback.stderr
    (directory / "trace.log").write_text(result.stderr)
    (directory / "transfer-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (directory / "native-fields.bin").write_bytes(expected)
    (directory / "resident-fields.bin").write_bytes(observed)
    (directory / "result.json").write_text(json.dumps({
        "full_float64_arrays_bounds_and_halos": "bitwise_equal",
        "native_sha256": sha256(expected).hexdigest(), "resident_sha256": sha256(observed).hexdigest(),
        "native_prefix_and_remaining_suffix_executed_once": True,
        "native_entry_read_proof_available": read_modify_write,
        "automatic_pricing_qualified": False, "performance_samples": 0,
    }, indent=2) + "\n")
