"""Original reached reads admit missing host sections after an opaque prefix."""

import json
import os
import re
from functools import partial
from hashlib import sha256

import pytest

from compiler.tests.support.numerical_cuda import _transfers, build_reached_numerical_binaries
from compiler.tests.test_native_source_reads import emit_native_metadata_fixture
from compiler.tests.test_selection_activation import emit_activation_fixture
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
write(error_unit,'(a,i0)')'BEGIN_ACTIVATION ',step
call advance(n,step==2)
write(error_unit,'(a,i0)')'END_ACTIVATION ',step
if(visits/=2*step) error stop 'native prefix or correction replayed'
do i=0,n-1
expected=3.d0*old(i)
if(step==2.and.i==0) expected=4.d0*old(i)
if(star(i)/=expected) error stop 'resident producer or diffusion changed'
if(i>=8) expected=expected-95.d0
if(out(i)/=expected) error stop 'entry coverage or accumulation changed'
enddo
if(viscosity(-1)/=-91.d0.or.viscosity(16)/=-91.d0) error stop 'producer halo changed'
if(star(-1)/=-93.d0.or.star(16)/=-93.d0) error stop 'intermediate halo changed'
if(out(-1)/=-95.d0.or.out(16)/=-95.d0) error stop 'output halo changed'
if(n==8.and.any(out(8:16)/=-95.d0)) error stop 'undefined outer sections changed'
write(unit)visits,lbound(old),ubound(old),lbound(viscosity),ubound(viscosity)
write(unit)lbound(star),ubound(star),lbound(out),ubound(out)
write(unit)old,viscosity,star,out
enddo
close(unit)
end program
"""


def emit(directory, *, precision, device_budget):
    assert precision == 8
    assert device_budget is None
    path = directory / "prefix.f90"
    builder, owner, outputs, report = emit_activation_fixture(path)
    assert owner.execution_plan.validate().numerical_invocations == 3
    assert builder.facts["captures"]["interval_fields::out"]["initialized"] == "none"
    return [path], outputs, report, builder


def emit_with_metadata(directory, *, precision, device_budget, protected=False):
    assert precision == 8
    assert device_budget is None
    path = directory / "prefix.f90"
    builder, owner, outputs, report = emit_native_metadata_fixture(path, protected=protected)
    assert owner.execution_plan.validate().numerical_invocations == 3
    return [path], outputs, report, builder


@pytest.mark.cuda
@pytest.mark.parametrize("metadata", [None, "demand", "precise"],
                         ids=["numerical_reads", "native_demand_reads", "native_precise_reads"])
def test_original_reads_survive_borrowing_and_retain_current_device_sections(tmp_path_factory, metadata):
    directory, candidate, native, _precision, _fallback, resources = build_reached_numerical_binaries(
        tmp_path_factory, 8, False,
        emit_fixture=partial(emit_with_metadata, protected=metadata == "precise") if metadata else emit,
        prefix="selection_activation_",
        driver_source=lambda _precision: DRIVER, expected_gpu_leaves=3,
        expected_capture_states={"interval_fields::out": "none",
                                 **({"interval_fields::tiles": "none"} if metadata else {})})
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"}
    environment.pop("FORT_RUNTIME_TRACE", None)
    run([str(native)], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate)], cwd=directory, env={**environment, "FORT_RUNTIME_TRACE": "1"})
    observed = (directory / "fields.bin").read_bytes()
    assert observed == expected
    traces = re.findall(r"BEGIN_ACTIVATION (\d+)\n(.*?)END_ACTIVATION \1\n", result.stderr, re.DOTALL)
    assert [number for number, _trace in traces] == ["1", "2", "3"]
    identities = {root.split("::")[-1]: item["registration_identity"] for root, item in resources.items()}
    ledger = []
    for (number, trace), launches in zip(traces, (3, 3, 3), strict=True):
        assert trace.count("FORT_SCOPED launch ") == launches, trace
        assert trace.count("FORT_SCOPED close ") == 1, trace
        traffic = _transfers(trace)
        size = 128 if number == "3" else 64
        expected_uploads = {identities["old"]: size, identities["out"]: size}
        expected_downloads = {identities[name]: size for name in ("viscosity", "star", "out")}
        if number == "2":
            # The original native point correction uses current host old,
            # publishes eight stale viscosity bytes, then uploads that update.
            expected_uploads[identities["viscosity"]] = 8
        assert dict(traffic["upload"]) == expected_uploads, trace
        assert dict(traffic["download"]) == expected_downloads, trace
        ledger.append({"invocation": int(number), "launches": launches,
                       "upload_bytes": dict(traffic["upload"]), "download_bytes": dict(traffic["download"]),
                       "predicted_upload_bytes": expected_uploads, "predicted_download_bytes": expected_downloads})
    for overrides in ({"CUDA_VISIBLE_DEVICES": ""}, {"OMP_NUM_THREADS": "2"}):
        fallback = run([str(candidate)], cwd=directory,
                       env={**environment, **overrides, "FORT_RUNTIME_TRACE": "1"})
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
        "launches_per_invocation": [3, 3, 3], "native_prefix_and_remaining_suffix_executed_once": True,
        "automatic_pricing_qualified": False, "performance_samples": 0,
    }, indent=2) + "\n")
