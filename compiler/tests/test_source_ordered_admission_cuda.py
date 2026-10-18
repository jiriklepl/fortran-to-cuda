"""A declined original child closes only its reached path, without replay."""

import json
import os
import re
from hashlib import sha256

import pytest

from compiler.tests.support.numerical_cuda import _transfers, build_reached_numerical_binaries
from compiler.tests.test_source_ordered_admission import SOURCE, fixture
from compiler.tests.test_source_scopes import run

# Make both the GPU prefix and retained native tail observably non-idempotent.
# The tail still contains a supported numerical anchor, but source-ordered
# admission must retain its original CALL after the preceding declined child.
EXECUTION_SOURCE = SOURCE.replace(
    "integer::native_visits=0", "integer::native_visits=0,tail_visits=0"
).replace(
    "v(i)=2.d0*a(i)", "v(i)=v(i)+2.d0*a(i)"
).replace(
    "s(i)=s(i)+a(i)\nenddo\n!$omp end parallel do\nend subroutine",
    "s(i)=s(i)+a(i)\nenddo\n!$omp end parallel do\ntail_visits=tail_visits+1\nend subroutine",
)

DRIVER = """program verify
use iso_fortran_env,only:error_unit
use ordered_fields
implicit none
integer::i,step,unit,expected_visits
logical::unsupported
real(8)::expected_star,total_old
expected_visits=0
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do step=1,4
unsupported=step==2.or.step==3
do i=-1,16
old(i)=real(i+7*step,8)*0.125d0
enddo
viscosity=-91.d0
star=-93.d0
out=-95.d0
total_old=sum(old)
write(error_unit,'(a,i0)')'BEGIN_ORDERED ',step
call advance(16,unsupported)
write(error_unit,'(a,i0)')'END_ORDERED ',step
if(unsupported) expected_visits=expected_visits+1
if(native_visits/=expected_visits.or.tail_visits/=expected_visits) error stop 'native suffix replayed'
do i=0,15
if(viscosity(i)/=-91.d0+2.d0*old(i)) error stop 'producer replayed'
if(unsupported) then
expected_star=-93.d0+total_old+old(i)
else
expected_star=-91.d0+3.d0*old(i)
endif
if(star(i)/=expected_star) error stop 'child or native tail replayed'
if(out(i)/=-95.d0+expected_star) error stop 'accumulation replayed'
enddo
if(viscosity(-1)/=-91.d0.or.viscosity(16)/=-91.d0) error stop 'producer halo changed'
if(star(-1)/=-93.d0.or.star(16)/=-93.d0) error stop 'intermediate halo changed'
if(out(-1)/=-95.d0.or.out(16)/=-95.d0) error stop 'output halo changed'
write(unit)step,native_visits,tail_visits
write(unit)lbound(old),ubound(old),lbound(viscosity),ubound(viscosity)
write(unit)lbound(star),ubound(star),lbound(out),ubound(out)
write(unit)old,viscosity,star,out
enddo
close(unit)
end program
"""


def emit(directory, *, precision, device_budget):
    assert precision == 8
    assert device_budget is None
    path, builder, owner, originals = fixture(directory, source=EXECUTION_SOURCE)
    assert owner.run(originals)
    outputs, report = builder.finish()
    assert report["scope_count"] == 1
    assert owner.execution_plan.validate().numerical_invocations == 3
    assert {member.routine.qualified for member in owner.execution_plan.members} == {
        "ordered_fields::advance", "ordered_fields::producer",
        "ordered_fields::diffusion", "ordered_fields::accumulate",
    }
    suffix, = owner.closed_suffixes
    assert [str(node.items[0]).lower() for node in suffix.validate(owner).originals] == ["closed_tail"]
    return [path], outputs, report, builder


@pytest.mark.cuda
def test_declined_child_and_native_tail_execute_once_without_losing_sibling_gpu_work(tmp_path_factory):
    directory, candidate, native, _precision, _fallback, resources = build_reached_numerical_binaries(
        tmp_path_factory, 8, False, emit_fixture=emit, prefix="source_ordered_admission_",
        driver_source=lambda _precision: DRIVER, expected_gpu_leaves=3, expected_capture_states={})
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"}
    environment.pop("FORT_RUNTIME_TRACE", None)
    reference = run([str(native)], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate)], cwd=directory, env={**environment, "FORT_RUNTIME_TRACE": "1"})
    observed = (directory / "fields.bin").read_bytes()
    assert observed == expected
    assert "array temporary" not in (reference.stderr + result.stderr).lower()
    traces = re.findall(r"BEGIN_ORDERED (\d+)\n(.*?)END_ORDERED \1\n", result.stderr, re.DOTALL)
    assert [int(number) for number, _trace in traces] == [1, 2, 3, 4]
    outside = re.sub(r"BEGIN_ORDERED (\d+)\n(.*?)END_ORDERED \1\n", "", result.stderr, flags=re.DOTALL)
    assert "FORT_SCOPED " not in outside
    identities = {name.split("::")[-1]: item["registration_identity"] for name, item in resources.items()}
    ledger = []
    for (number, trace), launches in zip(traces, (3, 1, 1, 3), strict=True):
        assert trace.count("FORT_SCOPED launch ") == launches, trace
        assert trace.count("FORT_SCOPED close ") == 1, trace
        assert len(set(re.findall(r"FORT_SCOPED .*?\bcontext=(\S+)", trace))) == 1, trace
        traffic = _transfers(trace)
        assert dict(traffic["upload"]) == {
            identities["old"]: 128, identities["viscosity"]: 128,
            **({identities["out"]: 128} if launches == 3 else {}),
        }, trace
        assert dict(traffic["download"]) == {
            identities["viscosity"]: 128,
            **({identities["star"]: 128, identities["out"]: 128} if launches == 3 else {}),
        }, trace
        ledger.append({"invocation": int(number), "launches": launches,
                       "upload_bytes": dict(traffic["upload"]), "download_bytes": dict(traffic["download"])})
    (directory / "trace.log").write_text(result.stderr)
    (directory / "transfer-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    (directory / "native-fields.bin").write_bytes(expected)
    (directory / "resident-fields.bin").write_bytes(observed)
    for overrides in ({"CUDA_VISIBLE_DEVICES": ""}, {"OMP_NUM_THREADS": "2"}):
        fallback = run([str(candidate)], cwd=directory, env={**environment, **overrides, "FORT_RUNTIME_TRACE": "1"})
        assert (directory / "fields.bin").read_bytes() == expected
        if "CUDA_VISIBLE_DEVICES" in overrides:
            assert "FORT_SCOPED launch " not in fallback.stderr
            assert not any(_transfers(fallback.stderr).values())
    (directory / "result.json").write_text(json.dumps({
        "full_float64_arrays_bounds_halos_and_native_visits": "bitwise_equal",
        "native_sha256": sha256(expected).hexdigest(), "resident_sha256": sha256(observed).hexdigest(),
        "matched_bytes": len(observed), "launches_per_invocation": [3, 1, 1, 3],
        "non_idempotent_prefix_and_native_tail_executed_once": True,
        "disabled_device_fallback_and_alternate_team": "bitwise_equal",
        "automatic_pricing_qualified": False, "performance_samples": 0,
    }, indent=2) + "\n")
