"""Partial entry definitions upload only interiors and preserve native halos."""

import json
import os
from hashlib import sha256

import pytest

from compiler.frontend.reached_windows import plan_reached_window
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.tests.support.numerical_cuda import _transfers, build_reached_numerical_binaries
from compiler.tests.test_reached_entry_sections import ENTRY, ROOTS, entry_sections_fixture
from compiler.tests.test_source_scopes import run


def emit_entry_fixture(directory, *, precision=8, device_budget=None):
    path, builder, originals = entry_sections_fixture(directory, precision=precision, device_budget=device_budget)
    proof = plan_reached_window(builder.analysis, ENTRY, originals, ROOTS, builder.facts)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    outputs, report = owner.builder.finish()
    return (path,), outputs, report, owner


def entry_driver(precision):
    return f"""program verify
use frost_fields,only:a,b,advance,prefix_visits,tail_visits
implicit none
integer::extra,repeat,i,unit
character(16)::path
call get_command_argument(1,path)
if(path/='matching'.and.path/='mismatch') error stop 'unknown path'
extra=0
if(path=='mismatch') extra=1
! Zero-trip execution must not inquire into unallocated captures.
call advance(0)
if(allocated(a).or.allocated(b)) error stop 'unexpected allocation'
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do repeat=1,2
allocate(a(-4-extra:7),b(-7:7))
do i=lbound(a,1),ubound(a,1)
a(i)=real(i+13+repeat,{precision})*0.125_{precision}
enddo
b=-77.0_{precision}
call advance(3)
if(prefix_visits/=repeat+1.or.tail_visits/=3*repeat) error stop 'original work replayed'
if(lbound(a,1)/=-4-extra.or.ubound(a,1)/=7) error stop 'input bounds changed'
if(lbound(b,1)/=-7.or.ubound(b,1)/=7) error stop 'output bounds changed'
if(any(b(-7:-4)/=-77.0_{precision}).or.any(b(6:7)/=-77.0_{precision})) error stop 'halo changed'
write(unit)repeat,extra,prefix_visits,tail_visits
write(unit)lbound(a),ubound(a),lbound(b),ubound(b),a,b
deallocate(a,b)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=[(precision, fallback) for precision in (4, 8) for fallback in (False, True)])
def entry_binaries(tmp_path_factory, request):
    precision, fallback = request.param
    return build_reached_numerical_binaries(tmp_path_factory, precision, fallback,
        emit_fixture=emit_entry_fixture, driver_source=entry_driver, prefix="reached_entry_sections_",
        expected_gpu_leaves=2, expected_capture_states={ROOTS[0]: "sections", ROOTS[1]: "none"})


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("path", ["matching", "mismatch"])
def test_partial_entry_matches_native_fields_and_exact_traffic(entry_binaries, path, disabled):
    directory, candidate, native, precision, resource_fallback, resources = entry_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    reference = run([str(native), path], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate), path], cwd=directory, env=environment)
    actual = (directory / "fields.bin").read_bytes()
    assert actual == expected
    assert "array temporary" not in (reference.stderr + result.stderr).lower()
    active = path == "matching" and not disabled and not resource_fallback
    assert result.stderr.count("FORT_SCOPED launch ") == (4 if active else 0)
    transfers = _transfers(result.stderr)
    ids = {root: resources[root]["registration_identity"] for root in ROOTS}
    assert dict(transfers["upload"]) == ({ids[ROOTS[0]]: 18 * precision} if active else {})
    assert dict(transfers["download"]) == ({ids[root]: 18 * precision for root in ROOTS} if active else {})
    label = path + ("-disabled" if disabled else "-cuda")
    (directory / (label + ".stderr")).write_text(result.stderr)
    (directory / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "precision_bytes": precision, "path": path,
        "cuda_disabled": disabled, "resource_fallback": resource_fallback,
        "native_sha256": sha256(expected).hexdigest(), "candidate_sha256": sha256(actual).hexdigest(),
        "complete_fields_halos_bounds_and_visits_equal": True,
        "upload_bytes": sum(transfers["upload"].values()),
        "download_bytes": sum(transfers["download"].values()),
        "launches": result.stderr.count("FORT_SCOPED launch "), "performance_samples": 0,
    }, indent=2, sort_keys=True) + "\n")
