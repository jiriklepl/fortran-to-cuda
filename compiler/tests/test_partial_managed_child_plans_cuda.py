"""Unmanaged original actuals stay native across reached CUDA continuations."""

import json
import os
from hashlib import sha256

import pytest

from compiler.tests.support.numerical_cuda import _transfers, build_reached_numerical_binaries
from compiler.tests.test_mapped_child_reached_plans import emit_mapped_child_window
from compiler.tests.test_partial_managed_child_plans import projected_sources
from compiler.tests.test_source_scopes import run


def emit_projected_fixture(directory, *, precision=8, device_budget=None):
    sources = projected_sources(body="if(n>0) then\nx(-3)=z(-7)\nendif\nbad_visits=bad_visits+1")
    # The driver gives b a different original lower bound. Keep the original
    # parent computation in range while the child still rebases both dummies.
    sources["parent_source"] = sources["parent_source"].replace("b(i)=", "b(i-3)=")
    sources["child_source"] = sources["child_source"].replace(
        "x(-3)=x(-3)+3.0_8", "if(n>0) then\nx(-3)=x(-3)+3.0_8\nendif")
    return emit_mapped_child_window(directory, precision=precision, device_budget=device_budget, **sources)


def projected_driver(precision):
    return f"""program verify
use outer_fields,only:advance,prefix_visits,tail_visits
use inner_fields,only:child_visits,condition_visits,bad_visits,good_visits
implicit none
real({precision}),allocatable::a(:),b(:),c(:)
integer,parameter::sizes(2)=[8,129]
integer::shape,repeat,n,i,unit,low,calls
logical::active
character(16)::path
call get_command_argument(1,path)
if(path/='unused'.and.path/='active'.and.path/='empty') error stop 'unknown path'
active=path/='unused'
calls=0
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
 n=sizes(shape)
 if(path=='empty') n=0
 low=-17+11*shape
 allocate(a(low:low+n-1),b(low-3:low+n-4))
 if(path=='active') then
  allocate(c(-23-shape:-20-shape))
 else
  allocate(c(-23-shape:-24-shape))
 endif
 do repeat=1,2
  do i=lbound(a,1),ubound(a,1)
   a(i)=real(i+21+repeat,{precision})*0.125_{precision}
  enddo
  b=-77.0_{precision}
  c=real(3*shape+repeat,{precision})*0.25_{precision}
  calls=calls+1
  call advance(a,b,c,n,3,active)
  if(prefix_visits/=calls.or.tail_visits/=3*calls) error stop 'original outer work replayed'
  if(child_visits/=3*calls.or.condition_visits/=3*calls) error stop 'original child work replayed'
  if(active.and.bad_visits/=3*calls) error stop 'original active branch replayed'
  if(.not.active.and.good_visits/=3*calls) error stop 'original inactive branch changed'
  if(size(c)>0) then
   if(any(c/=real(3*shape+repeat,{precision})*0.25_{precision})) error stop 'unmanaged input changed'
  endif
  write(unit)shape,repeat,n,prefix_visits,tail_visits,child_visits,condition_visits,bad_visits,good_visits
  write(unit)lbound(a),ubound(a),lbound(b),ubound(b),lbound(c),ubound(c)
  write(unit)a,b,c
 enddo
 deallocate(a,b,c)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=[(precision, fallback) for precision in (4, 8) for fallback in (False, True)])
def projected_binaries(tmp_path_factory, request):
    precision, fallback = request.param
    return build_reached_numerical_binaries(tmp_path_factory, precision, fallback,
        emit_fixture=emit_projected_fixture, driver_source=projected_driver, prefix="partial_managed_child_",
        expected_gpu_leaves=4,
        expected_capture_states={"argument::a": "whole", "argument::b": "whole"})


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("path", ["unused", "active", "empty"])
def test_projected_child_matches_complete_fields_bounds_and_visits(projected_binaries, path, disabled):
    directory, candidate, native, precision, resource_fallback, resources = projected_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    reference = run([str(native), path], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate), path], cwd=directory, env=environment)
    observed = (directory / "fields.bin").read_bytes()
    assert observed == expected
    assert "array temporary" not in (reference.stderr + result.stderr).lower()
    assert set(resources) == {"argument::a", "argument::b"}
    launches = result.stderr.count("FORT_SCOPED launch ")
    enabled = path != "empty" and not disabled and not resource_fallback
    assert launches == ((8 if path == "active" else 16) if enabled else 0)
    transfers = _transfers(result.stderr)
    identities = {item["registration_identity"] for item in resources.values()}
    assert all(identity in identities for counts in transfers.values() for identity in counts)
    if not enabled:
        assert not any(transfers.values())
    label = path + ("-disabled" if disabled else "-cuda")
    (directory / (label + ".stderr")).write_text(result.stderr)
    (directory / (label + "-native.bin")).write_bytes(expected)
    (directory / (label + "-candidate.bin")).write_bytes(observed)
    (directory / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "precision_bytes": precision, "path": path,
        "resource_fallback": resource_fallback, "cuda_disabled": disabled,
        "native_sha256": sha256(expected).hexdigest(), "candidate_sha256": sha256(observed).hexdigest(),
        "complete_fields_bounds_and_original_visits_equal": True,
        "unmanaged_resource_registered": False, "launches": launches,
        "upload_bytes": sum(transfers["upload"].values()), "download_bytes": sum(transfers["download"].values()),
        "performance_samples": 0,
    }, indent=2, sort_keys=True) + "\n")
