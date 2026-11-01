"""Empty original native loops preserve a live domain between real kernels."""
import json
import re

import pytest

from compiler.scopes.lexical import LexicalBoundaryPatch
from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_reached_loop_retirement import source


def halo_source():
    original = "do i=1,size(points)\nright(points(i)%xi+1)=sample(points(i),right)\nenddo"
    body = "right(points(i)%xi+3)=sample(points(i),right)"
    loops = ("if(change)then\ndo i=size(points),1,-1\n" + body + "\nenddo\nelse\n"
             "do i=1,size(points)\n" + body + "\nenddo\nendif")
    return (source().replace(original, loops)
        .replace("real(8),intent(out)::b(:)", "real(8),intent(inout)::b(:)")
        .replace("real(8),intent(out)::a(:)", "real(8),intent(inout)::a(:)")
        .replace("b(i)=real(i,8)", "b(i+2)=real(i,8)")
        .replace("a(i)=2.0_8*b(i)", "a(i+2)=2.0_8*b(i+2)")
        .replace("sample=values(point%xi)+2.0_8", "sample=values(point%xi+2)+2.0_8")
        .replace("allocate(left(-3:n-4),right(2:n+1))", "allocate(left(2:n+5),right(2:n+5))")
        .replace("left=0\nright=0", "left=-101\nright=-99"))


def sealed_emission(directory):
    from elmm_pipeline.local_adapter import _scope_package_build

    original, builder = fixture(directory, halo_source())
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    outputs = dict(outputs)
    owner = builder._conditional_return_draft.validate().owner
    middle, = [m for m in owner.members.values() if m.routine.qualified == "hierarchical::middle"]
    patches = [p for _first, _last, p in middle.patches
               if isinstance(p, LexicalBoundaryPatch) and p.counted_entry is not None]
    assert len(patches) == 2
    for patch in patches:
        assert patch.validate(middle, require_source=True) is patch
        assert patch.lifecycle.preserves_descriptors()
    assert builder.finish()
    scope, = report["scopes"]
    assert {name.split("#")[0] for name in scope["gpu_leaves"]} == {
        "hierarchical::producer", "hierarchical::consumer"}
    assert {row["resource"] for row in scope["checked_execution_plan"]["resources"]} == {
        "hierarchical::left", "hierarchical::right"}
    package = directory / "source-package"
    package.mkdir()
    for name, text in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = json.loads((package / "scope-manifest.json").read_text())
    _scope_package_build({"artifact_root": str(package), "scope_manifest": manifest,
                          "tool": "local", "memory_model": "scoped"})
    (directory / "loop-entry-proof.json").write_text(json.dumps([
        {"entry": dict(p.counted_entry.authority), "storage": p.lifecycle.public()} for p in patches], indent=2) + "\n")
    return original, outputs, manifest


def test_halo_loop_headers_and_strict_returned_package_are_authentic(tmp_path):
    sealed_emission(tmp_path)


DRIVER = """program exercise
use hierarchical
use iso_fortran_env,only:error_unit
use omp_lib,only:omp_get_max_threads
implicit none
integer::case_index,n,i,lo,hi,nonempty,descending,count,expected_index
real(8)::expected
if(omp_get_max_threads()/=4)error stop 'four threads required'
last_i=-999
write(error_unit,'(A)')'LOOP_ZERO_BEGIN'
call advance(257,0,.false.)
write(error_unit,'(A)')'LOOP_ZERO_END'
if(allocated(left).or.allocated(right).or.allocated(points))error stop 'zero path inspected storage'
if(last_i/=-999)error stop 'zero outer path entered original header'
do case_index=1,3
select case(case_index)
case(1)
n=1
case(2)
n=7
case default
n=257
end select
do nonempty=0,1
count=nonempty*n
allocate(points(count))
do i=1,count
points(i)%xi=i
enddo
do descending=0,1
last_i=-999
write(error_unit,'(A,3(1X,I0))')'LOOP_CASE_BEGIN',n,nonempty,descending
call advance(n,2,descending==1)
write(error_unit,'(A,3(1X,I0))')'LOOP_CASE_END',n,nonempty,descending
expected_index=count+1
if(descending==1)expected_index=0
if(last_i/=expected_index)error stop 'original final iterator changed'
if(any(shape(left)/=[n+4]).or.any(shape(right)/=[n+4]))error stop 'shape changed'
if(lbound(left,1)/=2.or.lbound(right,1)/=2)error stop 'allocation origin lost'
lo=lbound(left,1)
hi=ubound(left,1)
if(any(left(lo:lo+1)/=-101).or.any(left(hi-1:hi)/=-101))error stop 'left halo changed'
if(any(right(lo:lo+1)/=-99).or.any(right(hi-1:hi)/=-99))error stop 'right halo changed'
do i=1,n
expected=real(i+2*nonempty,8)
if(left(i+3)/=expected.or.right(i+3)/=2.0_8*expected)error stop 'original fields changed'
enddo
do i=1,count
if(points(i)%xi/=i.or.allocated(points(i)%unrelated))error stop 'readonly records changed'
enddo
write(*,'(8(I0,1X))')n,nonempty,descending,last_i,lbound(left,1),ubound(left,1),lbound(right,1),ubound(right,1)
write(*,'(*(ES26.17E3,1X))')left,right
deallocate(left,right)
enddo
deallocate(points)
enddo
enddo
print *, 'ORIGINAL_LOOP_FIELDS_HALOS_FINAL_ITERATOR_OK'
end program
"""


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "emit", sealed_emission)
        patch.setattr(base, "DRIVER", DRIVER)
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((directory / "scope-manifest.json").read_text())
    provenance = {row["id"]: row for row in manifest["runtime_provenance"]["records"]}
    return directory, candidate, native, provenance


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["traced", "untraced", "hidden"])
def test_zero_trip_loop_preserves_domain_and_nonempty_loop_retires_once(binaries, mode):
    directory, candidate, native, provenance = binaries
    env = environment(mode)
    reference = base.checked([str(native)], directory, mode + "-native", env)
    result = base.checked([str(candidate)], directory, mode + "-candidate", env)
    assert result.stdout == reference.stdout
    assert "ORIGINAL_LOOP_FIELDS_HALOS_FINAL_ITERATOR_OK" in result.stdout
    if mode == "untraced":
        assert "FORT_SCOPED " not in result.stderr
        return
    zero, = re.findall(r"LOOP_ZERO_BEGIN\n(.*?)LOOP_ZERO_END", result.stderr, re.S)
    assert not any(e["kind"] in {"domain_create", "register", "launch"} for e in events_in(zero))
    cases = re.findall(r"LOOP_CASE_BEGIN (\d+) ([01]) ([01])\n(.*?)LOOP_CASE_END \1 \2 \3", result.stderr, re.S)
    assert len(cases) == 12
    ledger = []
    for width, nonempty, descending, text in cases:
        n, reached = int(width), nonempty == "1"
        events = events_in(text)
        launches = [i for i, event in enumerate(events) if event["kind"] == "launch"]
        ledger.append({"n": n, "nonempty": reached, "descending": descending == "1", "events": events,
                       "upload_bytes": copy_bytes(events, "upload"), "download_bytes": copy_bytes(events, "download")})
        (directory / (mode + "-ledger.json")).write_text(json.dumps(ledger, indent=2) + "\n")
        if mode == "hidden":
            assert not launches
            assert copy_bytes(events, "upload") == copy_bytes(events, "download") == 0
            continue
        assert [procedure(events[i], provenance) for i in launches] == [
            "hierarchical::producer", "hierarchical::consumer"] * 2
        for stage in range(2):
            first, last = launches[2*stage:2*stage+2]
            between = events[first+1:last]
            assert (events[first]["domain"] != events[last]["domain"]) is reached
            assert any(e["kind"] == "domain_retire" for e in between) is reached
            assert copy_bytes(between, "download") == (8*n if reached else 0)
            assert copy_bytes(between, "upload") == (8*n if reached else 0)
        assert copy_bytes(events, "upload") == (16*n if reached else 0)
        assert copy_bytes(events, "download") == 32*n
