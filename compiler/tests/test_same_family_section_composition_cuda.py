"""Finite real-CUDA controls for an unchanged same-family native CALL."""

import json
import re
import shutil
import subprocess

import pytest

from compiler.scopes.conditional_return import SameFamilyNativeCallLifecycle
from compiler.scopes.lexical import LexicalBoundaryPatch
from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_hierarchical_child_regions_cuda import FAILURE, copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_same_family_native_call import WORKER, source_fixture

CALL = "call native_worker(right,left,n,scale1,scale2,scale3)"
NATIVE = """subroutine native_worker(values,readonly,n,f1,f2,f3)
integer,intent(in)::n
real(8),intent(inout)::values(:)
real(8),intent(in)::readonly(:)
procedure(scale1)::f1,f2,f3
integer::i
if(allocated(records))then
do i=1,size(records)
associate(xi=>records(i)%xi)
values(xi)=values(xi)+records(i)%flux*(f1(xi)+f2(xi)+f3(xi))*readonly(xi)
end associate
enddo
endif
end subroutine
pure real(8) function scale1(i)
integer,intent(in)::i
scale1=left(i+1)
end function
pure real(8) function scale2(i)
integer,intent(in)::i
scale2=2.0_8*left(i+1)
end function
pure real(8) function scale3(i)
integer,intent(in)::i
scale3=3.0_8*left(i+1)
end function
"""


def base_source():
    return (
        source_fixture()
        .replace(WORKER, NATIVE)
        .replace("call native_worker(left,right,n,scale)", CALL)
        .replace("real(8),intent(out)::b(:)", "real(8),intent(inout)::b(:)")
        .replace("real(8),intent(out)::a(:)", "real(8),intent(inout)::a(:)")
        .replace("b(i)=real(i,8)", "b(i+2)=real(i,8)")
        .replace("a(i)=2.0_8*b(i)", "a(i+2)=2.0_8*b(i+2)")
        .replace("observed=0.0_8\ndo i=1,n\nobserved=observed+right(lbound(right,1)+i-1)\nenddo\n", "")
        .replace("allocate(left(-3:n-4),right(2:n+1))", "allocate(left(2:n+5),right(2:n+5))")
        .replace("left=0\nright=0", "left=1\nright=-99")
        .replace("integer::stage\n", "integer::stage\nif(n<0)return\n")
    )


HELPERS = """subroutine helper(a,p,n,flag)
real(8),intent(inout)::a(:)
real(8),intent(in)::p(:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
!$omp parallel private(i)
if(flag.and.n>=6)then
!$omp sections
!$omp section
do i=n-2,n
a(i)=p(i-n+3)
enddo
!$omp section
do i=1,3
a(i)=p(i+n-3)
enddo
!$omp end sections nowait
endif
!$omp end parallel
end subroutine
subroutine relay(a,p,n,flag)
real(8),intent(inout)::a(:)
real(8),intent(in)::p(:)
integer,intent(in)::n
logical,intent(in)::flag
call helper(a,p,n,flag)
end subroutine
"""


def source():
    calls = CALL + "\ncall relay(left,right,n,change)\ncall relay(right,left,n,change)\ncall relay(left,right,n,change)"
    return base_source().replace(CALL, calls).replace("end module", HELPERS + "end module")


def sealed_emission(directory):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    original, builder = fixture(directory, source())
    from dataclasses import replace

    builder.config = replace(builder.config, host_threads=4)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    owner = builder._conditional_return_draft.validate().owner
    selected = [
        (member, patch)
        for member in owner.members.values()
        for _first, _last, patch in member.patches
        if isinstance(patch, LexicalBoundaryPatch)
        and type(patch.lifecycle) is SameFamilyNativeCallLifecycle
        and "native_worker" in str(patch.originals[0]).lower()
    ]
    assert len(selected) == 1
    member, patch = selected[0]
    assert patch.validate(member, require_source=True) is patch
    assert patch.completion is None
    assert patch.conditional_guard is None
    assert patch.lifecycle._view.public()["available"]
    assert (
        len(
            [
                i
                for i in patch.lifecycle._view._graph.host_instances
                if i.procedure.endswith(("scale1", "scale2", "scale3"))
            ]
        )
        == 3
    )
    outputs = dict(outputs)
    transformed = outputs[report["sources"][str(original)]["replacement"]]
    assert NATIVE in transformed
    assert transformed.count(CALL) == 1
    (scope,) = report["scopes"]
    assert {name.split("#")[0] for name in scope["gpu_leaves"]} == {"hierarchical::producer", "hierarchical::consumer"}
    assert {row["resource"] for row in scope["checked_execution_plan"]["resources"]} == {
        "hierarchical::left",
        "hierarchical::right",
    }
    assert scope["checked_execution_plan"]["storage_epoch"]["return_contract"]["version"] == 3
    tasks = [
        task
        for member in scope["checked_execution_plan"]["storage_epoch"]["members"]
        for task in member.get("source_device_tasks", [])
    ]
    (task,) = tasks
    assert task["version"] == 4
    assert task["kind"] == "original_atomic_conditional_joined_team"
    projection = task["execution_binding"]["call_projection"]
    calls = [edge for edge in projection["calls"] if edge["caller"] == "hierarchical::middle"]
    assert len(calls) == 3
    assert [[a["actual"].lower() for a in edge["source_call"]["original_arguments"]] for edge in calls] == [
        ["left", "right", "n", "change"],
        ["right", "left", "n", "change"],
        ["left", "right", "n", "change"],
    ]
    assert all(
        set(roots) == {"hierarchical::left", "hierarchical::right"} for roots in projection["field_roots"].values()
    )
    # SECTION helpers are transformed checked companions; their original ABI
    # fallback CALL remains, while the native worker above stays byte exact.
    assert "call helper(a,p,n,flag)" in transformed

    for binding, _slot in owner.resources.values():
        assert builder.capture(binding)["initialized"] == "none"
    validate_checked_execution_plans(json.loads(outputs["scope-manifest.json"]), capture_facts=builder.facts)
    package = directory / "source-package"
    package.mkdir()
    for name, text in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = json.loads((package / "scope-manifest.json").read_text())
    _scope_package_build(
        {"artifact_root": str(package), "scope_manifest": manifest, "tool": "local", "memory_model": "scoped"}
    )
    assert builder.finish()
    (directory / "same-family-proof.json").write_text(
        json.dumps(
            {
                "retirement": patch.lifecycle.public(),
                "structural_preservation": patch.lifecycle._view.public(),
                "original_native": NATIVE,
            },
            indent=2,
        )
        + "\n"
    )
    return original, outputs, manifest


def test_combined_native_call_and_guarded_sections_strict_package(tmp_path):
    sealed_emission(tmp_path)


DRIVER = """program exercise
use hierarchical
use iso_fortran_env,only:error_unit
implicit none
integer::record_mode,case_index,n,count,i,lo,hi,branch,change_number
logical::change
real(8),allocatable::ea(:),eb(:)
character(8)::argument
call get_command_argument(1,argument)
if(trim(argument)=='fault')then
allocate(records(7))
do i=1,7
records(i)%xi=i+2
records(i)%flux=1.0_8
enddo
write(error_unit,'(A)')'COMPOSE_FAULT_BEGIN'
call advance(7,1,.true.)
error stop 'failed retirement returned'
endif
write(error_unit,'(A)')'COMPOSE_COLD_BEGIN'
call advance(-1,1,.true.)
write(error_unit,'(A)')'COMPOSE_COLD_END'
if(allocated(left).or.allocated(right).or.allocated(records))error stop 'cold path touched storage'
write(error_unit,'(A)')'COMPOSE_ZERO_BEGIN'
call advance(257,0,.true.)
write(error_unit,'(A)')'COMPOSE_ZERO_END'
if(allocated(left).or.allocated(right).or.allocated(records))error stop 'zero outer path touched storage'
do record_mode=-1,1
do branch=0,1
change=branch==1
do case_index=1,3
select case(case_index)
case(1)
n=1
case(2)
n=7
case default
n=257
end select
if(record_mode>=0)then
count=record_mode*n
allocate(records(count))
do i=1,count
records(i)%xi=i+2
records(i)%flux=1.0_8
enddo
endif
allocate(ea(n+4),eb(n+4))
ea=1.0_8
eb=-99.0_8
do i=1,n
eb(i+2)=real(i,8)
if(record_mode==1)eb(i+2)=eb(i+2)+6.0_8
enddo
if(change.and.n>=6)then
do i=n-2,n
ea(i)=eb(i-n+3)
enddo
do i=1,3
ea(i)=eb(i+n-3)
enddo
do i=n-2,n
eb(i)=ea(i-n+3)
enddo
do i=1,3
eb(i)=ea(i+n-3)
enddo
do i=n-2,n
ea(i)=eb(i-n+3)
enddo
do i=1,3
ea(i)=eb(i+n-3)
enddo
endif
do i=1,n
ea(i+2)=2.0_8*eb(i+2)
enddo
write(error_unit,'(A,3(1X,I0))')'COMPOSE_CASE_BEGIN',record_mode,n,branch
call advance(n,1,change)
write(error_unit,'(A,3(1X,I0))')'COMPOSE_CASE_END',record_mode,n,branch
if(any(shape(left)/=[n+4]).or.any(shape(right)/=[n+4]))error stop 'shape changed'
if(lbound(left,1)/=2.or.lbound(right,1)/=2)error stop 'allocation origin lost'
if(any(left/=eb).or.any(right/=ea))error stop 'native/SECTION fields, halos or original call order changed'
if(record_mode>=0)then
if(size(records)/=record_mode*n)error stop 'record allocation changed'
do i=1,size(records)
if(records(i)%xi/=i+2.or.records(i)%flux/=1.0_8)error stop 'readonly metadata changed'
enddo
deallocate(records)
else
if(allocated(records))error stop 'unallocated metadata changed'
endif
write(*,'(6(I0,1X))')record_mode,n,branch,lbound(left,1),ubound(left,1),size(left)
write(*,'(*(ES26.17E3,1X))')left,right
if(change.and.n>=6)then
write(*,'(A)')'THREE_GUARDED_ORIGINAL_SECTION_CALLS_CHECKED'
endif
deallocate(left,right,ea,eb)
enddo
enddo
enddo
print *, 'NATIVE_CALL_AND_DISTINCT_SECTION_FIELDS_HALOS_BOUNDS_OK'
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
    wrapper = directory / "retirement-failure.cpp"
    wrapper.write_text(FAILURE)
    obj = directory / "retirement-failure.o"
    host = shutil.which("g++-14") or shutil.which("g++")
    base.checked([host, "-std=c++17", "-c", str(wrapper), "-o", str(obj)], directory, "compile-failure")
    command = json.loads((directory / "link-candidate.command.json").read_text())
    fault = directory / "fault-candidate"
    command[command.index("-o") + 1] = str(fault)
    base.checked([*command, str(obj), "-Wl,--wrap=fort_scope_domain_close_v1"], directory, "link-failure")
    return directory, candidate, native, fault, provenance


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["traced", "untraced", "hidden"])
def test_combined_native_call_and_distinct_guarded_sections(binaries, mode):
    directory, candidate, native, _fault, provenance = binaries
    env = environment(mode)
    reference = base.checked([str(native)], directory, mode + "-native", env)
    result = base.checked([str(candidate)], directory, mode + "-candidate", env)
    assert result.stdout == reference.stdout
    assert "NATIVE_CALL_AND_DISTINCT_SECTION_FIELDS_HALOS_BOUNDS_OK" in result.stdout
    if mode == "untraced":
        assert "FORT_SCOPED " not in result.stderr
        return
    for name in ["COLD", "ZERO"]:
        (text,) = re.findall("COMPOSE_" + name + "_BEGIN\n(.*?)COMPOSE_" + name + "_END", result.stderr, re.S)
        assert not any(e["kind"] in {"domain_create", "register", "launch", "external_commit"} for e in events_in(text))
    cases = re.findall(r"COMPOSE_CASE_BEGIN (-?\d+) (\d+) ([01])\n(.*?)COMPOSE_CASE_END \1 \2 \3", result.stderr, re.S)
    assert len(cases) == 18
    ledger = []
    for record_mode, width, branch, text in cases:
        events = events_in(text)
        n = int(width)
        active = branch == "1" and n >= 6
        numeric = [
            i
            for i, e in enumerate(events)
            if e["kind"] == "launch"
            and procedure(e, provenance) in {"hierarchical::producer", "hierarchical::consumer"}
        ]
        commits = [i for i, e in enumerate(events) if e["kind"] == "external_commit"]
        ledger.append({"record_mode": int(record_mode), "n": n, "branch": int(branch), "events": events})
        if mode == "hidden":
            assert not numeric
            assert not commits
            assert copy_bytes(events, "upload") == copy_bytes(events, "download") == 0
            continue
        assert [procedure(events[i], provenance) for i in numeric] == [
            "hierarchical::producer",
            "hierarchical::consumer",
        ]
        first, last = numeric
        between = events[first + 1 : last]
        assert events[first]["domain"] != events[last]["domain"]
        working_domains = {str(events[first]["domain"]), str(events[last]["domain"])}
        retired = [e for e in between if e["kind"] == "domain_retire"]
        assert sum(str(e["domain"]) in working_domains for e in retired) == 1
        empty_domains = {str(e["domain"]) for e in retired} - working_domains
        # An empty original team still executes, while its checked child
        # retains the borrowed domain for subsequent calls and GPU work.
        assert not empty_domains
        assert len(commits) == (3 if active else 0)
        if active:
            assert not empty_domains
            assert first < commits[0] < commits[1] < commits[2] < last
            assert {events[i]["domain"] for i in commits} == {events[last]["domain"]}
            # Every field is already present after first fresh SECTION use;
            # the swapped and repeated calls must reuse that same device state.
            inner = events[commits[0] + 1 : commits[-1]]
            assert copy_bytes(inner, "upload") == copy_bytes(inner, "download") == 0
        else:
            assert copy_bytes(between, "download") == 8 * n
            assert copy_bytes(between, "upload") == 8 * n
    (directory / (mode + "-combined-ledger.json")).write_text(json.dumps(ledger, indent=2) + "\n")


@pytest.mark.cuda
def test_combined_post_submission_close_failure_stops_before_native_or_section_replay(binaries):
    directory, _candidate, _native, fault, provenance = binaries
    result = subprocess.run(
        [str(fault), "fault"], cwd=directory, env=environment("traced"), capture_output=True, text=True, timeout=30
    )
    (directory / "failure-candidate.stdout").write_text(result.stdout)
    (directory / "failure-candidate.stderr").write_text(result.stderr)
    assert result.returncode != 0
    assert result.stderr.count("HIER_TEST_POST_SUBMISSION_RETIREMENT_ERROR") == 1
    assert "lexical owner publication failed" in result.stderr
    assert "failed retirement returned" not in result.stderr
    assert [procedure(e, provenance) for e in events_in(result.stderr) if e["kind"] == "launch"] == [
        "hierarchical::producer"
    ]
