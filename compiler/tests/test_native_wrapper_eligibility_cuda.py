"""Original native wrapper routing: source, observed host entry, and CUDA retirement controls."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.scopes.conditional_return import SameFamilyNativeCallLifecycle
from compiler.scopes.lexical import LexicalBoundaryPatch
from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_hierarchical_child_regions_cuda import FAILURE, copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_same_family_native_call import WORKER, source_fixture

CALL = "call native_wrapper(right,left,n)"
NATIVE = """subroutine native_wrapper(values,readonly,n)
integer,intent(in)::n
real(8),intent(inout)::values(:)
real(8),intent(in)::readonly(:)
call apply(values,readonly,n,scale1,scale2,scale3)
contains
subroutine apply(values,readonly,n,f1,f2,f3)
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
end subroutine
"""


def source():
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


def sealed_emission(directory):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    original, builder = fixture(directory, source())
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    owner = builder._conditional_return_draft.validate().owner
    selected = [
        (member, patch)
        for member in owner.members.values()
        for _first, _last, patch in member.patches
        if isinstance(patch, LexicalBoundaryPatch)
        and type(patch.lifecycle) is SameFamilyNativeCallLifecycle
        and "native_wrapper" in str(patch.originals[0]).lower()
    ]
    assert len(selected) == 1
    member, patch = selected[0]
    assert patch.validate(member, require_source=True) is patch
    assert patch.completion is None
    assert patch.conditional_guard is None
    assert patch.lifecycle._view.public()["available"]
    assert "hierarchical::native_wrapper" not in owner.members
    assert "hierarchical::native_wrapper::apply" not in owner.members
    host_procedures = {i.procedure for i in patch.lifecycle._view._graph.host_instances}
    assert {"hierarchical::native_wrapper", "hierarchical::native_wrapper::apply"} <= host_procedures
    assert {p for p in host_procedures if p.endswith(("scale1", "scale2", "scale3"))} == {
        "hierarchical::native_wrapper::scale1", "hierarchical::native_wrapper::scale2", "hierarchical::native_wrapper::scale3"
    }
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
    for binding, _slot in owner.resources.values():
        assert builder.capture(binding)["initialized"] == "none"
    validate_checked_execution_plans(report)
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
    (directory / "native-wrapper-proof.json").write_text(
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


def test_wrapper_source_keeps_native_route_and_strict_package(tmp_path):
    sealed_emission(tmp_path)


DRIVER = """program exercise
use hierarchical
use iso_fortran_env,only:error_unit
implicit none
integer::record_mode,case_index,n,count,i,lo,hi
real(8)::expected
character(8)::argument
call get_command_argument(1,argument)
if(trim(argument)=='fault')then
allocate(records(7))
do i=1,7
records(i)%xi=i+2
records(i)%flux=1.0_8
enddo
write(error_unit,'(A)')'FAMILY_FAULT_BEGIN'
call advance(7,1,.false.)
error stop 'failed retirement returned'
endif
write(error_unit,'(A)')'FAMILY_COLD_BEGIN'
call advance(-1,1,.false.)
write(error_unit,'(A)')'FAMILY_COLD_END'
if(allocated(left).or.allocated(right).or.allocated(records))error stop 'cold path touched storage'
write(error_unit,'(A)')'FAMILY_ZERO_BEGIN'
call advance(257,0,.false.)
write(error_unit,'(A)')'FAMILY_ZERO_END'
if(allocated(left).or.allocated(right).or.allocated(records))error stop 'zero outer path touched storage'
do record_mode=-1,1
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
write(error_unit,'(A,2(1X,I0))')'FAMILY_CASE_BEGIN',record_mode,n
call advance(n,1,.false.)
write(error_unit,'(A,2(1X,I0))')'FAMILY_CASE_END',record_mode,n
if(any(shape(left)/=[n+4]).or.any(shape(right)/=[n+4]))error stop 'shape changed'
if(lbound(left,1)/=2.or.lbound(right,1)/=2)error stop 'allocation origin lost'
lo=lbound(left,1)
hi=ubound(left,1)
if(any(left(lo:lo+1)/=-99).or.any(left(hi-1:hi)/=-99))error stop 'left halo changed'
if(any(right(lo:lo+1)/=1).or.any(right(hi-1:hi)/=1))error stop 'right halo changed'
do i=1,n
expected=real(i,8)
if(record_mode==1)expected=expected+6.0_8
if(left(i+3)/=expected.or.right(i+3)/=2.0_8*expected)error stop 'native indexed callback update/call count differs'
enddo
if(record_mode>=0)then
if(size(records)/=record_mode*n)error stop 'record allocation changed'
do i=1,size(records)
if(records(i)%xi/=i+2.or.records(i)%flux/=1.0_8)error stop 'readonly metadata changed'
enddo
deallocate(records)
else
if(allocated(records))error stop 'unallocated metadata changed'
endif
write(*,'(5(I0,1X))')record_mode,n,lo,hi,size(left)
write(*,'(*(ES26.17E3,1X))')left,right
deallocate(left,right)
enddo
enddo
print *, 'SAME_FAMILY_NATIVE_CALLBACK_FIELDS_HALOS_AND_BOUNDS_OK'
end program
"""


OBSERVER = r"""#include <cstdio>
extern "C" void __hierarchical_MOD_native_wrapper();
extern "C" __attribute__((no_instrument_function))
void __cyg_profile_func_enter(void *function, void *) {
  if (function == reinterpret_cast<void *>(&__hierarchical_MOD_native_wrapper))
    std::fputs("ORIGINAL_NATIVE_WRAPPER\n", stderr);
}
extern "C" __attribute__((no_instrument_function))
void __cyg_profile_func_exit(void *, void *) {}
"""


def observer_object(directory):
    observer = directory / "wrapper-entry-observer.cpp"
    observer.write_text(OBSERVER)
    obj = directory / "wrapper-entry-observer.o"
    host = shutil.which("g++-14") or shutil.which("g++")
    assert host, "qualified C++ compiler required"
    base.checked([host, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-c", str(observer), "-o", str(obj)],
                 directory, "compile-observer")
    return obj


def symbols(directory, executable, stem):
    nm = shutil.which("nm")
    assert nm, "qualified nm required"
    result = base.checked([nm, "-g", "--defined-only", str(executable)], directory, stem + "-symbols")
    matches = re.findall(r"^[0-9a-fA-F]+ [Tt] __hierarchical_MOD_native_wrapper$", result.stdout, re.M)
    assert len(matches) == 1, result.stdout


def wrapper_entries(text):
    marker = "ORIGINAL_NATIVE_WRAPPER"
    assert text.splitlines().count(marker) == 9, text
    for name in ("COLD", "ZERO"):
        (body,) = re.findall("FAMILY_" + name + "_BEGIN\n(.*?)FAMILY_" + name + "_END", text, re.S)
        assert marker not in body, body
    rows = re.findall(r"FAMILY_CASE_BEGIN (-?\d+) (\d+)\n(.*?)FAMILY_CASE_END \1 \2", text, re.S)
    assert [(int(m), int(n)) for m, n, _ in rows] == [(m, n) for m in (-1, 0, 1) for n in (1, 7, 257)]
    for _mode, _width, body in rows:
        assert body.splitlines().count(marker) == 1, body
    return [{"record_mode": int(m), "n": int(n), "original_wrapper_entries": 1} for m, n, _ in rows]


def test_native_wrapper_observer_preserves_fields_and_counts_empty_calls(tmp_path):
    original = tmp_path / "original.f90"
    driver = tmp_path / "driver.f90"
    original.write_text(source())
    driver.write_text(DRIVER)
    observer = observer_object(tmp_path)
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert fortran, "qualified Fortran compiler required"
    native = tmp_path / "native"
    base.checked([fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-finstrument-functions",
                  "-J", str(tmp_path), str(original), str(driver), str(observer), "-lstdc++", "-o", str(native)],
                 tmp_path, "link-native")
    symbols(tmp_path, native, "native")
    result = base.checked([str(native)], tmp_path, "observed-native", environment("hidden"))
    assert "SAME_FAMILY_NATIVE_CALLBACK_FIELDS_HALOS_AND_BOUNDS_OK" in result.stdout
    rows = wrapper_entries(result.stderr)
    (tmp_path / "native-wrapper-entries.json").write_text(json.dumps(rows, indent=2) + "\n")


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    observer_dir = tmp_path_factory.mktemp("native_wrapper_observer")
    observer = observer_object(observer_dir)
    checked = base.checked

    def instrument(command, build, stem, env=None):
        command = list(command)
        if Path(command[0]).name.startswith("gfortran"):
            command.insert(1, "-finstrument-functions")
            if "-c" not in command:
                command.extend([str(observer), "-lstdc++"])
        return checked(command, build, stem, env)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "emit", sealed_emission)
        patch.setattr(base, "DRIVER", DRIVER)
        patch.setattr(base, "checked", instrument)
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    symbols(directory, candidate, "candidate")
    symbols(directory, native, "native")
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
    symbols(directory, fault, "fault")
    return directory, candidate, native, fault, provenance


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["traced", "untraced", "hidden"])
def test_original_wrapper_call_publishes_then_rearms_real_children(binaries, mode):
    directory, candidate, native, _fault, provenance = binaries
    env = environment(mode)
    reference = base.checked([str(native)], directory, mode + "-native", env)
    result = base.checked([str(candidate)], directory, mode + "-candidate", env)
    assert result.stdout == reference.stdout
    reference_entries = wrapper_entries(reference.stderr)
    candidate_entries = wrapper_entries(result.stderr)
    assert candidate_entries == reference_entries
    (directory / (mode + "-wrapper-entries.json")).write_text(
        json.dumps({"native": reference_entries, "candidate": candidate_entries}, indent=2) + "\n"
    )
    assert "SAME_FAMILY_NATIVE_CALLBACK_FIELDS_HALOS_AND_BOUNDS_OK" in result.stdout
    if mode == "untraced":
        assert "FORT_SCOPED " not in result.stderr
        return
    for name in ["COLD", "ZERO"]:
        (text,) = re.findall("FAMILY_" + name + "_BEGIN\n(.*?)FAMILY_" + name + "_END", result.stderr, re.S)
        assert not any(e["kind"] in {"domain_create", "register", "launch"} for e in events_in(text))
    cases = re.findall(r"FAMILY_CASE_BEGIN (-?\d+) (\d+)\n(.*?)FAMILY_CASE_END \1 \2", result.stderr, re.S)
    assert len(cases) == 9
    ledger = []
    for record_mode, width, text in cases:
        events = events_in(text)
        n = int(width)
        launches = [i for i, e in enumerate(events) if e["kind"] == "launch"]
        ledger.append({"record_mode": int(record_mode), "n": n, "events": events})
        if mode == "hidden":
            assert not launches
            assert copy_bytes(events, "upload") == copy_bytes(events, "download") == 0
            continue
        assert [procedure(events[i], provenance) for i in launches] == [
            "hierarchical::producer",
            "hierarchical::consumer",
        ]
        first, last = launches
        between = events[first + 1 : last]
        # Whole CALL is the first slice: even an empty native loop retires.
        assert events[first]["domain"] != events[last]["domain"]
        assert sum(e["kind"] == "domain_retire" for e in between) == 1
        assert copy_bytes(between, "download") == 8 * n
        assert copy_bytes(between, "upload") == 8 * n
    (directory / (mode + "-ledger.json")).write_text(json.dumps(ledger, indent=2) + "\n")


@pytest.mark.cuda
def test_post_submission_close_failure_stops_before_native_call_or_replay(binaries):
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
    assert "ORIGINAL_NATIVE_WRAPPER" not in result.stderr
    assert [procedure(e, provenance) for e in events_in(result.stderr) if e["kind"] == "launch"] == [
        "hierarchical::producer"
    ]
