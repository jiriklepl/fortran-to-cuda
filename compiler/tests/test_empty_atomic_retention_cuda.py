"""Empty original OpenMP teams execute without publishing resident fields."""

import json
import re
import shutil

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from elmm_pipeline.local_adapter import _scope_package_build

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_same_family_section_composition_cuda import CALL, source


def emit(directory):
    from dataclasses import replace

    text = source().replace(CALL, "! No opaque native CALL in this retention control.")
    text = text.replace("subroutine producer(b,n)", "subroutine producer(a,b,n)\nreal(8),intent(in)::a(:)")
    text = text.replace("b(i+2)=real(i,8)", "b(i+2)=a(i+2)+real(i,8)")
    text = text.replace("call producer(right,n)", "call producer(left,right,n)")
    text = text.replace("a(i+2)=2.0_8*b(i+2)", "a(i+2)=a(i+2)+2.0_8*b(i+2)")
    original, builder = fixture(directory, text)
    builder.config = replace(builder.config, host_threads=4)
    outputs, report = builder.run()
    assert report["scope_count"] == 1
    manifest = json.loads(outputs["scope-manifest.json"])
    validate_checked_execution_plans(manifest, capture_facts=builder.facts)
    assert not any(
        "call_preservation" in route
        for member in manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]["return_contract"]["members"]
        for route in member["routes"]
    )
    package = directory / "source-package"
    package.mkdir()
    for name, value in outputs.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    _scope_package_build(
        {"artifact_root": str(package), "scope_manifest": manifest, "tool": "local", "memory_model": "scoped"}
    )
    assert builder.finish()
    return original, outputs, manifest


DRIVER = """program exercise
use hierarchical
use iso_fortran_env,only:error_unit
implicit none
integer::index,n,i,unit
real(8)::want_left,want_right
open(newunit=unit,file='empty-fields.bin',form='unformatted',access='stream',status='replace')
do index=1,3
select case(index)
case(1)
n=1
case(2)
n=7
case default
n=257
end select
write(error_unit,'(A,1X,I0)')'EMPTY_CASE_BEGIN',n
call advance(n,1,.false.)
write(error_unit,'(A,1X,I0)')'EMPTY_CASE_END',n
if(any(shape(left)/=[n+4]).or.any(shape(right)/=[n+4]))error stop 'shape changed'
if(lbound(left,1)/=2.or.lbound(right,1)/=2)error stop 'allocation bounds changed'
do i=2,n+5
want_left=-99.0_8
want_right=1.0_8
if(i>=4.and.i<=n+3)then
want_left=real(i-2,8)
want_right=real(2*i-3,8)
endif
if(left(i)/=want_left.or.right(i)/=want_right)error stop 'full field or halo differs'
enddo
write(unit)shape(left),lbound(left),ubound(left),shape(right),lbound(right),ubound(right),left,right
deallocate(left,right)
enddo
close(unit)
print *, 'EMPTY_ORIGINAL_TEAMS_FIELDS_HALOS_BOUNDS_OK'
end program
"""


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "emit", emit)
        patch.setattr(base, "DRIVER", DRIVER)
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((directory / "scope-manifest.json").read_text())
    provenance = {row["id"]: row for row in manifest["runtime_provenance"]["records"]}
    # Use the same GNU OpenMP ABI observer as the pristine-entry controls.
    # One marker per genuine original team proves that retaining residency
    # does not bypass its source execution.
    observer = directory / "empty-observer.cpp"
    observer.write_text(
        '#include <cstdio>\nextern "C" {\n'
        'void __real_GOMP_parallel(void (*fn)(void *),void *data,unsigned threads,unsigned flags);\n'
        'void __wrap_GOMP_parallel(void (*fn)(void *),void *data,unsigned threads,unsigned flags) {\n'
        'std::fputs("EMPTY_NATIVE_TEAM\\n",stderr);\n'
        '__real_GOMP_parallel(fn,data,threads,flags);\n}\n}\n'
    )
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    observer_object = directory / "empty-observer.o"
    base.checked([host, "-std=c++17", "-c", str(observer), "-o", str(observer_object)], directory, "observer-compile")
    driver = directory / "driver.f90"
    flags = [fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-I", str(directory)]
    observer_link = [str(observer_object), "-Wl,--wrap=GOMP_parallel"]
    base.checked(
        [*flags, str(driver), *map(str, sorted(directory.glob("object-*.o"))), *observer_link,
         "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(candidate)],
        directory, "observer-link-candidate",
    )
    base.checked(
        [*flags, "-J", str(directory), str(directory.parent / "source.f90"), str(driver), *observer_link,
         "-lstdc++", "-o", str(native)], directory, "observer-link-native",
    )
    return directory, candidate, native, provenance


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["traced", "untraced", "hidden"])
def test_empty_teams_execute_and_preserve_live_readonly_and_dirty_replicas(binaries, mode):
    directory, candidate, native, provenance = binaries
    env = environment(mode)
    reference = base.checked([str(native)], directory, mode + "-native", env)
    expected = (directory / "empty-fields.bin").read_bytes()
    (directory / (mode + "-native-fields.bin")).write_bytes(expected)
    result = base.checked([str(candidate)], directory, mode + "-candidate", env)
    actual = (directory / "empty-fields.bin").read_bytes()
    assert actual == expected
    (directory / (mode + "-fields.bin")).write_bytes(actual)
    assert result.stdout == reference.stdout
    if mode == "untraced":
        assert "FORT_SCOPED " not in result.stderr
    cases = re.findall(r"EMPTY_CASE_BEGIN (\d+)\n(.*?)EMPTY_CASE_END \1", result.stderr, re.S)
    assert len(cases) == 3
    assert reference.stderr.count("EMPTY_NATIVE_TEAM\n") == 9
    teams_per_case = 5 if mode == "hidden" else 3
    assert result.stderr.count("EMPTY_NATIVE_TEAM\n") == 3 * teams_per_case
    ledger = []
    for size, text in cases:
        assert text.count("EMPTY_NATIVE_TEAM\n") == teams_per_case
        if mode == "hidden":
            latest_position = None
            team_procedures = []
            for line in text.splitlines():
                if line.startswith("FORT_SCOPED position "):
                    (position,) = events_in(line)
                    latest_position = procedure(position, provenance)
                elif line == "EMPTY_NATIVE_TEAM":
                    assert latest_position is not None
                    team_procedures.append(latest_position)
            assert team_procedures == [
                "hierarchical::producer",
                "hierarchical::relay",
                "hierarchical::relay",
                "hierarchical::relay",
                "hierarchical::consumer",
            ]
            assert team_procedures.count("hierarchical::relay") == 3
        events = events_in(text)
        launches = [i for i, event in enumerate(events) if event["kind"] == "launch"]
        ledger.append({
            "size": int(size), "events": events, "original_teams_executed": 3,
            "total_GOMP_teams": teams_per_case,
            "numerical_fallback_teams": 2 if mode == "hidden" else 0,
        })
        if mode == "hidden":
            assert not launches
            assert copy_bytes(events, "upload") == 0
            assert copy_bytes(events, "download") == 0
        elif mode == "traced":
            assert [procedure(events[i], provenance) for i in launches] == [
                "hierarchical::producer", "hierarchical::consumer"
            ]
            first, last = launches
            assert events[first]["domain"] == events[last]["domain"]
            between = events[first + 1:last]
            assert not any(event["kind"] == "domain_retire" for event in between)
            assert copy_bytes(between, "upload") == 0
            assert copy_bytes(between, "download") == 0
            assert copy_bytes(events, "upload") == 8 * int(size)
            assert copy_bytes(events, "download") == 16 * int(size)
    (directory / (mode + "-ledger.json")).write_text(json.dumps(ledger, indent=2) + "\n")


def test_empty_team_retention_seals_original_source_and_strict_package(tmp_path):
    emit(tmp_path)
