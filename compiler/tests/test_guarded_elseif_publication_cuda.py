"""Real lazy native predicates preserve current replicas and publish aliases."""
import json
import re
import shutil
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_native_retirement_scalar_formals_cuda import physical_gpu_events


def source(variant):
    assert variant in {"metadata", "managed", "alias"}
    module_field = ("real(8),allocatable,target::flags(:)" if variant == "alias" else
                    "integer,allocatable::late(:)" if variant == "managed" else
                    "integer,allocatable::flags(:)")
    formals = "a,flags,n,selector" if variant == "managed" else "a,n,selector"
    declaration = ("integer,intent(inout)::flags(:)\n" if variant == "managed" else "")
    prefix = "do i=1,2\nflags(i)=flags(i)+1\nenddo\n" if variant == "managed" else ""
    late = "late(2)" if variant == "managed" else "flags(2)"
    final = "2.0_8*a(i)+real(flags(1),8)" if variant == "managed" else "2.0_8*a(i)"
    return f"""module guarded_device
implicit none
{module_field}
integer::visits=0
contains
subroutine advance({formals})
real(8),target,intent(inout)::a(-3:)
{declaration}integer,intent(in)::n,selector
integer::i
if(n<=0)return
{prefix}do i=0,n-1
a(i)=a(i)+1.0_8
enddo
if(selector==1)then
visits=visits+1
do i=0,n-1
a(i)=a(i)+10.0_8
enddo
else if(flags(1)==2)then
visits=visits+1
do i=0,n-1
a(i)=a(i)+20.0_8
enddo
else if({late}==3)then
visits=visits+1
do i=0,n-1
a(i)=a(i)+30.0_8
enddo
else
visits=visits+1
do i=0,n-1
a(i)=a(i)+40.0_8
enddo
endif
do i=0,n-1
a(i)={final}
enddo
end subroutine
end module
"""


def driver(variant):
    managed, alias = variant == "managed", variant == "alias"
    width_list = "[7,257]" if alias else "[1,7,257]"
    width_count = 2 if alias else 3
    declaration = ("real(8),pointer::a(:)\n" if alias else "real(8),allocatable,target::a(:)\n")
    declaration += "integer::managed_flags(2)\n" if managed else ""
    arguments = "a,managed_flags,n,mode" if managed else "a,n,mode"
    cold_arguments = "a,managed_flags,0,1" if managed else "a,0,1"
    cold = ("allocate(flags(-3:2));a=>flags" if alias else "allocate(a(-3:2))")
    cold_cleanup = "nullify(a);deallocate(flags)" if alias else "deallocate(a)"
    allocate = "allocate(flags(-3:n+2));a=>flags" if alias else "allocate(a(-3:n+2))"
    cleanup = "nullify(a);deallocate(flags)" if alias else "deallocate(a)"
    if alias:
        setup = """if(mode==2)a(1)=1.0_8
if(mode>=3)a(1)=0.0_8
if(mode==3)a(2)=2.0_8
if(mode==4)a(2)=0.0_8"""
    elif managed:
        setup = """if(allocated(late))deallocate(late)
managed_flags=[0,4]
if(mode==1)managed_flags=[9,10]
if(mode==2)then
managed_flags=[1,0]
allocate(late(1));late=0
else if(mode>=3)then
allocate(late(2));late=0
if(mode==3)late(2)=3
endif"""
    else:
        setup = """if(allocated(flags))deallocate(flags)
if(mode==2)then
allocate(flags(1));flags=2
else if(mode>=3)then
allocate(flags(2));flags=0
if(mode==3)flags(2)=3
endif"""
    flags_save = "before_flags=managed_flags\n" if managed else ""
    flags_check = ("if(any(managed_flags/=before_flags+1))error stop 'managed predicate changed incorrectly'\n"
                   if managed else "")
    flags_output = "write(*,'(*(I0,1X))')managed_flags\n" if managed else ""
    tail = "+real(before_flags(1)+1,8)" if managed else ""
    return f"""program exercise
use guarded_device
use iso_fortran_env,only:error_unit
use omp_lib,only:omp_get_max_threads
implicit none
{declaration}real(8),allocatable::before(:)
real(8)::expected
integer::n,mode,i,case_index,before_flags(2)
integer,parameter::sizes({width_count})={width_list}
if(omp_get_max_threads()/=4)error stop 'four native threads required'
{cold}
write(error_unit,'(A)')'GUARDED_ZERO_BEGIN'
call advance({cold_arguments})
write(error_unit,'(A)')'GUARDED_ZERO_END'
{cold_cleanup}
do case_index=1,size(sizes)
n=sizes(case_index)
do mode=1,4
{allocate}
allocate(before(-3:n+2))
do i=-3,n+2
a(i)=real(i+37*mode+n,8)
enddo
{setup}
before=a
{flags_save}visits=0
write(error_unit,'(A,1X,I0,1X,I0)')'GUARDED_CASE_BEGIN',mode,n
call advance({arguments})
write(error_unit,'(A,1X,I0,1X,I0)')'GUARDED_CASE_END',mode,n
if(visits/=1)error stop 'selected original branch did not execute exactly once'
if(lbound(a,1)/=-3.or.ubound(a,1)/=n+2)error stop 'physical bounds changed'
do i=-3,n+2
expected=before(i)
if(i>=0.and.i<n)expected=(before(i)+1.0_8+10.0_8*mode)*2.0_8{tail}
if(a(i)/=expected)error stop 'field, halo, branch or lazy predicate mismatch'
enddo
{flags_check}write(*,'(5(I0,1X))')mode,n,lbound(a,1),ubound(a,1),visits
write(*,'(*(ES26.17E3,1X))')a
{flags_output}deallocate(before)
{cleanup}
enddo
enddo
print *,'GUARDED_ELSEIF_FIELDS_HALOS_BOUNDS_AND_LAZY_BRANCHES_OK'
end program
"""


def sealed_emission(directory, variant):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    directory.mkdir(parents=True, exist_ok=True)
    original = directory / "original.f90"
    original.write_text(source(variant))
    fact = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    captures = {"argument::a": dict(fact)}
    if variant == "managed":
        captures["argument::flags"] = dict(fact)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(original): sha256(original.read_bytes()).hexdigest()}, "captures": captures}
    builder = ScopeBuilder([original], "guarded_device::advance", facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    validate_checked_execution_plans(json.loads(outputs["scope-manifest.json"]), capture_facts=facts)
    package = directory / "strict-package"
    for name, content in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    manifest = json.loads((package / "scope-manifest.json").read_text())
    _scope_package_build({"artifact_root": str(package), "scope_manifest": manifest})
    assert builder.finish()
    assert report["scopes"][0]["gpu_leaves"]
    transformed = outputs[report["sources"][str(original)]["replacement"]]
    assert "elseif(flags(1)==2)then" not in "".join(transformed.lower().split())
    return original, outputs, manifest


@pytest.mark.parametrize("variant", ["metadata", "managed", "alias"])
def test_guarded_device_fixture_source_and_native_driver(tmp_path, variant):
    original, _outputs, _manifest = sealed_emission(tmp_path, variant)
    for text in (source(variant), driver(variant)):
        assert all(line == line.rstrip() and len(line) <= 132 for line in text.splitlines())
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    assert fortran
    native_driver = tmp_path / "driver.f90"
    native_driver.write_text(driver(variant))
    executable = tmp_path / "native"
    base.checked([fortran, "-fopenmp", "-fcheck=all,array-temps", "-ffp-contract=off", "-J", str(tmp_path),
                  str(original), str(native_driver), "-o", str(executable)], tmp_path, "native-compile")
    result = base.checked([str(executable)], tmp_path, "native-exercise", environment("untraced"))
    assert "GUARDED_ELSEIF_FIELDS_HALOS_BOUNDS_AND_LAZY_BRANCHES_OK" in result.stdout


@pytest.fixture(scope="module", params=["metadata", "managed", "alias"])
def guarded_binaries(tmp_path_factory, request):
    variant = request.param
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, "emit", lambda directory: sealed_emission(directory, variant))
        patch.setattr(base, "DRIVER", driver(variant))
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((directory / "scope-manifest.json").read_text())
    resources = {row["resource"]: row["registration_identity"]
                 for row in manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]["resources"]}
    return variant, directory, candidate, native, resources


@pytest.mark.cuda
@pytest.mark.parametrize("mode", ["traced", "untraced", "hidden"])
def test_guarded_chain_keeps_lazy_reads_current_and_preserves_replicas(guarded_binaries, mode):
    variant, directory, candidate, native, resources = guarded_binaries
    env = environment(mode)
    reference = base.checked([str(native)], directory, mode + "-native", env)
    result = base.checked([str(candidate)], directory, mode + "-candidate", env)
    assert result.stdout == reference.stdout
    if mode == "untraced":
        assert "FORT_SCOPED " not in result.stderr
        return
    zero, = re.findall(r"GUARDED_ZERO_BEGIN\n(.*?)GUARDED_ZERO_END", result.stderr, re.S)
    assert not physical_gpu_events(zero)
    assert not any(row["kind"] in {"domain_create", "register"} for row in events_in(zero))
    cases = re.findall(r"GUARDED_CASE_BEGIN (\d+) (\d+)\n(.*?)GUARDED_CASE_END \1 \2", result.stderr, re.S)
    assert len(cases) == (8 if variant == "alias" else 12)
    ledger = []
    for selected, width, text in cases:
        branch, n = int(selected), int(width)
        events = events_in(text)
        ledger.append({"branch": branch, "n": n, "events": events})
        (directory / (mode + "-ledger.json")).write_text(json.dumps(ledger, indent=2) + "\n")
        if mode == "hidden":
            assert not physical_gpu_events(text)
            continue
        launches = [i for i, row in enumerate(events) if row["kind"] == "launch"]
        expected_launches = 4 if variant == "managed" else 2 if variant == "alias" and branch >= 3 else 3
        assert len(launches) == expected_launches, text
        a_events = [row for row in events if int(row.get("buffer", -1)) == resources["argument::a"]]
        if variant == "alias" and branch > 1:
            # The first reached opaque metadata reservation aliases dirty A.
            # Its native predicate must see the GPU prefix before selecting.
            between = events[launches[0]+1:launches[-1]]
            assert any(row["kind"] == "domain_retire" for row in between), text
            assert copy_bytes(between, "download") >= 8*n, text
            assert events[launches[0]]["domain"] != events[launches[-1]]["domain"], text
            # After the first alias publishes A, a reached second metadata
            # reservation owns its range. The selected A-writing body then
            # declines on that alias and runs natively before the GPU tail.
            assert sum(row["kind"] == "domain_retire" for row in between) == (2 if branch >= 3 else 1), text
            assert copy_bytes(events, "upload") == copy_bytes(events, "download") == 16*n, text
        else:
            assert len({events[i]["domain"] for i in launches}) == 1, text
            assert copy_bytes(a_events, "upload") == 8*n, text
            assert copy_bytes(a_events, "download") == 8*n, text
            assert copy_bytes(events[:launches[-1]], "download") == (4 if variant == "managed" and branch > 1 else 0)
        if variant == "managed":
            flag_events = [row for row in events if int(row.get("buffer", -1)) == resources["argument::flags"]]
            assert copy_bytes(flag_events, "upload") == 8, text
            assert copy_bytes(flag_events, "download") == 8, text
            assert not any(row["kind"] == "upload" and int(row.get("buffer", -1)) == resources["argument::flags"]
                           for row in events[launches[0]+1:]), text
        elif variant == "metadata":
            assert copy_bytes(events, "upload") == copy_bytes(events, "download") == 8*n, text
        if branch == 1:
            assert not any(row["kind"] == "domain_retire" for row in events[launches[0]+1:launches[-1]]), text
