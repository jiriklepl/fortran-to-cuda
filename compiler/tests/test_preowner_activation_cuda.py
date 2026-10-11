"""Original prefix copies acquire current descriptors before one CUDA owner."""

from __future__ import annotations

import json
import os
import re
import shutil
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.numerical_contract import (
    require_explicit_cuda_environment,
    require_numerical_build_contract,
)
from compiler.tests.test_preowner_activation import PREOWNER_SOURCE, emit_preowner_window
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run


def _source_with_native_boundaries():
    # False single-line guards keep the owner eligible. True guards disable
    # before each original call, including the separate opaque control.
    source = PREOWNER_SOURCE.replace("""if(bad) then
call opaque_prefix()
bad_visits=bad_visits+1
endif
if(skip .and. stage==1) then
cycle
endif""", """if(bad.and.stage==1) call original_notice()
if(bad.and.stage/=1) call original_joined_team()
if(bad) call opaque_prefix()
if(bad) bad_visits=bad_visits+1
if(skip .and. stage==1) cycle""")
    return source.removesuffix("end module\n") + """subroutine original_notice()
write(*,'(a)') 'original native notice'
bad_visits=bad_visits+1000
end subroutine
subroutine original_joined_team()
!$omp parallel default(shared)
!$omp single
bad_visits=bad_visits+10000
!$omp end single
!$omp end parallel
end subroutine
end module
"""


def _driver(precision):
    return f"""subroutine opaque_prefix()
! Deliberately absent from configured source analysis; original native once.
use renamed_prefix,only:bad_visits
bad_visits=bad_visits+100
end subroutine
program verify
use renamed_prefix,only:advance,workspace,prefix_visits,tail_visits,bad_visits,condition_visits,route
implicit none
real({precision}),allocatable::a(:),b(:)
integer,parameter::sizes(4)=[0,0,7,4096],extents(4)=[0,12,19,4108]
integer::shape,state,iteration,n,extent,i,unit,iterations,expected_lower
logical::bad,skip
character(16)::path
call get_command_argument(1,path)
if(path/='eligible'.and.path/='opaque'.and.path/='skip'.and.path/='zero') error stop 'unknown path'
bad=path=='opaque'
skip=path=='skip'
iterations=3
if(path=='zero') iterations=0
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
 n=sizes(shape)
 extent=extents(shape)
 allocate(a(-11-shape:-12-shape+extent))
 do state=1,3
  ! The conforming destination has DIFFERENT bounds from a and its dummy.
  ! With extent=n+12, both still contain the original -2:n-3 interior.
  if(state==2) allocate(b(-7:-8+extent))
  if(state==3) allocate(b(-13:-12))
  do iteration=1,2
   do i=lbound(a,1),ubound(a,1)
    a(i)=real(i+32*iteration,{precision})*0.125_{precision}
   enddo
   if(allocated(b)) b=-77._{precision}
   ! Mutable host-only metadata must be read anew at the original condition.
   route=[iteration,merge(shape,-shape,state/=2)]
   call advance(a,b,n,iterations,bad,skip)
   if(.not.allocated(b).or..not.allocated(workspace)) error stop 'original copy did not allocate'
   if(size(b)/=extent.or.size(workspace)/=extent) error stop 'stale destination shape'
   expected_lower=-2
   if(state==2) expected_lower=-7
   if(extent==0) expected_lower=1
   if(lbound(b,1)/=expected_lower) error stop 'original assignment lower bounds changed'
   if(extent>0.and.lbound(workspace,1)/=-2) error stop 'workspace descriptor rebased incorrectly'
   write(unit)shape,state,iteration,prefix_visits,tail_visits,bad_visits,condition_visits
   write(unit)lbound(a),ubound(a),lbound(b),ubound(b),lbound(workspace),ubound(workspace)
   write(unit)a,b,workspace
  enddo
  deallocate(b)
 enddo
 deallocate(a)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=((4, False), (4, True), (8, False), (8, True)))
def preowner_binaries(tmp_path_factory, request):
    precision, resource_fallback = request.param
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    # If the tools exist, device/probe/build failures are test failures.
    require_explicit_cuda_environment()
    directory = tmp_path_factory.mktemp("preowner_" + str(precision))
    architecture = actual_profile(directory, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    source = _source_with_native_boundaries()
    if precision == 4:
        source = source.replace("real(8)", "real(4)").replace("_8", "_4").replace(",8)", ",4)")
    original, outputs, report, owner = emit_preowner_window(
        directory, source, device_budget=0 if resource_fallback else None)
    assert original.read_text() == source
    scope, = report["scopes"]
    assert len(scope["gpu_leaves"]) == 2
    assert scope["reached_window"]["prefix_eligibility"]["activation_path_possible"]
    conditional = [point for point in scope["reached_window"]["prefix_eligibility"]["disable_points"]
                   if point.get("conditional_action")]
    assert len(conditional) == 4
    assert all(point["original_source_kind"] == "If_Stmt" for point in conditional)
    assert [point["conditional_action"]["action_source_kind"] for point in conditional].count("Call_Stmt") == 3
    assert [point["conditional_action"]["action_source_kind"] for point in conditional].count("Cycle_Stmt") == 1
    assert scope["reached_window"]["prefix_eligibility"]["disabled_path_possible"]
    disabled_calls = {point["reason"] for point in
                      scope["reached_window"]["prefix_eligibility"]["disable_points"]
                      if point.get("conditional_action", {}).get("action_source_kind") == "Call_Stmt"}
    assert any("Write_Stmt" in reason for reason in disabled_calls)
    assert any("PARALLEL DO" in reason for reason in disabled_calls)
    for root in ("argument::b", "renamed_prefix::workspace"):
        assert owner.builder.facts["captures"][root]["initialized"] == "none"
        assert owner.builder.facts["captures"][root]["lifetime"] == "selected_window"
    for name, text in outputs.items():
        artifact = directory / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    flags = [fortran, "-O3", "-cpp", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-ffp-contract=off"]
    objects, commands = [], []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] != role:
                continue
            artifact = directory / item["path"]
            target = artifact.with_suffix(".o")
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_" + architecture,
                            "-Xcompiler=-fopenmp", *contract["required_cuda_options"],
                            *("-Xcompiler=" + value for value in contract["required_host_options"])]
            else:
                compiler = flags
            command = [*compiler, "-I", str(directory), "-c", str(artifact), "-o", str(target)]
            run(command, cwd=directory)
            commands.append({"role": role, "argv": command,
                             "numerical_contract": item.get("numerical_contract")})
            objects.append(str(target))
    driver = directory / "driver.f90"
    driver.write_text(_driver(precision))
    library = Path(nvcc).resolve().parents[1] / "lib64"
    candidate, native = directory / "candidate", directory / "native"
    for role, command in (
        ("candidate-link", [*flags, "-I", str(directory), str(driver), *objects,
                            "-L" + str(library), "-Wl,-rpath," + str(library),
                            "-lcudart", "-lstdc++", "-o", str(candidate)]),
        ("native-compile-link", [*flags, str(original), str(driver), "-o", str(native)]),
    ):
        run(command, cwd=directory)
        commands.append({"role": role, "argv": command})
    (directory / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (directory / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return directory, candidate, native, precision, resource_fallback


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("path", ["eligible", "opaque", "skip", "zero"])
def test_original_prefix_current_bounds_and_absorbing_native_paths(preowner_binaries, disabled, path):
    directory, candidate, native, precision, resource_fallback = preowner_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    # Each path gets its own process/trace. Eligible calls cannot conceal a
    # transfer or launch on an absorbing-disabled or zero-trip path.
    reference = run([str(native), path], cwd=directory, env=environment)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate), path], cwd=directory, env=environment)
    actual = (directory / "fields.bin").read_bytes()
    label = path + ("-disabled" if disabled else "-cuda")
    (directory / (label + "-native.bin")).write_bytes(expected)
    (directory / (label + "-candidate.bin")).write_bytes(actual)
    (directory / (label + ".stderr")).write_text(result.stderr)
    assert actual == expected
    assert "array temporary" not in result.stderr.lower()
    assert "array temporary" not in reference.stderr.lower()
    launches = result.stderr.count("FORT_SCOPED launch ")
    uploaded = sum(map(int, re.findall(r"FORT_SCOPED upload .*bytes=(\d+)", result.stderr)))
    downloaded = sum(map(int, re.findall(r"FORT_SCOPED download .*bytes=(\d+)", result.stderr)))
    numerical = path == "eligible" and not disabled and not resource_fallback
    # Two nonempty interiors x three initial b states x two calls x two
    # kernels. Later original RK-like iterations execute native exactly once.
    assert launches == (24 if numerical else 0)
    if numerical:
        assert uploaded > 0
        assert downloaded > 0
    else:
        assert uploaded == downloaded == 0
    if path in {"opaque", "skip", "zero"}:
        assert "FORT_SCOPED create " not in result.stderr
    receipt = {"schema_version": 1, "precision_bytes": precision, "path": path,
               "cuda_disabled": disabled, "resource_preexecution_fallback": resource_fallback,
               "complete_arrays_halos_bounds_and_original_visits_equal": True,
               "conforming_lhs_retains_distinct_lower_bound": True,
               "empty_and_changing_original_allocations": True,
               "source_copy_destinations_initially_undefined_in_capture_facts": True,
               "dormant_metadata_reloaded_each_call": True,
               "source_backed_io_and_joined_team_execute_once_on_original_paths": True,
               "single_line_guards_execute_original_calls_once": True,
               "single_line_cycle_is_absorbing": True,
               "independently_traced_path": True, "numerical_launches": launches,
               "h2d_bytes": uploaded, "d2h_bytes": downloaded,
               "compared_bytes": len(actual), "native_sha256": sha256(expected).hexdigest(),
               "candidate_sha256": sha256(actual).hexdigest(),
               "trace_sha256": sha256(result.stderr.encode()).hexdigest(),
               "performance_samples": 0}
    (directory / (label + "-receipt.json")).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
