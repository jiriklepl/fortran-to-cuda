"""Counted calls, immutable operands and native IEEE teams share one owner."""

import json
import os
import re
import resource
import shutil
import signal
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import FACT, run


def sources(directory, precision):
    constants, child, owner = [directory / name for name in ("constants.f90", "child.f90", "owner.f90")]
    constants.write_text(f"""module coefficients
real({precision}),parameter::weights(-2:0)=[2._{precision},-0._{precision},0.5_{precision}]
end module
""")
    child.write_text(f"""module update_lib
contains
subroutine update(a,out,n,stage,weights)
use ieee_exceptions
implicit none
real({precision}),intent(in)::a(-2:),weights(-1:1)
real({precision}),intent(inout)::out(-2:)
integer,intent(in)::n,stage
integer::j
logical::modes(size(ieee_all))
logical::team_modes(size(ieee_all))
!$omp parallel private(team_modes)
call ieee_get_halting_mode(ieee_all,team_modes)
call ieee_set_halting_mode(ieee_all,.false.)
call ieee_set_halting_mode(ieee_all,team_modes)
!$omp end parallel
call ieee_get_halting_mode(ieee_all,modes)
call ieee_set_halting_mode(ieee_all,.false.)
do j=1,n
out(j)=a(j)*weights(stage)+real(j,{precision})
enddo
call ieee_set_halting_mode(ieee_all,modes)
end subroutine
end module
""")
    owner.write_text(f"""module owner_lib
use coefficients,only:renamed=>weights
use update_lib,only:renamed_update=>update
implicit none
integer::visits=0
contains
subroutine advance(a,b,out,n,escape)
real({precision}),intent(in)::a(-2:)
real({precision}),intent(inout)::b(-2:),out(-2:)
integer,intent(in)::n
logical,intent(in)::escape
integer::stage,i,effective_stage
visits=visits+1
do stage=-1,1
effective_stage=stage
if(n==0) effective_stage=huge(1)
call renamed_update(a,out,n,effective_stage,renamed)
if(escape.and.stage==0) then
call opaque(out,n)
endif
!$omp parallel private(i)
!$omp do
do i=1,n
b(i)=b(i)+out(i)
enddo
!$omp end do
!$omp end parallel
enddo
end subroutine
end module
""")
    return constants, child, owner


def driver_source(precision):
    return f"""subroutine opaque(out,n)
use owner_lib,only:visits
integer,intent(in)::n
real({precision}),intent(inout)::out(-2:n+7)
if(n>0) out(1)=out(1)+17._{precision}
visits=visits+100
end subroutine
program verify
use owner_lib,only:advance,visits
use ieee_exceptions
use ieee_arithmetic,only:ieee_value,ieee_positive_inf,ieee_negative_inf
use omp_lib,only:omp_get_thread_num
implicit none
real({precision}),allocatable::a(:),b(:),out(:)
integer,parameter::sizes(3)=[0,7,67]
integer::shape,iteration,i,n,unit,bad
logical::mode
character(16)::requested_case
call get_command_argument(1,requested_case)
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
!$omp parallel
if(requested_case=='mixed') then
call ieee_set_halting_mode(ieee_invalid,omp_get_thread_num()/=0)
else
call ieee_set_halting_mode(ieee_invalid,.true.)
endif
!$omp end parallel
do shape=1,size(sizes)
n=sizes(shape)
allocate(a(-7:n+2),b(-7:n+2),out(-7:n+2))
do iteration=1,2
do i=lbound(a,1),ubound(a,1)
a(i)=real(i+8,{precision})*0.25_{precision}+real(iteration,{precision})
enddo
b=-99._{precision}
out=-101._{precision}
if(requested_case=='trap') then
a=ieee_value(0._{precision},ieee_positive_inf)
b=ieee_value(0._{precision},ieee_negative_inf)
endif
call advance(a,b,out,n,iteration==2)
bad=0
!$omp parallel private(mode) reduction(+:bad)
call ieee_get_halting_mode(ieee_invalid,mode)
if(requested_case=='mixed'.and.omp_get_thread_num()==0) then
if(mode) bad=bad+1
else
if(.not.mode) bad=bad+1
endif
!$omp end parallel
if(bad/=0) error stop 'native thread environment changed'
write(unit)visits,a,b,out
enddo
deallocate(a,b,out)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=(4, 8))
def stage_binary(tmp_path_factory, request):
    precision = request.param
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/C++/Fortran toolchain unavailable")
    directory = tmp_path_factory.mktemp("reached_stage_"+str(precision))
    architecture = actual_profile(directory, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    paths = sources(directory, precision)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths},
             "captures": {"argument::"+name: FACT for name in ("a", "b", "out")}}
    outputs, report = ScopeBuilder(paths, "owner_lib::advance", facts=facts,
        options=CompilerOptions(), config=OffloadConfig(policy="sections", scope_execution="reached")).run()
    (directory / "manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    owner, = report["scopes"]
    assert owner["counted_controls"]
    assert len(owner["module_coordinators"]) == 1
    assert owner["boundaries"][0]["first_line"] > owner["counted_controls"][0]["first_line"]
    assert not owner["module_coordinators"][0]["boundaries"]
    assert len(owner["gpu_leaves"]) == 2
    for name, text in outputs.items():
        artifact = directory / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    commands = []

    def execute(command):
        commands.append(command)
        return run(command, cwd=directory)

    flags = [fortran, "-O3", "-fopenmp", "-fcheck=all,array-temps", "-ffp-contract=off"]
    constants_object = paths[0].with_suffix(".o")
    execute([*flags, "-c", str(paths[0]), "-o", str(constants_object)])
    objects = [str(constants_object)]
    source_order = {report["sources"][str(path)]["replacement"]: index
                    for index, path in enumerate(paths) if str(path) in report["sources"]}
    for role in ("common_runtime", "shared_entry", "original_source"):
        items = report["build_sources"]
        if role == "original_source":
            items = sorted(items, key=lambda item: source_order.get(item["path"], -1))
        for item in items:
            if item["role"] != role:
                continue
            artifact = directory / item["path"]
            target = artifact.with_suffix(".o")
            compiler = ([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_"+architecture,
                         "-Xcompiler=-fopenmp", *item.get("numerical_contract", {}).get("required_cuda_options", ()),
                         *["-Xcompiler="+option for option in item.get("numerical_contract", {}).get("required_host_options", ())]]
                        if item["language"] == "cuda" else flags)
            execute([*compiler, "-I", str(directory), "-c", str(artifact), "-o", str(target)])
            objects.append(str(target))
    driver = directory / "driver.f90"
    driver.write_text(driver_source(precision))
    library = Path(nvcc).resolve().parents[1] / "lib64"
    candidate, native = directory / "candidate", directory / "native"
    execute([*flags, "-I", str(directory), str(driver), *objects, "-L"+str(library),
             "-Wl,-rpath,"+str(library), "-lcudart", "-lstdc++", "-o", str(candidate)])
    execute([*flags, *map(str, paths), str(driver), "-o", str(native)])
    (directory / "commands.json").write_text(json.dumps(commands, indent=2)+"\n")
    run([str(native)], cwd=directory,
        env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"})
    expected = (directory / "fields.bin").read_bytes()
    (directory / "native-fields.bin").write_bytes(expected)
    return directory, candidate, expected, precision


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("halting_mode", ["uniform", "mixed"])
def test_reached_stage_complete_arrays_halos_and_escape_once(stage_binary, disabled, halting_mode):
    directory, candidate, expected, precision = stage_binary
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    if halting_mode == "mixed":
        run([str(directory / "native"), halting_mode], cwd=directory, env=environment)
        mixed_expected = (directory / "fields.bin").read_bytes()
        (directory / "native-mixed-fields.bin").write_bytes(mixed_expected)
        assert mixed_expected == expected
    result = run([str(candidate), halting_mode], cwd=directory, env=environment)
    label = ("disabled" if disabled else "cuda") + ("-mixed" if halting_mode == "mixed" else "")
    (directory / (label+".stderr")).write_text(result.stderr)
    actual = (directory / "fields.bin").read_bytes()
    (directory / (label+"-fields.bin")).write_bytes(actual)
    assert actual == expected
    launches = result.stderr.count("FORT_SCOPED launch ")
    # Child work disables traps; the parent accumulation restores the
    # original enabled trap and must execute coherently on the CPU. Each
    # nonempty shape runs three children, then two before the opaque escape.
    assert launches == (0 if disabled else 10)
    uploaded = sum(map(int, re.findall(r"FORT_SCOPED upload .*bytes=(\d+)", result.stderr)))
    downloaded = sum(map(int, re.findall(r"FORT_SCOPED download .*bytes=(\d+)", result.stderr)))
    receipt = {"precision": precision, "disabled": disabled, "halting_mode": halting_mode,
               "comparison": "complete native byte agreement",
               "bytes_compared": len(actual), "launches": launches, "h2d_bytes": uploaded, "d2h_bytes": downloaded,
               "native_sha256": sha256(expected).hexdigest(), "candidate_sha256": sha256(actual).hexdigest(),
               "performance_samples": 0}
    (directory / (label+"-receipt.json")).write_text(json.dumps(receipt, indent=2)+"\n")


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_restored_invalid_trap_stays_at_original_native_consumer(stage_binary, disabled):
    directory, candidate, _expected, precision = stage_binary
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
                   "FORT_RUNTIME_TRACE": "1", "GFORTRAN_ERROR_BACKTRACE": "0"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    # Avoid generated core files while retaining the original SIGFPE outcome.
    previous_limit = resource.getrlimit(resource.RLIMIT_CORE)
    outcomes = {}
    try:
        resource.setrlimit(resource.RLIMIT_CORE, (0, previous_limit[1]))
        for label, binary in (("native", directory / "native"), ("candidate", candidate)):
            result = subprocess.run([str(binary), "trap"], cwd=directory, env=environment,
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == -signal.SIGFPE, result.stdout + result.stderr
            outcomes[label] = {"returncode": result.returncode,
                               "launches": result.stderr.count("FORT_SCOPED launch ")}
            (directory / ("trap-"+label+("-disabled" if disabled else "")+".stderr")).write_text(result.stderr)
    finally:
        resource.setrlimit(resource.RLIMIT_CORE, previous_limit)
    assert outcomes["native"]["launches"] == 0
    assert outcomes["candidate"]["launches"] == (0 if disabled else 1)
    receipt = {"precision": precision, "disabled": disabled,
               "comparison": "original invalid trap after GPU child and native mode restore",
               "outcomes": outcomes, "performance_samples": 0}
    (directory / ("trap-"+("disabled" if disabled else "cuda")+"-receipt.json")).write_text(
        json.dumps(receipt, indent=2)+"\n")
