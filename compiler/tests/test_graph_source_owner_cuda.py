"""Mixed-source graph execution agrees with native Fortran across loop backedges."""

from __future__ import annotations

import json
import os
import re
import shutil

import pytest

from compiler.numerical_contract import require_numerical_build_contract
from compiler.tests.test_graph_source_owner import FACT, SOURCE, emit
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run

DRIVER = """program verify
use graph_fixture
implicit none
real(8),allocatable::a(:),b(:)
integer,allocatable::x(:)
integer::n,shape,repeat,i,unit
integer,parameter::sizes(3)=[0,1,17]
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
n=sizes(shape)
allocate(a(-3:n+4),b(-7:n),x(-11:n-4))
do repeat=1,2
do i=lbound(a,1),ubound(a,1)
a(i)=real(i+5,8)*0.25d0+repeat
enddo
b=-99.d0
x=[(i,i=1,size(x))]
call advance(a,b,x,n)
write(unit)n,lbound(a),ubound(a),lbound(b),ubound(b),lbound(x),ubound(x)
write(unit)a,b,x
enddo
deallocate(a,b,x)
enddo
close(unit)
end program
"""

MAPPED_SOURCE = """module graph_fixture
implicit none
contains
subroutine advance(a,b,x,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(inout)::x(:)
integer,intent(in)::n
integer::i,stage
do stage=1,3
do i=1,n
b(i)=2.0_8*a(i)
enddo
call helper(a,x,n)
call helper(b,x,n)
do i=1,n
a(i)=a(i)+b(i)
enddo
enddo
end subroutine
subroutine helper(y,q,n)
real(8),intent(inout)::y(:)
integer,intent(inout)::q(:)
integer,intent(in)::n
integer::i
do i=2,n
q(i)=q(i-1)+q(i)
enddo
do i=1,n
y(i)=y(i)+3.0_8
enddo
end subroutine
end module
"""

EPOCH_SOURCE = """module graph_fixture
implicit none
contains
subroutine advance(a,b,x,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(inout)::x(:)
integer,intent(in)::n
integer::i
interface
subroutine opaque(v,n)
real(8),intent(inout)::v(*)
integer,intent(in)::n
end subroutine
end interface
call opaque(a,n)
do i=1,n
b(i)=2.0_8*a(i)
enddo
call opaque(b,n)
do i=1,n
a(i)=b(i)+1.0_8
enddo
end subroutine
end module
"""

OPAQUE_SOURCE = """subroutine opaque(v,n)
implicit none
real(8),intent(inout)::v(*)
integer,intent(in)::n
integer::i
do i=1,n
v(i)=v(i)+4.0_8
enddo
end subroutine
"""

MULTI_NATIVE_SOURCE = """module graph_fixture
implicit none
contains
subroutine advance(a,b,x,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(inout)::x(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=a(i)+1.0_8
enddo
call native_lookup(a,b,x,n)
do i=1,n
a(i)=a(i)+b(i)
enddo
end subroutine
subroutine native_lookup(a,b,x,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::x(:),n
integer::i
do i=1,n
b(x(i))=a(x(i))+4.0_8
enddo
end subroutine
end module
"""


FAILED_EPOCH_SOURCE = EPOCH_SOURCE.replace("b(i)=2.0_8*a(i)",
    "b(i)=2.0_8*a(i)\na(i)=b(i)+1.0_8\nb(i)=a(i)+1.0_8")
HELPER_DESCRIPTOR_SOURCE = EPOCH_SOURCE.replace("call opaque(b,n)", "call helper(x,n)").replace(
    "contains\n", """contains
subroutine helper(x,n)
integer,intent(inout)::x(:)
integer,intent(in)::n
integer::i
do i=1,n
x(i)=x(i)+1
enddo
end subroutine
""", 1)


@pytest.fixture(scope="module", params=["mixed", "mapped", "epochs", "epochs_fallback", "helper_descriptor", "multi_native"])
def graph_source_binaries(request, tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    directory = tmp_path_factory.mktemp("mixed_source_graph")
    architecture = actual_profile(directory, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    source = (
        MULTI_NATIVE_SOURCE if request.param == "multi_native" else
        EPOCH_SOURCE if request.param == "epochs" else
        FAILED_EPOCH_SOURCE if request.param == "epochs_fallback" else
        HELPER_DESCRIPTOR_SOURCE if request.param == "helper_descriptor" else
        MAPPED_SOURCE
        if request.param == "mapped"
        else (
            SOURCE.replace("integer::i", "integer::i,stage")
            .replace("do i=1,n\nb(i)", "do stage=1,3\ndo i=1,n\nb(i)", 1)
            .replace("end subroutine", "enddo\nend subroutine", 1)
        )
    )
    captures = ({"argument::x": {**FACT, "expected_descriptor": {"lower_bounds": [-11], "extents": [9999]}}}
                if request.param == "helper_descriptor" else None)
    original, outputs, report = emit(directory, source, captures=captures)
    (owner,) = report["scopes"]
    if request.param in {"epochs", "epochs_fallback", "helper_descriptor"}:
        assert owner["boundaries"]
    else:
        assert not owner["boundaries"]
        if request.param != "multi_native":
            assert owner["counted_controls"]
    for name, text in outputs.items():
        artifact = directory / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    flags = [fortran, "-O2", "-cpp", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-ffp-contract=off"]
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] != role:
                continue
            artifact = directory / item["path"]
            target = artifact.with_suffix(".o")
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [
                    nvcc,
                    "-O2",
                    "-std=c++17",
                    "-ccbin",
                    host,
                    "-arch=sm_" + architecture,
                    "-Xcompiler=-fopenmp",
                    *contract["required_cuda_options"],
                    *("-Xcompiler=" + option for option in contract["required_host_options"]),
                ]
            else:
                compiler = flags
            run([*compiler, "-I", str(directory), "-c", str(artifact), "-o", str(target)], cwd=directory)
            objects.append(str(target))
    driver = directory / "driver.f90"
    driver.write_text(DRIVER)
    external = []
    if request.param in {"epochs", "epochs_fallback", "helper_descriptor"}:
        opaque = directory / "opaque.f90"
        opaque.write_text(OPAQUE_SOURCE)
        external.append(str(opaque))
    native, candidate = directory / "native", directory / "candidate"
    run([*flags, str(original), str(driver), *external, "-o", str(native)], cwd=directory)
    run(
        [
            *flags,
            str(driver),
            *external,
            *objects,
            "-L/usr/local/cuda/lib64",
            "-Wl,-rpath,/usr/local/cuda/lib64",
            "-lcudart",
            "-lstdc++",
            "-o",
            str(candidate),
        ],
        cwd=directory,
    )
    (directory / "manifest.json").write_text(json.dumps(report, indent=2))
    launches, submissions, uploads, downloads = {
        "mixed": (24, 4, 288, 576), "mapped": (48, 4, 288, 576), "epochs": (8, 8, 576, 576),
        "epochs_fallback": (4, 4, 288, 288), "helper_descriptor": (8, 8, 576, 576),
        "multi_native": (8, 8, 576, 576),
    }[request.param]
    return directory, native, candidate, launches, submissions, uploads, downloads


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_mixed_graph_keeps_versions_across_native_tasks_and_three_iterations(graph_source_binaries, disabled):
    directory, native, candidate, expected_launches, submissions, expected_uploads, expected_downloads = graph_source_binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        env["CUDA_VISIBLE_DEVICES"] = ""
    run([str(native)], cwd=directory, env=env)
    expected = (directory / "fields.bin").read_bytes()
    result = run([str(candidate)], cwd=directory, env=env)
    assert (directory / "fields.bin").read_bytes() == expected
    assert "array temporary" not in result.stderr.lower()
    launches = sum(line.startswith("FORT_SCOPED launch ") for line in result.stderr.splitlines())
    launches += sum(
        int(re.search(r"\bbytes=(\d+)", line).group(1))
        for line in result.stderr.splitlines()
        if line.startswith("FORT_SCOPED graph_submit ")
    )
    if disabled:
        assert not launches
        assert "FORT_SCOPED upload " not in result.stderr
        assert "FORT_SCOPED download " not in result.stderr
    else:
        assert launches == expected_launches
        reports = [
            json.loads(line.removeprefix("FORT_SCOPED evidence "))
            for line in result.stderr.splitlines()
            if line.startswith("FORT_SCOPED evidence ")
        ]
        graph_reports = [report for report in reports if report["event"] == "graph_statistics"]
        queue_reports = [report for report in reports if report["event"] == "command_queue_statistics"]
        # Four nonempty invocations each submit all three loop iterations.
        # Source units, helper calls and unrelated host work add no submit.
        assert (sum(report["submissions"] for report in graph_reports)
                + sum(report["direct_submissions"] for report in queue_reports)) == submissions
        assert (sum(report["logical_kernels"] for report in graph_reports)
                + sum(report["direct_kernels"] for report in queue_reports)) == expected_launches
        transfers = [
            line
            for line in result.stderr.splitlines()
            if line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download "))
        ]
        uploads = sum(int(re.search(r"\bbytes=(\d+)", line).group(1)) for line in transfers if " upload " in line)
        downloads = sum(int(re.search(r"\bbytes=(\d+)", line).group(1)) for line in transfers if " download " in line)
        assert uploads == expected_uploads
        assert downloads == expected_downloads
