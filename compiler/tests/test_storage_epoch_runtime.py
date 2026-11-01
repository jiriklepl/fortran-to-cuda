"""Generated mixed tasks retain replicas and publish complete original fields."""
import json
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.tests.test_graph_source_owner import FACT, SOURCE, emit, reusable_helper_source
from compiler.tests.test_source_scopes import run

MIXED_SOURCE = SOURCE.replace(
    "do i=2,n\nx(i)=x(i-1)+1\nenddo", "x(1)=int(a(1)+b(1))"
).replace("a(i)=b(i)+1.0_8", "a(i)=a(i)+b(i)+real(x(1),8)")
DRIVER = """program exercise
use graph_fixture
implicit none
real(8)::a(-3:4),b(-3:4)
integer::x(-8:-1),i
do i=-3,4
 a(i)=real(i+4,8)
enddo
b=-99
x=-7
call advance(a,b,x,8)
if(x(-8)/=3)error stop 'native probe'
if(any(x(-7:-1)/=-7))error stop 'untouched integer cells'
do i=-3,4
 if(b(i)/=2*real(i+4,8))error stop 'complete intermediate'
 if(a(i)/=3*real(i+4,8)+3)error stop 'complete output'
enddo
print *, 'FIELDS_OK'
end program
"""


def test_mixed_source_requires_gpu_host_gpu_order_without_retirement(tmp_path):
    _source, _outputs, manifest = emit(tmp_path, MIXED_SOURCE,
                                     captures={"argument::b": {**FACT, "initialized": "none"}})
    owner, = manifest["scopes"]
    assert not owner["boundaries"]
    procedure, = owner["graph_execution"]["procedures"]
    assert [item["target"] for item in procedure["schedule"]["placements"]] == ["gpu", "cpu", "gpu"]
    assert not owner["graph_execution"]["source_motion"]


def build_binaries(directory, source, outputs, manifest, driver_text):
    """Compile the exact emitted artifacts and the original reference body."""
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/Fortran toolchain unavailable")
    build = directory / "build"
    build.mkdir()
    (build / "scope-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["role"] != role:
                continue
            path = build / artifact["path"]
            assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
            target = build / f"object-{index}.o"
            command = ([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp", "-I", str(build)]
                       if artifact["language"] == "cuda" else
                       [fortran, "-std=f2018", "-fopenmp", "-fcheck=all", "-J", str(build), "-I", str(build)])
            run([*command, "-c", str(path), "-o", str(target)], cwd=build)
            objects.append(str(target))
    driver = build / "driver.f90"
    driver.write_text(driver_text)
    candidate, native = build / "candidate", build / "native"
    run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", str(driver), *objects,
         "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(candidate)], cwd=build)
    run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", str(source), str(driver), "-o", str(native)], cwd=build)
    return build, candidate, native


@pytest.fixture(scope="module")
def mixed_binaries(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mixed_epoch")
    source, outputs, manifest = emit(directory / "input", MIXED_SOURCE,
                                     captures={"argument::b": {**FACT, "initialized": "none"}})
    return build_binaries(directory, source, outputs, manifest, DRIVER)


@pytest.mark.native
def test_generated_mixed_domain_keeps_native_recovery(mixed_binaries):
    build, candidate, native = mixed_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "CUDA_VISIBLE_DEVICES": "-1",
                   "FORT_RUNTIME_TRACE": "1"}
    for executable in (native, candidate):
        result = run([str(executable)], cwd=build, env=environment)
        assert "FIELDS_OK" in result.stdout
        assert "FORT_SCOPED launch " not in result.stderr


@pytest.mark.cuda
def test_generated_mixed_domain_retains_readonly_mirrors(mixed_binaries):
    build, candidate, _ = mixed_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    result = run([str(candidate)], cwd=build, env=environment)
    (build / "gpu.stdout").write_text(result.stdout)
    (build / "gpu.stderr").write_text(result.stderr)
    assert "FIELDS_OK" in result.stdout
    lines = result.stderr.splitlines()
    assert sum(line.startswith("FORT_SCOPED launch ") for line in lines) == 2, result.stderr
    uploads = [line for line in lines if line.startswith("FORT_SCOPED upload ")]
    downloads = [line for line in lines if line.startswith("FORT_SCOPED download ")]
    def amount(line):
        return int(re.search(r"\bbytes=(\d+)\b", line)[1])
    assert sorted(map(amount, uploads)) == [4, 64], result.stderr
    assert sum(map(amount, downloads)) == 128, result.stderr
    # The native read obtains one GPU-produced cell while A's original host
    # replica survives. The remainder is published only at unmanaged return.
    first_launch = next(i for i, line in enumerate(lines) if line.startswith("FORT_SCOPED launch "))
    last_launch = max(i for i, line in enumerate(lines) if line.startswith("FORT_SCOPED launch "))
    assert [amount(line) for line in lines[first_launch:last_launch] if line.startswith("FORT_SCOPED download ")] == [8]


@pytest.fixture(scope="module")
def saved_binaries(tmp_path_factory):
    directory = tmp_path_factory.mktemp("saved_epoch")
    body = reusable_helper_source(False).replace(
        "integer::i\ndo i=1,n", "integer::i\ninteger,save::visits=0\nvisits=visits+1\ndo i=1,n"
    ).replace("field(i)=field(i)+1", "field(i)=field(i)+real(visits,8)")
    source, outputs, manifest = emit(directory / "input", body)
    assert not manifest["scopes"][0]["boundaries"]
    driver = """program exercise
use graph_fixture
implicit none
real(8)::a(-3:4),b(-3:4)
integer::x(8)
a=1
b=2
x=-7
call leaf(a,8)
if(any(a/=2))error stop 'original saved state'
call advance(a,b,x,8)
if(any(a/=4).or.any(b/=5))error stop 'shared ENTRY state'
call leaf(a,8)
if(any(a/=8))error stop 'original sees ENTRY state'
call advance(a,b,x,8)
if(any(a/=13).or.any(b/=11))error stop 'repeated ENTRY state'
if(any(x/=-7))error stop 'untouched argument'
print *, 'SAVED_FIELDS_OK'
end program
"""
    return build_binaries(directory, source, outputs, manifest, driver)


@pytest.mark.native
def test_saved_scalar_is_shared_by_original_and_checked_entries_without_cuda(saved_binaries):
    build, candidate, native = saved_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "CUDA_VISIBLE_DEVICES": "-1",
                   "FORT_RUNTIME_TRACE": "1"}
    for executable in (native, candidate):
        result = run([str(executable)], cwd=build, env=environment)
        assert "SAVED_FIELDS_OK" in result.stdout
        assert "FORT_SCOPED launch " not in result.stderr


@pytest.mark.cuda
def test_saved_scalar_stays_shared_while_checked_children_execute_on_gpu(saved_binaries):
    build, candidate, _ = saved_binaries
    result = run([str(candidate)], cwd=build,
                 env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"})
    (build / "gpu.stdout").write_text(result.stdout)
    (build / "gpu.stderr").write_text(result.stderr)
    assert "SAVED_FIELDS_OK" in result.stdout
    assert sum(line.startswith("FORT_SCOPED launch ") for line in result.stderr.splitlines()) == 4, result.stderr


@pytest.fixture(scope="module")
def borrowed_swap_binaries(tmp_path_factory):
    directory = tmp_path_factory.mktemp("borrowed_swap_epoch")
    body = """module graph_fixture
implicit none
contains
subroutine producer(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2*a(i)
enddo
end subroutine
subroutine consumer(a,b,x,n)
real(8),intent(inout)::a(:)
real(8),intent(in)::b(:)
integer,intent(in)::x(:),n
integer::i
do i=1,n
a(i)=a(i)+b(i)+real(x(1),8)
enddo
end subroutine
subroutine relocate(left,right)
real(8),allocatable,intent(inout)::left(:),right(:)
real(8),allocatable::temporary(:)
call move_alloc(left,temporary)
call move_alloc(right,left)
call move_alloc(temporary,right)
end subroutine
subroutine advance(a,b,x,n)
real(8),allocatable,intent(inout)::a(:),b(:)
integer,intent(inout)::x(:)
integer,intent(in)::n
call producer(a,b,n)
x(1)=int(a(1)+b(1))
call consumer(a,b,x,n)
call relocate(a,b)
end subroutine
end module
"""
    source, outputs, manifest = emit(directory / "input", body)
    owner, = manifest["scopes"]
    assert not owner["boundaries"]
    assert len(owner["module_coordinators"]) == 2
    driver = """program exercise
use graph_fixture
implicit none
real(8),allocatable::a(:),b(:)
integer::x(-8:-1),i
allocate(a(-1:9),b(-3:10))
do i=-1,9
a(i)=real(i+5,8)
enddo
b=-999
x=-7
call advance(a,b,x,8)
if(x(-8)/=22.or.any(x(-7:-1)/=-7))error stop 'host observation'
if(lbound(a,1)/=-3.or.ubound(a,1)/=10)error stop 'first descriptor'
if(lbound(b,1)/=-1.or.ubound(b,1)/=9)error stop 'second descriptor'
do i=-3,10
if(i<=4)then
if(a(i)/=2*real(i+7,8))error stop 'producer allocation'
else
if(a(i)/=-999)error stop 'producer halo'
endif
enddo
do i=-1,9
if(i<=6)then
if(b(i)/=3*real(i+5,8)+22)error stop 'consumer allocation'
else
if(b(i)/=real(i+5,8))error stop 'consumer halo'
endif
enddo
print *, 'BORROWED_SWAP_FIELDS_OK'
end program
"""
    return build_binaries(directory, source, outputs, manifest, driver)


@pytest.mark.native
def test_borrowed_children_host_probe_and_swap_recover_without_cuda(borrowed_swap_binaries):
    build, candidate, native = borrowed_swap_binaries
    for executable in (native, candidate):
        result = run([str(executable)], cwd=build,
                     env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"})
        assert "BORROWED_SWAP_FIELDS_OK" in result.stdout


@pytest.mark.cuda
def test_borrowed_children_preserve_readonly_mirrors_through_host_probe_and_swap(borrowed_swap_binaries):
    build, candidate, _ = borrowed_swap_binaries
    result = run([str(candidate)], cwd=build,
                 env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"})
    (build / "gpu.stdout").write_text(result.stdout)
    (build / "gpu.stderr").write_text(result.stderr)
    assert "BORROWED_SWAP_FIELDS_OK" in result.stdout
    lines = result.stderr.splitlines()
    assert sum(line.startswith("FORT_SCOPED launch ") for line in lines) == 2, result.stderr
    assert sum(line.startswith("FORT_SCOPED identity_swap ") for line in lines) == 1, result.stderr
    assert sum(int(re.search(r"\bbytes=(\d+)\b", line)[1]) for line in lines
               if line.startswith("FORT_SCOPED upload ")) == 68, result.stderr
    assert sum(int(re.search(r"\bbytes=(\d+)\b", line)[1]) for line in lines
               if line.startswith("FORT_SCOPED download ")) == 128, result.stderr
