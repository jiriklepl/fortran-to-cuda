"""Public source artifacts preserve complete fields, holes and native snapshots."""

from hashlib import sha256
import ctypes
import os
from pathlib import Path
import shutil
import struct

import pytest

from compiler.tests.test_source_scopes import FACT, ROOT, generate, run


SOURCE = """module renamed_boundaries
contains
subroutine advance(a,b,lo,hi,scale,mode,checksum)
real(8),intent(inout)::a(-3:,-2:)
real(8),intent(in)::b(-7:,-6:)
integer,intent(in)::lo,hi,mode
real(8),intent(in)::scale
real(8),intent(out)::checksum
if(mode==1)then
 a=scale
else if(mode==2)then
 a=a*scale
else if(mode==3)then
 a(lo:hi:2,-2)=b(lo-4:hi-4:2,-6)+real(lbound(a,1),8)
else if(mode==4)then
 a(hi:lo:-2,-2)=b(hi-4:lo-4:-2,-6)
else
 a(lo:hi,-2)=a(lo-1:hi-1,-2)
endif
checksum=sum(a)
a=a+scale
end subroutine
end module
"""


DRIVER = """program exercise
use renamed_boundaries
implicit none
real(8),allocatable::a(:,:),b(:,:)
real(8)::checksum,scale
integer::n,i,j,k,mode,lo,unit
integer,parameter::sizes(4)=[-4,2,9,2]
character(1024)::output
call get_command_argument(1,output)
open(newunit=unit,file=trim(output),access='stream',form='unformatted',status='replace')
do k=1,size(sizes)
 n=sizes(k)
 allocate(a(-3:n,-2:2),b(-7:n-4,-6:-2))
 do mode=1,5
  if(n<-3.and.mode>2)cycle
  do j=-2,2
   do i=-3,n
    a(i,j)=real(i*17+j*3,8)/13.d0
    b(i-4,j-4)=real((i-4)*23+(j-4)*7,8)/11.d0
   enddo
  enddo
  lo=-3
  if(mode==5)lo=-2
  scale=real(mode,8)/7.d0
  call advance(a,b,lo,n,scale,mode,checksum)
  write(unit)a,checksum
 enddo
 deallocate(a,b)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module")
def array_operation_binaries(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/Fortran toolchain unavailable")
    directory = tmp_path_factory.mktemp("array_source_artifacts")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler",
                    ignore=shutil.ignore_patterns("__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::a": FACT, "argument::b": FACT}}
    results, shared = {}, {}
    for policy in ("sections", "auto"):
        original, output, manifest = generate(directory / policy, SOURCE, entry="advance", mode=policy,
                                             facts=facts, checkout=checkout)
        assert manifest["scope_count"], manifest["boundaries"]
        regions = manifest["inline_numerical_regions"]["regions"]
        assert any(region["operation_kind"] == "array_assignment" for region in regions)
        objects = []
        # Only public build roles and hashes are consumed by this independent
        # caller. No generated-CUDA or compiler-IR inspection is required.
        for role in ("common_runtime", "shared_entry", "original_source"):
            for item in manifest["build_sources"]:
                if item["role"] != role:
                    continue
                source = output / item["path"]
                assert sha256(source.read_bytes()).hexdigest() == manifest["artifacts_sha256"][item["path"]]
                target = source.with_suffix(".o")
                key = (manifest["runtime"]["runtime_id"], item["path"], sha256(source.read_bytes()).hexdigest())
                if role == "common_runtime" and item["language"] == "cuda" and key in shared:
                    objects.append(shared[key])
                    continue
                if item["language"] == "cuda":
                    command = [nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp", "-I", str(output)]
                else:
                    command = [fortran, "-std=f2018", "-fopenmp", "-fcheck=all"]
                run([*command, "-c", str(source), "-o", str(target)], cwd=output)
                objects.append(str(target))
                if role == "common_runtime" and item["language"] == "cuda":
                    shared[key] = str(target)
        driver = output / "caller.f90"
        driver.write_text(DRIVER)
        target, native = output / "candidate", output / "reference"
        run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", str(driver), *objects,
             "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++",
             "-o", str(target)], cwd=output)
        run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all", str(original), str(driver), "-o", str(native)], cwd=output)
        results[policy] = (target, native, output, manifest)
    return results


def compare(target, reference, output, *, disabled=False):
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    run([str(reference), "reference.bin"], cwd=output, env=environment)
    result = run([str(target), "candidate.bin"], cwd=output, env=environment)
    expected, actual = ((output / name).read_bytes() for name in ("reference.bin", "candidate.bin"))
    assert len(expected) == len(actual)
    count = len(expected) // 8
    values = struct.unpack("=" + "d" * count, expected)
    assert struct.unpack("=" + "d" * count, actual) == pytest.approx(values, rel=1e-12, abs=1e-12)
    return result


@pytest.mark.native
def test_source_array_operations_complete_fields_and_snapshots_without_cuda(array_operation_binaries):
    for target, reference, output, _ in array_operation_binaries.values():
        result = compare(target, reference, output, disabled=True)
        assert "FORT_SCOPED launch" not in result.stderr


@pytest.mark.cuda
def test_source_array_operations_complete_fields_and_holes_on_gpu(array_operation_binaries):
    count = ctypes.c_int()
    cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
    if cuda.cudaGetDeviceCount(ctypes.byref(count)) != 0 or count.value == 0:
        pytest.skip("CUDA device unavailable")
    target, reference, output, manifest = array_operation_binaries["sections"]
    result = compare(target, reference, output)
    assert "FORT_SCOPED launch" in result.stderr
    assert any("loop-carried conflict" in boundary["reason"] for boundary in manifest["boundaries"])


@pytest.mark.cuda
def test_missing_calibration_keeps_array_operations_native(array_operation_binaries):
    target, reference, output, _ = array_operation_binaries["auto"]
    result = compare(target, reference, output)
    assert "FORT_SCOPED launch" not in result.stderr
