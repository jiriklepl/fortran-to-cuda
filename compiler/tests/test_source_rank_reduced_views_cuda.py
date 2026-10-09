"""Independent public-build boundary views agree on complete fields and halos."""
import ctypes
from hashlib import sha256
import os
import shutil

import pytest

from compiler.tests.test_source_rank_reduced_views import SOURCE
from compiler.tests.test_source_scopes import FACT, ROOT, generate, run


DRIVER = """program verify
use renamed_planes
implicit none
real(8),allocatable::a(:,:,:),b(:,:,:)
integer::shape,repeat,i,j,k,n,m,p,plane,unit
integer,parameter::sizes(4)=[0,7,13,7]
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,4
 n=sizes(shape)
 m=5+shape
 p=7+shape
 allocate(a(-7:n-8,-11:m-12,13:p+12),b(-2:n-3,4:m+3,-17:p-18))
 do repeat=1,2
  do k=1,p
   do j=1,m
    do i=1,n
     a(i-8,j-12,k+12)=real(i+17*j+31*k,8)*0.25d0+repeat
     b(i-3,j+3,k-18)=-117.d0
    enddo
   enddo
  enddo
  plane=7+mod(shape+repeat,m)
  if(n>2.and.p>2)call step(a,b,plane)
  write(unit)a,b
 enddo
 deallocate(a,b)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module")
def plane_binaries(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/C++/Fortran toolchain unavailable")
    directory = tmp_path_factory.mktemp("rank_reduced_source_build")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::" + name: FACT for name in ("a", "b")}}
    results, cache = {}, {}
    for mode in ("sections", "auto"):
        original, output, manifest = generate(directory / mode, SOURCE, entry="step", mode=mode, facts=facts, checkout=checkout)
        assert manifest["scope_count"] == 1, manifest["boundaries"]
        assert manifest["scopes"][0]["borrowed_views"]["abi_version"] == 2
        objects = []
        headers = tuple((name, digest) for name, digest in sorted(manifest["artifacts_sha256"].items())
                        if name.endswith((".h", ".hpp", ".cuh")))
        for role in ("common_runtime", "shared_entry", "original_source"):
            for item in manifest["build_sources"]:
                if item["role"] != role:
                    continue
                source = output / item["path"]
                digest = sha256(source.read_bytes()).hexdigest()
                assert digest == manifest["artifacts_sha256"][item["path"]]
                key = (item["language"], digest, headers if item["language"] == "cuda" else ())
                target = source.with_suffix(".o")
                if role == "common_runtime" and item["language"] == "cuda" and key in cache:
                    objects.append(cache[key])
                    continue
                flags = ([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
                         if item["language"] == "cuda" else [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"])
                run([*flags, "-I", str(output), "-c", str(source), "-o", str(target)], cwd=output)
                objects.append(str(target))
                if role == "common_runtime" and item["language"] == "cuda":
                    cache[key] = str(target)
        driver = output / "caller.f90"
        driver.write_text(DRIVER)
        target, reference = output / "candidate", output / "native"
        run([fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", str(driver), *objects,
             "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(target)], cwd=output)
        run([fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", str(original), str(driver),
             "-o", str(reference)], cwd=output)
        environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
        run([str(reference)], cwd=output, env=environment)
        expected = (output / "fields.bin").read_bytes()
        results[mode] = target, output, expected
    return results


def compare(plane_binaries, *, disabled):
    for mode, (target, output, expected) in plane_binaries.items():
        environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
        if disabled:
            environment["CUDA_VISIBLE_DEVICES"] = "-1"
        result = run([str(target)], cwd=output, env=environment)
        assert (output / "fields.bin").read_bytes() == expected
        assert "array temporary" not in result.stderr.lower()
        if not disabled and mode == "sections":
            assert "FORT_SCOPED launch " in result.stderr
        else:
            assert "FORT_SCOPED launch " not in result.stderr


@pytest.mark.native
def test_boundary_views_without_a_device_preserve_original_fallback(plane_binaries):
    compare(plane_binaries, disabled=True)


@pytest.mark.cuda
def test_gpu_and_native_boundary_operations_preserve_every_plane_and_halo(plane_binaries):
    count = ctypes.c_int()
    cuda = ctypes.CDLL("/usr/local/cuda/lib64/libcudart.so")
    if cuda.cudaGetDeviceCount(ctypes.byref(count)) != 0 or count.value == 0:
        pytest.skip("CUDA device unavailable")
    compare(plane_binaries, disabled=False)
