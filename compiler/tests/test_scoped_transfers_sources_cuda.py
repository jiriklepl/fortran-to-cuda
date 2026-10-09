"""Public source transfer modes preserve opposite faces and complete 3D fields."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from hashlib import sha256

import pytest

from compiler.tests.test_module_allocatable_scopes_runtime import ROOT, _run
from compiler.tests.test_scoped_planning_entries import calibration
from compiler.tests.test_source_scopes import FACT

SOURCE = """module original
implicit none
contains
subroutine producer(a,b,nx,ny,nz)
real(8),intent(in)::a(:,:,:)
real(8),intent(inout)::b(:,:,:)
integer,intent(in)::nx,ny,nz
integer::j,k
if(nx>=4.and.ny>=3.and.nz>=3) then
do k=2,nz-1
do j=2,ny-1
b(2,j,k)=2*a(2,j,k)+real(j+3*k,8)
b(nx-1,j,k)=3*a(nx-1,j,k)+real(2*j+k,8)
enddo
enddo
endif
end subroutine
subroutine consumer(a,b,out,nx,ny,nz)
real(8),intent(in)::a(:,:,:),b(:,:,:)
real(8),intent(inout)::out(:,:,:)
integer,intent(in)::nx,ny,nz
integer::j,k
if(nx>=4.and.ny>=3.and.nz>=3) then
do k=2,nz-1
do j=2,ny-1
out(2,j,k)=a(2,j,k)+b(2,j,k)+real(j+k,8)
out(nx-1,j,k)=a(nx-1,j,k)+b(nx-1,j,k)+real(j-k,8)
enddo
enddo
endif
end subroutine
subroutine step(a,b,out,nx,ny,nz)
real(8),intent(in)::a(-2:,3:,-4:)
real(8),intent(inout)::b(-2:,3:,-4:),out(-2:,3:,-4:)
integer,intent(in)::nx,ny,nz
call producer(a,b,nx,ny,nz)
call consumer(a,b,out,nx,ny,nz)
end subroutine
end module
"""

DRIVER = """program caller
use original,only:step
implicit none
real(8),allocatable::a(:,:,:),b(:,:,:),out(:,:,:)
integer,parameter::xs(3)=[0,8,13],ys(3)=[6,7,9],zs(3)=[5,6,8]
integer::shape,sign,nx,ny,nz,i,j,k,lx,ly,lz
do shape=1,3
nx=xs(shape)
ny=ys(shape)
nz=zs(shape)
lx=-9+shape
ly=2-3*shape
lz=-13-shape
allocate(a(lx:lx+nx-1,ly:ly+ny-1,lz:lz+nz-1))
allocate(b(lx:lx+nx-1,ly:ly+ny-1,lz:lz+nz-1))
allocate(out(lx:lx+nx-1,ly:ly+ny-1,lz:lz+nz-1))
do sign=-1,1,2
do k=lz,lz+nz-1
do j=ly,ly+ny-1
do i=lx,lx+nx-1
a(i,j,k)=real(sign*64,8)+real(i-lx,8)*0.25d0+real(j-ly,8)*0.5d0+real(k-lz,8)
enddo
enddo
enddo
b=-99.d0
out=-101.d0
call step(a,b,out,nx,ny,nz)
print '(a,5i5)', 'CALL_OK',shape,sign,lbound(b)
print '(100000f14.3)', b
print '(100000f14.3)', out
enddo
deallocate(a,b,out)
enddo
print *, 'FIELDS_OK'
end program
"""

DIAGNOSTIC = r"""#include <cuda_runtime_api.h>
#include <cstdio>
static unsigned long long device_calls = 0, count_calls = 0, copies_3d = 0;
extern "C" cudaError_t __real_cudaGetDevice(int *);
extern "C" cudaError_t __real_cudaGetDeviceCount(int *);
extern "C" cudaError_t __real_cudaMemcpy3DAsync(const cudaMemcpy3DParms *, cudaStream_t);
extern "C" cudaError_t __wrap_cudaGetDevice(int *value) {
    ++device_calls; return __real_cudaGetDevice(value);
}
extern "C" cudaError_t __wrap_cudaGetDeviceCount(int *value) {
    ++count_calls; return __real_cudaGetDeviceCount(value);
}
extern "C" cudaError_t __wrap_cudaMemcpy3DAsync(const cudaMemcpy3DParms *p, cudaStream_t stream) {
    ++copies_3d;
    std::fprintf(stderr, "FIXTURE_COPY3D {\"width\":%zu,\"height\":%zu,\"depth\":%zu,"
                        "\"source_pitch\":%zu,\"destination_pitch\":%zu,"
                        "\"source_height\":%zu,\"destination_height\":%zu}\n",
                 p->extent.width, p->extent.height, p->extent.depth,
                 p->srcPtr.pitch, p->dstPtr.pitch, p->srcPtr.ysize, p->dstPtr.ysize);
    return __real_cudaMemcpy3DAsync(p, stream);
}
struct FinalEvidence {
    ~FinalEvidence() {
        std::fprintf(stderr, "FIXTURE_CUDA device=%llu count=%llu copies3d=%llu\n",
                     device_calls, count_calls, copies_3d);
    }
} evidence;
"""


def generate(case, checkout, transfers, policy):
    case.mkdir()
    original = case / "original.f90"
    original.write_text(SOURCE)
    facts = case / "captures.json"
    facts.write_text(json.dumps({"schema_version": 1, "participation": "serial",
                                 "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
                                 "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}}))
    profile = case / "old-scoped-profile.json"
    profile.write_text(json.dumps(calibration()))
    output = case / "output"
    result = _run([sys.executable, "-m", "compiler", "--form-scopes", "--scope-facts", str(facts),
                   "--input", str(original), "--kernel", "step", "--memory-model", "scoped",
                   "--gpu-policy", policy, "--scope-transfers", transfers,
                   *(["--calibration-profile", str(profile)] if policy == "auto" else []),
                   "--json", "--output-dir", str(output)], checkout,
                  env={**os.environ, "PYTHONPATH": str(checkout)})
    public = json.loads(result.stdout)
    assert public["supported"], public
    manifest = json.loads((output / "scope-manifest.json").read_text())
    assert manifest == public["scopes"]
    scope, = manifest["scopes"]
    assert scope["transfer_configuration"]["requested"] == transfers
    assert set(scope["gpu_leaves"]) == {"original::producer", "original::consumer"}
    if policy == "auto":
        assert json.loads(profile.read_text())["scoped"]["runtime_id"] == manifest["runtime"]["runtime_id"]
        assert not scope["estimate_available"]
        assert scope["transfer_configuration"]["placement_estimate_reason"] == "transfer_estimates_unavailable"
    return original, output, manifest


@pytest.mark.parametrize(("transfers", "policy"), [("pinned", "sections"), ("auto", "sections"),
                                                   ("pipelined", "sections"), ("pinned", "auto")])
def test_thin_opposite_3d_faces_have_public_source_transfer_configuration(tmp_path, transfers, policy):
    generate(tmp_path / "source", ROOT, transfers, policy)


@pytest.fixture(scope="module")
def transfer_binaries(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, Fortran and OpenMP toolchains are required")
    directory = tmp_path_factory.mktemp("source-transfers")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    fortran_flags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    cuda_flags = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
    cache, targets, reuse = {}, {}, []
    for label, transfers, policy in [("pinned", "pinned", "sections"), ("auto", "auto", "sections"),
                                      ("pipelined", "pipelined", "sections"), ("pinned-native", "pinned", "auto")]:
        case = directory / label
        original, output, manifest = generate(case, checkout, transfers, policy)
        objects = []
        for role in ("common_runtime", "shared_entry", "original_source"):
            for item in manifest["build_sources"]:
                if item["role"] != role:
                    continue
                artifact = item["path"]
                target = output / (artifact.replace("/", "_") + ".o")
                if item["language"] == "cuda":
                    headers = [name for name in manifest["artifacts_sha256"] if name.endswith((".h", ".hpp", ".cuh"))]
                    hashes = {name: sha256((output / name).read_bytes()).hexdigest() for name in sorted(headers)}
                    key = (role, manifest["artifacts_sha256"][artifact], tuple(hashes.items()), tuple(cuda_flags))
                    reused = key in cache
                    if reused:
                        target = cache[key]
                    else:
                        _run([*cuda_flags, "-I", str(output), "-c", str(output / artifact), "-o", str(target)], output)
                        cache[key] = target
                    reuse.append({"case": label, "source_sha256": key[1], "headers_sha256": hashes,
                                  "flags": cuda_flags, "reused": reused, "object": str(target)})
                else:
                    _run([*fortran_flags, "-c", str(output / artifact), "-o", str(target)], output)
                objects.append(str(target))
        diagnostic = output / "copy-evidence.cpp"
        diagnostic.write_text(DIAGNOSTIC)
        diagnostic_object = output / "copy-evidence.o"
        _run([host, "-std=c++17", "-I/usr/local/cuda/include", "-c", str(diagnostic), "-o", str(diagnostic_object)], output)
        driver = output / "driver.f90"
        driver.write_text(DRIVER)
        target = output / "verify"
        _run([*fortran_flags, str(driver), *objects, str(diagnostic_object),
              "-Wl,--wrap=cudaGetDevice", "-Wl,--wrap=cudaGetDeviceCount", "-Wl,--wrap=cudaMemcpy3DAsync",
              "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(target)], output)
        native = output / "native"
        _run([*fortran_flags, str(original), str(driver), "-o", str(native)], output)
        reference = _run([str(native)], output, env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"})
        (output / "native.stdout").write_text(reference.stdout)
        targets[label] = target, output, reference.stdout
    (directory / "cuda-object-reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")
    return targets


@pytest.mark.cuda
@pytest.mark.parametrize("label", ["pinned", "auto", "pipelined", "pinned-native"])
def test_transfer_modes_preserve_complete_fields_and_pitched_opposite_faces(transfer_binaries, label):
    target, output, reference = transfer_binaries[label]
    result = _run([str(target)], output, env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
                                            "FORT_RUNTIME_TRACE": "1"})
    (output / "scoped.stdout").write_text(result.stdout)
    (output / "scoped.stderr").write_text(result.stderr)
    assert result.stdout == reference
    assert result.stdout.count("CALL_OK") == 6
    assert "array temporary" not in result.stderr.lower()
    device, count, copies = map(int, re.search(r"FIXTURE_CUDA device=(\d+) count=(\d+) copies3d=(\d+)", result.stderr).groups())
    if label == "pinned-native":
        assert (device, count, copies) == (0, 0, 0)
        assert "FORT_SCOPED launch" not in result.stderr
        assert "FORT_SCOPED upload" not in result.stderr
        return
    assert copies > 0
    operations = [json.loads(line.removeprefix("FIXTURE_COPY3D ")) for line in result.stderr.splitlines()
                  if line.startswith("FIXTURE_COPY3D ")]
    assert any(operation["width"] == 8 and operation["depth"] > 1 and
               max(operation["source_pitch"], operation["destination_pitch"]) > 8 and
               max(operation["source_height"], operation["destination_height"]) > operation["height"]
               for operation in operations)
    events = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in result.stderr.splitlines()
              if line.startswith("FORT_SCOPED evidence ")]
    stats = [event for event in events if event["event"] == "transfer_statistics"]
    assert len(stats) == 6
    assert all(event["complete"] and event["stats_version"] == 1 for event in stats)
    if label == "pinned":
        assert all(event["effective_mode"] == 1 and event["reason"] == "none" for event in stats)
        expected_uploads = [0, 0, 2 * 5 * 4 * 8, 2 * 5 * 4 * 8, 2 * 7 * 6 * 8, 2 * 7 * 6 * 8]
        assert [event["pinned_upload_bytes"] for event in stats] == expected_uploads
        assert [event["pinned_download_bytes"] for event in stats] == [2 * value for value in expected_uploads]
        assert all(event["packed_bytes"] == event["pinned_upload_bytes"] for event in stats)
        assert all(event["unpacked_bytes"] == event["pinned_download_bytes"] for event in stats)
        assert all(event["event_waits"] == event["events"] for event in stats)
        assert max(event["process_peak_bytes"] for event in stats) <= 64 * 1024 * 1024
    else:
        reason = "transfer_estimates_unavailable" if label == "auto" else "pipelined_not_available"
        assert all(event["effective_mode"] == 0 and event["reason"] == reason for event in stats)
        assert all(event["pinned_upload_bytes"] == event["pinned_download_bytes"] == 0 for event in stats)
