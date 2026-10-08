"""Public CUDA scopes validate definitions and preserve native physical sections."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CASES = (("rmw", "step_rmw", 1), ("unknown", "step_unknown", 2),
         ("out", "step_out", 3), ("faces", "step_faces", 4), ("parallel", "step_parallel", 5))
PROGRAM = """module original
implicit none
contains
subroutine partial_producer(a,b,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::b(:)
integer,intent(in)::n
integer::i
do i=3,6
b(i)=2*a(i)+real(i,8)
enddo
end subroutine
subroutine partial_consumer(a,b,out,n)
real(8),contiguous,intent(in)::a(:),b(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
integer::i
do i=3,6
out(i)=a(i)+b(i)
enddo
end subroutine
subroutine full_producer(a,b,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2*a(i)+real(i,8)
enddo
end subroutine
subroutine full_consumer(a,b,out,n)
real(8),contiguous,intent(in)::a(:),b(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=a(i)+b(i)
enddo
end subroutine
subroutine native_rmw(b)
real(8),contiguous,intent(inout)::b(-2:)
b(0:3)=b(0:3)+7.d0+real(lbound(b,1),8)
end subroutine
subroutine native_unknown(b,last)
real(8),contiguous,intent(inout)::b(-2:)
integer,intent(in)::last
b(0:last)=9.d0+real(lbound(b,1),8)
end subroutine
subroutine native_out(b)
real(8),contiguous,intent(out)::b(-2:)
b(0:3)=9.d0+real(lbound(b,1),8)
end subroutine
subroutine native_faces(b)
real(8),contiguous,intent(inout)::b(-2:)
b(lbound(b,1))=b(lbound(b,1))+7.d0+real(lbound(b,1),8)
b(ubound(b,1))=b(ubound(b,1))+11.d0+real(ubound(b,1),8)
end subroutine
subroutine native_parallel(b)
real(8),contiguous,intent(inout)::b(:)
integer::i
!$omp parallel do
do i=1,size(b)
b(i)=3*b(i)
enddo
!$omp end parallel do
end subroutine
subroutine step_rmw(a,b,out,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::b(:),out(:)
integer,intent(in)::n
call partial_producer(a,b,n)
call native_rmw(b)
call partial_consumer(a,b,out,n)
end subroutine
subroutine step_unknown(a,b,out,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::b(:),out(:)
integer,intent(in)::n
call partial_producer(a,b,n)
call native_unknown(b,3)
call partial_consumer(a,b,out,n)
end subroutine
subroutine step_out(a,b,out,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::b(:),out(:)
integer,intent(in)::n
call partial_producer(a,b,n)
call native_out(b)
call partial_consumer(a,b,out,n)
end subroutine
subroutine step_faces(a,b,out,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::b(:),out(:)
integer,intent(in)::n
call full_producer(a,b,n)
call native_faces(b)
call full_consumer(a,b,out,n)
end subroutine
subroutine step_parallel(a,b,out,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::b(:),out(:)
integer,intent(in)::n
call full_producer(a,b,n)
call native_parallel(b)
call full_consumer(a,b,out,n)
end subroutine
end module
"""

DRIVER = """program caller
use original
implicit none
real(8),allocatable::a(:),b(:),output(:)
real(8)::expected_b,expected_out
integer::which,shape,n,repeat,i,actual
logical::inside
character(8)::argument
call get_command_argument(1,argument)
read(argument,*) which
do shape=1,2
 n=4+8*shape
 allocate(a(-4:n-5),b(-4:n-5),output(-4:n-5))
 do repeat=1,2
  do i=1,n
   a(i-5)=real(i,8)*0.25d0+repeat
  enddo
  b=-99.d0
  output=-199.d0
  select case(which)
  case(1)
   call step_rmw(a,b,output,n)
  case(2)
   call step_unknown(a,b,output,n)
  case(3)
   call step_out(a,b,output,n)
  case(4)
   call step_faces(a,b,output,n)
  case(5)
   call step_parallel(a,b,output,n)
  case default
   error stop 'unknown fixture case'
  end select
  do i=1,n
   actual=i-5
   inside=i>=3.and.i<=6
   expected_b=-99.d0
   expected_out=-199.d0
   if(which>=4) then
    expected_b=2*a(actual)+real(i,8)
    if(which==5) then
     expected_b=3*expected_b
    else
     if(i==1) expected_b=expected_b+5.d0
     if(i==n) expected_b=expected_b+real(n+8,8)
    endif
    expected_out=a(actual)+expected_b
   else if(inside) then
    expected_b=7.d0
    if(which==1) expected_b=2*a(actual)+real(i,8)+5.d0
    expected_out=a(actual)+expected_b
   endif
   ! Native OUT undefines its other elements; never inspect those holes.
   if(which/=3.or.inside) then
    if(b(actual)/=expected_b) error stop 'intermediate or halo disagreement'
   endif
   if(output(actual)/=expected_out) error stop 'output or halo disagreement'
  enddo
  print *, 'CALL_OK',n,repeat
 enddo
 deallocate(a,b,output)
enddo
print *, 'FIELDS_OK'
end program
"""


def _memory_guard():
    available = next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    if available < 3*1024**3:
        pytest.skip("less than 3 GiB available for the bounded CUDA acceptance build")


def _run(command, cwd, *, env=None):
    _memory_guard()
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stdout+result.stderr
    return result


@pytest.fixture(scope="module")
def compiled_native_sections(tmp_path_factory):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    host = shutil.which("g++-14") or shutil.which("g++")
    nvcc = shutil.which("nvcc")
    if not fortran or not host or not nvcc:
        pytest.skip("CUDA and Fortran toolchains are required")
    directory = tmp_path_factory.mktemp("native-section-scopes")
    checkout = Path(os.environ.get("FORT_TEST_COMPILER_CHECKOUT", str(ROOT))).resolve()
    original = directory/"original.f90"
    original.write_text(PROGRAM)
    driver = directory/"caller.f90"
    driver.write_text(DRIVER)
    stable = {"storage": "stable", "escapes": False, "allocation_changes": False}
    captures = {"schema_version": 1, "participation": "serial",
                "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
                "captures": {"argument::a": {**stable, "initialized": "whole"},
                             "argument::b": {**stable, "initialized": "none"},
                             "argument::out": {**stable, "initialized": "whole"}}}
    facts = directory/"captures.json"
    facts.write_text(json.dumps(captures, indent=2))
    native = directory/"native"
    _run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", str(original), str(driver),
          "-o", str(native)], directory)
    cache, proofs, targets = {}, [], {}
    cuda_flags = [nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
    for label, entry, which in CASES:
        output = directory/label
        response = subprocess.run([sys.executable, "-m", "compiler", "--form-scopes", "--scope-facts", str(facts),
                                   "--input", str(original), "--kernel", entry, "--memory-model", "scoped",
                                   "--gpu-policy", "sections", "--json", "--output-dir", str(output)],
                                  cwd=checkout, capture_output=True, text=True, timeout=120,
                                  env={**os.environ, "PYTHONPATH": str(checkout), "PYTHONHASHSEED": "0"})
        directory.joinpath(label+"-generation.stdout").write_text(response.stdout)
        directory.joinpath(label+"-generation.stderr").write_text(response.stderr)
        assert response.returncode == 0, response.stdout+response.stderr
        public = json.loads(response.stdout)
        assert public["supported"], public
        manifest = public["scopes"]
        assert manifest["scope_count"] == 1, manifest["boundaries"]
        assert not manifest["automatic_estimate_available"]
        scope, = manifest["scopes"]
        assert scope["definition_preflight"]["query_available"], scope["definition_preflight"]
        assert "ordered_definition_validation" in manifest["runtime"]["capabilities"]
        for path, digest in manifest["artifacts_sha256"].items():
            assert sha256((output/path).read_bytes()).hexdigest() == digest
        objects = []
        for role in ("common_runtime", "shared_entry", "original_source"):
            for item in manifest["build_sources"]:
                if item["role"] != role:
                    continue
                artifact = item["path"]
                target = output/(artifact.replace("/", "_")+".o")
                if item["language"] == "cuda":
                    # Common runtime exposes its complete public header set;
                    # workers also bind every published numerical header.
                    headers = (manifest["runtime"]["headers"] if role == "common_runtime" else
                               [name for name in manifest["artifacts_sha256"] if name.endswith((".h", ".hpp", ".cuh"))])
                    header_hashes = {name: sha256((output/name).read_bytes()).hexdigest() for name in sorted(headers)}
                    key = (role, manifest["artifacts_sha256"][artifact], tuple(header_hashes.items()), tuple(cuda_flags))
                    reused = key in cache
                    if reused:
                        target = cache[key]
                    else:
                        _run([*cuda_flags, "-I", str(output), "-c", str(output/artifact), "-o", str(target)], output)
                        cache[key] = target
                    proofs.append({"case": label, "artifact": artifact, "source_sha256": key[1],
                                   "all_required_public_headers_sha256": header_hashes,
                                   "compiler_flags": cuda_flags, "verified_include_directory": str(output),
                                   "reused": reused, "object": str(target)})
                else:
                    _run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", "-c",
                          str(output/artifact), "-o", str(target)], output)
                objects.append(str(target))
        target = output/"caller"
        _run([fortran, "-std=f2018", "-fopenmp", "-fcheck=all,array-temps", str(driver), *objects,
              "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++",
              "-o", str(target)], output)
        reference = _run([str(native), str(which)], directory, env={**os.environ, "OMP_NUM_THREADS": "4"})
        output.joinpath("native.stdout").write_text(reference.stdout)
        output.joinpath("native.stderr").write_text(reference.stderr)
        assert "FIELDS_OK" in reference.stdout
        targets[label] = (target, output, which)
    directory.joinpath("cuda-object-reuse.json").write_text(json.dumps(proofs, indent=2))
    return targets


def _execute(compiled, label, *, trace=True):
    target, output, which = compiled[label]
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
    env.pop("FORT_RUNTIME_TRACE", None)
    if trace:
        env["FORT_RUNTIME_TRACE"] = "1"
    result = _run([str(target), str(which)], output, env=env)
    suffix = "traced" if trace else "untraced"
    output.joinpath(suffix+".stdout").write_text(result.stdout)
    output.joinpath(suffix+".stderr").write_text(result.stderr)
    assert "FIELDS_OK" in result.stdout
    assert result.stdout.count("CALL_OK") == 4
    assert "array temporary" not in result.stderr.lower()
    lines = result.stderr.splitlines()
    if not trace:
        assert not any(line.startswith("FORT_SCOPED") for line in lines)
        return lines, []
    evidence = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in lines
                if line.startswith("FORT_SCOPED evidence ")]
    validations = [row for row in evidence if row["event"] == "definition_validation"]
    assert len(validations) == 4, result.stderr
    assert all(row["mutates_live_state"] is False for row in validations)
    output.joinpath("validation.json").write_text(json.dumps(validations, indent=2))
    return lines, validations


def _trace_bytes(lines, operation):
    return sum(int(re.search(r"bytes=(\d+)", line).group(1)) for line in lines
               if line.startswith("FORT_SCOPED "+operation+" "))


def _successful(compiled, label):
    lines, validations = _execute(compiled, label)
    launches = [index for index, line in enumerate(lines) if line.startswith("FORT_SCOPED launch ")]
    if not launches:
        pytest.skip("CUDA execution unavailable")
    assert len(launches) == 8
    assert all(row["status"] == 0 for row in validations)
    assert all(row["reason"] == "definition_plan_valid" for row in validations)
    proofs = [index for index, line in enumerate(lines) if '"event":"definition_validation"' in line]
    assert all(proofs[index] < launches[2*index] for index in range(4))
    _execute(compiled, label, trace=False)
    return lines, launches


@pytest.mark.cuda
def test_native_rmw_reads_only_the_gpu_defined_negative_bound_section(compiled_native_sections):
    lines, _ = _successful(compiled_native_sections, "rmw")
    assert _trace_bytes(lines, "upload") == 4*64
    assert _trace_bytes(lines, "download") == 4*64


@pytest.mark.cuda
def test_unknown_native_preservation_falls_back_before_any_gpu_work(compiled_native_sections):
    lines, validations = _execute(compiled_native_sections, "unknown")
    manifest = json.loads((compiled_native_sections["unknown"][1]/"scope-manifest.json").read_text())
    effects = next(item for item in manifest["native_effects"]["procedures"]
                   if item["procedure"] == "original::native_unknown")
    assert not effects["native_sections"]["available"]
    assert not any(operation["kind"] == "read" and operation.get("rank") for operation in effects["operations"])
    assert all(row["status"] == 7 for row in validations)
    # The validator uses this public reason for reads and output preservation.
    assert all(row["reason"] == "uninitialized_read" for row in validations)
    assert not any(line.startswith("FORT_SCOPED launch ") for line in lines)
    assert _trace_bytes(lines, "upload") == 0
    assert _trace_bytes(lines, "download") == 0
    _execute(compiled_native_sections, "unknown", trace=False)


@pytest.mark.cuda
def test_native_partial_out_forgets_old_definition_and_defines_only_written_section(compiled_native_sections):
    lines, launches = _successful(compiled_native_sections, "out")
    for index in range(4):
        between = lines[launches[2*index]+1:launches[2*index+1]]
        assert sum(line.startswith("FORT_SCOPED forget_definition ") for line in between) == 1
        assert _trace_bytes(between, "download") == 0
    assert _trace_bytes(lines, "upload") == 4*64
    assert _trace_bytes(lines, "download") == 4*32


@pytest.mark.cuda
def test_opposite_native_faces_mirror_only_faces_and_keep_gpu_interiors_current(compiled_native_sections):
    lines, launches = _successful(compiled_native_sections, "faces")
    for index in range(4):
        between = lines[launches[2*index]+1:launches[2*index+1]]
        assert _trace_bytes(between, "download") == 16
        assert _trace_bytes(between, "upload") == 16
    # The final host-visible b interior and complete output remain in the cost.
    assert _trace_bytes(lines, "upload") == 2*(12*8+16)+2*(20*8+16)
    assert _trace_bytes(lines, "download") == 2*(12*16)+2*(20*16)


@pytest.mark.cuda
def test_native_parallel_do_finishes_before_host_commit_and_gpu_consumer(compiled_native_sections):
    lines, launches = _successful(compiled_native_sections, "parallel")
    output = compiled_native_sections["parallel"][1]
    manifest = json.loads((output/"scope-manifest.json").read_text())
    effects = next(item for item in manifest["native_effects"]["procedures"]
                   if item["procedure"] == "original::native_parallel")
    assert effects["native_completion"]["available"]
    assert not effects["native_sections"]["available"]
    for index, n in enumerate((12, 12, 20, 20)):
        between = lines[launches[2*index]+1:launches[2*index+1]]
        assert _trace_bytes(between, "download") == n*8
        assert _trace_bytes(between, "upload") == n*8
