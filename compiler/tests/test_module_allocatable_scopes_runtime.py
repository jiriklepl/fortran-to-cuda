"""Source-proved module allocations keep native effects in coherent scopes."""

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
SOURCE = """module original
implicit none
real(8),allocatable,target::field(:)
contains
subroutine produce(a,b,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::b(:)
integer,intent(in)::n
integer::i
do i=2,n-1
b(i)=a(i-1)+2*a(i)+a(i+1)+real(i,8)
enddo
end subroutine
subroutine consume(a,b,out,n)
real(8),contiguous,intent(in)::a(:),b(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
integer::i
do i=2,n-1
out(i)=a(i)+b(i)
enddo
end subroutine
subroutine native_hidden()
if(allocated(field)) then
 if(size(field)>2) then
  field(lbound(field,1)+1)=field(lbound(field,1)+1)+sum(field)+real(lbound(field,1),8)
 endif
endif
end subroutine
subroutine produce_team(a,b,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::b(:)
integer,intent(in)::n
integer::i
!$omp do
do i=2,n-1
b(i)=a(i-1)+2*a(i)+a(i+1)+real(i,8)
enddo
!$omp end do
end subroutine
subroutine consume_team(a,b,out,n)
real(8),contiguous,intent(in)::a(:),b(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
integer::i
!$omp do
do i=2,n-1
out(i)=a(i)+b(i)
enddo
!$omp end do
end subroutine
subroutine native_hidden_team(n)
integer,intent(in)::n
integer::i
!$omp do
do i=1,min(1,n)
if(n>2) field(lbound(field,1)+1)=field(lbound(field,1)+1)+sum(field)+real(lbound(field,1),8)
enddo
!$omp end do
end subroutine
subroutine step_serial(a,out,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
if(allocated(field)) then
 call produce(a,field,n)
 call native_hidden()
 call consume(a,field,out,n)
else
 call native_hidden()
endif
end subroutine
subroutine step_team(a,out,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
call produce_team(a,field,n)
call native_hidden_team(n)
call consume_team(a,field,out,n)
end subroutine
subroutine qualified(a,out,n)
real(8),contiguous,intent(in)::a(:)
real(8),contiguous,intent(inout)::out(:)
integer,intent(in)::n
if(.not.allocated(field)) then
 call native_hidden()
 return
endif
!$omp parallel default(none) shared(a,out,n,field) num_threads(4)
call step_team(a,out,n)
!$omp end parallel
end subroutine
end module
"""

PARTIAL_SOURCE = SOURCE.replace("""subroutine native_hidden()
if(allocated(field)) then
 if(size(field)>2) then
  field(lbound(field,1)+1)=field(lbound(field,1)+1)+sum(field)+real(lbound(field,1),8)
 endif
endif
end subroutine""", """subroutine native_hidden()
integer::i
if(allocated(field)) then
 if(size(field)>2) then
  do i=1,ubound(field,1)
   field(i)=100.d0+real(i,8)
  enddo
 endif
endif
end subroutine""")

DRIVER = """program verify
use original
use omp_lib,only:omp_set_dynamic
implicit none
real(8),allocatable::a(:),out(:),expected(:)
integer,parameter::sizes(3)=[0,19,27]
integer::shape,repeat,n,lower,i,index
real(8)::wanted
character(16)::path,mode
call get_command_argument(1,path)
call get_command_argument(2,mode)
call omp_set_dynamic(.false.)
if(mode=='unallocated') then
 allocate(a(-4:0),out(-4:0))
 a=3.d0
 out=-70.d0
 if(path=='serial'.or.path=='serial_partial') then
  call step_serial(a,out,5)
 else
  call qualified(a,out,5)
 endif
 if(allocated(field)) error stop 'native guard allocated its capture'
 if(any(a/=3.d0).or.any(out/=-70.d0)) error stop 'native guarded field mismatch'
 print *, 'UNALLOCATED_OK'
 stop
endif
do shape=1,size(sizes)
 n=sizes(shape)
 do repeat=1,2
  lower=-4-3*shape-repeat
  allocate(a(lower:lower+n-1),field(lower:lower+n-1),out(lower:lower+n-1),expected(lower:lower+n-1))
  do i=1,n
   a(lower+i-1)=real(i,8)*0.25d0+repeat
  enddo
  field=-50.d0
  out=-70.d0
  expected=field
  do i=2,n-1
   index=lower+i-1
   expected(index)=a(index-1)+2*a(index)+a(index+1)+real(i,8)
  enddo
  if(path=='serial_partial') then
   do i=1,ubound(expected,1)
    expected(i)=100.d0+real(i,8)
   enddo
  else if(n>2) then
   expected(lower+1)=expected(lower+1)+sum(expected)+real(lbound(field,1),8)
  endif
  if(path=='serial'.or.path=='serial_partial') then
   call step_serial(a,out,n)
  else
   call qualified(a,out,n)
  endif
  do i=1,n
   index=lower+i-1
   if(a(index)/=real(i,8)*0.25d0+repeat) error stop 'immutable input changed'
   if(field(index)/=expected(index)) error stop 'module field or halo disagreement'
   wanted=-70.d0
   if(i>=2.and.i<=n-1) wanted=a(index)+expected(index)
   if(out(index)/=wanted) error stop 'output or halo disagreement'
  enddo
  print *, 'CALL_OK',n,repeat,lbound(field,1)
  deallocate(a,field,out,expected)
 enddo
enddo
print *, 'FIELDS_OK'
end program
"""


def _guard():
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    assert available >= 3 * 1024**3, "memory guard stopped module-allocation acceptance before a new process"


def _run(command, directory, *, env=None, timeout=180):
    _guard()
    result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=timeout)
    with (directory / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps({"argv": list(map(str, command)), "returncode": result.returncode,
                                 "stdout": result.stdout, "stderr": result.stderr}) + "\n")
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _facts(original, label):
    stable = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
             "captures": {root: dict(stable) for root in ("argument::a", "argument::out", "original::field")}}
    if label == "team":
        lines = original.read_text().splitlines(keepends=True)
        first = next(i for i, line in enumerate(lines, 1) if line.startswith("call step_team"))
        facts["schema_version"] = 2
        facts["captures"] = {root: {**stable, "association": "shared_whole_storage", "descriptor_uniform": True}
                             for root in facts["captures"]}
        facts["captures"]["argument::n"] = {**stable, "association": "shared_immutable_control"}
        facts["participation"] = {"kind": "omp_full_team", "dispatch": "qualified_companion",
                                  "entry": "original::step_team", "expected_omp_level": 1, "host_threads": 4,
                                  "call_sites": [{"source": str(original), "caller": "original::qualified",
                                                  "first_line": first, "last_line": first,
                                                  "span_sha256": sha256(lines[first-1].encode()).hexdigest(),
                                                  "team_first_line": first-1, "team_last_line": first+1,
                                                  "uniform_guard": "unconditional"}]}
    return facts


@pytest.fixture(scope="module")
def module_scope_binaries(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, Fortran and OpenMP toolchains are required")
    directory = tmp_path_factory.mktemp("module-allocation-scopes")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    original, driver = directory / "original.f90", directory / "caller.f90"
    original.write_text(SOURCE)
    driver.write_text(DRIVER)
    fortran_flags = [fortran, "-O0", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    cuda_flags = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
    cache, reuse, targets = {}, [], {}
    for label in ("serial", "serial_partial", "team"):
        # Preserve one canonical public input path so identical workers can be
        # reused only after byte, header and flag proof. Keep each source version.
        original.write_text(PARTIAL_SOURCE if label == "serial_partial" else SOURCE)
        (directory / (label + "-original.f90")).write_text(original.read_text())
        native = directory / (label + "-native")
        _run([*fortran_flags, str(original), str(driver), "-o", str(native)], directory)
        output = directory / label
        facts = directory / (label + "-captures.json")
        facts.write_text(json.dumps(_facts(original, label), indent=2) + "\n")
        command = [sys.executable, "-m", "compiler", "--form-scopes", "--scope-facts", str(facts),
                   "--input", str(original), "--kernel", "step_team" if label == "team" else "step_serial",
                   "--memory-model", "scoped",
                   "--gpu-policy", "sections", "--host-threads", "4", "--json", "--output-dir", str(output)]
        if label == "team":
            command.append("--gpu-collective")
        response = _run(command, checkout, env={**os.environ, "PYTHONPATH": str(checkout), "PYTHONHASHSEED": "0"})
        (directory / (label + "-public.json")).write_text(response.stdout)
        public = json.loads(response.stdout)
        assert public["supported"], public
        manifest = public["scopes"]
        assert manifest["scope_count"] == 1, manifest["boundaries"]
        scope, = manifest["scopes"]
        assert scope["definition_preflight"]["query_available"]
        assert not manifest["automatic_estimate_available"]
        assert len(scope["gpu_leaves"]) == 2
        native_effects = next(item for item in manifest["native_effects"]["procedures"]
                              if item["procedure"] == "original::native_hidden" + ("_team" if label == "team" else ""))
        assert native_effects["complete"]
        assert not native_effects["native_sections"]["available"]
        if label == "serial_partial":
            assert not any(row["kind"] == "read" and row.get("rank") for row in native_effects["operations"])
            assert "original::field" not in native_effects["guaranteed_whole_overwrites"]
        authorizations = manifest["native_effects"]["capture_lifetime_authorizations"]
        assert [item["resource"] for item in authorizations] == ["original::field"]
        assert authorizations[0]["source_sha256"] == sha256(original.read_bytes()).hexdigest()
        for path, digest in manifest["artifacts_sha256"].items():
            assert sha256((output / path).read_bytes()).hexdigest() == digest
        objects = []
        for role in ("common_runtime", "shared_entry", "original_source"):
            for item in manifest["build_sources"]:
                if item["role"] != role:
                    continue
                artifact = item["path"]
                target = output / (artifact.replace("/", "_") + ".o")
                if item["language"] == "cuda":
                    headers = (manifest["runtime"]["headers"] if role == "common_runtime" else
                               [name for name in manifest["artifacts_sha256"] if name.endswith((".h", ".hpp", ".cuh"))])
                    hashes = {name: sha256((output / name).read_bytes()).hexdigest() for name in sorted(headers)}
                    key = (role, manifest["artifacts_sha256"][artifact], tuple(hashes.items()), tuple(cuda_flags))
                    reused = key in cache
                    if reused:
                        target = cache[key]
                    else:
                        _run([*cuda_flags, "-I", str(output), "-c", str(output / artifact), "-o", str(target)], output)
                        cache[key] = target
                    reuse.append({"variant": label, "artifact": artifact, "source_sha256": key[1],
                                  "all_public_headers_sha256": hashes, "compiler_flags": cuda_flags,
                                  "verified_include_directory": str(output), "reused": reused, "object": str(target)})
                else:
                    _run([*fortran_flags, "-c", str(output / artifact), "-o", str(target)], output)
                objects.append(str(target))
        executable = output / "verify"
        _run([*fortran_flags, str(driver), *objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64",
              "-lcudart", "-lstdc++", "-o", str(executable)], output)
        references = {}
        for mode in ("fields", "unallocated"):
            reference = _run([str(native), label, mode], directory, timeout=30)
            (output / (mode + "-native.stdout")).write_text(reference.stdout)
            (output / (mode + "-native.stderr")).write_text(reference.stderr)
            assert "array temporary" not in reference.stderr.lower()
            references[mode] = reference.stdout
        targets[label] = executable, output, scope, references
    (directory / "cuda-object-reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")
    return targets


def _execute(compiled, label, mode, *, trace=True):
    executable, output, scope, references = compiled[label]
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
    env.pop("FORT_RUNTIME_TRACE", None)
    if trace:
        env["FORT_RUNTIME_TRACE"] = "1"
    result = _run([str(executable), label, mode], output, env=env, timeout=30)
    prefix = mode + ("-traced" if trace else "-untraced")
    (output / (prefix + ".stdout")).write_text(result.stdout)
    (output / (prefix + ".stderr")).write_text(result.stderr)
    assert result.stdout == references[mode]
    assert "array temporary" not in result.stderr.lower()
    if not trace:
        assert "FORT_SCOPED" not in result.stderr
    return result.stderr.splitlines(), scope


def _transfers(lines, operation, identity):
    return [int(re.search(r"bytes=(\d+)", line).group(1)) for line in lines
            if line.startswith("FORT_SCOPED " + operation + " ") and f"buffer={identity} " in line]


@pytest.mark.cuda
@pytest.mark.parametrize("label", ["serial", "serial_partial", "team"])
def test_hidden_native_module_effects_preserve_bounds_fields_and_gpu_mirror(module_scope_binaries, label):
    lines, scope = _execute(module_scope_binaries, label, "fields")
    launches = [i for i, line in enumerate(lines) if line.startswith("FORT_SCOPED launch ")]
    if not launches:
        pytest.skip("CUDA execution unavailable")
    assert len(launches) == 8
    validations = [(i, json.loads(line.removeprefix("FORT_SCOPED evidence "))) for i, line in enumerate(lines)
                   if line.startswith("FORT_SCOPED evidence ") and '"event":"definition_validation"' in line]
    assert len(validations) == 6
    assert all(row["status"] == 0 and row["mutates_live_state"] is False for _, row in validations)
    assert len({row["context"] for _, row in validations}) == 6
    assert all(validations[i+2][0] < launches[2*i] < launches[2*i+1] for i in range(4))
    identities = {row["resource"]: row["registration_identity"] for row in scope["resources"]}
    for i, n in enumerate((19, 19, 27, 27)):
        prepared = lines[validations[i+2][0]:launches[2*i]]
        between = lines[launches[2*i]+1:launches[2*i+1]]
        # A halo footprint can use multiple disjoint physical copies. Require
        # one array's worth of data before its first worker and no repeated
        # bytes when the next worker follows the native operation.
        assert sum(_transfers(prepared, "upload", identities["argument::a"])) == n*8
        assert _transfers(between, "upload", identities["argument::a"]) == []
        assert _transfers(between, "download", identities["argument::a"]) == []
    assert sum(_transfers(lines, "upload", identities["argument::a"])) == 2*(19+27)*8
    assert _transfers(lines, "download", identities["argument::a"]) == []
    expected = [17*8, 17*8, 25*8, 25*8]
    # Native SUM reads GPU values, and the write-only positive-index sweep must
    # preserve negative cells. Conservative native writes invalidate only the
    # field's device mirror; the unrelated immutable input stays current.
    assert _transfers(lines, "download", identities["original::field"]) == expected
    assert _transfers(lines, "upload", identities["original::field"]) == expected
    assert _transfers(lines, "download", identities["argument::out"]) == expected
    _execute(module_scope_binaries, label, "fields", trace=False)


@pytest.mark.cuda
@pytest.mark.parametrize("label", ["serial", "serial_partial", "team"])
def test_unallocated_module_capture_keeps_original_guard_without_cuda(module_scope_binaries, label):
    lines, _ = _execute(module_scope_binaries, label, "unallocated")
    assert not any(line.startswith("FORT_SCOPED") for line in lines)
    _execute(module_scope_binaries, label, "unallocated", trace=False)
