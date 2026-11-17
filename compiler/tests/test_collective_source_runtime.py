"""Public source companions preserve fields and use one coordinated context.

This opt-in acceptance fixture builds independent public compiler outputs. It
does not inspect compiler IR or generated CUDA to infer execution decisions.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SOURCE = """module operators
implicit none
contains
subroutine produce(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:)
integer,intent(in)::n
integer::i
!$omp do
do i=2,n-1
b(i)=2*a(i)+real(i,8)
enddo
!$omp end do
end subroutine
subroutine native_permute(b,permutation,n)
real(8),intent(inout)::b(:)
integer,intent(in)::permutation(:),n
integer::i
!$omp do
do i=2,n-1
b(permutation(i))=b(permutation(i))+1
enddo
!$omp end do
end subroutine
subroutine consume(a,b,c,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(inout)::c(:)
integer,intent(in)::n
integer::i
!$omp do
do i=2,n-1
c(i)=a(i)+b(i)
enddo
!$omp end do
end subroutine
subroutine step(a,b,c,permutation,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::permutation(:),n
call produce(a,b,n)
call consume(a,b,c,n)
end subroutine
subroutine step_native(a,b,c,permutation,n)
real(8),intent(in)::a(:)
real(8),intent(inout)::b(:),c(:)
integer,intent(in)::permutation(:),n
call produce(a,b,n)
call native_permute(b,permutation,n)
call consume(a,b,c,n)
end subroutine
end module
module callers
use operators,only:step,step_native
implicit none
contains
subroutine qualified(a,b,c,permutation,n)
real(8),allocatable,intent(in)::a(:)
real(8),allocatable,intent(inout)::b(:),c(:)
integer,allocatable,intent(in)::permutation(:)
integer,intent(in)::n
!$omp parallel default(none) shared(a,b,c,permutation,n) num_threads(4)
call step(a,b,c,permutation,n)
!$omp end parallel
end subroutine
subroutine qualified_native(a,b,c,permutation,n)
real(8),allocatable,intent(in)::a(:)
real(8),allocatable,intent(inout)::b(:),c(:)
integer,allocatable,intent(in)::permutation(:)
integer,intent(in)::n
!$omp parallel default(none) shared(a,b,c,permutation,n) num_threads(4)
call step_native(a,b,c,permutation,n)
!$omp end parallel
end subroutine
end module
"""

HIDDEN_SOURCE = SOURCE.replace("implicit none\ncontains", "implicit none\nreal(8),allocatable::scratch(:)\ncontains", 1).replace(
    "subroutine consume(", """subroutine native_guard(c,n)
real(8),intent(inout)::c(:)
integer,intent(in)::n
integer::i
!$omp do
do i=1,merge(n,0,allocated(scratch))
if(scratch(i)>0) c(i)=c(i)+scratch(i)
enddo
!$omp end do
end subroutine
subroutine consume(""", 1).replace(
    "call produce(a,b,n)\ncall consume(a,b,c,n)",
    "call produce(a,b,n)\ncall native_guard(c,n)\ncall consume(a,b,c,n)", 1).replace(
    "use operators,only:step,step_native", "use operators,only:step,step_native,scratch").replace(
    "shared(a,b,c,permutation,n)", "shared(a,b,c,permutation,n,scratch)")

DRIVER = """program verify_fields
use callers
use iso_c_binding
use omp_lib,only:omp_set_dynamic
implicit none
interface
subroutine counter_snapshot(created,registered,closed) bind(C)
import c_int64_t
integer(c_int64_t),intent(out)::created,registered,closed
end subroutine
end interface
real(8),allocatable::a(:),b(:),c(:)
integer,allocatable::permutation(:)
integer::n,repetition,shape,k,i,checks
integer,parameter::sizes(3)=[0,19,27]
integer(c_int64_t)::created,registered,closed
real(8)::wanted_a,wanted_b,wanted_c
character(16)::mode
call get_command_argument(1,mode)
call omp_set_dynamic(.false.)
checks=0
do shape=1,size(sizes)
n=sizes(shape)
do repetition=1,2
allocate(a(-2:n-3),b(-2:n-3),c(-2:n-3),permutation(-2:n-3))
do k=-2,n-3
i=k+3
a(k)=real(3*i+repetition,8)
b(k)=-50.d0
c(k)=-70.d0
permutation(k)=n-i+1
enddo
if(mode=='native') then
call qualified_native(a,b,c,permutation,n)
else if(mode=='serial') then
call step(a,b,c,permutation,n)
else
call qualified(a,b,c,permutation,n)
endif
do k=-2,n-3
i=k+3
wanted_a=real(3*i+repetition,8)
wanted_b=-50.d0
wanted_c=-70.d0
if(i>=2.and.i<=n-1) then
wanted_b=2*wanted_a+real(i,8)
if(mode=='native') wanted_b=wanted_b+1
wanted_c=wanted_a+wanted_b
endif
if(a(k)/=wanted_a.or.b(k)/=wanted_b.or.c(k)/=wanted_c) error stop 'full field or halo mismatch'
if(permutation(k)/=n-i+1) error stop 'immutable permutation changed'
enddo
deallocate(a,b,c,permutation)
checks=checks+1
enddo
enddo
call counter_snapshot(created,registered,closed)
print '(a,a,a,i0)', 'FIELDS_OK ',trim(mode),' checks=',checks
print '(a,i0,a,i0,a,i0)', 'COUNTERS created=',created,' registered=',registered,' closed=',closed
end program
"""

COUNTERS = r'''#include <atomic>
#include <cstdint>
#include "scoped_runtime.h"
static std::atomic<std::int64_t> creates{0},registrations{0},closes{0};
extern "C" int __real_fort_scope_create(int,fort_scope_t *);
extern "C" int __real_fort_scope_register(fort_scope_t,std::uint64_t,std::uint64_t,
                                         const fort_scope_layout *,int,fort_buffer_t *);
extern "C" int __real_fort_scope_register_sections(fort_scope_t,std::uint64_t,std::uint64_t,
                                                  const fort_scope_layout *,const fort_scope_section *,
                                                  std::size_t,fort_buffer_t *);
extern "C" int __real_fort_scope_close(fort_scope_t);
extern "C" int __wrap_fort_scope_create(int device,fort_scope_t *scope) {
    const int status=__real_fort_scope_create(device,scope);if(!status) ++creates;return status;
}
extern "C" int __wrap_fort_scope_register(fort_scope_t scope,std::uint64_t identity,std::uint64_t generation,
                                         const fort_scope_layout *layout,int initialized,fort_buffer_t *buffer) {
    const int status=__real_fort_scope_register(scope,identity,generation,layout,initialized,buffer);
    if(!status) ++registrations;return status;
}
extern "C" int __wrap_fort_scope_register_sections(fort_scope_t scope,std::uint64_t identity,std::uint64_t generation,
                                                  const fort_scope_layout *layout,const fort_scope_section *sections,
                                                  std::size_t count,fort_buffer_t *buffer) {
    const int status=__real_fort_scope_register_sections(scope,identity,generation,layout,sections,count,buffer);
    if(!status) ++registrations;return status;
}
extern "C" int __wrap_fort_scope_close(fort_scope_t scope) {
    const int status=__real_fort_scope_close(scope);if(!status) ++closes;return status;
}
extern "C" void counter_snapshot(std::int64_t *created,std::int64_t *registered,std::int64_t *closed) {
    *created=creates.load();*registered=registrations.load();*closed=closes.load();
}
'''

NATIVE_COUNTERS = r'''#include <cstdint>
extern "C" void counter_snapshot(std::int64_t *created,std::int64_t *registered,std::int64_t *closed) {
    *created=*registered=*closed=0;
}
'''


def _guard():
    minimum = int(os.environ.get("FORT_TEST_MIN_AVAILABLE_BYTES", "0"))
    if minimum:
        available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                         if line.startswith("MemAvailable:"))
        assert available >= minimum, "memory guard stopped source-team acceptance before a new process"


def _run(command, directory, *, env=None, timeout=180, expected_returncode=0):
    _guard()
    result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=timeout)
    log = directory / "command-log.jsonl"
    with log.open("a") as stream:
        stream.write(json.dumps({"argv": list(map(str, command)), "returncode": result.returncode,
                                 "stdout": result.stdout, "stderr": result.stderr}) + "\n")
    assert result.returncode == expected_returncode, result.stdout + result.stderr
    return result


def _facts(original, *, native=False, budget=None):
    text = original.read_text().splitlines(keepends=True)
    entry = "step_native" if native else "step"
    first = next(i for i, line in enumerate(text, 1) if line.strip() == f"call {entry}(a,b,c,permutation,n)"
                 and text[i-2].startswith("!$omp parallel"))
    stable = {"storage": "stable", "initialized": "whole", "allocation_changes": False, "escapes": False}
    facts = {"schema_version": 2, "sources": {str(original): sha256(original.read_bytes()).hexdigest()},
             "captures": {**{"argument::" + name: {**stable, "association": "shared_whole_storage",
                                                    "descriptor_uniform": True}
                              for name in ("a", "b", "c", "permutation")},
                          "argument::n": {**stable, "association": "shared_immutable_control"}},
             "participation": {"kind": "omp_full_team", "dispatch": "qualified_companion",
                               "entry": "operators::" + entry, "host_threads": 4, "expected_omp_level": 1,
                               "call_sites": [{"source": str(original),
                                               "caller": "callers::qualified_native" if native else "callers::qualified",
                                               "first_line": first, "last_line": first,
                                               "span_sha256": sha256(text[first-1].encode()).hexdigest(),
                                               "team_first_line": first-1, "team_last_line": first+1,
                                               "uniform_guard": "unconditional"}]}}
    if budget is not None:
        facts["device_budget_bytes"] = budget
    return facts


@pytest.fixture(scope="module")
def source_team_binaries(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA/OpenMP/Fortran toolchain unavailable")
    configured = os.environ.get("FORT_TEST_COLLECTIVE_EVIDENCE_DIR")
    directory = Path(configured).resolve() if configured else tmp_path_factory.mktemp("collective_source")
    directory.mkdir(parents=True, exist_ok=True)
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    original = directory / "original.f90"
    original.write_text(SOURCE)
    driver = directory / "driver.f90"
    driver.write_text(DRIVER)
    native_counter = directory / "native-counters.cpp"
    native_counter.write_text(NATIVE_COUNTERS)
    native_counter_object = directory / "native-counters.o"
    _run([host, "-std=c++17", "-c", str(native_counter), "-o", str(native_counter_object)], directory)
    native_binary = directory / "native-reference"
    fortran_flags = ["-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    _run([fortran, *fortran_flags, str(original), str(driver), str(native_counter_object), "-lstdc++",
          "-o", str(native_binary)], directory)
    for mode in ("basic", "native", "serial"):
        reference = _run([str(native_binary), mode], directory,
                         env={**os.environ, "OMP_DYNAMIC": "FALSE", "OMP_NUM_THREADS": "4"})
        assert "FIELDS_OK" in reference.stdout
        assert "array temporary" not in reference.stderr.lower()

    binaries, manifests, cuda_objects, reuse = {}, {}, {}, []
    compiler_env = {**os.environ, "PYTHONPATH": str(checkout), "PYTHONHASHSEED": "0"}
    cuda_flags = ["-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
    for label, entry, policy, budget in (("basic", "step", "sections", None),
                                        ("native", "step_native", "sections", None),
                                        ("resource", "step", "sections", 0),
                                        ("auto", "step", "auto", None)):
        output = directory / label
        output.mkdir()
        facts = output / "facts.json"
        facts.write_text(json.dumps(_facts(original, native=entry == "step_native", budget=budget)))
        response = _run([sys.executable, "-m", "compiler.tests.support.reference_scopes", "--input", str(original), "--kernel", "operators::" + entry,
                         "--form-scopes", "--scope-facts", str(facts), "--memory-model", "scoped",
                         "--gpu-policy", policy, "--gpu-collective", "--host-threads", "4", "--json",
                         "--output-dir", str(output)], checkout, env=compiler_env)
        manifest = json.loads(response.stdout)["scopes"]
        manifests[label] = manifest
        (output / "public-response.json").write_text(response.stdout)
        if label == "auto":
            assert manifest["scope_count"] == 0
            assert manifest["participation"]["schema_version"] == 2
            assert manifest["boundaries"][0]["reason"] == (
                "no calibration profile supplied; automatic policy retains native execution; "
                "no calibration profile or exact-window evidence supplied; "
                "automatic policy retains unchanged native source")
            assert not manifest["build_sources"]
            binaries[label] = native_binary
            continue
        assert manifest["scope_count"] == 1, manifest["boundaries"]
        assert manifest["scopes"][0]["gpu_leaves"] == ["operators::consume", "operators::produce"]
        if label == "native":
            assert any(item["procedure"] == "operators::native_permute" and item["available"]
                       for item in manifest["collective_roles"])
        headers = tuple(sorted((str(path.relative_to(output)), sha256(path.read_bytes()).hexdigest())
                               for path in output.rglob("*") if path.suffix in {".h", ".hpp", ".cuh"}))
        objects = []
        for role in ("common_runtime", "shared_entry", "original_source"):
            for item in manifest["build_sources"]:
                if item["role"] != role:
                    continue
                source = output / item["path"]
                digest = sha256(source.read_bytes()).hexdigest()
                assert digest == manifest["artifacts_sha256"][item["path"]]
                target = output / (item["path"].replace("/", "_") + ".o")
                if item["language"] == "cuda":
                    key = (digest, headers, tuple(cuda_flags), manifest["runtime"]["runtime_id"])
                    if key in cuda_objects:
                        target = cuda_objects[key]
                        reuse.append({"case": label, "source": item["path"], "sha256": digest,
                                      "object": str(target), "proof": "identical source, all public headers, runtime identity and CUDA flags"})
                    else:
                        _run([nvcc, *cuda_flags, "-I", str(output), "-c", str(source), "-o", str(target)], output)
                        cuda_objects[key] = target
                else:
                    _run([fortran, *fortran_flags, "-c", str(source), "-o", str(target)], output)
                objects.append(str(target))
        counters = output / "counters.cpp"
        counters.write_text(COUNTERS)
        counters_object = output / "counters.o"
        _run([host, "-std=c++17", "-I", str(output), "-c", str(counters), "-o", str(counters_object)], output)
        binary = output / "verify"
        wrapping = ["-Wl,--wrap=" + name for name in ("fort_scope_create", "fort_scope_register",
                                                     "fort_scope_register_sections", "fort_scope_close")]
        _run([fortran, *fortran_flags, str(driver), *objects, str(counters_object), *wrapping,
              "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++",
              "-o", str(binary)], output)
        binaries[label] = binary
    (directory / "object-reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")
    return directory, binaries, manifests


def _execute(compiled, label, *, mode=None, env=None):
    directory, binaries, _manifests = compiled
    runtime_env = {**os.environ, "OMP_DYNAMIC": "FALSE", "OMP_NUM_THREADS": "4", "FORT_RUNTIME_TRACE": "1",
                   **(env or {})}
    result = _run([str(binaries[label]), mode or ("native" if label == "native" else "basic")], directory,
                  env=runtime_env, timeout=60)
    assert "FIELDS_OK" in result.stdout
    assert "array temporary" not in result.stderr.lower()
    (directory / f"run-{label}-{mode or 'default'}.stdout").write_text(result.stdout)
    (directory / f"run-{label}-{mode or 'default'}.stderr").write_text(result.stderr)
    return result


def _transfers(stderr, operation, identity):
    result = []
    for line in stderr.splitlines():
        if line.startswith("FORT_SCOPED " + operation + " "):
            fields = dict(field.split("=", 1) for field in line.split()[2:])
            if int(fields["buffer"]) == identity:
                result.append(int(fields["bytes"]))
    return result
