"""Reached source segments retain fields across branches and native teams."""

from __future__ import annotations

import json
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.tests.test_module_allocatable_scopes_runtime import ROOT, _run
from compiler.tests.test_source_continuation import TREE
from compiler.tests.test_source_scopes import FACT, generate

SOURCE = TREE.replace("real(8),intent(out)::b(:)", "real(8),intent(inout)::b(:)").replace(
    "real(8),intent(out)::out(:)", "real(8),intent(inout)::out(:)").replace(
    "do i=1,n", "do i=2,n-1").replace(
    "if(b(-2)>0) b(-2)=b(-2)+7",
    "if(n>2) then\nif(b(-1)>0.and.lbound(b,1)==-2) b(-1)=b(-1)+7+real(lbound(b,1),8)\nendif")

JOINED = SOURCE.replace("logical::flag", "logical::flag\ninteger::i").replace(
    "n=n-1\ncall consumer", "!$omp parallel private(i) shared(b,n)\n!$omp do\n"
    "do i=-1,n-4\nb(i)=b(i)+1\nenddo\n!$omp end do nowait\n"
    "!$omp end parallel\nn=n-1\ncall consumer")
PARTIAL = SOURCE.replace("flag=sum(b)>0", "flag=.true.")
GUARDED = SOURCE.replace("call consumer(a,b,out,n)\nelse", "call consumer(a,b,out,2147483647)\nelse")
HIDDEN_CONTROL = SOURCE.replace("implicit none", "implicit none\ninteger,save::limit=0,next_limit=0", 1).replace(
    "n=n-1\ncall consumer(a,b,out,n)", "next_limit=n-1\ncall set_limit()\nif(limit>0) then\ncall consumer(a,b,out,limit)\nendif").replace(
    "end module", "subroutine set_limit()\nlimit=next_limit\nend subroutine\nend module")
ALLOCATED_FORMALS = SOURCE.replace("real(8),intent(in)::a(-2:)", "real(8),allocatable,intent(in)::a(:)").replace(
    "real(8),intent(inout)::b(-2:),out(-2:)", "real(8),allocatable,intent(inout)::b(:),out(:)").replace(
    "if(flag) then", "if(allocated(b).and.flag) then").replace("lbound(b,1)==-2", "lbound(b,1)==-7").replace(
    "logical::flag\ncall producer", "logical::flag\nif(.not.allocated(b)) return\ncall producer")
ALLOCATED_MODULE = SOURCE.replace("implicit none", "implicit none\nreal(8),allocatable::field(:)", 1)
_module_prefix, _module_entry = ALLOCATED_MODULE.split("subroutine step", 1)
_module_specification, _module_body = _module_entry.split("call producer", 1)
_module_body = re.sub(r"\bb\b", "field", "call producer" + _module_body)
ALLOCATED_MODULE = _module_prefix + "subroutine step" + _module_specification + _module_body.replace(
    "if(flag) then", "if(allocated(field).and.flag) then").replace("lbound(field,1)==-2", "lbound(field,1)==-7")
ALLOCATED_MODULE = ALLOCATED_MODULE.replace("logical::flag\ncall producer", "logical::flag\nif(.not.allocated(field)) return\ncall producer")
HIDDEN_ARRAY = ALLOCATED_MODULE.replace("allocatable::field", "allocatable,target::field").replace(
    "n=n-1\ncall consumer", "call touch_field()\nn=n-1\ncall consumer").replace(
    "end module", "subroutine touch_field()\nif(size(field)>2) field(lbound(field,1)+1)=field(lbound(field,1)+1)+11\nend subroutine\nend module")

DRIVER = """program caller
use original,only:step
implicit none
real(8),allocatable::a(:),b(:),out(:)
integer::shape,sign,n,i,threads
character(16)::mode
call get_command_argument(1,mode)
do shape=1,4
 n=[0,2,8,32](shape)
 allocate(a(-7:n-8),b(-7:n-8),out(-7:n-8))
 do sign=-1,1,2
  do i=-7,n-8
   a(i)=real(sign*64,8)+real(i+8,8)*0.25d0
  enddo
  b=-99.d0
  out=-101.d0
  if(trim(mode)=='team') then
   !$omp parallel num_threads(4) private(threads)
   !$omp single
   call step(a,b,out,n)
   !$omp end single
   !$omp end parallel
  else
   call step(a,b,out,n)
  endif
  print '(a,4i5)', 'CALL_OK',shape,sign,n,lbound(b,1)
  print '(1000f14.3)', b
  print '(1000f14.3)', out
  n=size(a)
 enddo
 deallocate(a,b,out)
enddo
print *, 'FIELDS_OK'
end program
""".replace("integer::shape,sign,n,i,threads", "integer::shape,sign,n,i,threads\ninteger,parameter::sizes(4)=[0,2,8,32]").replace(
    "n=[0,2,8,32](shape)", "n=sizes(shape)")
MODULE_DRIVER = DRIVER.replace("use original,only:step", "use original,only:step,field").replace(
    "allocate(a(-7:n-8),b(-7:n-8),out(-7:n-8))", "allocate(a(-7:n-8),b(-7:n-8),out(-7:n-8),field(-7:n-8))").replace(
    "b=-99.d0", "b=-99.d0\n  field=-99.d0").replace("'CALL_OK',shape,sign,n,lbound(b,1)", "'CALL_OK',shape,sign,n,lbound(field,1)").replace(
    "print '(1000f14.3)', b", "print '(1000f14.3)', field").replace("deallocate(a,b,out)", "deallocate(a,b,out,field)")
ALLOCATED_DRIVER = DRIVER.replace("do shape=1,4", "if(trim(mode)=='unallocated') then\n"
                                  "n=8\ncall step(a,b,out,n)\nprint *, 'FIELDS_OK'\nstop\nendif\ndo shape=1,4")
MODULE_DRIVER = MODULE_DRIVER.replace("do shape=1,4", "if(trim(mode)=='unallocated') then\n"
                                     "n=8\nallocate(a(8),b(8),out(8))\ncall step(a,b,out,n)\n"
                                     "print *, 'FIELDS_OK'\nstop\nendif\ndo shape=1,4")

DIAGNOSTIC = r"""#include "scoped_runtime.h"
#include <cstdio>
extern "C" int __real_fort_scope_close(fort_scope_t);
extern "C" int __wrap_fort_scope_close(fort_scope_t context) {
    fort_scope_plan_report report{};
    const int status = fort_scope_plan_report_v2(context, &report);
    std::fprintf(stderr, "FIXTURE_CLOSE context=%lld status=%d segments=%llu owner_available=%u\n",
                 (long long)context, status, (unsigned long long)report.owner_segments, report.owner_available);
    return __real_fort_scope_close(context);
}
"""


@pytest.fixture(scope="module")
def continuation_binaries(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, Fortran and OpenMP toolchains are required")
    directory = tmp_path_factory.mktemp("source-continuation")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    fortran_flags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    cuda_flags = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
    cache, targets, reuse = {}, {}, []
    for label, policy, source in [("tree", "sections", SOURCE), ("automatic", "auto", SOURCE),
                                  ("joined", "sections", JOINED), ("partial", "sections", PARTIAL),
                                  ("guarded", "sections", GUARDED), ("resource-before", "sections", SOURCE),
                                  ("hidden-control", "sections", HIDDEN_CONTROL),
                                  ("allocated-formals", "sections", ALLOCATED_FORMALS),
                                  ("allocated-module", "sections", ALLOCATED_MODULE), ("hidden-array", "sections", HIDDEN_ARRAY)]:
        case = directory / label
        facts = {"schema_version": 1, "participation": "serial",
                 "captures": {"argument::a": FACT, "argument::b": FACT, "argument::out": FACT}}
        if label == "partial":
            facts["captures"].update({"argument::b": {**FACT, "initialized": "none"},
                                      "argument::out": {**FACT, "initialized": "none"}})
        if label == "resource-before":
            facts["device_budget_bytes"] = 0
        if label in {"allocated-module", "hidden-array"}:
            facts["captures"]["original::field"] = FACT
        original, output, manifest = generate(case, source, mode=policy, facts=facts, checkout=checkout)
        scope, = manifest["scopes"]
        assert scope["ownership"]["planning_mode"] == "continuation", manifest["boundaries"]
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
        diagnostic = output / "close-evidence.cpp"
        diagnostic.write_text(DIAGNOSTIC)
        diagnostic_object = output / "close-evidence.o"
        _run([host, "-std=c++17", "-I", str(output), "-c", str(diagnostic), "-o", str(diagnostic_object)], output)
        driver = output / "driver.f90"
        driver.write_text(MODULE_DRIVER if label in {"allocated-module", "hidden-array"} else ALLOCATED_DRIVER if label == "allocated-formals" else DRIVER)
        target = output / "verify"
        _run([*fortran_flags, str(driver), *objects, str(diagnostic_object), "-Wl,--wrap=fort_scope_close",
              "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(target)], output)
        native = output / "native"
        _run([*fortran_flags, str(original), str(driver), "-o", str(native)], output)
        references = {}
        for mode in ("serial", "team", *(["unallocated"] if label.startswith("allocated") else [])):
            result = _run([str(native), mode], output, env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"})
            (output / (mode + "-native.stdout")).write_text(result.stdout)
            references[mode] = result.stdout
        targets[label] = target, output, references
    (directory / "cuda-object-reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")
    return targets


@pytest.mark.cuda
@pytest.mark.parametrize("label", ["tree", "automatic", "joined", "partial", "guarded", "resource-before",
                                 "hidden-control", "allocated-formals", "allocated-module", "hidden-array"])
@pytest.mark.parametrize("mode", ["serial", "team", "no-device"])
def test_source_continuation_matches_complete_native_fields(continuation_binaries, label, mode):
    target, output, references = continuation_binaries[label]
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if mode == "no-device":
        environment["CUDA_VISIBLE_DEVICES"] = ""
    result = _run([str(target), "team" if mode == "team" else "serial"], output, env=environment)
    (output / (mode + "-scoped.stdout")).write_text(result.stdout)
    (output / (mode + "-scoped.stderr")).write_text(result.stderr)
    assert result.stdout == references["team" if mode == "team" else "serial"]
    assert "array temporary" not in result.stderr.lower()
    assert result.stdout.count("CALL_OK") == 8
    if mode == "serial":
        if label == "automatic":
            # No profile supplies any GPU estimate. The compiler proves this
            # before creating an owner, while retaining the original fields.
            assert "FIXTURE_CLOSE" not in result.stderr
            assert "FORT_SCOPED" not in result.stderr
            return
        assert result.stderr.count("FIXTURE_CLOSE") == 8, result.stderr
        if label == "resource-before":
            assert "FORT_SCOPED launch" not in result.stderr
            assert "FORT_SCOPED upload" not in result.stderr
        else:
            assert result.stderr.count("FORT_SCOPED launch") >= 8, result.stderr
    else:
        assert "FORT_SCOPED launch" not in result.stderr
        assert "FORT_SCOPED upload" not in result.stderr


@pytest.mark.cuda
@pytest.mark.parametrize("label", ["allocated-formals", "allocated-module"])
def test_unallocated_original_guard_precedes_all_owner_descriptor_reads(continuation_binaries, label):
    target, output, references = continuation_binaries[label]
    result = _run([str(target), "unallocated"], output, env={**os.environ, "OMP_NUM_THREADS": "4", "FORT_RUNTIME_TRACE": "1"})
    (output / "unallocated-scoped.stdout").write_text(result.stdout)
    (output / "unallocated-scoped.stderr").write_text(result.stderr)
    assert result.stdout == references["unallocated"]
    assert "FIXTURE_CLOSE" not in result.stderr
    assert "FORT_SCOPED" not in result.stderr
