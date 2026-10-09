"""Original saved storage/counters survive compiler-owned inline GPU workers."""

import json
import os
import shutil
from hashlib import sha256

import pytest

from compiler.tests.test_source_inline_dispatch import PROGRAM
from compiler.tests.test_source_scopes import FACT, ROOT, generate, run

SOURCE = PROGRAM.replace("subroutine step(a,out,n)", "subroutine step(a,out,n,count)").replace(
    "integer,intent(in)::n", "integer,intent(in)::n\ninteger,intent(out)::count").replace(
        "enddo\nend subroutine", "enddo\ncount=calls\nend subroutine")

STRUCTURED_SOURCE = SOURCE.replace("integer::i", "integer::i,extent").replace(
    "do i=-2,n-3\nout(i)=scratch(i)+a(i)\nenddo",
    "if(n>0) then\nout(-2)=out(-2)+1\nextent=n\n"
    "if(out(-2)>-1000.d0) then\ndo i=-2,extent-3\nout(i)=scratch(i)+a(i)\nenddo\nendif\nendif")
DOUBLE_PRIVATE_SOURCE = SOURCE.replace("integer::i", "integer::i\nreal(8)::temporary").replace(
    "scratch(i)=2*a(i)+real(i,8)",
    "temporary=a(i)+1.0000000000000002d0\nscratch(i)=temporary+real(i,8)")
CASES = {"saved-allocation": SOURCE, "reached-branches-partial-write": STRUCTURED_SOURCE,
         "double-private-temporary": DOUBLE_PRIVATE_SOURCE}

DRIVER = """program verify
use saved_work,only:step
implicit none
real(8),allocatable::a(:),out(:)
integer::n,shape,repeat,i,low,unit,count,expected
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
expected=0
do shape=0,3
 n=16*shape
 if(shape==3) n=16
 low=-5-shape
 allocate(a(low:low+n+3),out(low-3:low+n))
 do repeat=1,2
  do i=0,n+3
   a(low+i)=real(i,8)*0.25d0+repeat
  enddo
  out=-117.d0
  call step(a,out,n,count)
  expected=expected+1
  if(count/=expected) error stop 'original saved counter was duplicated or replayed'
  if(any(out(low+n-3:low+n)/=-117.d0)) error stop 'output halo changed'
  write(unit) n,low,count,a,out
  print *, 'CALL_OK'
 enddo
 deallocate(a,out)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module")
def inline_binaries(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, C++ and Fortran toolchains required")
    directory = tmp_path_factory.mktemp("source-inline")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    common = directory / "common-build"
    common.mkdir()
    facts = {"schema_version": 1, "participation": "serial", "captures": {
        "argument::a": FACT, "argument::out": FACT,
        "saved_work::step::scratch": {**FACT, "initialized": "none"}}}
    cflags = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
    fflags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
    binaries, cache, reuse = {}, {}, []
    for label, source in CASES.items():
        binaries[label] = {}
        for mode in ("auto", "sections"):
            case = directory / label / mode
            original, output, manifest = generate(case, source, mode=mode, facts=facts, checkout=checkout)
            assert manifest["scope_count"] == 1, manifest["boundaries"]
            assert len(manifest["inline_numerical_regions"]["regions"]) == 2
            build = case / "build"
            build.mkdir()
            objects = []
            headers = tuple((name, digest) for name, digest in sorted(manifest["artifacts_sha256"].items())
                            if name.endswith((".h", ".hpp", ".cuh")))
            for role in ("common_runtime", "shared_entry", "original_source"):
                for item in manifest["build_sources"]:
                    if item["role"] != role:
                        continue
                    path = output / item["path"]
                    flags = cflags if item["language"] == "cuda" else fflags
                    key = (item["language"], sha256(path.read_bytes()).hexdigest(),
                           headers if item["language"] == "cuda" else (), tuple(flags))
                    cached = role != "original_source" and key in cache
                    if cached:
                        target = cache[key]
                    else:
                        target = build / (str(len(objects)) + ".o")
                        command = [*flags]
                        if item["language"] == "fortran":
                            command += ["-J", str(build if role == "original_source" else common), "-I", str(common)]
                        run([*command, "-c", str(path), "-o", str(target)], cwd=build)
                        if role != "original_source":
                            cache[key] = target
                    reuse.append({"case": label, "mode": mode, "path": item["path"], "sha256": key[1],
                                  "headers": headers if item["language"] == "cuda" else (), "flags": flags,
                                  "reused": cached, "object": str(target)})
                    objects.append(str(target))
            driver = build / "driver.f90"
            driver.write_text(DRIVER)
            binary = build / "verify"
            run([*fflags, "-I", str(build), "-I", str(common), str(driver), *objects, "-L/usr/local/cuda/lib64",
                 "-Wl,-rpath,/usr/local/cuda/lib64", "-lcudart", "-lstdc++", "-o", str(binary)], cwd=build)
            native_build = case / "native-build"
            native_build.mkdir()
            native = native_build / "verify"
            run([*fflags, str(original), str(driver), "-o", str(native)], cwd=native_build)
            reference = run([str(native)], cwd=native_build, env=environment)
            binaries[label][mode] = (binary, build, native_build / "fields.bin", reference.stdout, manifest)
    (directory / "object-reuse.json").write_text(json.dumps(reuse, indent=2) + "\n")
    return binaries


@pytest.mark.cuda
@pytest.mark.parametrize("label", CASES)
def test_saved_inline_regions_match_original_state_complete_arrays_and_halos(inline_binaries, label):
    for mode in ("auto", "sections"):
        binary, build, reference, stdout, manifest = inline_binaries[label][mode]
        result = run([str(binary)], cwd=build, env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
                                                  "FORT_RUNTIME_TRACE": "1"})
        (build / "formed.stdout").write_text(result.stdout)
        (build / "formed.stderr").write_text(result.stderr)
        assert result.stdout == stdout
        assert result.stdout.count("CALL_OK") == 8
        assert (build / "fields.bin").read_bytes() == reference.read_bytes()
        assert "array temporary" not in result.stderr.lower()
        if mode == "auto":
            assert not manifest["automatic_estimate_available"]
            for event in ("alloc", "h2d", "d2h", "launch"):
                assert "FORT_SCOPED " + event + " " not in result.stderr
        else:
            assert "FORT_SCOPED launch " in result.stderr
