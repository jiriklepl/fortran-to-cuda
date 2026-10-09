"""Complete fields and halos survive native descriptor calls inside residency."""

import json
import os
import shutil
from hashlib import sha256

import pytest

from compiler.tests.test_source_descriptor_forwarding import program
from compiler.tests.test_source_scopes import FACT, ROOT, generate, run

CASES = {
    "omitted": (False, False, False),
    "present": (True, False, False),
    "forwarded-omitted": (False, True, False),
    "forwarded-present": (True, True, False),
    "readonly-allocation": (True, False, True),
}

DRIVER = """program verify
use original,only:step
implicit none
real(8),allocatable::a(:),b(:),output(:)
integer::n,shape,repeat,i,low,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=0,3
 n=16*shape
 if(shape==3) n=16
 low=-2-shape
 allocate(a(low:low+n-1),b(low-2:low+n-3),output(low-4:low+n-5))
 do repeat=1,2
  do i=0,n-1
   a(low+i)=real(i,8)*0.25d0+repeat
  enddo
  b=-99.d0
  output=-117.d0
  call step(a,b,output,n)
  if(n>4) then
   if(any(b(low-2:low-1)/=-99.d0).or.any(b(low+n-4:low+n-3)/=-99.d0)) error stop 'B halo changed'
   if(any(output(low-4:low-3)/=-117.d0).or.any(output(low+n-6:low+n-5)/=-117.d0)) error stop 'output halo changed'
  endif
  write(unit) n,low,a,b,output
  print *, 'CALL_OK'
 enddo
 deallocate(a,b,output)
enddo
close(unit)
end program
"""


def partial_program(supplied, nested, allocatable):
    source = program(supplied=supplied, nested=nested, allocatable=allocatable)
    source = source.replace("intent(out)::b(:)", "intent(inout)::b(:)")
    source = source.replace("intent(out)::out(:)", "intent(inout)::out(:)")
    source = source.replace("intent(out)::b(:),out(:)", "intent(inout)::b(:),out(:)")
    source = source.replace("do i=1,n", "do i=3,n-2")
    inquiry = "allocated" if allocatable else "present"
    extra = "+real(lbound(extra,1),8)" if allocatable else ""
    section = "lbound(extra,1)+2:ubound(extra,1)-2" if allocatable else "3:size(extra)-2"
    source = source.replace(f"b=3*b\nif ({inquiry}(extra)) b=b+extra", f"""if(size(b)>4) then
b(3:size(b)-2)=3*b(3:size(b)-2)
if ({inquiry}(extra)) b(3:size(b)-2)=b(3:size(b)-2)+extra({section}){extra}
endif""")
    return source


@pytest.fixture(scope="module")
def descriptor_binaries(tmp_path_factory):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, C++ and Fortran toolchains required")
    directory = tmp_path_factory.mktemp("native-descriptor-scopes")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    common = directory / "common-build"
    common.mkdir()
    cflags = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
    fflags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
    binaries, cache, reuse = {}, {}, []
    facts = {"schema_version": 1, "participation": "serial", "captures": {
        "argument::" + name: FACT for name in ("a", "b", "out")}}
    for label, parameters in CASES.items():
        binaries[label] = {}
        source = partial_program(*parameters)
        for mode in ("auto", "sections"):
            case = directory / label / mode
            original, output, manifest = generate(case, source, mode=mode, facts=facts, checkout=checkout)
            assert manifest["scope_count"] == 1, manifest["boundaries"]
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
            run([*fflags, "-I", str(build), str(driver), *objects, "-L/usr/local/cuda/lib64",
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
def test_native_descriptor_calls_preserve_complete_fields_and_halos(descriptor_binaries, label):
    for mode in ("auto", "sections"):
        binary, directory, reference, stdout, manifest = descriptor_binaries[label][mode]
        result = run([str(binary)], cwd=directory, env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE",
                                                       "FORT_RUNTIME_TRACE": "1"})
        (directory / "formed.stdout").write_text(result.stdout)
        (directory / "formed.stderr").write_text(result.stderr)
        assert result.stdout == stdout
        assert result.stdout.count("CALL_OK") == 8
        assert (directory / "fields.bin").read_bytes() == reference.read_bytes()
        assert "array temporary" not in result.stderr.lower()
        if mode == "auto":
            assert not manifest["automatic_estimate_available"]
            assert "FORT_SCOPED " not in result.stderr
        else:
            assert "FORT_SCOPED launch " in result.stderr
