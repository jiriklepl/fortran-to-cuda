"""Execute compiler-formed rectangular scopes through public build artifacts."""

import json
import os
import shutil
from hashlib import sha256

import pytest

from compiler.tests.test_source_rectangular_views import PROGRAM, nested_program
from compiler.tests.test_source_scopes import FACT, ROOT, generate, run

DRIVER = """program verify
use windows,only:step
implicit none
real(8),allocatable::a(:,:),b(:,:),output(:,:)
integer::n,m,shape,repeat,i,j,low,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=0,3
 n=10+4*shape
 m=9+2*shape
 if(shape==0) n=0
 if(shape==3) then
 n=10
 m=9
 endif
 low=-5-shape
 allocate(a(low:low+n-1,low-2:low+m-3),b(low-3:low+n-4,low-5:low+m-6), &
          output(low-7:low+n-8,low-9:low+m-10))
 do repeat=1,2
  do j=0,m-1
  do i=0,n-1
   a(low+i,low-2+j)=real(i+16*j,8)*0.25d0+repeat
  enddo
  enddo
  b=-99.d0
  output=-117.d0
  call step(a,b,output,1)
  write(unit) n,m,low,a,b,output
  print *, 'CALL_OK'
 enddo
 deallocate(a,b,output)
enddo
close(unit)
end program
"""


def guarded(source):
    source = source.replace("call produce(b=b", "if(size(a,1)>3.and.size(a,2)>3) then\ncall produce(b=b")
    return source.replace("             out(-1:ubound(out,1)-edge,-2:ubound(out,2)-1))",
                          "             out(-1:ubound(out,1)-edge,-2:ubound(out,2)-1))\nendif")


ALIASES = """module windows
implicit none
contains
subroutine paired(left,right)
real(8),intent(inout)::left(:,:),right(:,:)
integer::i,j
do j=1,size(left,2)
do i=1,size(left,1)
left(i,j)=left(i,j)+real(i+8*j,8)
right(i,j)=2*right(i,j)+real(2*i-j,8)
enddo
enddo
end subroutine
subroutine mixture(left,right,result)
real(8),intent(in)::left(:,:),right(:,:)
real(8),intent(out)::result(:,:)
integer::i,j
do j=1,size(result,2)
do i=1,size(result,1)
result(i,j)=left(i,j)+2*right(i,j)+real(i+7*j+size(left),8)
enddo
enddo
end subroutine
subroutine step(a,b,out,edge)
real(8),intent(inout)::a(-2:,-3:),b(-2:,-3:),out(-2:,-3:)
integer,intent(in)::edge
if(size(a,1)>8.and.size(a,2)>6) then
CALLS
endif
end subroutine
end module
"""


CASES = {
    "original-coordinates-native-consumer": guarded(PROGRAM),
    "disjoint-same-root-writes": ALIASES.replace("CALLS", """call paired(a(-1:2,-2:0),a(3:6,1:3))
call paired(a(-1:2,-2:0),a(3:6,1:3))"""),
    "overlapping-immutable-reads": ALIASES.replace("CALLS", """call mixture(a(-1:2,-2:0),a(0:3,-1:1),out(-1:2,-2:0))
call mixture(a(0:3,-1:1),a(-1:2,-2:0),out(3:6,1:3))"""),
    "projected-native-middle": guarded(PROGRAM).replace("call inspect(b)", "call inspect(b(-1:1,-2:2))"),
    "projected-native-partial-out": guarded(PROGRAM).replace(
        "intent(inout)::b(:,:)\nb=3*b", "intent(out)::b(:,:)\nb=5.d0").replace(
            "call inspect(b)", "call inspect(b(-1:1,-2:2))"),
    "nested-views-native-middle": nested_program().replace(
        "call consume(a(2:size(a,1)-edge", "call inspect(b(2:size(b,1)-edge,2:size(b,2)-1))\ncall consume(a(2:size(a,1)-edge").replace(
            "call outer(a,b,out,edge)\ncall outer(a,b,out,edge)",
            "if(size(a,1)>6.and.size(a,2)>6) then\ncall outer(a,b,out,edge)\ncall outer(a,b,out,edge)\nendif"),
}


@pytest.fixture(scope="module")
def rectangular_binaries(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, C++ and Fortran toolchains required")
    directory = tmp_path_factory.mktemp("source-rectangles")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    common = directory / "common-build"
    common.mkdir()
    cflags = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp"]
    fflags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
    facts = {"schema_version": 1, "participation": "serial", "captures": {
        "argument::" + name: FACT for name in ("a", "b", "out")}}
    binaries, cache, reuse = {}, {}, []
    for label, source in CASES.items():
        binaries[label] = {}
        for mode in ("auto", "sections"):
            case = directory / label / mode
            original, output, manifest = generate(case, source, mode=mode, facts=facts, checkout=checkout)
            assert manifest["scope_count"] == 1, manifest["boundaries"]
            assert manifest["scopes"][0]["borrowed_views"]["abi_version"] == 2
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
def test_source_rectangular_scopes_preserve_full_fields_and_halos(rectangular_binaries, label):
    for mode in ("auto", "sections"):
        binary, directory, reference, stdout, manifest = rectangular_binaries[label][mode]
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
