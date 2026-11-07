"""Original root and nested source teams execute through the graph transaction."""

import json
import os
import re
import shutil
from hashlib import sha256

import pytest

from compiler.tests.test_source_scopes import run
from compiler.tests.test_source_tile_tasks import emit


@pytest.fixture(scope="module", params=[(False, "whole"), (True, "whole"), (False, "none"), (True, "none")],
                ids=["root", "nested", "root-unknown", "nested-unknown"])
def compiled_source_task(request, tmp_path_factory):
    directory = tmp_path_factory.mktemp("compiled_source_task")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA and Fortran toolchains required")
    child, initialized = request.param
    original, outputs, manifest = emit(directory, child=child, initialized=initialized)
    build = directory / "build"
    build.mkdir()
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (build / "scope-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    from compiler.tests.test_reached_native_specification_cuda import validate_package

    validate_package(build)
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["role"] != role:
                continue
            path = build / artifact["path"]
            assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
            target = build / f"object-{index}.o"
            if artifact["language"] == "cuda":
                command = [nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp",
                           *artifact["numerical_contract"]["required_cuda_options"], "-I", str(build)]
            else:
                command = [fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build)]
            result = run([*command, "-c", str(path), "-o", str(target)], cwd=build)
            (build / f"compile-{index}.stdout").write_text(result.stdout)
            (build / f"compile-{index}.stderr").write_text(result.stderr)
            objects.append(str(target))
    procedure = "outer" if child else "advance"
    driver = build / "driver.f90"
    driver.write_text(f"""program exercise
use stencil
use iso_fortran_env, only:error_unit
implicit none
real(8),allocatable::a(:),b(:),expected(:),original_a(:)
integer::shape,n,i,repeat
do shape=1,4
 select case(shape)
 case(1)
 n=0
 case(2)
 n=1
 case(3)
 n=7
 case default
 n=19
 end select
 allocate(a(-3:n+4),b(-3:n+4),expected(-3:n+4),original_a(-3:n+4))
 do i=-3,n+4
  a(i)=real(17*i*i-3*i+11,8)/64.0_8
  b(i)=real(500-i,8)/32.0_8
 enddo
 original_a=a
 do repeat=1,2
  expected=b
  do i=1,n
   expected(i)=b(i)+(3.0_8*a(i+1)-3.0_8*a(i-1))
  enddo
  write(error_unit,'(A,1X,I0,1X,I0)')'SOURCE_CASE_BEGIN',n,repeat
  call {procedure}(a,b,n,3)
  write(error_unit,'(A,1X,I0,1X,I0)')'SOURCE_CASE_END',n,repeat
  if(any(a/=original_a))error stop 'readonly input changed'
  if(any(b/=expected))error stop 'full field or halo mismatch'
 enddo
 deallocate(a,b,expected,original_a)
enddo
print *, 'GENERATED_SOURCE_TEAM_FULL_FIELDS_AND_HALOS_OK'
end program
""")
    candidate, native = build / "candidate", build / "native"
    command = [fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-I", str(build), str(driver)]
    run([*command, *objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64",
         "-lcudart", "-lstdc++", "-o", str(candidate)], cwd=build)
    run([fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build),
         str(original), str(driver), "-o", str(native)], cwd=build)
    return build, candidate, native


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True], ids=["enabled", "hidden"])
def test_source_only_child_and_root_preserve_full_original_fields_and_fallback_once(compiled_source_task, hidden):
    build, candidate, native = compiled_source_task
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "FORT_RUNTIME_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    mode = "hidden" if hidden else "enabled"
    reference = run([str(native)], cwd=build, env=environment)
    result = run([str(candidate)], cwd=build, env=environment)
    for role, process in (("native", reference), ("candidate", result)):
        (build / f"{mode}-{role}.stdout").write_text(process.stdout)
        (build / f"{mode}-{role}.stderr").write_text(process.stderr)
    assert result.stdout == reference.stdout
    assert "GENERATED_SOURCE_TEAM_FULL_FIELDS_AND_HALOS_OK" in result.stdout
    cases = re.findall(r"SOURCE_CASE_BEGIN (\d+) (\d+)\n(.*?)SOURCE_CASE_END \1 \2", result.stderr, re.S)
    assert len(cases) == 8
    for size, _repeat, events in cases:
        size = int(size)
        launches = [line for line in events.splitlines() if line.startswith("FORT_SCOPED launch ")]
        copies = [line for line in events.splitlines() if line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download "))]
        if hidden or not size:
            assert not launches, events
            assert not copies, events
        else:
            # These bounded cases fit all source tiles in one provider batch.
            # Original private storage and per-tile arithmetic are unchanged.
            assert len(launches) == 1, events
            uploads = [line for line in copies if line.startswith("FORT_SCOPED upload ")]
            downloads = [line for line in copies if line.startswith("FORT_SCOPED download ")]
            assert len(uploads) == 2, events
            assert len(downloads) == 1, events
            assert [int(re.search(r"\bbytes=(\d+)\b", line)[1]) for line in uploads] == [8*(size+3), 8*size]
            assert [int(re.search(r"\bbytes=(\d+)\b", line)[1]) for line in downloads] == [8*size]
