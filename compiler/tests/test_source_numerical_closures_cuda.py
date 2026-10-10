"""Guarded source closures agree with native fields across runtime tile changes."""

import json
import os
import shutil
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.numerical_contract import require_numerical_build_contract
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_numerical_closures import SOURCE
from compiler.tests.test_source_scopes import FACT, generate

DRIVER = """program check
use renamed_numerics,only:advance
use iso_fortran_env,only:error_unit
use ieee_arithmetic
use ieee_exceptions
implicit none
real(8),allocatable::a(:)
integer::shape,repeat,low,n,tile,i,unit,ordinal
logical::trapping
type(ieee_round_type)::rounding
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
ordinal=0
do shape=0,3
 n=9*shape
 low=-7-shape
 allocate(a(low:low+n+7))
 do repeat=1,4
  tile=3
  if(repeat==2) tile=7
  if(repeat==3) tile=-3
  if(repeat==4) tile=1
  do i=low,low+n+7
   a(i)=real(i-low,8)*0.25d0+1
  enddo
  if(repeat==4) call ieee_set_rounding_mode(ieee_up)
  ordinal=ordinal+1
  write(error_unit,*) 'BEGIN_CALL',ordinal
  call advance(a,n,tile)
  write(error_unit,*) 'END_CALL',ordinal
  call ieee_get_rounding_mode(rounding)
  if(repeat==4.and.rounding/=ieee_up) error stop 'original rounding mode changed'
  call ieee_set_rounding_mode(ieee_nearest)
  call ieee_get_halting_mode(ieee_invalid,trapping)
  if(trapping) error stop 'host trap state changed'
  write(unit) n,tile,a
 enddo
 deallocate(a)
enddo
close(unit)
print *, 'FIELDS_OK'
end program
"""


@pytest.mark.cuda
@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("helper", ["subroutine", "function"])
def test_guarded_private_helper_source_matches_native_complete_fields_and_halos(tmp_path, precision, helper):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, C++ and Fortran toolchains required")
    source = SOURCE.replace("do block=-3,n,tile", "i=0\n!$omp parallel do private(g,s,i,block) schedule(runtime)\ndo block=-3,n,tile").replace(
        "enddo\ncontains", "enddo\n!$omp end parallel do\ncontains")
    if helper == "function":
        source = source.replace("call invariant(s,g)", "s=invariant(g)").replace(
            "pure subroutine invariant(s,g)", "pure function invariant(g) result(s)").replace(
            "real(8),intent(out)::s", "real(8)::s").replace(
            "s=dot_product(g,g)\nend subroutine", "s=dot_product(g,g)\nend function")
    driver_text = DRIVER
    if precision == 4:
        source = source.replace("real(8)", "real(4)").replace(",8)", ",4)").replace("d0", "_4")
        driver_text = driver_text.replace("real(8)", "real(4)").replace(",8)", ",4)").replace("d0", "_4")
    facts = {"schema_version": 1, "participation": "serial", "captures": {"argument::a": FACT}}
    original, output, manifest = generate(tmp_path / "formed", source, entry="advance", facts=facts)
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    build = tmp_path / "build"
    build.mkdir()
    architecture = actual_profile(build, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    cflags = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_" + architecture, "-Xcompiler=-fopenmp"]
    fflags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    objects, commands = [], []

    def execute(command, *, cwd=build, env=None):
        result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, timeout=120)
        commands.append({"argv": command, "cwd": str(cwd), "exit_code": result.returncode,
                         "stdout": result.stdout, "stderr": result.stderr})
        (build / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            path = output / item["path"]
            target = build / (str(len(objects)) + ".o")
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                flags = [*cflags, *contract["required_cuda_options"],
                         *("-Xcompiler=" + flag for flag in contract["required_host_options"])]
            else:
                flags = [*fflags, "-J", str(build), "-I", str(build)]
            execute([*flags, "-c", str(path), "-o", str(target)])
            objects.append(str(target))
    driver = build / "driver.f90"
    driver.write_text(driver_text)
    binary = build / "check"
    library = Path(nvcc).resolve().parents[1] / "lib64"
    execute([*fflags, "-I", str(build), str(driver), *objects, "-L" + str(library),
             "-Wl,-rpath," + str(library), "-lcudart", "-lstdc++", "-o", str(binary)])
    native = tmp_path / "native"
    native.mkdir()
    native_binary = native / "check"
    execute([*fflags, str(original), str(driver), "-o", str(native_binary)], cwd=native)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"}
    reference = execute([str(native_binary)], cwd=native, env=environment)
    measured = execute([str(binary)], env={**environment, "FORT_RUNTIME_TRACE": "1"})
    (build / "build-commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (build / "scope-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (build / "candidate.stderr").write_text(measured.stderr)
    assert measured.stdout == reference.stdout
    assert (build / "fields.bin").read_bytes() == (native / "fields.bin").read_bytes()
    assert "array temporary" not in measured.stderr.lower()
    calls = measured.stderr.split("BEGIN_CALL")[1:]
    assert len(calls) == 16
    for ordinal, call in enumerate(calls, 1):
        trace = call.split("END_CALL")[0]
        if ordinal % 4 in {0, 3}:  # Non-nearest rounding or nonpositive tile.
            assert "FORT_SCOPED launch " not in trace
            assert "FORT_SCOPED upload " not in trace
            assert "FORT_SCOPED download " not in trace
        else:
            assert "FORT_SCOPED launch " in trace
    fields = (build / "fields.bin").read_bytes()
    (build / "native-fields.bin").write_bytes((native / "fields.bin").read_bytes())
    receipt = {"schema_version": 1, "precision_bytes": precision, "helper": helper,
               "complete_arrays_and_halos_equal": True, "guarded_calls": len(calls),
               "native_fields_sha256": sha256((native / "fields.bin").read_bytes()).hexdigest(),
               "candidate_fields_sha256": sha256(fields).hexdigest(), "field_bytes": len(fields),
               "numerical_contract_consumed": True, "architecture": architecture,
               "launches": measured.stderr.count("FORT_SCOPED launch "),
               "new_application_observations": 0, "new_performance_samples": 0}
    (build / "closure-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
