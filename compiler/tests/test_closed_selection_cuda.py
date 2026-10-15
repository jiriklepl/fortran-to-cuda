"""Execute both reached original branches and verify exact-once native fallback."""

import json
import os
import re
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.numerical_contract import require_explicit_cuda_environment, require_numerical_build_contract
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_closed_selection import SOURCE
from compiler.tests.test_source_scopes import FACT

DRIVER = """program verify
use iso_fortran_env,only:int64,error_unit
use selected_groups
use reference_groups,only:native_advance=>advance,native_order=>order
implicit none
real(8)::a(0:16,0:8,0:4),b(0:16,0:8,0:4),expected(0:16,0:8,0:4)
integer::i,j,k,step,output
do k=0,4
do j=0,8
do i=0,16
a(i,j,k)=real(i+3*j+7*k,8)/16.0_8
enddo
enddo
enddo
b=-113.0_8
expected=b
do step=1,2
order=2*step
native_order=order
call native_advance(a,expected)
write(error_unit,*)'BEGIN_SELECTION',step
call advance(a,b)
write(error_unit,*)'END_SELECTION',step
if(any(transfer(b,[0_int64],size(b))/=transfer(expected,[0_int64],size(expected)))) &
  error stop 'complete fields differ or original group replayed'
if(any(b(16,:,:)/=-113.0_8).or.any(b(:,8,:)/=-113.0_8).or.any(b(:,:,4)/=-113.0_8)) &
  error stop 'unwritten halos changed'
enddo
open(newunit=output,file='fields.bin',access='stream',form='unformatted',status='replace')
write(output)lbound(b),ubound(b),b
close(output)
print *,'CLOSED_SELECTION_FIELDS_BITWISE_OK'
end program
"""


@pytest.mark.cuda
def test_closed_original_branches_execute_and_fallback_exactly_once(tmp_path):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fc, nvcc, host)):
        pytest.skip("native Fortran and CUDA toolchain required")
    require_explicit_cuda_environment()
    source = tmp_path / "original.f90"
    original = SOURCE.replace("intent(out)", "intent(inout)").replace(
        "b(i,j,k)=coefficient", "b(i,j,k)=b(i,j,k)+coefficient")
    source.write_text(original)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(source): sha256(source.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT, "argument::b": FACT, "selected_groups::tiles": FACT}}
    outputs, manifest = ScopeBuilder([source], "selected_groups::advance", facts=facts,
        options=CompilerOptions(gpu_policy="sections"), config=OffloadConfig(policy="sections")).run()
    assert manifest["scope_count"] == 2
    assert not manifest["boundaries"]
    build = tmp_path / "build"
    build.mkdir()
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static"}
    environment.pop("FORT_RUNTIME_TRACE", None)
    commands = []

    def run(argv, extra=None):
        ordinal = len(commands)
        commands.append({"argv": argv, "environment_overrides": extra or {}})
        (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(argv, cwd=build, env={**environment, **(extra or {})},
                                capture_output=True, text=True, timeout=180, check=False)
        (tmp_path / f"command-{ordinal:02}.stdout").write_text(result.stdout)
        (tmp_path / f"command-{ordinal:02}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    fflags = [fc, "-O3", "-fopenmp", "-ffp-contract=off", "-fcheck=all", "-J", str(build), "-I", str(build)]
    objects, compiled = [], set()
    for role in ("common_runtime", "shared_entry", "original_source"):
        for descriptor in manifest["build_sources"]:
            if descriptor["role"] != role:
                continue
            path, target = build / descriptor["path"], build / f"artifact-{len(objects):02}.o"
            assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][descriptor["path"]]
            flags = fflags
            if descriptor["language"] == "cuda":
                contract = descriptor["numerical_contract"]
                require_numerical_build_contract(contract)
                flags = [nvcc, "-O3", "-std=c++17", "-ccbin", host, "-Xcompiler=-fopenmp",
                         "-arch=" + os.environ.get("FORT_TEST_CUDA_ARCH", "native"),
                         *contract["required_cuda_options"],
                         *("-Xcompiler=" + option for option in contract["required_host_options"])]
            run([*flags, "-c", str(path), "-o", str(target)])
            objects.append(str(target))
            compiled.add(descriptor["path"])
    assert compiled == {item["path"] for item in manifest["build_sources"]}
    for name, text in (("reference", original.replace("selected_groups", "reference_groups")), ("driver", DRIVER)):
        path, target = build / (name + ".f90"), build / (name + ".o")
        path.write_text(text)
        run([*fflags, "-c", str(path), "-o", str(target)])
        objects.append(str(target))
    executable = build / "verify"
    run([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", *objects, "-lgfortran", "-o", str(executable)])
    observed = run([str(executable)], {"FORT_RUNTIME_TRACE": "1"})
    assert "CLOSED_SELECTION_FIELDS_BITWISE_OK" in observed.stdout
    expected = (build / "fields.bin").read_bytes()
    calls = observed.stderr.split("BEGIN_SELECTION")[1:]
    assert len(calls) == 2
    ledger = []
    for call in calls:
        trace = call.split("END_SELECTION", 1)[0]
        events = [dict(re.findall(r"(\w+)=([^ ]+)", line)) | {"kind": line.split()[1]}
                  for line in trace.splitlines() if line.startswith("FORT_SCOPED ") and "provenance_version=1" in line]
        assert sum(item["kind"] == "launch" for item in events) == 1, trace
        assert sum(item["kind"] == "close" for item in events) == 1, trace
        transfers = [item for item in events if item["kind"] in {"upload", "download"}]
        assert sum(int(item["bytes"]) for item in transfers if item["kind"] == "upload") == 2 * 16 * 8 * 4 * 8, trace
        assert sum(int(item["bytes"]) for item in transfers if item["kind"] == "download") == 16 * 8 * 4 * 8, trace
        ledger.append(transfers)
    (tmp_path / "transfer-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
    for extra in ({}, {"FORT_RUNTIME_TRACE": "1", "OMP_NUM_THREADS": "2"},
                  {"FORT_RUNTIME_TRACE": "1", "CUDA_VISIBLE_DEVICES": ""}):
        result = run([str(executable)], extra)
        assert "CLOSED_SELECTION_FIELDS_BITWISE_OK" in result.stdout
        assert (build / "fields.bin").read_bytes() == expected
        assert not re.search(r"FORT_SCOPED (?:launch|upload|download) ", result.stderr)
