"""A public CLI consumer transports native scalar constants into CUDA work."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.numerical_contract import require_explicit_cuda_environment, require_numerical_build_contract


@pytest.mark.cuda
@pytest.mark.parametrize("precision", [4, 8])
@pytest.mark.parametrize("integer", [False, True])
def test_public_reached_scope_captures_native_constants_and_preserves_complete_fields(tmp_path, precision, integer):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fc, nvcc, host)):
        pytest.skip("native Fortran and CUDA toolchain required")
    require_explicit_cuda_environment()
    constant_type = "integer" if integer else f"real({precision})"
    initializer = f"int(acos(-1._{precision}))" if integer else f"acos(-1._{precision})"
    constants = tmp_path / "constants.f90"
    constants.write_text(f"""module renamed_values
implicit none
{constant_type},parameter::angle={initializer}
end module
""")
    helpers = tmp_path / "helpers.f90"
    helpers.write_text(f"""module renamed_helpers
use renamed_values,only:helper_angle=>angle
implicit none
contains
pure real({precision}) function weight(x)
real({precision}),intent(in)::x
weight=x*helper_angle
end function
end module
""")
    source = tmp_path / "caller.f90"
    source.write_text(f"""module renamed_owner
use renamed_helpers,only:evaluate=>weight
use renamed_values,only:original_value=>angle
implicit none
contains
subroutine advance(a,out,n)
real({precision}),intent(in)::a(-2:)
real({precision}),intent(inout)::out(-2:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=local_value(a(i))
enddo
contains
pure real({precision}) function local_value(x)
real({precision}),intent(in)::x
local_value=evaluate(x)
end function
end subroutine
end module
""")
    native = tmp_path / "native-owner.f90"
    native.write_text(source.read_text().replace("module renamed_owner", "module reference_owner"))
    paths = source, constants, helpers
    stable = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::a": stable, "argument::out": stable},
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}}
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts, indent=2) + "\n")
    output = tmp_path / "generated"
    build = tmp_path / "build"
    build.mkdir()
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    commands = []

    def run(arguments, *, trace=False):
        effective = {**environment, **({"FORT_RUNTIME_TRACE": "1"} if trace else {})}
        ordinal = len(commands)
        commands.append({"command": arguments, "cwd": str(build),
                         "environment_overrides": {key: effective[key] for key in
                             ("OMP_NUM_THREADS", "OMP_DYNAMIC", "OMP_SCHEDULE", "PYTHONPATH")},
                         "trace": trace})
        (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(arguments, cwd=build, env=effective, text=True, capture_output=True,
                                timeout=120, check=False)
        (tmp_path / f"command-{ordinal:02}.stdout").write_text(result.stdout)
        (tmp_path / f"command-{ordinal:02}.stderr").write_text(result.stderr)
        (tmp_path / f"command-{ordinal:02}.result.json").write_text(
            json.dumps({"exit_code": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    report = json.loads(run([sys.executable, "-m", "compiler", "--form-scopes", "--input", str(source),
                            "--source-file", str(constants), "--source-file", str(helpers),
                            "--kernel", "renamed_owner::advance", "--scope-facts", str(facts_path),
                            "--scope-execution", "reached", "--gpu-policy", "sections", "--memory-model", "scoped",
                            "--json", "--output-dir", str(output)]).stdout)
    assert report["supported"], report
    manifest = json.loads((output / "scope-manifest.json").read_text())
    assert manifest == report["scopes"]
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    region, = manifest["inline_numerical_regions"]["regions"]
    assert region["used"]
    assert region["immutable_scalar_captures"] == ["renamed_values::angle"]
    for path in paths:
        assert sha256(path.read_bytes()).hexdigest() == facts["sources"][str(path)]

    fflags = [fc, "-O3", "-fopenmp", "-ffp-contract=off", "-fcheck=all", "-J", str(build), "-I", str(build)]
    objects = []
    compiled_paths = set()

    def compile_descriptor(descriptor):
        path = output / descriptor["path"]
        assert path.is_file()
        assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][descriptor["path"]]
        target = build / f"artifact-{len(objects):02}.o"
        if descriptor["language"] == "cuda":
            contract = descriptor["numerical_contract"]
            require_numerical_build_contract(contract)
            required = [*contract["required_cuda_options"],
                        *("-Xcompiler=" + option for option in contract["required_host_options"])]
            flags = [nvcc, "-O3", "-std=c++17", "-ccbin", host, "-Xcompiler=-fopenmp",
                     "-arch=" + os.environ.get("FORT_TEST_CUDA_ARCH", "native"), *required]
        else:
            assert descriptor["language"] == "fortran"
            flags = fflags
        run([*flags, "-c", str(path), "-o", str(target)])
        objects.append(str(target))
        compiled_paths.add(descriptor["path"])

    # Public descriptors establish build order. Dependency source remains with
    # the independent consumer; no pipeline imports or generated-source parsing.
    for role in ("common_runtime", "shared_entry"):
        for descriptor in manifest["build_sources"]:
            if descriptor["role"] == role:
                compile_descriptor(descriptor)
    for path in (constants, helpers, native):
        target = build / f"dependency-{len(objects):02}.o"
        run([*fflags, "-c", str(path), "-o", str(target)])
        objects.append(str(target))
    for descriptor in manifest["build_sources"]:
        if descriptor["role"] == "original_source":
            compile_descriptor(descriptor)
    assert compiled_paths == {item["path"] for item in manifest["build_sources"]}

    bits = "int32" if precision == 4 else "int64"
    driver = tmp_path / "driver.f90"
    driver.write_text(f"""program verify
use iso_fortran_env,only:{bits},error_unit
use reference_owner,only:native_advance=>advance
use renamed_owner,only:generated_advance=>advance
implicit none
integer,parameter::sizes(3)=[17,0,3]
real({precision}),allocatable,target::a(:),out(:),expected(:),input_reference(:)
integer::n,i,step,repeat,ordinal
allocate(a(-2:19),out(-2:19),expected(-2:19),input_reference(-2:19))
ordinal=0
do repeat=1,2
do step=1,size(sizes)
n=sizes(step)
ordinal=ordinal+1
do i=lbound(a,1),ubound(a,1)
a(i)=real(3*i-7+repeat,{precision})/16._{precision}
enddo
a(-1)=-0._{precision}
a(4)=0._{precision}
input_reference=a
out=-117._{precision}
expected=out
call native_advance(a,expected,n)
write(error_unit,*) 'BEGIN_SCALAR_CALL',ordinal,n
call generated_advance(a,out,n)
write(error_unit,*) 'END_SCALAR_CALL',ordinal,n
if(any(transfer(out,[0_{bits}],size(out))/=transfer(expected,[0_{bits}],size(expected)))) &
 error stop 'native scalar value or complete output/halos differ'
if(any(transfer(a,[0_{bits}],size(a))/=transfer(input_reference,[0_{bits}],size(input_reference)))) &
 error stop 'read-only input or halos changed'
enddo
enddo
print *,'SIX_CALLS_FIELDS_BITWISE_OK'
end program
""")
    target = build / "driver.o"
    run([*fflags, "-c", str(driver), "-o", str(target)])
    executable = build / "verify"
    run([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", *objects, str(target), "-lgfortran", "-o", str(executable)])
    observed = run([str(executable)], trace=True)
    assert "SIX_CALLS_FIELDS_BITWISE_OK" in observed.stdout
    calls = observed.stderr.split("BEGIN_SCALAR_CALL")[1:]
    assert len(calls) == 6, observed.stderr
    for ordinal, call in enumerate(calls):
        trace = call.split("END_SCALAR_CALL", 1)[0]
        if ordinal % 3 == 1:
            assert "FORT_SCOPED launch " not in trace
            assert "FORT_SCOPED upload " not in trace
            assert "FORT_SCOPED download " not in trace
        else:
            assert "FORT_SCOPED launch " in trace, observed.stderr
