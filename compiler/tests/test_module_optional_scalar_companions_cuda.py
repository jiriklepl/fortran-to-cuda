"""An independent public consumer preserves optional presence through residency."""

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
@pytest.mark.parametrize("omitted_middle", [False, True])
def test_optional_companion_presence_and_native_escape_preserve_complete_fields(tmp_path, precision, omitted_middle):
    fc = shutil.which("gfortran-15") or shutil.which("gfortran")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    if not all((fc, nvcc, host)):
        pytest.skip("native Fortran and CUDA toolchain required")
    require_explicit_cuda_environment()
    child = tmp_path / "child.f90"
    child.write_text(f"""module renamed_child
implicit none
integer::child_visits=0
contains
subroutine adjust(x,bias,n,escape)
real({precision}),intent(inout)::x(-4:)
integer,optional,intent(in)::bias
integer,intent(in)::n
logical,intent(in)::escape
integer::j,factor
child_visits=child_visits+1
factor=3
if(present(bias)) factor=bias
do j=-4,n-5
x(j)=x(j)+real(factor+j,{precision})
enddo
if(escape) call opaque(x,n)
end subroutine
end module
module renamed_forward
use renamed_child,only:adjust
implicit none
integer::forward_visits=0
contains
subroutine forward(x,bias,n,escape)
real({precision}),intent(inout)::x(-4:)
integer,optional,intent(in)::bias
integer,intent(in)::n
logical,intent(in)::escape
forward_visits=forward_visits+1
call adjust(escape=escape,x=x,n=n,bias=bias)
end subroutine
end module
""")
    source = tmp_path / "caller.f90"
    source.write_text(f"""module renamed_owner
use renamed_forward,only:forward
implicit none
integer::visits=0
contains
subroutine step(a,b,out,n,escape,bias)
real({precision}),intent(in)::a(-2:)
real({precision}),intent(inout)::b(-2:),out(-2:)
integer,intent(in)::n
logical,intent(in)::escape
integer,optional,intent(in)::bias
integer::i
visits=visits+1
do i=-2,n-3
b(i)=2*a(i)+real(i,{precision})
enddo
i=0
call forward(escape=escape,x=b,n=n,bias=bias)
call forward(b,n=n,escape=.false.{'' if omitted_middle else ',bias=bias'})
do i=-2,n-3
out(i)=a(i)+b(i)
enddo
end subroutine
end module
""")
    native_child, native_owner = tmp_path / "native-child.f90", tmp_path / "native-owner.f90"
    native_child.write_text(child.read_text().replace("renamed_child", "reference_child")
                            .replace("renamed_forward", "reference_forward"))
    native_owner.write_text(source.read_text().replace("renamed_forward", "reference_forward")
                            .replace("renamed_owner", "reference_owner"))
    callback = tmp_path / "callback.f90"
    callback.write_text(f"""module callback_audit
integer::opaque_visits=0
end module
subroutine opaque(x,n)
use callback_audit,only:opaque_visits
real({precision})::x(*)
integer::n
opaque_visits=opaque_visits+1
if(n>0) x(1)=x(1)+41
end subroutine
""")
    stable = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "captures": {"argument::" + name: stable for name in ("a", "b", "out")},
             "sources": {str(path): sha256(path.read_bytes()).hexdigest() for path in (source, child)}}
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps(facts, indent=2) + "\n")
    output, build = tmp_path / "generated", tmp_path / "build"
    build.mkdir()
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    commands = []

    def run(arguments, *, trace=False, native_fallback=False):
        ordinal = len(commands)
        commands.append({"argv": arguments, "cwd": str(build), "trace": trace, "native_fallback": native_fallback})
        (tmp_path / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        result = subprocess.run(arguments, cwd=build,
            env={**environment, **({"FORT_RUNTIME_TRACE": "1"} if trace else {}),
                 **({"CUDA_VISIBLE_DEVICES": ""} if native_fallback else {})},
            capture_output=True, text=True, timeout=120, check=False)
        (tmp_path / f"command-{ordinal:02}.stdout").write_text(result.stdout)
        (tmp_path / f"command-{ordinal:02}.stderr").write_text(result.stderr)
        (tmp_path / f"command-{ordinal:02}.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    report = json.loads(run([sys.executable, "-m", "compiler", "--form-scopes", "--input", str(source),
        "--source-file", str(child), "--kernel", "renamed_owner::step", "--scope-facts", str(facts_path),
        "--scope-execution", "reached", "--gpu-policy", "sections", "--memory-model", "scoped",
        "--json", "--output-dir", str(output)]).stdout)
    assert report["supported"], report
    manifest = json.loads((output / "scope-manifest.json").read_text())
    assert manifest == report["scopes"]
    owner, = manifest["scopes"]
    companions = {item["procedure"]: item for item in owner["module_coordinators"]}
    assert set(companions) == {"renamed_child::adjust", "renamed_forward::forward"}
    assert len(companions["renamed_forward::forward"]["calls"]) == 2
    assert companions["renamed_child::adjust"]["boundaries"], "opaque work must end ownership on its original guard"
    # A statically omitted helper actual retains the existing conservative
    # producer liveness boundary. The child and downstream consumer still run.
    assert len(owner["gpu_leaves"]) == (2 if omitted_middle else 3)
    fflags = [fc, "-O3", "-fopenmp", "-ffp-contract=off", "-fcheck=all", "-J", str(build), "-I", str(build)]
    objects, compiled = [], set()

    def compile_descriptor(descriptor):
        path = output / descriptor["path"]
        assert sha256(path.read_bytes()).hexdigest() == manifest["artifacts_sha256"][descriptor["path"]]
        target = build / f"artifact-{len(objects):02}.o"
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

    for role in ("common_runtime", "shared_entry"):
        for descriptor in manifest["build_sources"]:
            if descriptor["role"] == role:
                compile_descriptor(descriptor)
    for path in (callback, native_child, native_owner):
        target = build / f"dependency-{len(objects):02}.o"
        run([*fflags, "-c", str(path), "-o", str(target)])
        objects.append(str(target))
    for original in (child, source):
        replacement = manifest["sources"][str(original)]["replacement"]
        descriptor, = [item for item in manifest["build_sources"] if item["path"] == replacement]
        assert descriptor["role"] == "original_source"
        compile_descriptor(descriptor)
    assert compiled == {item["path"] for item in manifest["build_sources"]}
    bits = "int32" if precision == 4 else "int64"
    driver = tmp_path / "driver.f90"
    driver.write_text(f"""program verify
use iso_fortran_env,only:{bits},error_unit
use callback_audit,only:opaque_visits
use reference_owner,only:native_step=>step,native_visits=>visits
use reference_child,only:native_child_visits=>child_visits
use reference_forward,only:native_forward_visits=>forward_visits
use renamed_owner,only:generated_step=>step,generated_visits=>visits
use renamed_child,only:generated_child_visits=>child_visits,generated_adjust=>adjust
use reference_child,only:native_adjust=>adjust
use renamed_forward,only:generated_forward_visits=>forward_visits
implicit none
integer,parameter::sizes(4)=[17,0,3,17]
real({precision})::a(-2:19),b(-2:19),out(-2:19),expected_b(-2:19),expected_out(-2:19),input_reference(-2:19)
integer::step,i,n,expected_opaque
logical::escape
do step=1,4
n=sizes(step)
escape=step==3
do i=-2,19
a(i)=real(3*i-7+step,{precision})/16._{precision}
enddo
input_reference=a
b=-113
out=-117
expected_b=b
expected_out=out
native_visits=0
native_child_visits=0
native_forward_visits=0
opaque_visits=0
if(mod(step,2)==1) then
call native_step(a,expected_b,expected_out,n,escape,7)
else
call native_step(a,expected_b,expected_out,n,escape)
endif
expected_opaque=opaque_visits
generated_visits=0
generated_child_visits=0
generated_forward_visits=0
opaque_visits=0
write(error_unit,*) 'BEGIN_OPTIONAL_CALL',step,n,escape
if(mod(step,2)==1) then
call generated_step(a,b,out,n,escape,7)
else
call generated_step(a,b,out,n,escape)
endif
write(error_unit,*) 'END_OPTIONAL_CALL',step,n,escape
if(generated_visits/=1.or.generated_child_visits/=2.or.native_visits/=1.or.native_child_visits/=2) &
 error stop 'original source prefix or child was replayed'
if(generated_forward_visits/=2.or.native_forward_visits/=2) error stop 'nested forward was replayed'
if(opaque_visits/=expected_opaque.or.opaque_visits/=merge(1,0,escape)) error stop 'native escape count differs'
if(any(transfer(b,[0_{bits}],size(b))/=transfer(expected_b,[0_{bits}],size(expected_b)))) &
 error stop 'complete intermediate or halo differs'
if(any(transfer(out,[0_{bits}],size(out))/=transfer(expected_out,[0_{bits}],size(expected_out)))) &
 error stop 'complete output or halo differs'
if(any(transfer(a,[0_{bits}],size(a))/=transfer(input_reference,[0_{bits}],size(input_reference)))) &
 error stop 'read-only input changed'
enddo
b=-113
expected_b=b
call native_adjust(expected_b,n=3,escape=.false.)
write(error_unit,*) 'BEGIN_NATIVE_ENTRY'
call generated_adjust(b,n=3,escape=.false.)
write(error_unit,*) 'END_NATIVE_ENTRY'
if(any(transfer(b,[0_{bits}],size(b))/=transfer(expected_b,[0_{bits}],size(expected_b)))) &
 error stop 'original native entry or omitted compiler controls differ'
print *,'FOUR_OPTIONAL_CALLS_FIELDS_BITWISE_OK'
end program
""")
    target = build / "driver.o"
    run([*fflags, "-c", str(driver), "-o", str(target)])
    executable = build / "verify"
    run([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", *objects, str(target), "-lgfortran", "-o", str(executable)])
    observed = run([str(executable)], trace=True)
    assert "FOUR_OPTIONAL_CALLS_FIELDS_BITWISE_OK" in observed.stdout
    calls = observed.stderr.split("BEGIN_OPTIONAL_CALL")[1:]
    assert len(calls) == 4
    for ordinal, call in enumerate(calls):
        trace = call.split("END_OPTIONAL_CALL", 1)[0]
        if ordinal == 1:
            assert "FORT_SCOPED launch " not in trace
            assert "FORT_SCOPED upload " not in trace
            assert "FORT_SCOPED download " not in trace
        else:
            expected = (2 if ordinal == 2 else 4) - int(omitted_middle)
            assert trace.count("FORT_SCOPED launch ") == expected, observed.stderr
    direct = observed.stderr.split("BEGIN_NATIVE_ENTRY", 1)[1].split("END_NATIVE_ENTRY", 1)[0]
    assert "FORT_SCOPED launch " not in direct
    fallback = run([str(executable)], trace=True, native_fallback=True)
    assert "FOUR_OPTIONAL_CALLS_FIELDS_BITWISE_OK" in fallback.stdout
    assert "FORT_SCOPED launch " not in fallback.stderr
