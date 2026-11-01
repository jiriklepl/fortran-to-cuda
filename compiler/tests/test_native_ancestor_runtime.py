"""Checked returned-domain controls retain mirrors or disable an unsafe suffix."""
import json
import os
import re
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.tests.test_native_ancestor import SOURCE, fixture

DRIVER = """module opaque_state
integer::visits=0
end module
program exercise
use ancestor_fixture
use opaque_state
implicit none
integer,parameter::n=19,extent=n+8
real(8)::a(-3:n+4),b(7:n+14),ea(-3:n+4),eb(7:n+14),expected_probe
integer::i,flag_integer
logical::flag
character(1)::argument
call get_command_argument(1,argument)
read(argument,*)flag_integer
flag=flag_integer/=0
do i=1,extent
 a(i-4)=real(17*i-3,8)/32.0_8
 b(i+6)=real(500-i,8)/16.0_8
enddo
ea=a
eb=b
expected_probe=0
do i=1,n
 eb(i+6)=2.0_8*a(i-4)
 expected_probe=expected_probe+eb(i+6)
 if(flag) eb(i+6)=eb(i+6)+3.0_8
 ea(i-4)=eb(i+6)+1.0_8
enddo
call advance(a,b,n,flag)
if(any(a/=ea).or.any(b/=eb))error stop 'ancestor complete original fields/halos'
if(observed/=expected_probe)error stop 'original native readonly probe'
if(opaque_visits/=flag_integer)error stop 'reached original call did not execute exactly once'
print *, 'NATIVE_ANCESTOR_FULL_FIELDS_OK',flag_integer
end program
subroutine opaque(x,n)
use opaque_state
implicit none
integer,intent(in)::n
real(8),intent(inout)::x(*)
integer::i
visits=visits+1
do i=1,n
 x(i)=x(i)+3.0_8
enddo
end subroutine
"""


def build_native_ancestor(directory, *, source, driver_source, selected_window=False):
    """Build only published artifacts and an independent original control."""
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/GNU toolchains required")
    original, builder = fixture(directory, source=source)
    if selected_window:
        for fact in builder.facts["captures"].values():
            fact.update(lifetime="selected_window", initialized="none")
    outputs, manifest = builder.run()
    assert manifest["scope_count"] == 1
    build = directory / "build"
    build.mkdir()
    for name, text in outputs.items():
        path = build / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (build / "scope-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    commands = []

    def command(argv):
        index = len(commands)
        commands.append(argv)
        (build / f"command-{index}.json").write_text(json.dumps({"argv": argv, "cwd": str(build)}) + "\n")
        result = subprocess.run(argv, cwd=build, capture_output=True, text=True, timeout=120)
        (build / f"command-{index}.stdout").write_text(result.stdout)
        (build / f"command-{index}.stderr").write_text(result.stderr)
        (build / f"command-{index}.completion.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr

    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for index, artifact in enumerate(manifest["build_sources"]):
            if artifact["role"] != role:
                continue
            source, target = build / artifact["path"], build / f"object-{index}.o"
            assert sha256(source.read_bytes()).hexdigest() == manifest["artifacts_sha256"][artifact["path"]]
            if artifact["language"] == "cuda":
                options = [nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp",
                           *artifact["numerical_contract"]["required_cuda_options"], "-I", str(build)]
            else:
                options = [fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build)]
            command([*options, "-c", str(source), "-o", str(target)])
            objects.append(str(target))
    driver = build / "driver.f90"
    driver.write_text(driver_source)
    candidate, native = build / "candidate", build / "native"
    command([fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build), "-I", str(build),
             str(driver), *objects, "-L/usr/local/cuda/lib64", "-Wl,-rpath,/usr/local/cuda/lib64",
             "-lcudart", "-lstdc++", "-o", str(candidate)])
    command([fortran, "-fopenmp", "-fcheck=all", "-ffp-contract=off", "-J", str(build),
             str(original), str(driver), "-o", str(native)])
    return build, candidate, native


@pytest.fixture(scope="module")
def native_ancestor_binaries(tmp_path_factory):
    # A later caller task verifies that the child's retired control propagates
    # out of the child, rather than merely disabling its local continuation.
    source = SOURCE.replace("call middle(a,b,n,flag)", "call middle(a,b,n,flag)\ncall consumer(a,b,n)", 1)
    return build_native_ancestor(tmp_path_factory.mktemp("native_ancestor"), source=source, driver_source=DRIVER)


@pytest.mark.cuda
@pytest.mark.parametrize("hidden", [False, True], ids=["enabled", "hidden"])
@pytest.mark.parametrize("cut", [False, True], ids=["readonly-live", "reached-retirement"])
def test_native_ancestor_keeps_checked_child_domain_or_retires_before_unknown_writer(native_ancestor_binaries, hidden, cut):
    build, candidate, native = native_ancestor_binaries
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "OMP_SCHEDULE": "static",
                   "FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    if hidden:
        environment["CUDA_VISIBLE_DEVICES"] = ""
    mode = ("hidden" if hidden else "enabled") + ("-cut" if cut else "-live")
    results = []
    for name, binary in (("native", native), ("candidate", candidate)):
        argv = [str(binary), str(int(cut))]
        (build / f"{mode}-{name}.command.json").write_text(json.dumps({"argv": argv,
            "OMP_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": environment.get("CUDA_VISIBLE_DEVICES")}) + "\n")
        result = subprocess.run(argv, cwd=build, env=environment, capture_output=True, text=True, timeout=30)
        (build / f"{mode}-{name}.stdout").write_text(result.stdout)
        (build / f"{mode}-{name}.stderr").write_text(result.stderr)
        (build / f"{mode}-{name}.completion.json").write_text(json.dumps({"returncode": result.returncode}) + "\n")
        assert result.returncode == 0, result.stdout + result.stderr
        results.append(result)
    reference, result = results
    assert reference.stdout == result.stdout
    assert "NATIVE_ANCESTOR_FULL_FIELDS_OK" in result.stdout
    lines = result.stderr.splitlines()
    launches = [index for index, line in enumerate(lines) if line.startswith("FORT_SCOPED launch ")]
    if hidden:
        assert not launches, result.stderr
        assert not any(line.startswith(("FORT_SCOPED upload ", "FORT_SCOPED download ")) for line in lines)
    else:
        assert len(launches) == 3, result.stderr
        between = lines[launches[0] + 1:launches[1]]
        downloads = [line for line in between if line.startswith("FORT_SCOPED download ")]
        assert downloads
        assert sum(int(re.search(r"\bbytes=(\d+)", line)[1]) for line in downloads) >= 19*8
        if cut:
            assert any(line.startswith("FORT_SCOPED upload ") for line in between), result.stderr
            assert any('"event":"domain_retire"' in line for line in between), result.stderr
            assert sum('"event":"domain_create"' in line for line in lines) == 2, result.stderr
        else:
            assert not any(line.startswith("FORT_SCOPED upload ") for line in between), result.stderr
            assert not any('"event":"domain_retire"' in line for line in between), result.stderr
            assert sum('"event":"domain_create"' in line for line in lines) == 1, result.stderr
