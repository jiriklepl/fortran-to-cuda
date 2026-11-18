"""An automatically placed reached child builds independently and preserves native fields."""

import json
import os
import shutil
from hashlib import sha256

import pytest

from compiler.numerical_contract import require_numerical_build_contract
from compiler.tests.borrowed_source_coordinators_support import source
from compiler.tests.scoped_batch_sources_cuda_support import actual_profile
from compiler.tests.test_source_scopes import FACT, ROOT, generate, run

DRIVER = """program verify
use original,only:step
implicit none
real(8),allocatable::a(:),b(:),out(:)
integer,parameter::sizes(3)=[0,8,262144]
integer::shape,repeat,sign,n,i,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
 allocate(a(-7:sizes(shape)-8),b(-7:sizes(shape)-8),out(-7:sizes(shape)-8))
 do repeat=1,2
  do sign=-1,1,2
   n=size(a)
   do i=lbound(a,1),ubound(a,1)
    a(i)=real(sign*(i+8),8)*0.25d0+repeat
   enddo
   b=-99.d0
   out=-101.d0
   call step(a,b,out,n)
   write(unit)n,a,b,out
  enddo
 enddo
 deallocate(a,b,out)
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=("graph", "graph_rollback"))
def auto_child_binary(tmp_path_factory, request):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA/C++/Fortran toolchain unavailable")
    directory = tmp_path_factory.mktemp("auto_borrowed_child")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns(
        "__pycache__", ".*cache", "CODE_MAP.md", "MEMORY_MODEL_PLAN.md"))
    profile = directory / "profile.json"
    flags = [fortran, "-O3", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    # The profile is bound to the actual hardware and tool identities so that
    # AUTO ranks the child; its rates are not performance evidence.
    calibration = actual_profile(directory, nvcc, host)
    if len(os.sched_getaffinity(0)) != calibration["cpu_threads"]:
        pytest.skip("AUTO CUDA correctness requires the profiled four-CPU placement")
    profile.write_text(json.dumps(calibration))
    text = source()
    if request.param == 'graph_rollback':
        text = text.replace('call child(a,b,out,n)', 'if(n<0) then\ncall child(a,b,b,n)\nendif\ncall child(a,b,out,n)')
    original, output, manifest = generate(directory / "case", text, mode="auto", profile=profile,
        checkout=checkout, facts={"schema_version": 1, "participation": "serial",
                                 "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}})
    assert manifest["scopes"][0]["estimate_available"]
    objects, commands = [], []
    headers = tuple((name, digest) for name, digest in sorted(manifest["artifacts_sha256"].items())
                    if name.endswith((".h", ".hpp", ".cuh")))
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            artifact = output / item["path"]
            assert sha256(artifact.read_bytes()).hexdigest() == manifest["artifacts_sha256"][item["path"]]
            target = artifact.with_suffix(".o")
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host,
                            "-arch=sm_" + calibration['hardware']['compute_capability'].replace('.', ''),
                            "-Xcompiler=-fopenmp", *contract["required_cuda_options"],
                            *("-Xcompiler=" + option for option in contract["required_host_options"])]
            else:
                compiler = flags
            command = [*compiler, "-I", str(output), "-c", str(artifact), "-o", str(target)]
            run(command, cwd=output)
            commands.append({"path": item["path"], "role": role, "argv": command,
                             "numerical_contract": item.get("numerical_contract")})
            objects.append(str(target))
    driver = output / "driver.f90"
    driver.write_text(DRIVER)
    target, native = output / "candidate", output / "native"
    cuda_lib = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(nvcc))), "lib64")
    command = [*flags, str(driver), *objects, "-L" + cuda_lib, "-Wl,-rpath," + cuda_lib,
               "-lcudart", "-lstdc++", "-o", str(target)]
    run(command, cwd=output)
    commands.append({"role": "candidate-link", "argv": command})
    command = [*flags, str(original), str(driver), "-o", str(native)]
    run(command, cwd=output)
    commands.append({"role": "native-compile-link", "argv": command})
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"}
    run([str(native)], cwd=output, env=environment)
    expected = (output / "fields.bin").read_bytes()
    (output / "native-fields.bin").write_bytes(expected)
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "artifact-evidence.json").write_text(json.dumps({"headers": headers,
        "runtime_id": manifest["runtime"]["runtime_id"], "profile": str(profile)}, indent=2))
    return target, output, expected


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_auto_child_preserves_reached_native_fallback(auto_child_binary, disabled):
    target, output, expected = auto_child_binary
    environment = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        environment["CUDA_VISIBLE_DEVICES"] = "-1"
    result = run([str(target)], cwd=output, env=environment)
    actual = (output / "fields.bin").read_bytes()
    assert actual == expected
    assert "array temporary" not in result.stderr.lower()
    if disabled:
        assert "FORT_SCOPED launch " not in result.stderr
    else:
        assert "FORT_SCOPED launch " in result.stderr
    label = "disabled" if disabled else "enabled"
    (output / (label + ".stderr")).write_text(result.stderr)
    (output / (label + "-receipt.json")).write_text(json.dumps({
        "schema_version": 1, "disabled": disabled,
        "native_fields_sha256": sha256(expected).hexdigest(),
        "candidate_fields_sha256": sha256(actual).hexdigest(),
        "runtime_trace_sha256": sha256(result.stderr.encode()).hexdigest(),
        "launches": result.stderr.count("FORT_SCOPED launch "),
        "complete_arrays_and_halos_equal": True,
        "independent_compiler_checkout": True,
        "performance_sample": False,
    }, indent=2, sort_keys=True) + "\n")
