"""A one-shot window publishes before allocation movement and loop backedges."""

from __future__ import annotations

import json
import os
import shutil
from hashlib import sha256

import pytest

from compiler.numerical_contract import require_numerical_build_contract
from compiler.tests.test_module_reached_windows import MODULE_BOUNDARY_SOURCE, module_fixture
from compiler.tests.test_reached_window_discovery import discovery_fixture
from compiler.tests.test_reached_window_owner import SOURCE, emit_window
from compiler.tests.test_scoped_batch_sources_cuda import actual_profile
from compiler.tests.test_source_scopes import run

DRIVER = """program verify
use renamed_stage,only:step,prefix_visits,tail_visits,prefix_total
implicit none
real(8),allocatable::a(:),b(:)
integer,parameter::sizes(3)=[0,8,4096],counts(3)=[0,1,3]
integer::shape,stages,iteration,n,i,unit
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
 n=sizes(shape)
 do stages=1,size(counts)
  allocate(a(-7:n+2),b(-7:n+2))
  do iteration=1,2
   do i=lbound(a,1),ubound(a,1)
    a(i)=real(i+16,8)*0.125_8+real(iteration,8)
   enddo
   b=-77.0_8
   call step(a,b,n,counts(stages))
   write(unit)prefix_visits,tail_visits,prefix_total,lbound(a),ubound(a),a,b
  enddo
  deallocate(a,b)
 enddo
enddo
close(unit)
end program
"""


@pytest.fixture(scope="module", params=(("explicit", False), ("explicit", True),
                                      ("discovered", False), ("discovered", True),
                                      ("module_discovered", False), ("module_discovered", True),
                                      ("module_native_boundary", False),
                                      ("explicit_live_iterator", False)))
def window_binaries(tmp_path_factory, request):
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not all((nvcc, host, fortran)):
        pytest.skip("CUDA, C++ and Fortran toolchain required")
    directory = tmp_path_factory.mktemp("one_shot_window")
    architecture = actual_profile(directory, nvcc, host)["hardware"]["compute_capability"].replace(".", "")
    emission, resource_fallback = request.param
    if emission in {"module_discovered", "module_native_boundary"}:
        options = {"source": MODULE_BOUNDARY_SOURCE} if emission == "module_native_boundary" else {}
        original, builder = module_fixture(directory, device_budget=0 if resource_fallback else None, **options)
        outputs, report = builder.run()
        assert report["reached_window_discovery"]["selected"]
        assert builder.analysis.stable_module_allocatables == frozenset()
    elif emission == "discovered":
        original, builder = discovery_fixture(directory, device_budget=0 if resource_fallback else None)
        outputs, report = builder.run()
        assert report["reached_window_discovery"]["selected"]
    elif emission == "explicit_live_iterator":
        source = SOURCE.replace("tail_visits=tail_visits+1", "tail_visits=tail_visits+i")
        original, outputs, report, _owner = emit_window(directory, source)
        assert len(report["scopes"][0]["gpu_leaves"]) == 1
    else:
        original, outputs, report, _owner = emit_window(directory, device_budget=0 if resource_fallback else None)
    for name, text in outputs.items():
        artifact = directory / name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(text)
    objects, compile_commands = [], []
    flags = [fortran, "-O3", "-cpp", "-std=f2018", "-fopenmp", "-fcheck=all,array-temps"]
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in report["build_sources"]:
            if item["role"] != role:
                continue
            artifact = directory / item["path"]
            target = artifact.with_suffix(".o")
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_" + architecture,
                            "-Xcompiler=-fopenmp", *contract["required_cuda_options"],
                            *("-Xcompiler=" + option for option in contract["required_host_options"])]
            else:
                compiler = flags
            command = [*compiler, "-I", str(directory), "-c", str(artifact), "-o", str(target)]
            run(command, cwd=directory)
            compile_commands.append({"path": item["path"], "language": item["language"], "argv": command,
                                     "numerical_contract": item.get("numerical_contract")})
            objects.append(str(target))
    driver = directory / "driver.f90"
    driver_text = DRIVER
    if emission.startswith("module_"):
        driver_text = driver_text.replace("tail_visits,prefix_total", "tail_visits,prefix_total,a,b").replace(
            "real(8),allocatable::a(:),b(:)\n", "").replace("call step(a,b,n,counts(stages))",
                                                           "call step(n,counts(stages))")
    driver.write_text(driver_text)
    candidate, native = directory / "candidate", directory / "native"
    cuda_root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    cuda_lib = os.path.join(cuda_root, "lib64")
    run([*flags, str(driver), *objects, "-L" + cuda_lib, "-Wl,-rpath," + cuda_lib,
         "-lcudart", "-lstdc++", "-o", str(candidate)], cwd=directory)
    run([*flags, str(original), str(driver), "-o", str(native)], cwd=directory)
    run([str(native)], cwd=directory)
    expected = (directory / "fields.bin").read_bytes()
    (directory / "compile-commands.json").write_text(json.dumps(compile_commands, indent=2) + "\n")
    return directory, candidate, expected, report, resource_fallback, emission


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True])
def test_one_shot_window_matches_complete_native_arrays_and_all_original_visits(window_binaries, disabled):
    directory, candidate, expected, report, resource_fallback, emission = window_binaries
    env = {**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "FORT_RUNTIME_TRACE": "1"}
    if disabled:
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    result = run([str(candidate)], cwd=directory, env=env)
    actual = (directory / "fields.bin").read_bytes()
    assert actual == expected
    assert "array temporary" not in result.stderr.lower()
    # Two kernels run only on the first traversal of each nonempty window.
    # Extra original stages still execute once, after mandatory publication and
    # MOVE_ALLOC. Zero-trip enclosing loops never activate the window at all.
    launches = result.stderr.count("FORT_SCOPED launch ")
    numerical_launches = 8 if emission == "explicit_live_iterator" else 16
    assert launches == (0 if disabled or resource_fallback else numerical_launches)
    contexts = result.stderr.count("FORT_SCOPED create ")
    # At most one context may be opened at each first reached traversal. A
    # failed pre-execution resource guard cannot retry on later backedges.
    assert contexts <= 12
    receipt = {"schema_version": 1, "disabled": disabled, "fields_bytes": len(actual),
               "native_fields_sha256": sha256(expected).hexdigest(),
               "candidate_fields_sha256": sha256(actual).hexdigest(),
               "launches": launches, "contexts_created": contexts, "resource_fallback": resource_fallback,
               "complete_arrays_and_halos_equal": True,
               "original_prefix_and_tail_counters_equal": True,
               "reached_window": report["scopes"][0]["reached_window"],
               "compiler_owned_discovery": report["scopes"][0]["reached_window"]["automatic_discovery"],
               "emission": emission,
               "performance_sample": False}
    (directory / ("window-disabled.json" if disabled else "window-enabled.json")).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n")
