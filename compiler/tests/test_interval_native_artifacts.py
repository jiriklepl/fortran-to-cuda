"""Native interval captures must build without speculative GPU companions."""

import json
import os
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_interval_draft import SOURCE
from compiler.tests.test_selection_cost import build_validation, process_fixture  # noqa: F401
from compiler.tests.test_source_scopes import FACT


def cross_module_builder(tmp_path, config):
    first = SOURCE.index("subroutine producer(")
    last = SOURCE.index("subroutine momentum(")
    helper = tmp_path / "helpers.f90"
    helper.write_text("module numerical_helpers\ncontains\n" + SOURCE[first:last] + "end module\n")
    path = tmp_path / "interval.f90"
    path.write_text((SOURCE[:first] + SOURCE[last:]).replace(
        "module interval_fields\n", "module interval_fields\nuse numerical_helpers, only: producer\n"))
    sources = [helper, path]
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(source): sha256(source.read_bytes()).hexdigest() for source in sources},
             "captures": {"interval_fields::" + name: dict(FACT)
                          for name in ("old", "viscosity", "star", "out")}}
    selected = ScopeBuilder(sources, "interval_fields::advance", facts=facts,
                            options=CompilerOptions(gpu_policy=config.policy), config=config)
    return sources, selected


@pytest.mark.parametrize("capture", [None, "native"])
def test_public_native_interval_builds_and_executes_cross_module_calls(tmp_path, build_validation, capture):  # noqa: F811
    fortran = shutil.which("gfortran-14") or shutil.which("gfortran")
    cxx = shutil.which("g++-14") or shutil.which("g++")
    if not fortran or not cxx:
        pytest.skip("Fortran and C++ compilers required")
    config = OffloadConfig(policy="auto", scope_execution="reached", exact_window={
        "coverage": "checked_interval", **({"calibration": capture} if capture else {}),
        "workloads": process_fixture()[0]["workloads"], "build_contract": {"checked_fixture": True},
        "validation_policy": {"checked_policy_fixture": True}})
    sources, selected = cross_module_builder(tmp_path, config)
    outputs, report = selected.run()
    scope, = report["scopes"]
    assert scope["checked_execution_plan"]["numerical_invocations"] == 3
    assert scope["owner_variant"] is None
    generated = tmp_path / "generated"
    for name, contents in outputs.items():
        target = generated / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    driver = tmp_path / "driver.f90"
    driver.write_text("""program check
use interval_fields
implicit none
old=2.d0
viscosity=-17.d0
star=-19.d0
out=1.d0
call advance(16,.true.)
if(visits/=1) error stop 'native prefix replayed'
if(out(0)/=9.d0.or.any(out(1:15)/=7.d0)) error stop 'arithmetic changed'
if(out(-1)/=1.d0.or.out(16)/=1.d0) error stop 'halos changed'
print *, 'native interval passed'
end program
""")

    def run(argv):
        result = subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, check=False, timeout=30,
                                env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_SCHEDULE": "static"})
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    objects = []
    for index, source in enumerate(sources):
        replacement = report["sources"].get(str(source), {}).get("replacement")
        source = generated / replacement if replacement else source
        obj = tmp_path / f"source-{index}.o"
        run([fortran, "-fopenmp", "-fcheck=all", "-c", str(source), "-o", str(obj)])
        objects.append(str(obj))
    for index, name in enumerate(outputs):
        if name.startswith("exact/") and name.endswith(".cu"):
            obj = tmp_path / f"observer-{index}.o"
            run([cxx, "-std=c++17", "-fopenmp", "-x", "c++", "-c", str(generated / name), "-o", str(obj)])
            objects.append(str(obj))
    if capture:
        # Native observation links the runtime's CPU reference implementation;
        # this test executes no GPU work and supplies no calibration evidence.
        obj = tmp_path / "runtime.o"
        run([cxx, "-std=c++17", "-fopenmp", "-DFORT_SCOPE_CPU_TEST", "-x", "c++", "-c",
             str(generated / "scoped_runtime.cu"), "-o", str(obj)])
        objects.append(str(obj))
    binary = tmp_path / "check"
    run([fortran, "-fopenmp", "-fcheck=all", str(driver), *objects, "-lstdc++", "-o", str(binary)])
    result = run([str(binary)])
    assert "native interval passed" in result.stdout
    if capture:
        rows = [json.loads(line.removeprefix("FORT_SELECTION_OBSERVATION "))
                for line in result.stderr.splitlines() if line.startswith("FORT_SELECTION_OBSERVATION ")]
        observation, = rows
        assert observation["complete"] is True
        assert observation["launches"] == 0
