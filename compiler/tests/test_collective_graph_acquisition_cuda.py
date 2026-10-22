"""A public collective graph drains GPU dependencies before preparing any host buffer."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from compiler.numerical_contract import require_numerical_build_contract
from compiler.tests.test_collective_source_runtime import DRIVER, NATIVE_COUNTERS, ROOT, SOURCE, _facts, _run

SOURCE_WITH_SHARED_READ = SOURCE.replace(
    "subroutine native_permute(b,permutation,n)\nreal(8),intent(inout)::b(:)",
    "subroutine native_permute(a,b,permutation,n)\nreal(8),intent(in)::a(:)\nreal(8),intent(inout)::b(:)",
).replace("b(permutation(i))=b(permutation(i))+1", "b(permutation(i))=b(permutation(i))+a(i)").replace(
    "call native_permute(b,permutation,n)", "call native_permute(a,b,permutation,n)")
CHECKED_DRIVER = DRIVER.replace("wanted_b=wanted_b+1", "wanted_b=wanted_b+real(3*(n-i+1)+repetition,8)")


@pytest.fixture(scope="module")
def collective_graph_binary(tmp_path_factory):
    nvcc, host, fortran = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++"), shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA, C++ and Fortran toolchains required")
    directory = tmp_path_factory.mktemp("collective_graph_acquisition")
    original, driver = directory / "original.f90", directory / "driver.f90"
    original.write_text(SOURCE_WITH_SHARED_READ)
    driver.write_text(CHECKED_DRIVER)
    facts = directory / "facts.json"
    facts.write_text(json.dumps(_facts(original, native=True)))
    output = directory / "generated"
    response = _run([sys.executable, "-m", "compiler", "--form-scopes", "--gpu-collective",
                     "--input", str(original), "--kernel", "operators::step_native", "--scope-facts", str(facts),
                     "--scope-execution", "graph", "--json", "--output-dir", str(output)], ROOT)
    manifest = json.loads(response.stdout)["scopes"]
    assert manifest["scope_execution"] == "graph"
    assert manifest["scope_count"] == 1, manifest["boundaries"]
    owner, = manifest["scopes"]
    assert owner["gpu_leaves"] == ["operators::consume", "operators::produce"]
    replacement = output / manifest["sources"][str(original)]["replacement"]
    emitted = replacement.read_text()
    assert "fort_scope_host_begin(" not in emitted
    assert emitted.count("fort_scope_access_begin_v1(") == 1
    # Native A is already host-current, but native B conflicts with the queued
    # producer that also reads A. Preparing A first used to reject that drain.
    assert "%access = fort_access" in emitted
    build, native_build = directory / "build", directory / "native"
    build.mkdir()
    native_build.mkdir()
    flags = ["-O2", "-std=f2018", "-fopenmp", "-ffp-contract=off", "-fcheck=all,array-temps"]
    counters = directory / "counters.cpp"
    counters.write_text(NATIVE_COUNTERS)
    counter_object = directory / "counters.o"
    _run([host, "-c", str(counters), "-o", str(counter_object)], directory)
    native = native_build / "verify"
    _run([fortran, *flags, "-J", str(native_build), "-I", str(native_build), str(original), str(driver),
          str(counter_object), "-lstdc++", "-o", str(native)], native_build)
    _run([str(native), "native"], native_build, env={**os.environ, "OMP_DYNAMIC": "FALSE", "OMP_NUM_THREADS": "4"})
    objects = []
    for role in ("common_runtime", "shared_entry", "original_source"):
        for item in manifest["build_sources"]:
            if item["role"] != role:
                continue
            source = output / item["path"]
            target = build / f"object-{len(objects)}.o"
            if item["language"] == "cuda":
                contract = item["numerical_contract"]
                require_numerical_build_contract(contract)
                compiler = [nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fopenmp",
                            *contract["required_cuda_options"],
                            *("-Xcompiler=" + option for option in contract["required_host_options"]), "-I", str(output)]
            else:
                compiler = [fortran, *flags, "-J", str(build), "-I", str(build)]
            _run([*compiler, "-c", str(source), "-o", str(target)], build, timeout=120)
            objects.append(str(target))
    binary = build / "verify"
    library = str(Path(nvcc).resolve().parents[1] / "lib64")
    _run([fortran, *flags, "-J", str(build), "-I", str(build), str(driver), *objects, str(counter_object),
          "-L" + library, "-Wl,-rpath," + library, "-lcudart", "-lstdc++", "-o", str(binary)], build)
    return directory, binary


@pytest.mark.cuda
@pytest.mark.parametrize("disabled", [False, True], ids=["cuda", "hidden-device"])
def test_collective_graph_acquires_native_effects_together(collective_graph_binary, disabled):
    directory, binary = collective_graph_binary
    result = _run([str(binary), "native"], directory,
                  env={**os.environ, "OMP_DYNAMIC": "FALSE", "OMP_NUM_THREADS": "4", "FORT_RUNTIME_TRACE": "1",
                       **({"CUDA_VISIBLE_DEVICES": ""} if disabled else {})}, timeout=30)
    assert "FIELDS_OK native checks=6" in result.stdout
    assert "array temporary" not in result.stderr.lower()
    rows = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in result.stderr.splitlines()
            if line.startswith("FORT_SCOPED evidence ")]
    queued = [row for row in rows if row["event"] == "command_queue_statistics"]
    if disabled:
        assert not queued
    else:
        assert len(queued) == 4
        assert all(row["accepted_tasks"] == 2 and row["submissions"] == 2 and not row["pending_tasks"] for row in queued)
        assert sum(row["conflict_flushes"] for row in queued) == 4
