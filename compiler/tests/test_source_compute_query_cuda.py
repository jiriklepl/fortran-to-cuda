"""Consume generated v3 query artifacts without executing numerical kernels."""

import ctypes as c
import os
import shutil
import subprocess

import pytest

from compiler.emission import generate_sources, read_common_header
from compiler.offload.config import OffloadConfig
from compiler.tests.test_scoped_planning_runtime import TOKEN, Costs, Decision, Scope, costs
from compiler.tests.test_scoped_runtime import Layout, Stats
from compiler.tests.test_source_compute_costs import profile_v2, source_analysis


@pytest.fixture(scope="module")
def query_library(tmp_path_factory):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14") or shutil.which("g++")
    if not nvcc or not host:
        pytest.skip("CUDA and host C++ toolchains required")
    function, plan, _ = source_analysis()
    generated = generate_sources(function, plan,
        offload_config=OffloadConfig("auto", profile_v2(), native_participation="serial"),
        memory_model="scoped")
    directory = tmp_path_factory.mktemp("source-compute-query")
    for name, text in generated.artifacts.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (directory / "common_functions.cuh").write_text(read_common_header())
    library = directory / "query.so"
    command = [nvcc, "-O1", "-std=c++17", "-ccbin", host, "-arch=" + os.environ.get("FORT_TEST_CUDA_ARCH", "native"), "-shared",
               "-Xcompiler=-fPIC", "-Xcompiler=-fopenmp", "-I", str(directory),
               str(directory / "scoped_runtime.cu"), str(directory / "shared_entry.cu"), "-o", str(library)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert result.returncode == 0, result.stderr
    lib = c.CDLL(str(library))
    lib.fort_scope_error.restype = c.c_char_p
    signatures = {
        "create": [c.c_int, c.POINTER(TOKEN)],
        "register": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.c_int, c.POINTER(TOKEN)],
        "close": [TOKEN], "stats_get": [TOKEN, c.POINTER(Stats)],
        "plan_reset": [TOKEN], "plan_validate": [TOKEN],
        "plan_select": [TOKEN, c.POINTER(Costs), c.c_int, c.POINTER(Decision)],
    }
    for name, signature in signatures.items():
        getattr(lib, "fort_scope_" + name).argtypes = signature
    query = getattr(lib, generated.scoped["planning"]["entry"])
    query.argtypes = [TOKEN, TOKEN, c.POINTER(c.c_int)]
    return lib, query


@pytest.mark.cuda
@pytest.mark.parametrize(("items", "available"), [
    (0, True), (65535, False), (65536, False),
    (196607, False), (196608, True), (524288, True), (1048576, True), (1048577, False),
])
def test_generated_query_enforces_measured_range_without_invalidating_definitions(query_library, items, available):
    lib, query = query_library
    scope = Scope(lib)
    values = (c.c_double * items)(*[1.] * items)
    status, handle = scope.register(values, [items])
    scope.check(status)
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    scope.check(query(scope.handle, handle, c.byref(c.c_int(items))))
    # This in-place array's working set is N*8, although its read+write traffic
    # is N*16. The memory profile starts at 3*65536*8 unique bytes. Both the item
    # and physical-working-set ranges apply independently; unavailable compute
    # costs must not make ordered definition proofs disappear.
    scope.check(lib.fort_scope_plan_validate(scope.handle))
    decision = Decision()
    scope.check(lib.fort_scope_plan_select(scope.handle, c.byref(costs()), -1, c.byref(decision)))
    assert bool(decision.available) == available
    assert not available or items == 0 or decision.gpu_units == 1
    statistics = scope.stats()
    assert statistics.allocations == 0
    assert statistics.upload_bytes == statistics.download_bytes == statistics.launches == 0
    scope.close()
