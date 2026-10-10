"""Original source costs and generated public queries retain their proofs.

All calibration observations here are synthetic independent fixtures. The
compiled tests contain metadata/query helpers and the CPU coherence reference
backend only; no numerical worker, CUDA compiler or GPU work is involved.
"""

import ctypes as c
import os
import re
import shutil
import subprocess
import sys
from copy import deepcopy

import pytest

from compiler.analysis import build_execution_plan
from compiler.driver.options import CompilerOptions
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_source
from compiler.offload import analyze_offload
from compiler.offload.calibrate import profile_with_scoped_measurements
from compiler.offload.config import OffloadConfig
from compiler.offload.cpu_dependency_calibration import (
    ACCESS_CLASS,
    DOMAINS,
    _hash_json,
    cpu_dependency_costs,
    profile_from_dependency_measurements,
)
from compiler.offload.cpu_protocol_calibration import profile_from_cpu_protocol_measurements
from compiler.offload.numerical_calibration import memory_compute_seconds, profile_from_compute_measurements
from compiler.offload.source_compute import apply_source_compute_costs
from compiler.tests.test_cpu_dependency_calibration import observations
from compiler.tests.test_cpu_protocol_calibration import evidence
from compiler.tests.test_offload_profile import scoped_observations
from compiler.tests.test_scoped_compute_costs import Compute
from compiler.tests.test_scoped_runtime import TOKEN, Layout, Scope, Stats
from compiler.tests.test_source_compute_memory import function_text

BODY = "x=a(i)\ny=b(i)\nout(i)=(x*y+x)/3d0+y"


def source_analysis(body=BODY, *, rank=1):
    dimensions = ":" if rank == 1 else ":,:"
    iteration = "do i=1,n" if rank == 1 else "do j=1,n\ndo i=1,n"
    ending = "enddo" if rank == 1 else "enddo\nenddo"
    source = f"""module independent_source_dependencies
contains
subroutine evaluate(a,b,out,n)
real(8),intent(in)::a({dimensions}),b({dimensions})
real(8),intent(inout)::out({dimensions})
integer,intent(in)::n
integer::i,j
real(8)::x,y
{iteration}
{body}
{ending}
end subroutine
end module
"""
    function = lower_source(source, "evaluate", source_name="independent_source_dependencies.f90")
    plan = build_execution_plan(function, options=CompilerOptions(opt_level=0))
    analysis = analyze_offload(function, plan)
    assert analysis.available, analysis.reason
    assert len(analysis.units) == 1
    return function, plan, analysis


def accepted_profile(*, affinity=None, environment=None):
    """Reidentify raw synthetic fixtures, never patch saved acceptance flags."""
    original, rows, identity = observations()
    # Rebuild dependent evidence after changing its raw numerical identity.
    # Keeping the old optional protocol here correctly fails validation.
    original.pop("cpu_execution_protocol", None)
    original.pop("cpu_dependency", None)
    original["toolchain"].update(host_cxx_version="GCC 14.4.0", nvcc_version="NVCC V13.4.92")
    raw_numerical = deepcopy(original["numerical"]["measurements"])
    if affinity is not None:
        raw_numerical[0]["cpu_affinity"] = list(affinity)
    base = profile_from_compute_measurements(original, raw_numerical, calibration={})
    base = profile_with_scoped_measurements(base, scoped_observations(),
        runtime_id=read_scoped_runtime()[1]["runtime_id"], cold_startup_seconds=[0.1] * 5)
    records, protocol_identity, proofs = evidence(base)
    if environment is not None:
        protocol_identity.update(environment)
    profile = profile_from_cpu_protocol_measurements(base, records, protocol_identity, proofs)
    protocol = profile["cpu_execution_protocol"]
    identity.update(
        cpu_affinity=deepcopy(protocol_identity["cpu_affinity"]),
        host=deepcopy(protocol_identity["host"]),
        base_numerical_identity=_hash_json(profile["numerical"]["identity"]),
        cpu_protocol_identity=_hash_json(protocol),
    )
    for name in ("actual_team_threads", "thread_limit", "omp_wait_policy", "gomp_spincount"):
        identity[name] = protocol_identity[name]
    return profile_from_dependency_measurements(profile, rows, identity)


@pytest.fixture(scope="module")
def profile():
    return accepted_profile()


def emit(profile, body=BODY, *, rank=1, root_views=False, participation="serial"):
    function, plan, _ = source_analysis(body, rank=rank)
    return generate_scoped(function, plan,
        OffloadConfig("auto", profile, native_participation=participation), "common_functions.cuh",
        runtime_id=read_scoped_runtime()[1]["runtime_id"], root_views=root_views, root_view_abi=2)


@pytest.mark.parametrize("participation", ["serial", "fork_join"])
def test_constant_free_pointwise_source_uses_full_cpu_work_span_and_independent_gpu(profile, participation):
    _, _, analysis = source_analysis()
    original, = analysis.units
    before_gpu = deepcopy(profile["numerical"])
    modeled, = apply_source_compute_costs(analysis, profile, participation).units
    assert modeled.region is original.region
    assert modeled.footprints == original.footprints
    model = modeled.compute_model
    assert model is not None, modeled.work_estimate_reason
    assert model["cost_model"] == "source_work_span_cpu_v1"
    assert model["dependency_identity"] == original.compute_dependencies.identity
    for role, backend in (("native_fortran", "native_" + participation), ("generated_cpu", "generated_cpu")):
        expected = cpu_dependency_costs(profile, original.compute_dependencies, backend,
            workload_class=original.workload_class, workload_features=original.workload_features,
            primitive_domains=DOMAINS, access_class=ACCESS_CLASS)
        assert model[role]["compute_seconds_per_item"] == expected["seconds_per_item"]
        assert expected["seconds_per_item"] == max(expected["work_seconds"], expected["span_seconds"])
        assert model[role]["fixed_seconds"] == expected["fixed_seconds"]
        assert model[role]["memory_cost_model"] == expected["memory_cost_model"]
        assert model[role]["arithmetic_seconds_per_operation"] == 0
    assert model["gpu"]["backend_identity"] == "gpu"
    assert model["gpu_generator_id"] == before_gpu["generator_id"]
    assert model["fortran"] == before_gpu["identity"]["fortran"]
    assert profile["numerical"] == before_gpu
    public, = emit(profile, participation=participation).report["planning"]["units"]
    assert public["compute_model"] == model


@pytest.mark.parametrize(("body", "rank", "reason"), [
    ("x=a(i)\ny=b(i)\nout(i)=out(i)+x+y", 1, "memory access class"),
    ("x=a(i)+a(i+1)\ny=b(i)\nout(i)=x+y", 1, "memory access class"),
    (BODY.replace("(i)", "(i,j)"), 2, "dense traversal"),
    ("x=a(i)\ny=b(i)\nout(i)=sqrt(x*x+y*y)", 1, "primitive interval proof"),
    ("x=a(i)\ny=b(i)\nout(i)=x/y", 1, "primitive interval proof"),
    ("x=a(i)\ny=b(i)\nout(i)=x+y+real(i/2,8)", 1, "unpriced"),
    ("x=a(i)\ny=b(i)\nx=x*x\nout(i)=y+y", 1, "unconsumed private source work"),
    ("x=a(i)\ny=b(i)\nout(i)=y+y", 1, "unconsumed private source work"),
])
def test_unvalidated_accesses_and_primitive_domains_remain_native(profile, body, rank, reason):
    generated = emit(profile, body, rank=rank)
    unit, = generated.report["planning"]["units"]
    assert unit["compute_model"] is None
    assert unit["work_per_iteration"] is None
    assert reason in unit["work_estimate_reason"]
    assert generated.report["planning"]["query_available"]
    assert not generated.report["automatic_estimate_available"]


def test_borrowed_root_view_costs_remain_unavailable_without_address_compute_evidence(profile):
    generated = emit(profile, root_views=True)
    unit, = generated.report["planning"]["units"]
    assert unit["compute_model"] is None
    assert "borrowed-view address compute calibration unavailable" in unit["work_estimate_reason"]
    assert generated.report["planning"]["query_available"]
    assert not generated.report["automatic_estimate_available"]


@pytest.mark.parametrize("mutation", ["malformed", "identity", "rejected_holdout"])
def test_present_but_unavailable_dependency_evidence_does_not_fall_back_to_v2(profile, mutation):
    changed = deepcopy(profile)
    if mutation == "malformed":
        changed["cpu_dependency"] = None
    elif mutation == "identity":
        changed["cpu_dependency"]["generator_id"] = "f" * 64
    else:
        rows = deepcopy(changed["cpu_dependency"]["measurements"])
        # A source-independent held-out original Fortran helper rejects the
        # ordinary class; no basis observation or GPU evidence is changed.
        from compiler.offload.cpu_dependency_workloads import recipes_for_precision

        rejected = next(recipe.name for recipe in recipes_for_precision(64)
                        if recipe.role == "structural_holdout" and recipe.helper_form == "separate"
                        and recipe.workload_class == "ordinary_expression_v2")
        for row in rows:
            if row["recipe"] == rejected and row["backend"] == "native_serial":
                for sample in row["samples"]:
                    sample["elapsed_seconds"] *= 3
                    sample["wall_seconds"] *= 3
        changed = profile_from_dependency_measurements(changed, rows,
            changed["cpu_dependency"]["execution_identity"])
    failed = emit(changed)
    assert not failed.report["automatic_estimate_available"]
    assert failed.report["planning"]["units"][0]["compute_model"] is None
    legacy = deepcopy(changed)
    del legacy["cpu_dependency"]
    restored = emit(legacy)
    assert restored.report["automatic_estimate_available"]
    assert restored.report["planning"]["units"][0]["compute_model"] is not None
    assert restored.report["planning"]["units"][0]["compute_model"]["backend_id"] != "source-work-span-cpu-v1"
    assert restored.report["planning"]["units"][0]["compute_model"]["generator_id"] == profile["numerical"]["generator_id"]


def query_text(generated, namespace):
    source = generated.cuda
    starts = [match.start() for match in re.finditer(
        r"(?m)^static (?:bool fort_compute_memory_|bool scoped_compute_placement_compatible|offload::Data plan_unit_)",
        source)]
    starts.append(source.index('extern "C" int ' + generated.report["planning"]["entry"] + "("))
    selected = "\n".join(function_text(source, start) for start in starts)
    assert "__global__" not in selected
    assert "<<<" not in selected
    return f"namespace generated_kernels::{namespace} {{\n{selected}\n}}\n"


def load_queries(path, entry):
    library = c.CDLL(str(path))
    library.fort_scope_error.restype = c.c_char_p
    for name, signature in {
        "create": [c.c_int, c.POINTER(TOKEN)],
        "register": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.c_int, c.POINTER(TOKEN)],
        "close": [TOKEN], "stats_get": [TOKEN, c.POINTER(Stats)],
        "plan_reset": [TOKEN], "plan_validate": [TOKEN],
    }.items():
        getattr(library, "fort_scope_" + name).argtypes = signature
    library.capture_get.argtypes = [c.POINTER(Compute)]
    query = getattr(library, entry)
    query.argtypes = [TOKEN, TOKEN, TOKEN, TOKEN, c.POINTER(c.c_int)]
    return library, query


def query_costs(library, query):
    scope = Scope(library)
    items = 65536
    handles = []
    try:
        for identity in range(1, 4):
            values = (c.c_double * items)()
            status, handle = scope.register(values, [items], identity=identity, initialized=identity != 3)
            scope.check(status)
            handles.append(handle)
        scope.check(library.fort_scope_plan_reset(scope.handle))
        scope.check(query(scope.handle, *handles, c.byref(c.c_int(items))))
        captured = Compute()
        library.capture_get(c.byref(captured))
        scope.check(library.fort_scope_plan_validate(scope.handle))
        stats = scope.stats()
        assert stats.upload_bytes == stats.download_bytes == stats.launches == stats.allocations == 0
        return captured
    finally:
        scope.close()


@pytest.fixture(scope="module")
def queries(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None or not hasattr(os, "sched_getaffinity"):
        pytest.skip("C++ compiler and Linux CPU placement required")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 4:
        pytest.skip("run CPU-only source query tests on exactly four CPUs")
    profiles = {"unset": accepted_profile(affinity=affinity),
                "explicit": accepted_profile(affinity=affinity,
                    environment={"omp_wait_policy": "PASSIVE", "gomp_spincount": "100"})}
    generated = {name: emit(profile) for name, profile in profiles.items()}
    source = r'''#include <climits>
#include <sched.h>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include "offload.hpp"
#include "scoped_runtime.h"
static fort_scope_compute_costs_v1 observed{};
extern "C" int capture_compute(fort_scope_t context, std::uint32_t kind, std::uint64_t unit,
    const fort_scope_plan_binding *bindings, std::size_t count, double flops, double memory_bytes,
    int gpu_available, const fort_scope_compute_costs_v1 *compute) {
    observed=*compute;
    return fort_scope_plan_add_compute_costs_v3(context,kind,unit,bindings,count,
                                               flops,memory_bytes,gpu_available,compute);
}
#define fort_scope_plan_add_compute_costs_v3 capture_compute
#include "scoped_entry.hpp"
#include "view_entry.hpp"
#undef fort_scope_plan_add_compute_costs_v3
#define FORT_SHARED_CHECK(expr) do { int status=(expr); if(status) return status; } while(false)
extern "C" void capture_get(fort_scope_compute_costs_v1 *result) { *result=observed; }
'''
    for name, value in generated.items():
        original_entry = value.report["planning"]["entry"]
        entry = original_entry + "_" + name
        source += query_text(value, name + "_dependency_query").replace(original_entry, entry)
        source += f'''extern "C" int {entry}_nested(fort_scope_t context,
    fort_buffer_t a,fort_buffer_t b,fort_buffer_t output,const int*n) {{
    int status=0;
    #pragma omp parallel num_threads(4)
    {{
        #pragma omp single
        status=generated_kernels::{name}_dependency_query::{entry}(context,a,b,output,n);
    }}
    return status;
}}
'''
        assert "scoped_compute_placement_compatible()" in function_text(value.cuda,
            value.cuda.index('extern "C" int ' + value.report["planning"]["selector"] + "("))
    directory = tmp_path_factory.mktemp("dependency-source-query")
    (directory / "queries.cpp").write_text(source)
    target = directory / "queries.so"
    runtime = read_scoped_runtime()[0]
    # The runtime helper returns artifact texts. Compile the original reference
    # implementation from its maintained location, with GPU operations disabled.
    from compiler.tests.test_scoped_runtime import RUNTIME

    assert "scoped_runtime.cu" in runtime
    command = [compiler, "-std=c++17", "-O0", "-fPIC", "-shared", "-pthread", "-fopenmp",
               "-DFORT_SCOPE_CPU_TEST", "-I", str(RUNTIME), "-x", "c++",
               str(RUNTIME / "scoped_runtime.cu"), str(directory / "queries.cpp"), "-o", str(target)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stderr
    entries = {name: value.report["planning"]["entry"] + "_" + name for name, value in generated.items()}
    return target, entries, generated


@pytest.fixture
def omp_query_state(queries, monkeypatch):
    path, entries, _ = queries
    library, query = load_queries(path, entries["unset"])
    previous_threads, previous_dynamic = library.omp_get_max_threads(), library.omp_get_dynamic()
    library.omp_set_num_threads(4)
    library.omp_set_dynamic(0)
    monkeypatch.delenv("OMP_WAIT_POLICY", raising=False)
    monkeypatch.delenv("GOMP_SPINCOUNT", raising=False)
    try:
        yield library, query
    finally:
        library.omp_set_num_threads(previous_threads)
        library.omp_set_dynamic(previous_dynamic)


def test_public_plain_query_preserves_complete_role_costs(queries, omp_query_state):
    _, _, generated = queries
    model = generated["unset"].report["planning"]["units"][0]["compute_model"]
    captured = query_costs(*omp_query_state)
    assert captured.known_backends == 7
    items, traffic = 65536, 65536 * 3 * 8
    for role in ("native_fortran", "generated_cpu", "gpu"):
        coefficients = model[role]
        memory = memory_compute_seconds(coefficients["memory_cost_model"], traffic, traffic)
        compute = items * coefficients.get("compute_seconds_per_item",
            coefficients["intrinsic_seconds_per_item"] +
            generated["unset"].report["planning"]["units"][0]["compute_arithmetic_operations_per_iteration"] *
            coefficients["arithmetic_seconds_per_operation"])
        assert getattr(captured, role + "_seconds") == pytest.approx(
            max(compute, memory) + coefficients["fixed_seconds"])


def test_every_public_query_rechecks_wait_spin_and_team_budget(queries, omp_query_state, monkeypatch):
    # These are live identity checks, not OpenMP runtime reconfiguration.
    # libgomp may retain launch-time WAIT/SPIN state privately; callers must
    # keep the calibrated environment fixed from process launch. Changing
    # getenv strings here only exercises mismatch detection and rechecking.
    path, entries, _ = queries
    library, query = omp_query_state
    assert query_costs(library, query).known_backends == 7
    for variable in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT"):
        monkeypatch.setenv(variable, "unexpected")
        assert query_costs(library, query).known_backends == 0
        monkeypatch.delenv(variable)
        assert query_costs(library, query).known_backends == 7
    library.omp_set_num_threads(1)
    assert query_costs(library, query).known_backends == 0
    library.omp_set_num_threads(4)
    library.omp_set_dynamic(1)
    assert query_costs(library, query).known_backends == 0
    library.omp_set_dynamic(0)
    _, explicit = load_queries(path, entries["explicit"])
    assert query_costs(library, explicit).known_backends == 0
    monkeypatch.setenv("OMP_WAIT_POLICY", "PASSIVE")
    monkeypatch.setenv("GOMP_SPINCOUNT", "100")
    assert query_costs(library, explicit).known_backends == 7
    monkeypatch.setenv("GOMP_SPINCOUNT", "101")
    assert query_costs(library, explicit).known_backends == 0


def test_public_query_declines_an_existing_native_team_but_preserves_effects(queries, omp_query_state):
    path, entries, _ = queries
    library, query = omp_query_state
    assert query_costs(library, query).known_backends == 7
    _, nested = load_queries(path, entries["unset"] + "_nested")
    assert query_costs(library, nested).known_backends == 0
    assert query_costs(library, query).known_backends == 7


def test_public_query_declines_actual_thread_limit_below_calibrated_budget(queries):
    path, entries, _ = queries
    environment = dict(os.environ, OMP_THREAD_LIMIT="1", OMP_NUM_THREADS="4", OMP_DYNAMIC="FALSE",
                       OMP_PROC_BIND="false")
    for variable in ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT", "GOMP_CPU_AFFINITY", "OMP_PLACES"):
        environment.pop(variable, None)
    program = """import sys
from compiler.tests.test_dependency_source import load_queries, query_costs
library, query=load_queries(sys.argv[1],sys.argv[2])
assert library.omp_get_thread_limit()==1
assert library.omp_get_max_threads()==4
assert query_costs(library,query).known_backends==0
print('coherent native query')
"""
    result = subprocess.run([sys.executable, "-c", program, str(path), entries["unset"]],
        env=environment, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "coherent native query"
