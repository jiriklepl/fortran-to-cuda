"""Generated source queries keep memory applicability separate from effects.

Only the emitted metadata, memory helpers and public query are compiled here.
The real reference runtime validates their effects; no numerical worker, CUDA
compiler, GPU initialization or device operation participates in these tests.
"""

import ctypes as c
import re
import shutil
import subprocess
from copy import deepcopy

import pytest

from compiler.analysis import build_execution_plan
from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.scoped import generate_scoped
from compiler.frontend import lower_source
from compiler.offload.config import OffloadConfig
from compiler.tests.test_schedule_calibration import schedule_profile
from compiler.tests.test_scoped_compute_costs import Compute
from compiler.tests.test_scoped_planning_runtime import Costs, Decision
from compiler.tests.test_scoped_rank_reduced_views import View
from compiler.tests.test_scoped_runtime import RUNTIME, TOKEN, Access, Layout, Section, Stats
from compiler.tests.test_source_compute_costs import SOURCE, profile_v2

ALIASES = """module distinct_inputs
contains
subroutine evaluate(a,b,out,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(out)::out(:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=sqrt(a(i)*b(i)+1d0)+cos(a(i))
enddo
end subroutine
end module
"""

INDIRECT = """module unknown_read_footprint
contains
subroutine evaluate(out,b,n)
real(8),intent(out)::out(:)
real(8),intent(in)::b(:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=sqrt(b(i*i)*b(i*i)+1d0)+cos(b(i*i))
enddo
end subroutine
end module
"""

POINTWISE = """module total_memory_query
contains
subroutine evaluate(a,b,out,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(out)::out(:)
integer,intent(in)::n
integer::i
do i=1,n
out(i)=2d0*a(i)+b(i)
enddo
end subroutine
end module
"""


def function_text(source, start):
    """Select complete compiler-emitted functions, including nested lambdas."""
    first = source.index("{", start)
    depth = 1
    end = first + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def query_source(emitted, namespace):
    query = emitted.report["planning"]["entry"]
    source = emitted.cuda
    starts = [match.start() for match in re.finditer(
        r"(?m)^static (?:bool fort_compute_memory_|bool scoped_compute_schedule_compatible|offload::Data plan_unit_)", source)]
    starts.append(source.index('extern "C" int ' + query + "("))
    selected = "\n".join(function_text(source, start) for start in starts)
    assert "__global__" not in selected
    assert "<<<" not in selected
    # These tests isolate physical memory contracts. Runtime placement and
    # environment rejection are exercised with the real helper separately;
    # this extracted query deliberately supplies compatible participation.
    placement = "static bool scoped_compute_placement_compatible() { return true; }"
    return f"namespace generated_kernels::{namespace} {{\n{placement}\n{selected}\n}}\n"


@pytest.fixture(scope="module")
def queries(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    profile = profile_v2()
    runtime_id = read_scoped_runtime()[1]["runtime_id"]
    emitted = {}
    specifications = {"plain": (SOURCE, False), "aliases": (ALIASES, False),
                      "unknown": (INDIRECT, False), "view": (SOURCE, True),
                      "view_aliases": (ALIASES, True),
                      "runtime_schedule": (SOURCE.replace("generic_source_compute", "schedule_source_compute"), False),
                      "per_role": (SOURCE.replace("generic_source_compute", "role_ranges"), False),
                      **{name: (POINTWISE, False) for name in (
                          "gpu_total", "gpu_total_no_cpu", "gpu_total_no_gpu", "gpu_total_no_native", "gpu_total_only",
                          "gpu_total_small_gain", "gpu_total_missing_contract")}}
    from compiler.offload import source_compute

    original_model = source_compute.numerical_compute_model

    def per_role_model(*args, **kwargs):
        # Exercise the emitter's public model contract with backend-specific
        # applicability. This is synthetic evidence, never a measured profile.
        result = deepcopy(original_model(*args, **kwargs))
        memory = result["generated_cpu"]["memory_cost_model"]
        for knot in memory["knots"]:
            knot["working_set_bytes"] *= 4
        memory["working_set_range"] = [value * 4 for value in memory["working_set_range"]]
        return result

    def total_model(name, *args, **kwargs):
        # Synthetic accepted payloads isolate composition in the actual public
        # query/planner. They establish no hardware calibration acceptance.
        result = deepcopy(original_model(*args, **kwargs))
        for role in ("native_fortran", "generated_cpu", "gpu"):
            coefficients = result[role]
            coefficients["arithmetic_seconds_per_operation"] = 1e-12
            coefficients["intrinsic_seconds_per_item"] = 0.0
            for knot in coefficients["memory_cost_model"]["knots"]:
                knot["seconds_per_traffic_byte"] = 1e-18
            coefficients["fixed_seconds"] = .1 if role == "generated_cpu" else .01
        result["native_fortran"]["fixed_seconds"] = .006 if name == "gpu_total_small_gain" else .01
        gpu = result["gpu"]
        gpu["fixed_seconds"] = .001
        gpu["launch_enqueue_included"] = True
        gpu["memory_combination"] = "max_fixed_compute_total_memory_v1"
        memory = gpu["memory_cost_model"]
        gpu["memory_cost_model"] = {"kind": "piecewise_gpu_total_memory_v1",
            "traffic_equals_working_set": True, "working_set_range": memory["working_set_range"],
            "knots": [{"working_set_bytes": row["working_set_bytes"], "total_seconds": .005}
                      for row in memory["knots"]]}
        if name == "gpu_total_no_cpu":
            result["generated_cpu"] = None
        elif name == "gpu_total_no_gpu":
            result["gpu"] = None
        elif name in {"gpu_total_no_native", "gpu_total_only"}:
            result["native_fortran"] = None
            if name == "gpu_total_only":
                result["generated_cpu"] = None
                result["fortran"] = result["generator_id"] = result["cpu_protocol_environment"] = None
        elif name == "gpu_total_missing_contract":
            del gpu["memory_combination"]
        return result

    for name, (source, root_views) in specifications.items():
        function = lower_source(source, "evaluate", source_name=name + ".f90")
        plan = build_execution_plan(function)
        with pytest.MonkeyPatch.context() as patch:
            selected_profile, participation = profile, "serial"
            if name == "per_role":
                patch.setattr(source_compute, "numerical_compute_model", per_role_model)
            elif name == "runtime_schedule":
                selected_profile, participation = schedule_profile(patch), "fork_join_runtime"
            elif name.startswith("gpu_total"):
                patch.setattr(source_compute, "numerical_compute_model",
                              lambda *args, name=name, **kwargs: total_model(name, *args, **kwargs))
            emitted[name] = generate_scoped(function, plan,
                OffloadConfig("auto", selected_profile, native_participation=participation), "common_functions.cuh",
                runtime_id=runtime_id, root_views=root_views, root_view_abi=2)

    source = r'''#include <climits>
#include <omp.h>
#include "offload.hpp"
#include "scoped_runtime.h"
static fort_scope_compute_costs_v1 observed{};
static double observed_traffic = 0;
static std::uint64_t observed_calls = 0;
extern "C" int capture_compute(fort_scope_t context, std::uint32_t kind, std::uint64_t unit,
    const fort_scope_plan_binding *bindings, std::size_t count, double flops, double memory_bytes,
    int gpu_available, const fort_scope_compute_costs_v1 *compute) {
    observed = *compute; observed_traffic = memory_bytes; ++observed_calls;
    return fort_scope_plan_add_compute_costs_v3(context,kind,unit,bindings,count,
                                               flops,memory_bytes,gpu_available,compute);
}
#define fort_scope_plan_add_compute_costs_v3 capture_compute
#include "scoped_entry.hpp"
#include "view_entry.hpp"
#undef fort_scope_plan_add_compute_costs_v3
#define FORT_SHARED_CHECK(expr) do { int status=(expr); if(status) return status; } while(false)
extern "C" void capture_reset() { observed={}; observed_traffic=0; observed_calls=0; }
extern "C" void capture_get(fort_scope_compute_costs_v1 *value,double *traffic,std::uint64_t *calls) {
    *value=observed; *traffic=observed_traffic; *calls=observed_calls;
}
'''
    source += "\n".join(query_source(value, name + "_query") for name, value in emitted.items())
    directory = tmp_path_factory.mktemp("source-memory-query")
    (directory / "queries.cpp").write_text(source)
    target = directory / "queries.so"
    command = [compiler, "-std=c++17", "-O0", "-fPIC", "-shared", "-pthread", "-fopenmp",
               "-DFORT_SCOPE_CPU_TEST", "-I", str(RUNTIME), "-x", "c++",
               str(RUNTIME / "scoped_runtime.cu"), str(directory / "queries.cpp"), "-o", str(target)]
    built = subprocess.run(command, text=True, capture_output=True, timeout=60)
    assert built.returncode == 0, built.stderr
    lib = c.CDLL(str(target))
    lib.fort_scope_error.restype = c.c_char_p
    for name, signature in {
        "create": [c.c_int, c.POINTER(TOKEN)],
        "register": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.c_int, c.POINTER(TOKEN)],
        "register_sections": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.POINTER(Section), c.c_size_t,
                              c.POINTER(TOKEN)],
        "close": [TOKEN], "stats_get": [TOKEN, c.POINTER(Stats)],
        "host_begin": [TOKEN, TOKEN, c.POINTER(Access)], "host_end": [TOKEN, TOKEN],
        "plan_reset": [TOKEN], "plan_validate": [TOKEN],
        "plan_select": [TOKEN, c.POINTER(Costs), c.c_int, c.POINTER(Decision)],
    }.items():
        getattr(lib, "fort_scope_" + name).argtypes = signature
    lib.capture_get.argtypes = [c.POINTER(Compute), c.POINTER(c.c_double), c.POINTER(TOKEN)]
    callables = {}
    for name, value in emitted.items():
        query = getattr(lib, value.report["planning"]["entry"])
        count = {"aliases": 3, "unknown": 2, "view_aliases": 3}.get(name, 3 if name.startswith("gpu_total") else 1)
        handles = [c.POINTER(View) if specifications[name][1] else TOKEN] * count
        query.argtypes = [TOKEN, *handles, c.POINTER(c.c_int)]
        callables[name] = query
    return lib, callables, emitted


def recorded(lib):
    value, traffic, calls = Compute(), c.c_double(), TOKEN()
    lib.capture_get(c.byref(value), c.byref(traffic), c.byref(calls))
    return value, traffic.value, calls.value


def validate_native_only_query(scope):
    scope.check(scope.lib.fort_scope_plan_validate(scope.handle))
    stats = scope.stats()
    assert stats.upload_bytes == stats.download_bytes == stats.allocations == stats.launches == 0
