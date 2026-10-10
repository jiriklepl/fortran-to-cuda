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
from compiler.tests.test_scoped_planning_runtime import Costs, Decision, costs
from compiler.tests.test_scoped_rank_reduced_views import View, view
from compiler.tests.test_scoped_runtime import RUNTIME, TOKEN, Access, Layout, Scope, Section, Stats, access
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
                      "per_role": (SOURCE.replace("generic_source_compute", "role_ranges"), False)}
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

    for name, (source, root_views) in specifications.items():
        function = lower_source(source, "evaluate", source_name=name + ".f90")
        plan = build_execution_plan(function)
        with pytest.MonkeyPatch.context() as patch:
            selected_profile, participation = profile, "serial"
            if name == "per_role":
                patch.setattr(source_compute, "numerical_compute_model", per_role_model)
            elif name == "runtime_schedule":
                selected_profile, participation = schedule_profile(patch), "fork_join_runtime"
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
        count = {"aliases": 3, "unknown": 2, "view_aliases": 3}.get(name, 1)
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


@pytest.mark.parametrize(("items", "bits"), [(0, 0), (65536, 0), (196607, 0), (196608, 7),
                                           (524288, 7), (1048576, 7), (1048577, 0)])
def test_read_write_union_enforces_physical_range_without_losing_definitions(queries, items, bits):
    lib, callables, emitted = queries
    unit, = emitted["plain"].report["planning"]["units"]
    assert unit["physical_working_set"]["source_sections_exact"]
    assert unit["physical_working_set"]["range_check_required"]
    for role in ("native_fortran", "generated_cpu", "gpu"):
        memory = unit["compute_model"][role]["memory_cost_model"]
        assert memory["working_set_range"][0] == 3 * 65536 * 8
        assert memory["working_set_range"][1] == memory["knots"][-1]["working_set_bytes"]
    scope = Scope(lib)
    values = (c.c_double * items)()
    status, handle = scope.register(values, [items])
    scope.check(status)
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    lib.capture_reset()
    scope.check(callables["plain"](scope.handle, handle, c.byref(c.c_int(items))))
    value, traffic, calls = recorded(lib)
    assert value.known_backends == bits
    assert traffic == items * 8 * 2  # RMW traffic; its physical union is items*8.
    assert calls == bool(items)
    validate_native_only_query(scope)
    decision = Decision()
    scope.check(lib.fort_scope_plan_select(scope.handle, c.byref(costs()), -1, c.byref(decision)))
    assert bool(decision.available) == (bool(bits) or items == 0)
    if items:
        assert values[0] == 0
    scope.close()


@pytest.mark.parametrize("aliased", [False, True])
def test_read_only_aliases_decline_costs_without_erasing_ordered_output_effects(queries, aliased):
    lib, callables, emitted = queries
    unit, = emitted["aliases"].report["planning"]["units"]
    assert unit["physical_working_set"]["source_sections_exact"]
    assert "distinct touched canonical roots" in unit["physical_working_set"]["runtime_requirements"]
    scope = Scope(lib)
    items = 262144
    first, second, output = ((c.c_double * items)() for _ in range(3))
    status, a = scope.register(first, [items])
    scope.check(status)
    if aliased:
        b = a
    else:
        status, b = scope.register(second, [items], identity=2)
        scope.check(status)
    status, out = scope.register(output, [items], identity=3, initialized=False)
    scope.check(status)
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    lib.capture_reset()
    scope.check(callables["aliases"](scope.handle, a, b, out, c.byref(c.c_int(items))))
    value, traffic, calls = recorded(lib)
    assert value.known_backends == (0 if aliased else 7)
    assert traffic == items * 8 * 3
    assert calls == 1
    validate_native_only_query(scope)
    # Simulated writes prove the reached sequence without defining live storage.
    assert lib.fort_scope_host_begin(scope.handle, out, c.byref(access(flags=1))) == 7
    scope.close()


def test_conservative_whole_read_keeps_definitions_but_has_no_physical_estimate(queries):
    lib, callables, emitted = queries
    public = emitted["unknown"].report
    assert public["planning"]["query_available"]
    unit, = public["planning"]["units"]
    assert unit["compute_model"] is None
    assert not unit["physical_working_set"]["source_sections_exact"]
    assert "exact physical working-set sections" in unit["work_estimate_reason"]
    assert not public["automatic_estimate_available"]
    assert "exact physical working-set sections" in public["automatic_reason"]
    scope = Scope(lib)
    items = 262144
    source, output = (c.c_double * items)(), (c.c_double * items)()
    status, out = scope.register(output, [items], initialized=False)
    scope.check(status)
    status, b = scope.register(source, [items], identity=2)
    scope.check(status)
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    lib.capture_reset()
    scope.check(callables["unknown"](scope.handle, out, b, c.byref(c.c_int(items))))
    assert recorded(lib)[0].known_backends == 0
    validate_native_only_query(scope)
    assert lib.fort_scope_host_begin(scope.handle, out, c.byref(access(flags=1))) == 7
    scope.close()


def test_shared_borrowed_read_views_preserve_exact_effects_around_undefined_storage(queries):
    lib, callables, _ = queries
    scope = Scope(lib)
    items = 262144
    source, output = (c.c_double * (3 * items))(), (c.c_double * items)()
    status, inputs = scope.register(source, [3 * items],
        defined=[([0], [items]), ([2 * items], [3 * items])])
    scope.check(status)
    status, out = scope.register(output, [items], identity=2, initialized=False)
    scope.check(status)
    first = view(inputs, [0], [items], [0])
    second = view(inputs, [2 * items], [items], [0])
    destination = view(out, [0], [items], [0])
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    lib.capture_reset()
    scope.check(callables["view_aliases"](scope.handle, c.byref(first), c.byref(second),
                                        c.byref(destination), c.byref(c.c_int(items))))
    value, traffic, calls = recorded(lib)
    assert value.known_backends == 0
    assert traffic == items * 8 * 3
    assert calls == 1
    # Cost unavailability must not widen the actual read union into its hole.
    validate_native_only_query(scope)
    assert lib.fort_scope_host_begin(scope.handle, inputs, c.byref(access(flags=1))) == 7
    assert lib.fort_scope_host_begin(scope.handle, out, c.byref(access(flags=1))) == 7
    scope.close()


def test_each_backend_requires_its_own_memory_applicability(queries):
    lib, callables, _ = queries
    scope = Scope(lib)
    items = 262144
    status, handle = scope.register((c.c_double * items)(), [items])
    scope.check(status)
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    lib.capture_reset()
    scope.check(callables["per_role"](scope.handle, handle, c.byref(c.c_int(items))))
    value, _, calls = recorded(lib)
    assert calls == 1
    assert value.known_backends == 5
    assert value.native_fortran_seconds > 0
    assert value.gpu_seconds > 0
    assert value.generated_cpu_seconds == 0
    validate_native_only_query(scope)
    decision = Decision()
    scope.check(lib.fort_scope_plan_select(scope.handle, c.byref(costs()), -1, c.byref(decision)))
    assert not decision.available
    assert not decision.gpu_units
    scope.close()


def test_runtime_schedule_is_rechecked_for_every_public_query_without_losing_definitions(queries):
    lib, callables, emitted = queries
    public = emitted["runtime_schedule"].report
    assert public["compute_estimates"]["runtime_schedule_checked"]
    unit, = public["planning"]["units"]
    assert unit["compute_model"]["runtime_schedule"] == {"kind": "static", "chunk": 0}
    lib.omp_get_schedule.argtypes = [c.POINTER(c.c_int), c.POINTER(c.c_int)]
    lib.omp_get_schedule.restype = None
    lib.omp_set_schedule.argtypes = [c.c_int, c.c_int]
    lib.omp_set_schedule.restype = None
    previous_kind, previous_chunk = c.c_int(), c.c_int()
    lib.omp_get_schedule(c.byref(previous_kind), c.byref(previous_chunk))
    scope = Scope(lib)
    try:
        items = 262144
        values = (c.c_double * items)()
        status, handle = scope.register(values, [items])
        scope.check(status)
        # libgomp uses the OpenMP standard enum values and monotonic high bit.
        # Change only this calling task's ICV, then restore it unconditionally.
        for kind, chunk, bits in ((1, 0, 7), (1, 1, 0), (2, 0, 0), (0x80000001, 0, 7)):
            lib.omp_set_schedule(kind, chunk)
            scope.check(lib.fort_scope_plan_reset(scope.handle))
            lib.capture_reset()
            scope.check(callables["runtime_schedule"](scope.handle, handle, c.byref(c.c_int(items))))
            value, traffic, calls = recorded(lib)
            assert value.known_backends == bits
            assert traffic == items * 8 * 2
            assert calls == 1
            validate_native_only_query(scope)
            decision = Decision()
            scope.check(lib.fort_scope_plan_select(scope.handle, c.byref(costs()), -1, c.byref(decision)))
            assert bool(decision.available) == bool(bits)
            assert values[0] == 0
    finally:
        lib.omp_set_schedule(previous_kind.value, previous_chunk.value)
        scope.close()


@pytest.mark.parametrize(("items", "bits"), [(196607, 0), (196608, 7)])
def test_validated_rank_reduced_view_prices_touched_elements_not_root_span(queries, items, bits):
    lib, callables, emitted = queries
    public = emitted["view"].report
    assert public["compute_estimates"]["memory_range_enforced"]
    assert "exact physical sections" in public["compute_estimates"]["memory_working_set"]
    assert public["borrowed_views"]["abi_version"] == 2
    scope = Scope(lib)
    status, handle = scope.register((c.c_double * (items * 2))(), [2, items], lower=[-3, 11])
    scope.check(status)
    spec = view(handle, [1, 0], [items], [1])
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    lib.capture_reset()
    scope.check(callables["view"](scope.handle, c.byref(spec), c.byref(c.c_int(items))))
    value, traffic, calls = recorded(lib)
    assert value.known_backends == bits
    assert calls == 1
    assert traffic == items * 8 * 2
    validate_native_only_query(scope)
    scope.close()
