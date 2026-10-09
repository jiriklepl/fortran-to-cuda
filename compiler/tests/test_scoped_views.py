"""Canonical borrowed views, partial definitions and full-layout indexing."""
# ruff: noqa: F811 - pytest discovers the imported module-scoped fixture.
from __future__ import annotations

import ctypes as c
import shutil
import subprocess

import pytest

from compiler.tests.test_scoped_planning_runtime import (
    costs,
    planning,  # noqa: F401
    record,
    select,
)
from compiler.tests.test_scoped_runtime import (
    RUNTIME,
    SIZE,
    TOKEN,
    Layout,
    Scope,
    Section,
    access,
    runtime,  # noqa: F401
)


class View(c.Structure):
    _fields_ = [("version", c.c_uint32), ("rank", c.c_uint32), ("buffer", TOKEN), ("generation", TOKEN),
                ("origins", c.POINTER(SIZE)), ("extents", c.POINTER(SIZE)), ("lower", c.POINTER(c.c_int64))]


class ViewLayout(c.Structure):
    _fields_ = [("root", Layout), ("origins", c.POINTER(SIZE)), ("extents", c.POINTER(SIZE)),
                ("strides", c.POINTER(SIZE)), ("lower", c.POINTER(c.c_int64)), ("elements", SIZE), ("offset", SIZE)]


def view(buffer, origin, extent, lower=None, generation=1):
    result = View(1, len(origin), buffer, generation)
    result.references = [(SIZE * len(origin))(*origin), (SIZE * len(extent))(*extent),
                         (c.c_int64 * len(origin))(*(lower or [1] * len(origin)))]
    result.origins, result.extents, result.lower = result.references
    return result


def get(scope, spec):
    scope.lib.fort_scope_view_get_v1.argtypes = [TOKEN, c.POINTER(View), c.POINTER(ViewLayout)]
    output = ViewLayout()
    return scope.lib.fort_scope_view_get_v1(scope.handle, c.byref(spec), c.byref(output)), output


@pytest.mark.native
def test_view_preserves_child_bounds_and_root_pitches_without_allocation(runtime):
    scope = Scope(runtime)
    data = (c.c_double * 35)(*range(35))
    status, buffer = scope.register(data, [5, 7], lower=[-3, 12], generation=17)
    scope.check(status)
    spec = view(buffer, [1, 2], [3, 4], lower=[-11, 1], generation=17)
    status, layout = get(scope, spec)
    scope.check(status)
    assert list(layout.extents[:2]) == [3, 4]
    assert list(layout.lower[:2]) == [-11, 1]
    assert list(layout.strides[:2]) == [8, 40]
    assert list(layout.root.lower[:2]) == [-3, 12]
    assert layout.offset == 88
    assert layout.elements == 12
    assert layout.root.host == c.addressof(data)
    assert scope.stats().allocations == 0
    scope.close()


@pytest.mark.native
def test_empty_view_has_no_address_arithmetic_and_stale_view_fails(runtime):
    scope = Scope(runtime)
    status, buffer = scope.register(None, [0, 7])
    scope.check(status)
    spec = view(buffer, [0, 7], [0, 0], generation=1)
    status, layout = get(scope, spec)
    scope.check(status)
    assert layout.elements == layout.offset == 0
    assert not layout.root.host
    spec.generation = 2
    assert get(scope, spec)[0] == 2
    scope.close()
    assert get(scope, view(buffer, [0, 0], [0, 0]))[0] == 2


@pytest.mark.native
@pytest.mark.parametrize(("origin", "extent", "lower", "expected"), [
    ([8], [1], [1], 1), ([0], [9], [1], 1), ([0], [8], [2**31-2], 5),
    ([0], [8], [-2**31-1], 5),
])
def test_invalid_view_bounds_are_rejected_before_work(runtime, origin, extent, lower, expected):
    scope = Scope(runtime)
    status, buffer = scope.register((c.c_double * 8)(), [8])
    scope.check(status)
    assert get(scope, view(buffer, origin, extent, lower))[0] == expected
    assert scope.stats().uploads == scope.stats().allocations == 0
    scope.close()


@pytest.mark.native
def test_partial_definition_retains_unrelated_dirty_device_values(runtime):
    scope = Scope(runtime)
    data = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(data, [8], lower=[-4])
    scope.check(status)
    device = scope.gpu(buffer, access(flags=3))
    for i in range(8):
        device[i] += 100
    scope.gpu_end(buffer)
    partial = access(write=[([2], [5])])
    runtime.fort_scope_forget_sections_v1.argtypes = [TOKEN, TOKEN, c.POINTER(Section), SIZE]
    scope.check(runtime.fort_scope_forget_sections_v1(scope.handle, buffer, partial.writes, partial.write_count))
    # Reading a discarded element fails; the error does not replay old work.
    undefined = access(read=[([2], [3])])
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(undefined)) == 7
    replacement = access(write=[([2], [5])], overwrite=[([2], [5])])
    scope.cpu_begin(buffer, replacement)
    for i in range(2, 5):
        data[i] = -i
    scope.cpu_end(buffer)
    scope.close()
    assert list(data) == [100, 101, -2, -3, -4, 105, 106, 107]


@pytest.mark.native
def test_partial_definition_fragmentation_failure_is_atomic(runtime):
    scope = Scope(runtime)
    data = (c.c_double * 80)(*range(80))
    status, buffer = scope.register(data, [80])
    scope.check(status)
    device = scope.gpu(buffer, access(flags=3))
    for i in range(80):
        device[i] += 1000
    scope.gpu_end(buffer)
    partial = access(write=[([2*i+1], [2*i+2]) for i in range(32)])
    runtime.fort_scope_forget_sections_v1.argtypes = [TOKEN, TOKEN, c.POINTER(Section), SIZE]
    assert runtime.fort_scope_forget_sections_v1(scope.handle, buffer, partial.writes, partial.write_count) == 5
    scope.close()
    assert list(data) == [i+1000 for i in range(80)]


@pytest.mark.native
def test_partial_definition_queries_do_not_mutate_and_revalidate_reads(planning):
    scope = Scope(planning)
    data = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(data, [8])
    scope.check(status)
    planning.fort_scope_plan_forget_sections_v1.argtypes = [TOKEN, TOKEN, c.POINTER(Section), SIZE]
    cut = access(write=[([2], [5])])
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    scope.check(planning.fort_scope_plan_forget_sections_v1(scope.handle, buffer, cut.writes, 1))
    record(scope, 51, buffer, access(flags=1))
    assert planning.fort_scope_plan_validate(scope.handle) == 7
    assert scope.stats().uploads == scope.stats().allocations == 0
    assert list(data) == list(range(8))
    # Changing this query invalidates its failed proof; an exact write defines
    # the discarded section before the later read without losing other data.
    scope.check(planning.fort_scope_plan_reset(scope.handle))
    scope.check(planning.fort_scope_plan_forget_sections_v1(scope.handle, buffer, cut.writes, 1))
    record(scope, 52, buffer, access(write=[([2], [5])], overwrite=[([2], [5])]))
    record(scope, 53, buffer, access(flags=1))
    scope.check(planning.fort_scope_plan_validate(scope.handle))
    decision = select(scope, costs(), -1)
    assert decision.available
    assert decision.gpu_units == 2
    scope.close()
    assert list(data) == list(range(8))


@pytest.mark.native
def test_view_worker_disjoint_writes_and_exact_read_aliases(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    source = tmp_path / "views.cpp"
    source.write_text(r'''
#include "view_entry.hpp"
#include <cassert>
int main() {
  fort_scope_t context=0; assert(!fort_scope_create(0,&context));
  double a[30]; for(int i=0;i<30;++i) a[i]=i;
  size_t dimensions[]={5,6}; int64_t bounds[]={-2,10};
  fort_scope_layout root{2,FORT_SCOPE_REAL64,8,a,dimensions,bounds,1};
  fort_buffer_t handle=0; assert(!fort_scope_register(context,1,1,&root,1,&handle));
  size_t o1[]={1,0},o2[]={1,3},extents[]={3,3}; int64_t lower[]={1,-5};
  fort_scope_view_v1 v1{1,2,handle,1,o1,extents,lower},v2{1,2,handle,1,o2,extents,lower};
  fort_scope_view_layout_v1 l1{},l2{};
  assert(!fort_scope_view_get_v1(context,&v1,&l1));
  assert(!fort_scope_view_get_v1(context,&v2,&l2));
  fort_scoped::RootView<double,2> x(l1,a),y(l2,a);
  { fort_scoped::ViewAccessBatch<2> access(context);
    fort_scope_access effect{}; effect.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
    assert(!access.add(&v1,effect)); assert(!access.add(&v2,effect)); assert(!access.begin(false));
    access.executing();
    for(size_t i=0;i<9;++i) { x[i]=100+i; y[i]=200+i; }
    assert(!access.finish());
  }
  for(size_t j=0;j<6;++j) for(size_t i=0;i<5;++i)
    assert(a[i+5*j]==(i>=1 && i<4 ? (j<3?100:200)+(i-1)+3*(j%3) : i+5*j));
  { fort_scoped::ViewAccessBatch<2> access(context);
    fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
    assert(!access.add(&v1,read)); assert(!access.add(&v1,read)); assert(!access.begin(true));
    access.executing(); assert(!access.finish());
  }
  fort_scope_stats stats{}; assert(!fort_scope_stats_get(context,&stats));
  assert(stats.upload_bytes==9*8); // no root bounding box or full array upload
  { fort_scoped::ViewAccessBatch<2> access(context);
    fort_scope_access write{}; write.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
    assert(!access.add(&v1,write)); assert(access.add(&v1,write)==FORT_SCOPE_ALIAS);
  }
  assert(!fort_scope_close(context));
}
''')
    target = tmp_path / "views"
    command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread", "-DFORT_SCOPE_CPU_TEST",
               "-I", str(RUNTIME), "-x", "c++", str(source), str(RUNTIME / "scoped_runtime.cu"), "-o", str(target)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(target)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
