"""Rank-reduced rectangular aliases retain root pitches and exact coverage."""
# ruff: noqa: F811 - pytest discovers the imported module-scoped fixture.
import ctypes as c
import shutil
import subprocess

import pytest

from compiler.tests.test_scoped_runtime import RUNTIME, SIZE, TOKEN, Layout, Scope, runtime  # noqa: F401


class View(c.Structure):
    _fields_ = [("version", c.c_uint32), ("rank", c.c_uint32), ("root_rank", c.c_uint32), ("reserved", c.c_uint32),
                ("buffer", TOKEN), ("generation", TOKEN), ("origins", c.POINTER(SIZE)),
                ("extents", c.POINTER(SIZE)), ("lower", c.POINTER(c.c_int64)), ("axes", c.POINTER(c.c_uint32))]


class ViewLayout(c.Structure):
    _fields_ = [("root", Layout), ("rank", c.c_uint32), ("reserved", c.c_uint32),
                ("origins", c.POINTER(SIZE)), ("extents", c.POINTER(SIZE)), ("strides", c.POINTER(SIZE)),
                ("lower", c.POINTER(c.c_int64)), ("axes", c.POINTER(c.c_uint32)), ("elements", SIZE), ("offset", SIZE)]


def view(buffer, origin, extent, axes, lower=None, generation=1):
    result = View(2, len(extent), len(origin), 0, buffer, generation)
    result.references = [(SIZE * len(origin))(*origin), (SIZE * len(extent))(*extent),
                         (c.c_int64 * len(extent))(*(lower or [1] * len(extent))), (c.c_uint32 * len(axes))(*axes)]
    result.origins, result.extents, result.lower, result.axes = result.references
    return result


def get(scope, spec):
    scope.lib.fort_scope_view_get_v2.argtypes = [TOKEN, c.POINTER(View), c.POINTER(ViewLayout)]
    layout = ViewLayout()
    return scope.lib.fort_scope_view_get_v2(scope.handle, c.byref(spec), c.byref(layout)), layout


@pytest.mark.native
def test_plane_maps_retained_axes_and_original_pitches_without_allocating(runtime):
    scope = Scope(runtime)
    data = (c.c_double * 315)(*range(315))
    status, handle = scope.register(data, [5, 7, 9], lower=[-3, 11, -9], generation=17)
    scope.check(status)
    spec = view(handle, [1, 2, 3], [3, 4], [0, 2], lower=[-11, 4], generation=17)
    status, layout = get(scope, spec)
    scope.check(status)
    assert layout.rank == 2 and layout.root.rank == 3
    assert list(layout.extents[:2]) == [3, 4]
    assert list(layout.axes[:2]) == [0, 2]
    assert list(layout.strides[:3]) == [8, 40, 280]
    assert list(layout.root.lower[:3]) == [-3, 11, -9]
    assert list(layout.lower[:2]) == [-11, 4]
    assert layout.offset == 928 and layout.elements == 12
    assert layout.root.host == c.addressof(data)
    assert scope.stats().allocations == 0
    scope.close()


@pytest.mark.native
@pytest.mark.parametrize(("origin", "extent", "axes", "lower", "expected"), [
    ([0, 7, 0], [5, 9], [0, 2], None, 1),
    ([0, 0, 0], [5, 9], [0, 0], None, 1),
    ([0, 0, 0], [5, 9], [0, 3], None, 1),
    ([1, 0, 0], [5, 9], [0, 2], None, 1),
    ([0, 0, 0], [5, 9], [0, 2], [2**31-2, 1], 5),
    ([0, 0, 0], [5, 9], [0, 2], [-2**31-1, 1], 5),
])
def test_invalid_axis_plane_and_bounds_fail_before_work(runtime, origin, extent, axes, lower, expected):
    scope = Scope(runtime)
    status, handle = scope.register((c.c_double * 315)(), [5, 7, 9])
    scope.check(status)
    assert get(scope, view(handle, origin, extent, axes, lower))[0] == expected
    assert scope.stats().uploads == scope.stats().allocations == 0
    scope.close()


@pytest.mark.native
def test_empty_plane_never_computes_an_address_and_generation_is_checked(runtime):
    scope = Scope(runtime)
    maximum = SIZE(-1).value
    status, handle = scope.register(None, [0, maximum])
    scope.check(status)
    spec = view(handle, [0, maximum], [0], [0])
    status, layout = get(scope, spec)
    scope.check(status)
    assert layout.elements == layout.offset == 0
    assert not layout.root.host
    spec.generation = 2
    assert get(scope, spec)[0] == 2
    scope.close()


@pytest.mark.native
def test_rank_reduced_worker_preserves_other_planes_and_projects_partial_definitions(tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    source = tmp_path / "planes.cpp"
    source.write_text(r'''
#include "view_entry.hpp"
#include <cassert>
int main() {
  fort_scope_t context=0; assert(!fort_scope_create(0,&context));
  double a[150]; for(int i=0;i<150;++i) a[i]=i;
  size_t dimensions[]={5,6,5}; int64_t bounds[]={-2,10,-5};
  fort_scope_layout root{3,FORT_SCOPE_REAL64,8,a,dimensions,bounds,1};
  fort_buffer_t handle=0; assert(!fort_scope_register(context,1,1,&root,1,&handle));
  size_t o1[]={1,1,1},o2[]={1,4,1},extents[]={3,3}; int64_t lower[]={1,1}; uint32_t axes[]={0,2};
  fort_scope_view_v2 v1{2,2,3,0,handle,1,o1,extents,lower,axes},v2{2,2,3,0,handle,1,o2,extents,lower,axes};
  fort_scope_view_layout_v2 l1{},l2{};
  assert(!fort_scope_view_get_v2(context,&v1,&l1)); assert(!fort_scope_view_get_v2(context,&v2,&l2));
  fort_scoped::RootView<double,2> x(l1,a),y(l2,a);
  { fort_scoped::ViewAccessBatch<2> access(context);
    fort_scope_access effect{}; effect.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
    assert(!access.add(&v1,effect)); assert(!access.add(&v2,effect)); assert(!access.begin(false));
    access.executing(); for(size_t i=0;i<9;++i) { x[i]=100+i; y[i]=200+i; } assert(!access.finish());
  }
  for(size_t z=0;z<5;++z) for(size_t j=0;j<6;++j) for(size_t i=0;i<5;++i) {
    const size_t p=i+5*j+30*z;
    const bool selected=i>=1 && i<4 && z>=1 && z<4 && (j==1 || j==4);
    assert(a[p]==(selected ? (j==1?100:200)+(i-1)+3*(z-1) : p));
  }
  { fort_scoped::ViewAccessBatch<2> access(context);
    size_t lo[]={1,1},hi[]={3,3}; fort_scope_section section{lo,hi};
    fort_scope_access effect{}; effect.read_count=1; effect.reads=&section;
    assert(!access.add(&v1,effect)); assert(!access.add(&v1,effect)); assert(!access.begin(true));
    access.executing(); assert(!access.finish());
  }
  fort_scope_stats stats{}; assert(!fort_scope_stats_get(context,&stats));
  assert(stats.upload_bytes==4*8); // exact x/z subset, no enclosing volume
  { fort_scoped::ViewAccessBatch<2> access(context);
    fort_scope_access effect{}; effect.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
    assert(!access.add(&v1,effect)); assert(access.add(&v1,effect)==FORT_SCOPE_ALIAS);
  }
  assert(!fort_scoped::forget_view(context,&v1,false));
  { fort_scoped::ViewAccessBatch<2> access(context);
    fort_scope_access effect{}; effect.flags=FORT_SCOPE_READ_ALL;
    assert(!access.add(&v1,effect)); assert(access.begin(false)==FORT_SCOPE_UNINITIALIZED);
  }
  { fort_scoped::ViewAccessBatch<2> access(context);
    fort_scope_access effect{}; effect.flags=FORT_SCOPE_READ_ALL;
    assert(!access.add(&v2,effect)); assert(!access.begin(false)); access.executing(); assert(!access.finish());
  }
  // Redefine the discarded face before owner publication, without changing
  // the other face or any halo element.
  { fort_scoped::ViewAccessBatch<2> access(context);
    fort_scope_access effect{}; effect.flags=FORT_SCOPE_WRITE_ALL|FORT_SCOPE_OVERWRITE_ALL;
    assert(!access.add(&v1,effect)); assert(!access.begin(false)); access.executing();
    for(size_t i=0;i<9;++i) { x[i]=-1; } assert(!access.finish());
  }
  assert(!fort_scope_close(context));
}
''')
    target = tmp_path / "planes"
    result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-pthread",
                             "-DFORT_SCOPE_CPU_TEST", "-I", str(RUNTIME), "-x", "c++", str(source),
                             str(RUNTIME / "scoped_runtime.cu"), "-o", str(target)], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(target)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
