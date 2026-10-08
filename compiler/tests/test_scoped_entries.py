"""Generated modules share the public runtime, retaining original coordinates."""

from __future__ import annotations

import ctypes as c
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Access, Layout, Scope, Stats, access

ROOT = Path(__file__).resolve().parents[2]

SOURCES = {
    "producer": '''module source_producer
contains
subroutine producer(a,b,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
integer,intent(in)::n
integer::i
do i=-2,n-3
  b(i+3)=2*a(i+3)+real(i,8)
enddo
end subroutine
end module
''',
    "consumer": '''module source_consumer
contains
subroutine consumer(a,b,out,n)
real(8),intent(in)::a(:),b(:)
real(8),intent(out)::out(:)
integer,intent(in)::n
integer::i
do i=-2,n-3
  out(i+3)=a(i+3)+b(i+3)
enddo
end subroutine
end module
''',
    "guarded": '''module source_guarded
contains
subroutine guarded(a,n,m)
real(8),intent(inout)::a(:,:)
integer,intent(in)::n,m
integer::i,j
do j=1,n
  do i=1,m
    a(i,j)=a(i,j)+1
  enddo
enddo
end subroutine
end module
''',
    "two_stage": '''module source_two_stage
contains
subroutine two_stage(a,b,out,n)
real(8),intent(in)::a(:)
real(8),intent(out)::b(:)
real(8),intent(inout)::out(:)
integer,intent(in)::n
integer::i
do i=-2,n-3
  b(i+3)=a(i+3)+1
enddo
do i=-1,n-3
  out(i+3)=b(i+2)+a(i+3)
enddo
end subroutine
end module
''',
    "guarded_value": '''module source_guarded_value
contains
subroutine guarded_value(a,n,flag,protected)
real(8),intent(inout)::a(:)
integer,intent(in)::n
logical,intent(in)::flag
real(8),intent(in)::protected
integer::i
do i=1,n
  if(flag) a(i)=protected
enddo
do i=1,n
  a(i)=a(i)+1
enddo
end subroutine
end module
''',
    "guarded_index": '''module source_guarded_index
contains
subroutine guarded_index(a,n,flag,protected)
real(8),intent(inout)::a(:)
integer,intent(in)::n,protected
logical,intent(in)::flag
integer::i
do i=1,n
  if(flag) a(i+protected)=1
enddo
do i=1,n
  a(i)=a(i)+1
enddo
end subroutine
end module
''',
    "guarded_host": '''module source_guarded_host
contains
subroutine guarded_host(a,n,flag,protected)
real(8),intent(inout)::a(:)
integer,intent(in)::n,protected
logical,intent(in)::flag
integer::i
if(flag) then
  do i=1,protected
    a(i)=1
  enddo
endif
do i=1,n
  a(i)=a(i)+1
enddo
end subroutine
end module
''',
}


def _run(command, *, directory, env=None):
    result = subprocess.run(command, cwd=directory, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shared_entries")
    nvcc = shutil.which("nvcc")
    host = shutil.which("g++-14") or shutil.which("g++")
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not nvcc or not host or not fortran:
        pytest.skip("CUDA/Fortran toolchain unavailable")
    checkout = directory / "independent-compiler"
    shutil.copytree(ROOT / "compiler", checkout / "compiler", ignore=shutil.ignore_patterns("__pycache__", ".*cache"))
    env = {**os.environ, "PYTHONPATH": str(checkout)}
    objects, reports = [], {}
    for name in [*SOURCES, "native_host_coherence", "native_device_bounds"]:
        source = directory / (name + ".f90")
        source.write_text(SOURCES[name] if name in SOURCES else
                          (ROOT / "compiler/tests/fixtures" / (name + ".f90")).read_text())
        output = directory / name
        report = json.loads(_run([sys.executable, "-m", "compiler", "--input", str(source), "--kernel", name,
                                 "--fallback", "host", "--gpu-policy", "sections", "--memory-model", "scoped",
                                 "--json", "--output-dir", str(output)], directory=directory, env=env).stdout)
        reports[name] = report
        assert report["scoped"]["abi_version"] == 1
        assert report["scoped"]["entry_abi_version"] == 2
        assert report["scoped"]["runtime"]["link_once"] is True
        assert report["scoped"]["runtime"]["runtime_id"] == reports["producer"]["scoped"]["runtime"]["runtime_id"]
        target = output / "shared.o"
        _run([nvcc, "-O2", "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fPIC,-fopenmp",
              "-c", str(output / "shared_entry.cu"), "-o", str(target)], directory=directory)
        if name == "producer":
            _run([fortran, "-c", "-std=f2018", str(output / "fort_scoped_memory.f90"), "-o", "memory-interface.o"],
                 directory=directory)
        _run([fortran, "-c", "-std=f2018", str(output / "shared_interface.f90"), "-o", str(output / "interface.o")],
             directory=directory)
        objects.append(str(target))
    runtime = directory / "producer"
    target = directory / "runtime.o"
    _run([nvcc, "-std=c++17", "-ccbin", host, "-arch=sm_86", "-Xcompiler=-fPIC,-fopenmp",
          "-DFORT_SCOPE_TEST_FAULTS", "-c", str(runtime / "scoped_runtime.cu"), "-o", str(target)], directory=directory)
    objects.append(str(target))
    probe = directory / "probe.cu"
    probe.write_text('#include <cuda_runtime.h>\nextern "C" int gpu_count() { int n=0; return cudaGetDeviceCount(&n)==cudaSuccess?n:0; }\n')
    _run([nvcc, "-ccbin", host, "-Xcompiler=-fPIC", "-c", str(probe), "-o", "probe.o"], directory=directory)
    objects.append(str(directory / "probe.o"))
    library = directory / "generated.so"
    _run([nvcc, "-ccbin", host, "-Xcompiler=-fopenmp", "-shared", *objects, "-o", str(library)], directory=directory)
    lib = c.CDLL(str(library))
    lib.fort_scope_error.restype = c.c_char_p
    signatures = {
        "create": [c.c_int, c.POINTER(TOKEN)],
        "register": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.c_int, c.POINTER(TOKEN)],
        "device_begin": [TOKEN, TOKEN, c.POINTER(Access), c.POINTER(c.c_void_p)],
        "host_begin": [TOKEN, TOKEN, c.POINTER(Access)],
        "device_end": [TOKEN, TOKEN], "host_end": [TOKEN, TOKEN], "close": [TOKEN],
        "stats_get": [TOKEN, c.POINTER(Stats)],
    }
    for name, signature in signatures.items():
        getattr(lib, "fort_scope_" + name).argtypes = signature
    for report in reports.values():
        shared = report["scoped"]
        function = getattr(lib, shared["entry"])
        dtype = {"integer": c.c_int, "real": c.c_double, "real32": c.c_float, "logical": c.c_bool}
        function.argtypes = [TOKEN, c.c_int, *[TOKEN for _ in shared["array_parameters"]],
                             *[c.POINTER(dtype[s["dtype"]]) for s in shared["scalar_parameters"]]]
    # Native references are compiled from the original source, not generated IR.
    reference = directory / "reference.f90"
    reference.write_text('''subroutine reference_host(a,value,nx,ny,nz) bind(C)
  use iso_c_binding
  use native_host_module
  integer(c_int),intent(in)::nx,ny,nz
  real(c_double),intent(inout)::a(nx,ny,nz)
  real(c_double),intent(in)::value
  call native_host_coherence(a,value,nx,ny,nz)
end subroutine
subroutine reference_bounds(bounds,a,n) bind(C)
  use iso_c_binding
  use native_device_bounds_module
  integer(c_int),intent(in)::n
  integer(c_int),intent(inout)::bounds(4)
  real(c_double),intent(inout)::a(8)
  call native_device_bounds(bounds,a,n)
end subroutine
''')
    native = directory / "reference.so"
    _run([fortran, "-shared", "-fPIC", str(directory / "native_host_coherence.f90"),
          str(directory / "native_device_bounds.f90"), str(reference), "-o", str(native)], directory=directory)
    return directory, lib, reports, c.CDLL(str(native))


def invoke(scope, reports, entry, buffers, scalars, mode):
    function = getattr(scope.lib, reports[entry]["scoped"]["entry"])
    scope.check(function(scope.handle, mode, *buffers, *[c.byref(s) for s in scalars]))


def require_gpu(lib):
    if not lib.gpu_count():
        pytest.skip("CUDA device unavailable")


@pytest.mark.cuda
def test_installed_plan_executes_independent_gpu_entries_with_shared_contents(generated):
    from compiler.tests.test_scoped_planning_runtime import Costs, Decision, costs

    _, lib, reports, _ = generated
    require_gpu(lib)
    scope = Scope(lib)
    n = c.c_int(16)
    a = (c.c_double * 16)(*[i*0.25+1 for i in range(16)])
    b, out = (c.c_double * 16)(*([-99]*16)), (c.c_double * 16)(*([-99]*16))
    handles = []
    for i, values in enumerate((a, b, out)):
        status, handle = scope.register(values, [16], identity=i+1, initialized=i == 0, lower=[-2])
        scope.check(status)
        handles.append(handle)
    lib.fort_scope_plan_reset.argtypes = [TOKEN]
    lib.fort_scope_plan_select.argtypes = [TOKEN, c.POINTER(Costs), c.c_int, c.POINTER(Decision)]
    scope.check(lib.fort_scope_plan_reset(scope.handle))
    for entry, bindings in (("producer", handles[:2]), ("consumer", handles)):
        public = reports[entry]["scoped"]["planning"]
        query = getattr(lib, public["entry"])
        query.argtypes = [TOKEN, *[TOKEN for _ in bindings], c.POINTER(c.c_int)]
        scope.check(query(scope.handle, *bindings, c.byref(n)))
    assert list(b) == list(out) == [-99]*16
    assert scope.stats().allocations == scope.stats().launches == 0
    # Deterministic synthetic costs force the GPU branch for correctness only;
    # they are never used as calibration or performance evidence.
    calibration, decision = costs(), Decision()
    scope.check(lib.fort_scope_plan_select(scope.handle, c.byref(calibration), 1, c.byref(decision)))
    assert decision.available
    assert (decision.gpu_units, decision.cpu_units) == (2, 0)
    invoke(scope, reports, "producer", handles[:2], [n], 2)
    invoke(scope, reports, "consumer", handles, [n], 2)
    assert (scope.stats().uploads, scope.stats().downloads) == (1, 0)
    for handle in handles[1:]:
        scope.cpu_begin(handle, access(flags=1))
        scope.cpu_end(handle)
    stats = scope.stats()
    for field in ("upload_bytes", "download_bytes", "uploads", "downloads", "launches", "allocations", "peak_device_bytes"):
        assert getattr(stats, field) == getattr(decision, field), field
    scope.close()
    assert list(b) == [2*a[i]+i-2 for i in range(16)]
    assert list(out) == [3*a[i]+i-2 for i in range(16)]


@pytest.mark.cuda
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_independent_generated_entries_and_cpu_mirrors(generated, mode):
    _, lib, reports, _ = generated
    if mode == 1:
        require_gpu(lib)
    scope = Scope(lib)
    n = c.c_int(16)
    a = (c.c_double * 16)(*[i*0.25+1 for i in range(16)])
    b, out = (c.c_double * 16)(), (c.c_double * 16)()
    handles = []
    for i, values in enumerate([a, b, out]):
        status, handle = scope.register(values, [16], identity=i+1, initialized=i == 0, lower=[-2])
        scope.check(status)
        handles.append(handle)
    invoke(scope, reports, "producer", handles[:2], [n], mode)
    invoke(scope, reports, "consumer", handles, [n], mode)
    before = scope.stats()
    assert before.uploads == int(mode == 1)
    assert before.downloads == 0
    scope.cpu_begin(handles[1], access(flags=1))
    expected_b = [2*a[i]+i-2 for i in range(16)]
    assert list(b) == expected_b
    scope.cpu_end(handles[1])
    invoke(scope, reports, "consumer", handles, [n], mode)
    assert scope.stats().uploads == before.uploads
    scope.cpu_begin(handles[1], access(flags=3))
    for i in range(16):
        b[i] *= 10
    scope.cpu_end(handles[1])
    invoke(scope, reports, "consumer", handles, [n], mode)
    stats = scope.stats()
    assert (stats.uploads, stats.downloads, stats.allocations) == ((2, 1, 3) if mode == 1 else (0, 0, 0))
    scope.close()
    assert list(out) == [a[i]+10*expected_b[i] for i in range(16)]


@pytest.mark.cuda
@pytest.mark.parametrize("mode", [0, 1])
def test_host_preparation_and_gpu_produced_bounds(generated, mode):
    _, lib, reports, native = generated
    if mode == 1:
        require_gpu(lib)
    scope = Scope(lib)
    shape = (6, 4, 5)
    host = (c.c_double * 120)(*[i*0.125 for i in range(120)])
    expected = (c.c_double * 120)(*host)
    value, nx, ny, nz = c.c_double(0.75), *[c.c_int(k) for k in shape]
    native.reference_host(expected, c.byref(value), c.byref(nx), c.byref(ny), c.byref(nz))
    status, buffer = scope.register(host, shape)
    scope.check(status)
    invoke(scope, reports, "native_host_coherence", [buffer], [value, nx, ny, nz], mode)
    scope.close()
    assert list(host) == list(expected)

    scope = Scope(lib)
    bounds, expected_bounds = (c.c_int * 4)(), (c.c_int * 4)()
    host, expected = (c.c_double * 8)(*range(8)), (c.c_double * 8)(*range(8))
    n = c.c_int(4)
    native.reference_bounds(expected_bounds, expected, c.byref(n))
    extents, lower = (SIZE * 1)(4), (c.c_int64 * 1)(1)
    layout = Layout(1, 3, 4, c.cast(bounds, c.c_void_p), extents, lower, 1)
    bound_handle = TOKEN()
    scope.check(lib.fort_scope_register(scope.handle, 2, 1, c.byref(layout), 0, c.byref(bound_handle)))
    status, buffer = scope.register(host, [8])
    scope.check(status)
    invoke(scope, reports, "native_device_bounds", [bound_handle, buffer], [n], mode)
    scope.close()
    assert list(bounds) == list(expected_bounds)
    assert list(host) == list(expected)


@pytest.mark.cuda
def test_native_continuation_after_gpu_work_without_replay(generated, monkeypatch):
    _, lib, reports, _ = generated
    require_gpu(lib)
    scope = Scope(lib)
    n = c.c_int(16)
    a = (c.c_double * 16)(*range(16))
    b, out = (c.c_double * 16)(), (c.c_double * 16)(*[99]*16)
    handles = []
    for i, values in enumerate([a, b, out]):
        status, handle = scope.register(values, [16], identity=i+1, initialized=i != 1, lower=[-2])
        scope.check(status)
        handles.append(handle)
    assert reports["two_stage"]["scoped"]["parallel_regions"] == 2
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_ALLOC_AFTER", "2")
    invoke(scope, reports, "two_stage", handles, [n], 1)
    stats = scope.stats()
    assert stats.launches == 1
    assert stats.allocations == 2
    scope.close()
    assert list(b) == [i+1 for i in range(16)]
    assert list(out) == [99, *[2*i for i in range(1, 16)]]




@pytest.mark.cuda
def test_unavailable_cuda_continues_natively_without_replay(generated):
    directory, _, reports, _ = generated
    script = directory / "no_device.py"
    producer = reports["producer"]["scoped"]["entry"]
    consumer = reports["consumer"]["scoped"]["entry"]
    script.write_text(f'''import ctypes as c
from compiler.tests.test_scoped_runtime import Scope, TOKEN, Layout
lib=c.CDLL({str(directory / 'generated.so')!r})
lib.fort_scope_error.restype=c.c_char_p
lib.fort_scope_create.argtypes=[c.c_int,c.POINTER(TOKEN)]
lib.fort_scope_register.argtypes=[TOKEN,TOKEN,TOKEN,c.POINTER(Layout),c.c_int,c.POINTER(TOKEN)]
lib.fort_scope_close.argtypes=[TOKEN]
lib.fort_scope_stats_get.argtypes=[TOKEN,c.c_void_p]
produce=getattr(lib,{producer!r})
consume=getattr(lib,{consumer!r})
produce.argtypes=[TOKEN,c.c_int,TOKEN,TOKEN,c.POINTER(c.c_int)]
consume.argtypes=[TOKEN,c.c_int,TOKEN,TOKEN,TOKEN,c.POINTER(c.c_int)]
scope=Scope(lib)
n=c.c_int(16)
a=(c.c_double*16)(*range(16))
b=(c.c_double*16)()
out=(c.c_double*16)()
handles=[]
for identity,values in enumerate([a,b,out],1):
    status,handle=scope.register(values,[16],identity=identity,initialized=identity==1,lower=[-2])
    scope.check(status)
    handles.append(handle)
scope.check(produce(scope.handle,1,*handles[:2],c.byref(n)))
scope.check(consume(scope.handle,1,*handles,c.byref(n)))
stats=scope.stats()
assert stats.uploads==stats.downloads==stats.allocations==stats.launches==0
scope.close()
assert list(b)==[3*i-2 for i in range(16)]
assert list(out)==[4*i-2 for i in range(16)]
''')
    # A fresh process prevents a CUDA device initialized by other tests from
    # surviving the visibility change.
    _run([sys.executable, str(script)], directory=ROOT,
         env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)})


@pytest.mark.cuda
def test_guarded_reference_abi_never_reads_protected_bound(generated):
    directory, _, reports, _ = generated
    entry = reports["guarded"]["scoped"]["entry"]
    # A child contains any SIGSEGV to this test process. The inaccessible bound
    # proves non-evaluation, rather than relying on a convenient sentinel value.
    script = directory / "protected.py"
    script.write_text(f'''import ctypes as c
from compiler.tests.test_scoped_runtime import Scope
lib=c.CDLL({str(directory / 'generated.so')!r})
lib.fort_scope_error.restype=c.c_char_p
from compiler.tests.test_scoped_runtime import TOKEN, Layout
lib.fort_scope_create.argtypes=[c.c_int,c.POINTER(TOKEN)]
lib.fort_scope_register.argtypes=[TOKEN,TOKEN,TOKEN,c.POINTER(Layout),c.c_int,c.POINTER(TOKEN)]
lib.fort_scope_close.argtypes=[TOKEN]
lib.fort_scope_stats_get.argtypes=[TOKEN,c.c_void_p]
function=getattr(lib,{entry!r})
function.argtypes=[TOKEN,c.c_int,TOKEN,c.POINTER(c.c_int),c.POINTER(c.c_int)]
libc=c.CDLL(None)
libc.mmap.restype=c.c_void_p
libc.mmap.argtypes=[c.c_void_p,c.c_size_t,c.c_int,c.c_int,c.c_int,c.c_long]
pointer=libc.mmap(None,4096,0,0x22,-1,0)
assert pointer and pointer!=c.c_void_p(-1).value
for mode in [0,1,2]:
    scope=Scope(lib)
    array=(c.c_double*16)()
    status,buffer=scope.register(array,[4,4],initialized=False)
    scope.check(status)
    n=c.c_int(0)
    scope.check(function(scope.handle,mode,buffer,c.byref(n),c.cast(pointer,c.POINTER(c.c_int))))
    stats=scope.stats()
    assert stats.uploads==stats.downloads==stats.allocations==stats.launches==0
    scope.close()
''')
    _run([sys.executable, str(script)], directory=ROOT,
         env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)})


@pytest.mark.cuda
@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("entry", ["guarded_value", "guarded_index", "guarded_host"])
def test_active_worker_keeps_conditional_scalar_protected(generated, mode, entry):
    directory, lib, reports, _ = generated
    if mode == 1:
        require_gpu(lib)
    public = reports[entry]["scoped"]
    assert [r["gpu_available"] for r in public["region_execution"]] == [entry == "guarded_host", True]
    assert public["region_execution"][0]["protected_scalars"] == ([] if entry == "guarded_host" else ["protected"])
    assert public["region_execution"][0]["native_footprints"] == (
        "whole-resource" if entry == "guarded_index" else "physical-sections")
    # The first mapped domain is active, unlike the zero-trip bound test above.
    # A value ABI would fault even though its original IF is false. The second
    # worker proves that protecting this input does not disable the whole entry.
    scalar_type = "c_double" if entry == "guarded_value" else "c_int"
    script = directory / f"protected-{entry}-{mode}.py"
    script.write_text(f'''import ctypes as c
from compiler.tests.test_scoped_runtime import Scope, TOKEN, Layout
lib=c.CDLL({str(directory / 'generated.so')!r})
lib.fort_scope_error.restype=c.c_char_p
lib.fort_scope_create.argtypes=[c.c_int,c.POINTER(TOKEN)]
lib.fort_scope_register.argtypes=[TOKEN,TOKEN,TOKEN,c.POINTER(Layout),c.c_int,c.POINTER(TOKEN)]
lib.fort_scope_close.argtypes=[TOKEN]
lib.fort_scope_stats_get.argtypes=[TOKEN,c.c_void_p]
function=getattr(lib,{public['entry']!r})
function.argtypes=[TOKEN,c.c_int,TOKEN,c.POINTER(c.c_int),c.POINTER(c.c_bool),c.POINTER(c.{scalar_type})]
libc=c.CDLL(None)
libc.mmap.restype=c.c_void_p
libc.mmap.argtypes=[c.c_void_p,c.c_size_t,c.c_int,c.c_int,c.c_int,c.c_long]
libc.munmap.argtypes=[c.c_void_p,c.c_size_t]
pointer=libc.mmap(None,4096,0,0x22,-1,0)
assert pointer and pointer!=c.c_void_p(-1).value
scope=Scope(lib)
array=(c.c_double*16)(*[7]*16)
status,buffer=scope.register(array,[16])
scope.check(status)
n,flag=c.c_int(16),c.c_bool(False)
scope.check(function(scope.handle,{mode},buffer,c.byref(n),c.byref(flag),c.cast(pointer,c.POINTER(c.{scalar_type}))))
assert scope.stats().launches=={int(mode == 1)}
scope.close()
assert list(array)==[8]*16
assert libc.munmap(pointer,4096)==0
''')
    _run([sys.executable, str(script)], directory=ROOT,
         env={**os.environ, "PYTHONPATH": str(ROOT)})


@pytest.mark.cuda
@pytest.mark.parametrize("mode", [0, 1])
def test_fortran_generated_modules_with_changing_shapes(generated, mode):
    directory, lib, reports, _ = generated
    if mode:
        require_gpu(lib)
    source = directory / "fortran_caller.f90"
    source.write_text(f'''program caller
  use iso_c_binding
  use fort_scoped_memory
  use {reports['producer']['scoped']['fortran_module']}, only: produce=>run
  use {reports['consumer']['scoped']['fortran_module']}, only: consume=>run
  implicit none
  real(c_double), allocatable, target :: a(:),b(:),output(:)
  integer(c_size_t),target :: extents(1)
  integer(c_int64_t),target :: bounds(1)=[-2_c_int64_t]
  integer(c_int64_t) :: context,ha,hb,ho
  integer(c_int) :: n,execution_mode
  integer :: iteration,i
  type(fort_scope_layout) :: layout
  type(fort_scope_stats) :: stats
  character(4) :: argument
  call get_command_argument(1,argument)
  read(argument,*) execution_mode
  do iteration=1,2
    n=8+8*iteration
    allocate(a(-2:n-3),b(-2:n-3),output(-2:n-3))
    do i=-2,n-3
      a(i)=real(i+3,c_double)*0.25_c_double
    enddo
    extents=n
    call check(fort_scope_create(0_c_int,context))
    layout=fort_scope_layout(1,FORT_SCOPE_REAL64,8_c_size_t,c_loc(a), &
                            c_loc(extents),c_loc(bounds),int(iteration,c_int64_t))
    call check(fort_scope_register(context,1_c_int64_t,int(iteration,c_int64_t),layout,1_c_int,ha))
    layout%host=c_loc(b)
    call check(fort_scope_register(context,2_c_int64_t,int(iteration,c_int64_t),layout,0_c_int,hb))
    layout%host=c_loc(output)
    call check(fort_scope_register(context,3_c_int64_t,int(iteration,c_int64_t),layout,0_c_int,ho))
    call check(produce(context,execution_mode,ha,hb,n))
    call check(consume(context,execution_mode,ha,hb,ho,n))
    call check(fort_scope_stats_get(context,stats))
    if (execution_mode==FORT_SCOPE_GPU) then
      if (stats%uploads/=1 .or. stats%downloads/=0 .or. stats%launches/=2) error stop 'transfer count'
    else
      if (stats%allocations/=0 .or. stats%launches/=0) error stop 'native CUDA activity'
    endif
    call check(fort_scope_close(context))
    do i=-2,n-3
      if (b(i)/=2*a(i)+real(i,c_double)) error stop 'producer field'
      if (output(i)/=3*a(i)+real(i,c_double)) error stop 'consumer field'
    enddo
    deallocate(a,b,output)
  enddo
contains
  subroutine check(status)
    integer(c_int),intent(in)::status
    if(status/=FORT_SCOPE_OK) then
      print *,status
      error stop 'runtime status'
    endif
  end subroutine
end program
''')
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    target = directory / ("fortran-caller-" + str(mode))
    _run([fortran, "-std=f2018", "-fcheck=all", str(source), str(directory / "generated.so"),
          "-Wl,-rpath," + str(directory), "-o", str(target)], directory=directory)
    _run([str(target), str(mode)], directory=directory)
