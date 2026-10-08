"""Public ABI acceptance: independent callers, partial mirrors, and lifetimes."""

from __future__ import annotations

import ctypes as c
import shutil
import subprocess
from pathlib import Path

import pytest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
SIZE = c.c_size_t
TOKEN = c.c_uint64


class Section(c.Structure):
    _fields_ = [("lower", c.POINTER(SIZE)), ("upper", c.POINTER(SIZE))]


class Access(c.Structure):
    _fields_ = [
        ("flags", c.c_uint32), ("read_count", SIZE), ("reads", c.POINTER(Section)),
        ("write_count", SIZE), ("writes", c.POINTER(Section)),
        ("overwrite_count", SIZE), ("overwrites", c.POINTER(Section)),
    ]


class Layout(c.Structure):
    _fields_ = [
        ("rank", c.c_uint32), ("type", c.c_uint32), ("element_bytes", SIZE),
        ("host", c.c_void_p), ("extents", c.POINTER(SIZE)), ("lower", c.POINTER(c.c_int64)),
        ("generation", TOKEN),
    ]


class Stats(c.Structure):
    _fields_ = [(name, TOKEN) for name in (
        "uploads", "downloads", "upload_bytes", "download_bytes", "allocations", "allocated_bytes",
        "peak_device_bytes", "launches", "waits", "reconciliations",
    )]


def access(*, read=(), write=(), overwrite=(), flags=0):
    result = Access(flags=flags)
    # Keep coordinate storage alive for the entire foreign call.
    result.references = []
    for label, boxes in (("read", read), ("write", write), ("overwrite", overwrite)):
        sections = []
        for lo, hi in boxes:
            lower, upper = (SIZE * len(lo))(*lo), (SIZE * len(hi))(*hi)
            result.references.extend([lower, upper])
            sections.append(Section(lower, upper))
        packed = (Section * len(sections))(*sections)
        result.references.append(packed)
        setattr(result, label + "s", packed)
        setattr(result, label + "_count", len(sections))
    return result


class Scope:
    def __init__(self, library):
        self.lib, self.handle = library, TOKEN()
        self.check(library.fort_scope_create(0, c.byref(self.handle)))
        self.references = []

    def check(self, status):
        assert status == 0, self.lib.fort_scope_error().decode()

    def register(self, values, shape, *, identity=1, generation=1, initialized=True, lower=None, defined=None):
        shape_values = (SIZE * len(shape))(*shape)
        bounds = (c.c_int64 * len(shape))(*(lower or [0] * len(shape)))
        layout = Layout(len(shape), 2, 8, c.cast(values, c.c_void_p) if values is not None else None,
                        shape_values, bounds, generation)
        buffer = TOKEN()
        self.references.extend([values, shape_values, bounds, layout])
        if defined is None:
            status = self.lib.fort_scope_register(self.handle, identity, generation, c.byref(layout),
                                                  initialized, c.byref(buffer))
        else:
            effects = access(read=defined)
            self.references.append(effects)
            status = self.lib.fort_scope_register_sections(self.handle, identity, generation, c.byref(layout),
                                                           effects.reads, effects.read_count, c.byref(buffer))
        return status, buffer

    def gpu(self, buffer, effects):
        pointer = c.c_void_p()
        self.check(self.lib.fort_scope_device_begin(self.handle, buffer, c.byref(effects), c.byref(pointer)))
        return c.cast(pointer, c.POINTER(c.c_double))

    def gpu_end(self, buffer):
        self.check(self.lib.fort_scope_device_end(self.handle, buffer))

    def cpu_begin(self, buffer, effects):
        self.check(self.lib.fort_scope_host_begin(self.handle, buffer, c.byref(effects)))

    def cpu_end(self, buffer):
        self.check(self.lib.fort_scope_host_end(self.handle, buffer))

    def stats(self):
        out = Stats()
        self.check(self.lib.fort_scope_stats_get(self.handle, c.byref(out)))
        return out

    def close(self):
        self.check(self.lib.fort_scope_close(self.handle))


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    directory = tmp_path_factory.mktemp("shared_runtime")
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    target = directory / "runtime.so"
    command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared", "-pthread",
               "-DFORT_SCOPE_CPU_TEST", "-x", "c++", str(RUNTIME / "scoped_runtime.cu"), "-o", str(target)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    lib = c.CDLL(str(target))
    lib.fort_scope_error.restype = c.c_char_p
    signatures = {
        "create": [c.c_int, c.POINTER(TOKEN)],
        "serial_caller": [],
        "register": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.c_int, c.POINTER(TOKEN)],
        "register_sections": [TOKEN, TOKEN, TOKEN, c.POINTER(Layout), c.POINTER(Section), SIZE, c.POINTER(TOKEN)],
        "forget_definition": [TOKEN, TOKEN],
        "set_device_budget": [TOKEN, SIZE],
        "layout_get": [TOKEN, TOKEN, c.POINTER(Layout)],
        "device_begin": [TOKEN, TOKEN, c.POINTER(Access), c.POINTER(c.c_void_p)],
        "host_begin": [TOKEN, TOKEN, c.POINTER(Access)],
        "device_end": [TOKEN, TOKEN], "host_end": [TOKEN, TOKEN], "cancel_access": [TOKEN, TOKEN],
        "unregister": [TOKEN, TOKEN], "close": [TOKEN], "abandon": [TOKEN], "wait": [TOKEN], "note_launch": [TOKEN],
        "stats_get": [TOKEN, c.POINTER(Stats)],
    }
    for name, signature in signatures.items():
        getattr(lib, "fort_scope_" + name).argtypes = signature
    return lib


@pytest.mark.native
def test_source_caller_without_openmp_support_selects_native(runtime):
    assert runtime.fort_scope_serial_caller() == 0


@pytest.mark.native
def test_read_mirror_retains_device_and_shares_identity(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(host, [8], lower=[-2])
    scope.check(status)
    status, duplicate = scope.register(host, [8], lower=[-2])
    scope.check(status)
    assert duplicate.value == buffer.value
    gpu = scope.gpu(buffer, access(flags=3))
    for i in range(8):
        gpu[i] += 10
    scope.gpu_end(buffer)
    scope.cpu_begin(buffer, access(flags=1))
    assert list(host) == [i + 10 for i in range(8)]
    scope.cpu_end(buffer)
    gpu = scope.gpu(duplicate, access(flags=1))
    assert gpu[7] == 17
    scope.gpu_end(duplicate)
    stats = scope.stats()
    assert (stats.uploads, stats.downloads, stats.allocations) == (1, 1, 1)
    scope.close()
    assert runtime.fort_scope_wait(scope.handle) == 2


@pytest.mark.native
def test_gpu_interior_cpu_opposite_faces_preserve_both(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 64)(*range(64))
    status, buffer = scope.register(host, [4, 4, 4], lower=[-2, -2, -2])
    scope.check(status)
    interior = [([1, 1, 1], [3, 3, 3])]
    gpu = scope.gpu(buffer, access(write=interior, overwrite=interior))
    for k in (1, 2):
        for j in (1, 2):
            for i in (1, 2):
                gpu[i+4*j+16*k] = 1000+i+4*j+16*k
    scope.gpu_end(buffer)
    faces = [([0, 0, 0], [4, 4, 1]), ([0, 0, 3], [4, 4, 4])]
    scope.cpu_begin(buffer, access(write=faces, overwrite=faces))
    for i in list(range(16)) + list(range(48, 64)):
        host[i] = -i-1
    scope.cpu_end(buffer)
    gpu = scope.gpu(buffer, access(flags=1))
    expected = list(range(64))
    for i in list(range(16)) + list(range(48, 64)):
        expected[i] = -i-1
    for k in (1, 2):
        for j in (1, 2):
            for i in (1, 2):
                expected[i+4*j+16*k] += 1000
    assert [gpu[i] for i in range(64)] == expected
    scope.gpu_end(buffer)
    scope.close()
    assert list(host) == expected


@pytest.mark.native
def test_conditional_holes_and_overwrite_skip_input(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(host, [8])
    scope.check(status)
    gpu = scope.gpu(buffer, access(write=[([2], [6])]))
    gpu[2], gpu[4] = 102, 104
    scope.gpu_end(buffer)
    scope.cpu_begin(buffer, access(read=[([1], [7])]))
    assert list(host) == [0, 1, 102, 3, 104, 5, 6, 7]
    scope.cpu_end(buffer)
    stats = scope.stats()
    assert stats.upload_bytes == stats.download_bytes == 4*8
    scope.close()

    scope = Scope(runtime)
    output = (c.c_double * 8)()
    status, buffer = scope.register(output, [8], initialized=False)
    scope.check(status)
    pointer = c.c_void_p()
    assert runtime.fort_scope_device_begin(scope.handle, buffer, c.byref(access(flags=1)), c.byref(pointer)) == 7
    gpu = scope.gpu(buffer, access(flags=6))
    for i in range(8):
        gpu[i] = i*2
    scope.gpu_end(buffer)
    assert scope.stats().upload_bytes == 0
    scope.close()
    assert list(output) == [i*2 for i in range(8)]


@pytest.mark.native
def test_fragmentation_reconciles_current_values(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 128)(*range(128))
    status, buffer = scope.register(host, [128])
    scope.check(status)
    # Alternate writers so enclosing stale-source copies would corrupt data.
    expected = list(range(128))
    for i in range(80):
        box = [([i], [i+1])]
        if i % 2:
            scope.cpu_begin(buffer, access(write=box, overwrite=box))
            host[i] = -100-i
            scope.cpu_end(buffer)
            expected[i] = -100-i
        else:
            gpu = scope.gpu(buffer, access(write=box, overwrite=box))
            gpu[i] = 1000+i
            scope.gpu_end(buffer)
            expected[i] = 1000+i
    assert scope.stats().reconciliations > 0
    scope.close()
    assert list(host) == expected


@pytest.mark.native
def test_generation_alias_empty_and_overflow(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*range(8))
    status, first = scope.register(host, [8])
    scope.check(status)
    assert scope.register(host, [8], identity=2)[0] == 3
    assert scope.register(host, [8], generation=2)[0] == 2
    scope.check(runtime.fort_scope_unregister(scope.handle, first))
    status, second = scope.register(host, [8], generation=2)
    scope.check(status)
    assert first.value != second.value
    assert runtime.fort_scope_host_begin(scope.handle, first, c.byref(access(flags=1))) == 2
    huge = (1 << (8*c.sizeof(SIZE))) - 1
    assert scope.register(host, [huge], identity=3)[0] == 1
    status, empty = scope.register(None, [huge, 0], identity=3)
    scope.check(status)
    gpu = scope.gpu(empty, access(flags=1))
    assert not gpu
    scope.gpu_end(empty)
    assert scope.stats().allocations == 0
    scope.close()


@pytest.mark.native
def test_partial_initialization_limit_is_preexecution_boundary(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 80)()
    status, buffer = scope.register(host, [80], initialized=False)
    scope.check(status)
    for i in range(32):
        box = [([2*i], [2*i+1])]
        scope.cpu_begin(buffer, access(write=box, overwrite=box))
        host[2*i] = i+10
        scope.cpu_end(buffer)
    box = [([64], [65])]
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(access(write=box, overwrite=box))) == 5
    # Failed preparation did not commit an unexecuted write or leave an access open.
    scope.close()
    assert [host[2*i] for i in range(32)] == [i+10 for i in range(32)]


@pytest.mark.native
def test_resource_failure_exports_existing_work_without_replay(runtime, monkeypatch):
    scope = Scope(runtime)
    host, other = (c.c_double * 8)(*range(8)), (c.c_double * 8)()
    status, buffer = scope.register(host, [8])
    scope.check(status)
    gpu = scope.gpu(buffer, access(flags=3))
    for i in range(8):
        gpu[i] += 10
    scope.gpu_end(buffer)
    status, unused = scope.register(other, [8], identity=2, initialized=False)
    scope.check(status)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_ALLOC", "1")
    pointer = c.c_void_p()
    assert runtime.fort_scope_device_begin(scope.handle, unused, c.byref(access(flags=6)), c.byref(pointer)) == 4
    # Materialize the previous result and continue at the failed operation.
    scope.cpu_begin(buffer, access(flags=1))
    assert list(host) == [i+10 for i in range(8)]
    scope.cpu_end(buffer)
    scope.cpu_begin(unused, access(flags=6))
    for i in range(8):
        other[i] = host[i]*2
    scope.cpu_end(unused)
    scope.close()
    assert list(other) == [2*(i+10) for i in range(8)]


@pytest.mark.native
def test_completion_failure_poisoned_scope_cannot_continue(runtime, monkeypatch):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(host, [8])
    scope.check(status)
    gpu = scope.gpu(buffer, access(flags=3))
    gpu[0] = 100
    scope.gpu_end(buffer)
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_WAIT", "1")
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(access(flags=1))) == 6
    assert runtime.fort_scope_close(scope.handle) == 6
    assert runtime.fort_scope_device_begin(scope.handle, buffer, c.byref(access(flags=1)), c.byref(c.c_void_p())) == 6
    scope.check(runtime.fort_scope_abandon(scope.handle))
    assert runtime.fort_scope_wait(scope.handle) == 2


@pytest.mark.native
def test_independent_concurrent_contexts(runtime):
    from concurrent.futures import ThreadPoolExecutor

    def run(seed):
        scope = Scope(runtime)
        host = (c.c_double * 32)(*[seed+i for i in range(32)])
        status, buffer = scope.register(host, [32])
        scope.check(status)
        gpu = scope.gpu(buffer, access(flags=3))
        for i in range(32):
            gpu[i] *= 2
        scope.gpu_end(buffer)
        scope.close()
        return list(host)

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(12)))
    assert results == [[2*(seed+i) for i in range(32)] for seed in range(12)]


@pytest.mark.native
def test_fortran_public_interface_and_negative_bounds(runtime, tmp_path):
    fortran = shutil.which("gfortran-15") or shutil.which("gfortran")
    if not fortran:
        pytest.skip("gfortran unavailable")
    source = tmp_path / "caller.f90"
    source.write_text('''program caller
  use iso_c_binding
  use fort_scoped_memory
  implicit none
  real(c_double), target :: host(-2:5)
  real(c_double), pointer :: device(:)
  integer(c_size_t), target :: extents(1)=[8_c_size_t]
  integer(c_int64_t), target :: lower(1)=[-2_c_int64_t]
  integer(c_int64_t) :: context, buffer
  type(fort_scope_layout) :: layout
  type(fort_scope_access) :: effects
  type(fort_scope_stats) :: stats
  type(c_ptr) :: pointer
  integer :: i
  do i=-2,5
    host(i)=real(i,c_double)
  end do
  call check(fort_scope_create(0_c_int,context))
  layout=fort_scope_layout(1, FORT_SCOPE_REAL64, 8_c_size_t, c_loc(host), &
                          c_loc(extents), c_loc(lower), 1_c_int64_t)
  call check(fort_scope_register(context,1_c_int64_t,1_c_int64_t,layout,1_c_int,buffer))
  effects%flags=ior(FORT_SCOPE_READ_ALL,FORT_SCOPE_WRITE_ALL)
  call check(fort_scope_device_begin(context,buffer,effects,pointer))
  call c_f_pointer(pointer,device,[8])
  device=device+10
  call check(fort_scope_device_end(context,buffer))
  effects%flags=FORT_SCOPE_READ_ALL
  call check(fort_scope_host_begin(context,buffer,effects))
  do i=-2,5
    if (host(i)/=real(i+10,c_double)) error stop 'incorrect mirror'
  end do
  call check(fort_scope_host_end(context,buffer))
  call check(fort_scope_device_begin(context,buffer,effects,pointer))
  call check(fort_scope_device_end(context,buffer))
  call check(fort_scope_stats_get(context,stats))
  if (stats%uploads/=1 .or. stats%downloads/=1) error stop 'redundant copies'
  call check(fort_scope_close(context))
contains
  subroutine check(status)
    integer(c_int), intent(in) :: status
    if (status/=FORT_SCOPE_OK) error stop 'runtime status'
  end subroutine
end program
''')
    command = [fortran, "-std=f2018", "-fcheck=all", str(RUNTIME / "scoped_memory.f90"), str(source),
               runtime._name, "-Wl,-rpath," + str(Path(runtime._name).parent), "-o", "caller"]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(tmp_path / "caller")], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr



@pytest.mark.native
def test_partial_initialization_never_copies_or_reads_undefined_halos(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(101, 102, 103, 104, 105, 106, 107, 108)
    defined = [([2], [6])]
    status, buffer = scope.register(host, [8], defined=defined, lower=[-2])
    scope.check(status)
    pointer = scope.gpu(buffer, access(read=defined, write=defined))
    for i in range(2, 6):
        pointer[i] += 10
    scope.gpu_end(buffer)
    assert scope.stats().upload_bytes == 4*8
    unknown = access(read=[([0], [1])])
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(unknown)) == 7
    # Re-registering cannot turn undefined sections into current host data.
    status, duplicate = scope.register(host, [8], lower=[-2], initialized=True)
    scope.check(status)
    assert duplicate.value == buffer.value
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(unknown)) == 7
    scope.close()
    assert list(host) == [101, 102, 113, 114, 115, 116, 107, 108]


@pytest.mark.native
def test_definition_change_discards_device_values_and_reuses_allocation(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(host, [8])
    scope.check(status)
    device = scope.gpu(buffer, access(flags=6))
    for i in range(8):
        device[i] = 1000+i
    scope.gpu_end(buffer)
    scope.check(runtime.fort_scope_forget_definition(scope.handle, buffer))
    assert scope.stats().downloads == 0
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(access(flags=1))) == 7
    middle = [([2], [6])]
    device = scope.gpu(buffer, access(write=middle, overwrite=middle))
    for i in range(2, 6):
        device[i] = 2000+i
    scope.gpu_end(buffer)
    assert scope.stats().allocations == 1
    scope.close()
    assert list(host) == [0, 1, 2002, 2003, 2004, 2005, 6, 7]


@pytest.mark.native
def test_definition_change_cannot_cancel_a_prepared_operation(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*range(8))
    status, buffer = scope.register(host, [8])
    scope.check(status)
    scope.cpu_begin(buffer, access(flags=3))
    assert runtime.fort_scope_forget_definition(scope.handle, buffer) == 8
    scope.cpu_end(buffer)
    scope.close()


@pytest.mark.native
def test_invalid_partial_initialization_is_rejected_before_registration(runtime):
    scope = Scope(runtime)
    host = (c.c_double * 8)(*range(8))
    status, _ = scope.register(host, [8], defined=[([0], [9])])
    assert status == 1
    status, buffer = scope.register(host, [8], defined=[])
    scope.check(status)
    assert runtime.fort_scope_host_begin(scope.handle, buffer, c.byref(access(flags=1))) == 7
    scope.close()



@pytest.mark.native
def test_declared_device_budget_prevents_allocation_and_keeps_native_continuation(runtime):
    scope = Scope(runtime)
    host, other = (c.c_double * 8)(*range(8)), (c.c_double * 8)()
    status, first = scope.register(host, [8])
    scope.check(status)
    status, second = scope.register(other, [8], identity=2, initialized=False)
    scope.check(status)
    scope.check(runtime.fort_scope_set_device_budget(scope.handle,64))
    device=scope.gpu(first,access(flags=3))
    for i in range(8):
        device[i]+=100
    scope.gpu_end(first)
    assert runtime.fort_scope_set_device_budget(scope.handle,128)==8
    assert runtime.fort_scope_device_begin(scope.handle,second,c.byref(access(flags=6)),c.byref(c.c_void_p()))==4
    assert scope.stats().allocated_bytes==64
    scope.cpu_begin(first,access(flags=1))
    scope.cpu_end(first)
    scope.cpu_begin(second,access(flags=6))
    for i in range(8):
        other[i]=host[i]*2
    scope.cpu_end(second)
    scope.close()
    assert list(other)==[2*(i+100) for i in range(8)]


@pytest.mark.native
def test_zero_device_budget_never_initializes_cuda(runtime):
    scope=Scope(runtime)
    host=(c.c_double * 8)(*range(8))
    status,buffer=scope.register(host,[8])
    scope.check(status)
    scope.check(runtime.fort_scope_set_device_budget(scope.handle,0))
    assert runtime.fort_scope_device_begin(scope.handle,buffer,c.byref(access(flags=1)),c.byref(c.c_void_p()))==4
    assert scope.stats().allocations==0
    scope.cpu_begin(buffer,access(flags=1))
    scope.cpu_end(buffer)
    scope.close()
