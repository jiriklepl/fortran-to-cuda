"""Conservative entry publication retains replicas and initialized holes."""
# ruff: noqa: F811
import ctypes as c
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_scoped_runtime import RUNTIME, TOKEN, Scope, access, runtime  # noqa: F401


@pytest.fixture
def observed(runtime):
    runtime.fort_scope_observe_all_v1.argtypes = [TOKEN]
    return runtime


@pytest.mark.native
def test_all_live_buffers_publish_once_and_keep_read_replicas(observed):
    scope = Scope(observed)
    arrays = []
    middle = [([2], [6])]
    for identity in (3, 1, 2):
        host = (c.c_double * 8)(*range(8))
        status, handle = scope.register(host, [8], identity=identity, initialized=False)
        scope.check(status)
        device = scope.gpu(handle, access(write=middle, overwrite=middle))
        for i in range(2, 6):
            device[i] = identity * 100 + i
        scope.gpu_end(handle)
        arrays.append((identity, host, handle))
    before = scope.stats()
    scope.check(observed.fort_scope_observe_all_v1(scope.handle))
    after = scope.stats()
    assert after.downloads == before.downloads + 3
    assert after.waits == before.waits + 1
    for identity, host, handle in arrays:
        assert list(host) == [0, 1, *[identity * 100 + i for i in range(2, 6)], 6, 7]
        assert observed.fort_scope_host_begin(scope.handle, handle, c.byref(access(flags=1))) == 7
        scope.gpu(handle, access(read=middle))
        scope.gpu_end(handle)
    scope.check(observed.fort_scope_observe_all_v1(scope.handle))
    assert scope.stats().uploads == before.uploads
    assert scope.stats().downloads == after.downloads
    assert scope.stats().allocations == before.allocations
    scope.close()


@pytest.mark.native
def test_empty_observation_and_stale_context(observed):
    scope = Scope(observed)
    before = bytes(scope.stats())
    scope.check(observed.fort_scope_observe_all_v1(scope.handle))
    assert bytes(scope.stats()) == before
    scope.close()
    assert observed.fort_scope_observe_all_v1(scope.handle) == 2


@pytest.mark.native
def test_active_access_rejected_without_publishing_an_earlier_buffer(observed):
    scope = Scope(observed)
    host = (c.c_double * 8)(*range(8))
    status, first = scope.register(host, [8])
    scope.check(status)
    scope.gpu(first, access(flags=3))[0] = 99
    scope.gpu_end(first)
    other = (c.c_double * 8)()
    status, second = scope.register(other, [8], identity=2)
    scope.check(status)
    scope.gpu(second, access(flags=1))
    before = bytes(scope.stats())
    assert observed.fort_scope_observe_all_v1(scope.handle) == 8
    assert bytes(scope.stats()) == before
    assert host[0] == 0
    scope.gpu_end(second)
    scope.check(observed.fort_scope_observe_all_v1(scope.handle))
    assert host[0] == 99
    scope.close()


@pytest.mark.native
def test_fortran_original_bounds_follow_publication_and_unreached_descriptors_stay_unread(runtime, tmp_path):
    """Real Fortran entry semantics, with the explicit simulated device backend."""
    from compiler.scopes.specification_entry import publication

    compiler = shutil.which("gfortran-15") or shutil.which("gfortran")
    if compiler is None:
        pytest.skip("GNU Fortran unavailable")
    source = tmp_path / "entry.f90"
    source.write_text("""module original_metadata
use iso_c_binding
implicit none
integer(c_int),allocatable,target::tile(:)
integer::seen=0
contains
subroutine child()
real(8)::scratch(tile(1))
seen=size(scratch)
end subroutine
end module
program entry
use original_metadata
use fort_scoped_memory
implicit none
integer(c_int64_t)::context=0,handle=0
integer(c_int64_t),target::lower(1)=[1]
integer(c_size_t),target::extent(1)=[1]
type(fort_scope_layout)::layout
type(fort_scope_access)::access
type(fort_scope_stats)::before,after
type(c_ptr)::pointer
integer(c_int),pointer::device(:)
integer::i
character(8)::mode
call get_command_argument(1,mode)
! False and zero-trip paths must not inspect the unallocated bound table.
do i=1,0
""" + "\n".join(publication("context")) + """
call child()
enddo
if(allocated(tile).or.seen/=0)error stop 'unreached descriptor'
allocate(tile(1));tile=2
call check(fort_scope_create(0_c_int,context))
layout%rank=1;layout%element_type=FORT_SCOPE_INTEGER32;layout%element_bytes=4
layout%host=c_loc(tile);layout%extents=c_loc(extent);layout%lower_bounds=c_loc(lower);layout%generation=1
call check(fort_scope_register(context,1_c_int64_t,1_c_int64_t,layout,1_c_int,handle))
access%flags=ior(FORT_SCOPE_WRITE_ALL,FORT_SCOPE_OVERWRITE_ALL)
call check(fort_scope_device_begin(context,handle,access,pointer))
call c_f_pointer(pointer,device,[1]);device=5
call check(fort_scope_device_end(context,handle))
if(tile(1)/=2)error stop 'device backend is not separate'
call check(fort_scope_stats_get(context,before))
if(mode=='fail')then
call check(fort_scope_close(context))
endif
""" + "\n".join(publication("context")) + """
call child()
if(seen/=5.or.tile(1)/=5)error stop 'stale specification bound'
access%flags=FORT_SCOPE_READ_ALL
call check(fort_scope_device_begin(context,handle,access,pointer))
call check(fort_scope_device_end(context,handle))
call check(fort_scope_stats_get(context,after))
if(after%uploads/=before%uploads.or.after%downloads/=before%downloads+1)error stop 'read replica lost'
if(after%allocations/=before%allocations)error stop 'allocation identity lost'
call check(fort_scope_close(context))
print *, 'ORIGINAL_ENTRY_COHERENT'
contains
subroutine check(status)
integer(c_int),value::status
if(status/=FORT_SCOPE_OK)error stop 'runtime operation failed'
end subroutine
end program
""")
    binary = tmp_path / "entry"
    library = Path(runtime._name)
    command = [compiler, "-O0", "-fcheck=all", str(RUNTIME / "scoped_memory.f90"), str(source),
               str(library), "-Wl,-rpath," + str(library.parent), "-lstdc++", "-pthread", "-o", str(binary)]
    built = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stdout + built.stderr
    good = subprocess.run([str(binary), "pass"], capture_output=True, text=True, timeout=15)
    assert good.returncode == 0, good.stdout + good.stderr
    assert "ORIGINAL_ENTRY_COHERENT" in good.stdout
    failed = subprocess.run([str(binary), "fail"], capture_output=True, text=True, timeout=15)
    assert failed.returncode != 0
    assert "original specification publication failed" in failed.stderr
    assert "ORIGINAL_ENTRY_COHERENT" not in failed.stdout
