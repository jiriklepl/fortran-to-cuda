"""Native acquisition reuses real preparations without installing virtual futures."""
# ruff: noqa: F811 - imported pytest fixtures.
from __future__ import annotations

import ctypes as c
import shutil
import subprocess

import pytest

from compiler.tests.test_scoped_native_unit_handoff import (
    add,
    apply,
    begin_plan,
    bindings,
    choose,
    native_unit,  # noqa: F401
    register,
    workers,
)
from compiler.tests.test_scoped_native_unit_handoff_cuda import CLIENT as NATIVE_CLIENT
from compiler.tests.test_scoped_planning_runtime import planning  # noqa: F401
from compiler.tests.test_scoped_runtime import RUNTIME, Scope, access, runtime  # noqa: F401


def test_unqueued_native_batch_retains_internal_definitions_until_body(native_unit):
    scope = Scope(native_unit)
    a_values, a = register(scope, 1)
    b_values, b = register(scope, 2, initialized=False)
    c_values, c_handle = register(scope, 3)
    producer = bindings((a, access(flags=1)), (b, access(flags=6)))
    consumer = bindings((b, access(flags=1)), (c_handle, access(flags=3)))
    aggregate = bindings((a, access(flags=1)), (b, access(flags=6)), (c_handle, access(flags=3)))
    begin_plan(scope)
    add(scope, 101, producer)
    add(scope, 102, consumer)
    assert choose(scope).cpu_units == 2
    assert apply(scope, workers((101, producer), (102, consumer)), aggregate) == (0, 1)
    assert list(b_values) == list(range(8))
    assert native_unit.fort_scope_host_begin(scope.handle, a, c.byref(access(flags=1))) == 8
    for i in range(8):
        b_values[i] = 2 * a_values[i]
        c_values[i] += b_values[i]
    for handle in (a, b, c_handle):
        scope.cpu_end(handle)
    scope.close()
    assert list(b_values) == [2 * i for i in range(8)]
    assert list(c_values) == [3 * i for i in range(8)]


def test_unqueued_late_failure_does_not_download_then_valid_batch_waits_once(native_unit):
    scope = Scope(native_unit)
    arrays = []
    for identity in range(1, 4):
        values, handle = register(scope, identity)
        device = scope.gpu(handle, access(flags=3))
        for i in range(8):
            device[i] += 100 * identity
        scope.gpu_end(handle)
        arrays.append((values, handle))
    _, undefined = register(scope, 4, initialized=False)
    reads = bindings(*((handle, access(flags=1)) for _, handle in arrays))
    begin_plan(scope)
    add(scope, 111, reads)
    choose(scope)
    before = bytes(scope.stats())
    invalid = bindings(*((handle, access(flags=1)) for _, handle in arrays), (undefined, access(flags=1)))
    assert apply(scope, workers((111, reads)), invalid) == (7, 0)
    assert bytes(scope.stats()) == before
    for values, _ in arrays:
        assert list(values) == list(range(8))
    waits = scope.stats().waits
    assert apply(scope, workers((111, reads)), reads) == (0, 1)
    assert scope.stats().waits == waits + 1
    assert scope.stats().download_bytes == 3 * 64
    for identity, (values, handle) in enumerate(arrays, 1):
        assert list(values) == [100 * identity + i for i in range(8)]
        scope.cpu_end(handle)
    uploads = scope.stats().upload_bytes
    for _, handle in arrays:
        scope.gpu(handle, access(flags=1))
        scope.gpu_end(handle)
    assert scope.stats().upload_bytes == uploads
    scope.close()


PARTIAL = r'''
void partial_queue() {
    Fixture f;
    f.first=0;f.last=4;f.lo[0]=0;f.hi[0]=4;enqueue(f,0);
    f.first=4;f.last=8;f.lo[0]=4;f.hi[0]=8;enqueue(f,0);
    assert(queued(f).pending_tasks==2 && !f.stats().launches);
    size_t lower[1]={0},upper[1]={4};fort_scope_section left{lower,upper};
    fort_scope_access read{0,1,&left,0,nullptr,0,nullptr};
    fort_scope_plan_binding native{f.b,read};
    check(fort_scope_task_graph_enable_v1(f.context));
    check(fort_scope_plan_reset_mode(f.context,FORT_SCOPE_PLAN_CONTINUE));
    check(fort_scope_plan_add(f.context,FORT_SCOPE_PLAN_WORKER,211,&native,1,1e9,32,0));
    check(fort_scope_plan_validate(f.context));auto c=costs();fort_scope_plan_decision decision{};
    check(fort_scope_plan_select(f.context,&c,1,&decision));assert(decision.cpu_units==1 && !decision.gpu_units);
    fort_scope_native_worker_v1 worker{211,&native,1};int applied=0;
    check(fort_scope_plan_native_begin_v1(f.context,&worker,1,&native,1,&applied));assert(applied);
    assert(queued(f).pending_tasks==1 && f.stats().launches==1);
    for(int i=0;i<4;++i)assert(f.host_b[i]==i+2);
    for(int i=4;i<8;++i)assert(f.host_b[i]==0);
    check(fort_scope_host_end(f.context,f.b));
    // A virtual right-hand producer is still unlaunched, so the real root
    // must remain only partially defined/current after the native commit.
    assert(fort_scope_plan_host_current(f.context,f.b)==FORT_SCOPE_BOUNDARY);
    assert(queued(f).pending_tasks==1 && f.stats().upload_bytes==32 && f.stats().download_bytes==32);
    f.observe(f.b);
    for(int i=0;i<8;++i)assert(f.host_b[i]==i+2);
    assert(f.stats().upload_bytes==64 && f.stats().download_bytes==64);
    f.first=0;f.last=8;f.lo[0]=0;f.hi[0]=8;enqueue(f,1);f.observe(f.out);
    for(int i=0;i<8;++i)assert(f.host_out[i]==3*(i+2));
    assert(f.stats().upload_bytes==64 && f.stats().download_bytes==128);
    assert(f.stats().launches==3 && !queued(f).pending_tasks);
    check(fort_scope_close(f.context));
}
'''
CLIENT = NATIVE_CLIENT.replace('int main(int argc,char **argv) {', PARTIAL + 'int main(int argc,char **argv) {')
CLIENT = CLIENT.replace('const std::string mode=argv[1];', '''const std::string request=argv[1];
    const bool materialized=request=="materialized_read" || request=="materialized_write";
    const std::string mode=request=="materialized_read"?"read":request=="materialized_write"?"write":request;''')
CLIENT = CLIENT.replace('    Fixture f;\n    double original[8]{};', '''    if(mode=="partial_queue") {partial_queue();return 0;}
    Fixture f;
    double original[8]{};''')
CLIENT = CLIENT.replace('enqueue(f,0); assert(queued(f).pending_tasks==1 && !f.stats().launches);', '''enqueue(f,0); assert(queued(f).pending_tasks==1 && !f.stats().launches);
    if(materialized) check(fort_scope_graph_flush_v1(f.context));''')


@pytest.fixture(scope='module')
def reuse_cuda_binary(tmp_path_factory):
    nvcc = shutil.which('nvcc')
    host = shutil.which('g++-14') or shutil.which('g++')
    if not nvcc or not host:
        pytest.skip('CUDA toolkit unavailable')
    directory = tmp_path_factory.mktemp('native_prepare_reuse')
    source = directory / 'client.cu'
    source.write_text(CLIENT)
    objects = []
    for name, path in [('runtime', RUNTIME / 'scoped_runtime.cu'), ('client', source)]:
        target = directory / (name + '.o')
        result = subprocess.run([nvcc, '-std=c++17', '-ccbin', host, '-arch=sm_86', '-I', str(RUNTIME),
                                 '-c', str(path), '-o', str(target)],
                                capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout + result.stderr
        objects.append(str(target))
    target = directory / 'run'
    result = subprocess.run([nvcc, '-ccbin', host, *objects, '-o', str(target)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return target


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['materialized_read', 'materialized_write', 'partial_queue'])
def test_actual_cuda_native_preparation_reuse(reuse_cuda_binary, mode):
    result = subprocess.run([str(reuse_cuda_binary), mode], capture_output=True, text=True, timeout=30)
    if result.returncode == 77:
        pytest.skip('CUDA device unavailable')
    assert result.returncode == 0, result.stdout + result.stderr
