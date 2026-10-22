"""Original source consumes only an exactly matched current all-CPU unit."""
# ruff: noqa: F811 - imported pytest fixtures.
from __future__ import annotations

import ctypes as c
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_scoped_continuation_runtime import Report, consume
from compiler.tests.test_scoped_graph_queue import QueueStats, enqueue, reserve, stats
from compiler.tests.test_scoped_graph_replay import Task
from compiler.tests.test_scoped_planning_runtime import Binding, costs, planning, select  # noqa: F401
from compiler.tests.test_scoped_runtime import SIZE, TOKEN, Scope, access, runtime  # noqa: F401


class Worker(c.Structure):
    _fields_ = [("unit", TOKEN), ("bindings", c.POINTER(Binding)), ("binding_count", SIZE)]


def bindings(*items):
    return (Binding * len(items))(*(Binding(handle, effects) for handle, effects in items))


def workers(*items):
    return (Worker * len(items))(*(Worker(unit, effects, len(effects)) for unit, effects in items))


@pytest.fixture
def native_unit(planning):
    signatures = {
        "plan_native_begin_v1": [TOKEN, c.POINTER(Worker), SIZE, c.POINTER(Binding), SIZE, c.POINTER(c.c_int)],
        "task_graph_enable_v1": [TOKEN], "plan_reset_mode": [TOKEN, c.c_uint32],
        "plan_report_v2": [TOKEN, c.POINTER(Report)],
        "plan_team_entry_v1": [TOKEN], "plan_team_native_call_v1": [TOKEN],
        "device_reserve_v1": [TOKEN, TOKEN, c.POINTER(c.c_void_p)],
        "graph_enqueue_v1": [TOKEN, TOKEN, c.POINTER(Task), c.POINTER(SIZE)],
        "graph_queue_stats_get_v1": [TOKEN, c.POINTER(QueueStats)],
        "graph_queue_cancel_v1": [TOKEN], "cancel_access": [TOKEN, TOKEN],
    }
    for name, signature in signatures.items():
        getattr(planning, "fort_scope_" + name).argtypes = signature
    return planning


def register(scope, identity, *, initialized=True):
    values = (c.c_double * 8)(*range(8))
    status, handle = scope.register(values, [8], identity=identity, initialized=initialized)
    scope.check(status)
    return values, handle


def begin_plan(scope):
    scope.check(scope.lib.fort_scope_task_graph_enable_v1(scope.handle))
    scope.check(scope.lib.fort_scope_plan_reset_mode(scope.handle, 1))


def add(scope, unit, effects, *, gpu=False):
    scope.check(scope.lib.fort_scope_plan_add(scope.handle, 1, unit, effects, len(effects), 1e9, 64, gpu))


def choose(scope):
    scope.check(scope.lib.fort_scope_plan_validate(scope.handle))
    return select(scope, costs())


def apply(scope, expected, native):
    applied = c.c_int(99)
    status = scope.lib.fort_scope_plan_native_begin_v1(
        scope.handle, expected, len(expected), native, len(native), c.byref(applied))
    return status, applied.value


def test_original_native_unit_commits_only_body_and_retains_gpu_suffix(native_unit):
    scope = Scope(native_unit)
    values, a = register(scope, 1, initialized=False)
    producer = bindings((a, access(flags=6)))
    consumer = bindings((a, access(flags=1)))
    begin_plan(scope)
    scope.check(native_unit.fort_scope_plan_team_entry_v1(scope.handle))
    add(scope, 11, producer)
    scope.check(native_unit.fort_scope_plan_team_native_call_v1(scope.handle))
    add(scope, 12, consumer)
    add(scope, 13, consumer, gpu=True)
    decision = choose(scope)
    assert (decision.cpu_units, decision.gpu_units) == (2, 1)
    assert apply(scope, workers((11, producer), (12, consumer)), producer) == (0, 1)
    # Matching worker metadata did not execute or commit a predicted producer.
    assert list(values) == list(range(8))
    assert scope.stats().uploads == scope.stats().downloads == scope.stats().allocations == 0
    assert native_unit.fort_scope_host_begin(scope.handle, a, c.byref(access(flags=1))) == 8
    for i in range(8):
        values[i] = 100 + i
    scope.cpu_end(a)
    consume(scope, 13, consumer, 1)
    device = scope.gpu(a, access(flags=1))
    assert list(device[:8]) == [100 + i for i in range(8)]
    scope.gpu_end(a)
    report = Report()
    scope.check(native_unit.fort_scope_plan_report_v2(scope.handle, c.byref(report)))
    assert report.available == report.owner_available == 0
    scope.close()


@pytest.mark.parametrize("change", ["late_id", "late_effects", "too_many"])
def test_late_range_mismatch_consumes_no_prefix(native_unit, change):
    scope = Scope(native_unit)
    _, a = register(scope, 1)
    first = bindings((a, access(read=[([0], [4])])))
    second = bindings((a, access(read=[([4], [8])])))
    begin_plan(scope)
    add(scope, 21, first)
    add(scope, 22, second)
    choose(scope)
    expected = workers((21, first), (23 if change == "late_id" else 22,
                                    first if change == "late_effects" else second))
    if change == "too_many":
        expected = workers((21, first), (22, second), (23, second))
    before = bytes(scope.stats())
    assert apply(scope, expected, bindings((a, access(flags=1)))) == (9, 0)
    assert bytes(scope.stats()) == before
    consume(scope, 21, first, 0)
    consume(scope, 22, second, 0)
    scope.close()


def test_current_mixed_range_declines_without_consuming_or_acquiring(native_unit):
    scope = Scope(native_unit)
    _, a = register(scope, 1)
    read = bindings((a, access(flags=1)))
    begin_plan(scope)
    add(scope, 31, read)
    add(scope, 32, read, gpu=True)
    assert choose(scope).gpu_units == 1
    # Native effects are not evaluated when the exact range has GPU work.
    bad_native = bindings((TOKEN(2**63), access(flags=1)))
    assert apply(scope, workers((31, read), (32, read)), bad_native) == (0, 0)
    consume(scope, 31, read, 0)
    consume(scope, 32, read, 1)
    scope.close()


@pytest.mark.parametrize("failure", ["undefined", "alias", "stale"])
def test_invalid_native_batch_prepares_nothing_and_preserves_choices(native_unit, failure):
    scope = Scope(native_unit)
    values, a = register(scope, 1)
    _, b = register(scope, 2, initialized=False)
    device = scope.gpu(a, access(flags=3))
    device[0] = 999
    scope.gpu_end(a)
    expected = bindings((a, access(flags=1)))
    begin_plan(scope)
    add(scope, 41, expected)
    choose(scope)
    second = a if failure == "alias" else TOKEN(2**63) if failure == "stale" else b
    native = bindings((a, access(flags=1)), (second, access(flags=1)))
    before = bytes(scope.stats())
    assert apply(scope, workers((41, expected)), native) == (
        {"undefined": 7, "alias": 3, "stale": 2}[failure], 0)
    assert bytes(scope.stats()) == before
    assert values[0] == 0
    # A failed later binding neither acquired A nor consumed the worker.
    consume(scope, 41, expected, 0)
    scope.cpu_begin(a, access(flags=1))
    assert values[0] == 999
    scope.cpu_end(a)
    scope.close()


def test_native_current_range_after_completed_prefix_preserves_read_mirrors(native_unit):
    scope = Scope(native_unit)
    values, a = register(scope, 1)
    device = scope.gpu(a, access(flags=3))
    device[0] = 77
    scope.gpu_end(a)
    read = bindings((a, access(flags=1)))
    begin_plan(scope)
    add(scope, 51, read)
    add(scope, 52, read)
    add(scope, 53, read, gpu=True)
    assert choose(scope).gpu_units == 1
    consume(scope, 51, read, 0)
    scope.cpu_begin(a, access(flags=1))
    scope.cpu_end(a)
    assert apply(scope, workers((52, read)), read) == (0, 1)
    assert values[0] == 77
    scope.cpu_end(a)
    before = scope.stats()
    consume(scope, 53, read, 1)
    device = scope.gpu(a, access(flags=1))
    assert device[0] == 77
    scope.gpu_end(a)
    assert scope.stats().upload_bytes == before.upload_bytes
    assert scope.stats().download_bytes == 64
    scope.close()


def test_native_unit_preserves_independent_accepted_queue(native_unit):
    scope = Scope(native_unit)
    _, queued = reserve(scope, 1)
    host, a = register(scope, 2)
    from compiler.tests.test_scoped_access_batch import pack

    scope.check(enqueue(scope, 61, pack((queued, access(flags=6)))))
    read = bindings((a, access(flags=1)))
    queued_read = bindings((queued, access(flags=1)))
    begin_plan(scope)
    add(scope, 62, read)
    add(scope, 63, queued_read, gpu=True)
    assert choose(scope).gpu_units == 1
    before = bytes(stats(scope))
    assert apply(scope, workers((62, read)), read) == (0, 1)
    assert bytes(stats(scope)) == before
    assert stats(scope).pending_tasks == 1
    assert list(host) == list(range(8))
    scope.cpu_end(a)
    consume(scope, 63, queued_read, 1)
    scope.check(native_unit.fort_scope_graph_queue_cancel_v1(scope.handle))
    scope.close()


def test_empty_unit_does_not_consume_future_worker(native_unit):
    scope = Scope(native_unit)
    _, a = register(scope, 1)
    read = bindings((a, access(flags=1)))
    begin_plan(scope)
    add(scope, 71, read)
    choose(scope)
    assert apply(scope, workers(), bindings()) == (0, 0)
    consume(scope, 71, read, 0)
    scope.close()


def test_prepared_access_rejection_leaves_current_choice_unconsumed(native_unit):
    scope = Scope(native_unit)
    _, a = register(scope, 1)
    read = bindings((a, access(flags=1)))
    begin_plan(scope)
    add(scope, 81, read)
    choose(scope)
    scope.cpu_begin(a, access(flags=1))
    assert apply(scope, workers((81, read)), read) == (8, 0)
    scope.check(native_unit.fort_scope_cancel_access(scope.handle, a))
    consume(scope, 81, read, 0)
    scope.close()


def test_invalid_late_native_read_does_not_submit_queued_producer(native_unit):
    scope = Scope(native_unit)
    _, queued = reserve(scope, 1)
    _, undefined = register(scope, 2, initialized=False)
    from compiler.tests.test_scoped_access_batch import pack

    scope.check(enqueue(scope, 91, pack((queued, access(flags=6)))))
    read = bindings((queued, access(flags=1)))
    begin_plan(scope)
    add(scope, 92, read)
    choose(scope)
    before = bytes(stats(scope)), bytes(scope.stats())
    native = bindings((queued, access(flags=1)), (undefined, access(flags=1)))
    assert apply(scope, workers((92, read)), native) == (7, 0)
    assert (bytes(stats(scope)), bytes(scope.stats())) == before
    consume(scope, 92, read, 0)
    scope.check(native_unit.fort_scope_graph_queue_cancel_v1(scope.handle))
    scope.close()


def test_collector_owns_coordinates_across_lexical_blocks_and_growth(native_unit, tmp_path):
    compiler = shutil.which("g++")
    if not compiler:
        pytest.skip("g++ unavailable")
    source = tmp_path / "collector.cpp"
    source.write_text('''#include "view_entry.hpp"
extern "C" int collect(fort_scope_t context,fort_buffer_t handle,int *applied) {
    fort_scoped::NativeWorkerCollector collector(context);
    for(size_t i=0;i<24;++i) {
        size_t lower=1,upper=7;
        fort_scope_section section{&lower,&upper};
        fort_scope_access access{}; access.read_count=1; access.reads=&section;
        fort_scoped::AccessBatch<1> batch(context);
        int status=batch.add(handle,access); if(status) return status;
        status=collector.add(100+i,batch); if(status) return status;
        lower=0; upper=8; // stack metadata dies and is reused after each iteration
    }
    fort_scope_access read{}; read.flags=FORT_SCOPE_READ_ALL;
    fort_scope_plan_binding native{handle,read};
    return collector.begin(&native,1,applied);
}
''')
    target = tmp_path / "collector.so"
    result = subprocess.run([
        compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared", "-I",
        str(Path(__file__).resolve().parents[1] / "runtime"), str(source),
        native_unit._name, "-o", str(target),
    ], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    glue = c.CDLL(str(target))
    glue.collect.argtypes = [TOKEN, TOKEN, c.POINTER(c.c_int)]
    scope = Scope(native_unit)
    _, a = register(scope, 1)
    expected = bindings((a, access(read=[([1], [7])])))
    begin_plan(scope)
    for unit in range(100, 124):
        add(scope, unit, expected)
    choose(scope)
    applied = c.c_int()
    scope.check(glue.collect(scope.handle, a, c.byref(applied)))
    assert applied.value == 1
    scope.cpu_end(a)
    scope.close()
