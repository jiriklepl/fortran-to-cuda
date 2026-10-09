"""Public batched execution owns definitions, exact mirrors and replay safety."""
# ruff: noqa: F811 - imported pytest fixtures.
from __future__ import annotations

import ctypes as c
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from compiler.tests.test_scoped_continuation_runtime import Report, begin, report
from compiler.tests.test_scoped_pinned_runtime import TransferStats, stats
from compiler.tests.test_scoped_planning_runtime import Binding, Costs, costs, planning, select  # noqa: F401
from compiler.tests.test_scoped_runtime import TOKEN, Access, Layout, Scope, access, runtime  # noqa: F401


class BatchCosts(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in ("version", "valid", "async_engine_count")] + [
        ("max_slot_bytes", c.c_size_t), ("staging_cold_seconds", c.c_double * 4),
        ("staging_reuse_seconds", c.c_double * 4)] + [(name, c.c_double) for name in (
            "event_record_seconds", "event_wait_seconds", "ready_event_seconds", "preparation_operation_seconds",
            "pack_bytes_per_second", "unpack_bytes_per_second", "pack_row_seconds", "unpack_row_seconds",
            "pinned_h2d_latency", "pinned_h2d_bandwidth", "pinned_d2h_latency", "pinned_d2h_bandwidth")]


class BatchBinding(c.Structure):
    _fields_ = [("buffer", TOKEN), ("axis", c.c_uint32), ("step", c.c_int64), ("access", Access)]


class Unit(c.Structure):
    _fields_ = [("kind", c.c_uint32), ("unit", TOKEN), ("bindings", c.POINTER(BatchBinding)),
                ("count", c.c_size_t), ("flops", c.c_double), ("memory_bytes", c.c_double)]


class Batch(c.Structure):
    _fields_ = [("version", c.c_uint32), ("execution_mode", c.c_uint32), ("iterations", c.c_size_t),
                ("units", c.POINTER(Unit)), ("unit_count", c.c_size_t), ("exports", c.POINTER(Binding)),
                ("export_count", c.c_size_t)]


class View(c.Structure):
    _fields_ = [("buffer", TOKEN), ("device", c.c_void_p), ("layout", Layout)]


class Window(c.Structure):
    _fields_ = [("version", c.c_uint32), ("begin", c.c_size_t), ("count", c.c_size_t),
                ("stream", c.c_void_p), ("views", c.POINTER(View)), ("view_count", c.c_size_t)]


class BatchReport(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in (
        "version", "available", "applied", "selected_transfers", "reason", "owner_cost_available")] + [
        (name, TOKEN) for name in ("preparation_operations", "batches", "chunk_iterations", "slot_bytes",
                                  "upload_bytes", "download_bytes", "prefix_upload_bytes", "prefix_uploads", "launches")] + [
        (name, c.c_double) for name in ("estimated_seconds", "baseline_seconds", "pinned_seconds",
                                      "pipelined_seconds", "execution_seconds", "terminal_delta_seconds")] + [
        (name, TOKEN) for name in ("completed_batches", "actual_upload_bytes", "actual_download_bytes", "actual_launches")]


Worker = c.CFUNCTYPE(c.c_int, c.POINTER(Window), c.c_void_p, c.POINTER(TOKEN))


@pytest.fixture
def batching(planning):
    signatures = {
        "set_transfers": [TOKEN, c.c_uint32],
        "set_transfer_costs_v1": [TOKEN, c.POINTER(BatchCosts), c.c_int],
        "transfer_stats_get_v1": [TOKEN, c.POINTER(TransferStats)],
        "plan_reset_mode": [TOKEN, c.c_uint32], "plan_report_v2": [TOKEN, c.POINTER(Report)],
        "batch_report_get_v1": [TOKEN, c.POINTER(BatchReport)],
        "batch_execute_v1": [TOKEN, c.POINTER(Batch), c.POINTER(Costs), c.POINTER(BatchCosts),
                             c.c_int, c.c_void_p, c.c_void_p, c.POINTER(BatchReport)],
    }
    for name, signature in signatures.items():
        getattr(planning, "fort_scope_" + name).argtypes = signature
    return planning


def transfer_costs():
    result = BatchCosts(1, 1, 2, 16 * 1024 * 1024)
    for name, field in BatchCosts._fields_[4:]:
        if isinstance(field, type) and issubclass(field, c.Array):
            for index in range(4):
                getattr(result, name)[index] = 1e-9
        else:
            setattr(result, name, 1e-9)
    for name in ("pack_bytes_per_second", "unpack_bytes_per_second", "pinned_h2d_bandwidth", "pinned_d2h_bandwidth"):
        setattr(result, name, 1e12)
    return result


def configured(library, mode=2, calibration=None):
    scope = Scope(library)
    scope.check(library.fort_scope_set_transfers(scope.handle, mode))
    if calibration is not None:
        scope.check(library.fort_scope_set_transfer_costs_v1(scope.handle, c.byref(calibration), 1))
    return scope


def chain(scope, *, iterations=10, halos=1, forget=False, metadata=False, execution_mode=1,
          reverse=False, immutable_halo=False, export=False):
    shape = [4, 5, iterations + 2 * halos]
    size = 20 * shape[2]
    arrays = [(c.c_double * size)(*[i + 1 for i in range(size)]),
              (c.c_double * size)(*[-101] * size), (c.c_double * size)(*[-202] * size)]
    handles = []
    for identity, values in enumerate(arrays, 1):
        status, handle = scope.register(values, shape, identity=identity, lower=[-2, -4, -7], initialized=identity != 2)
        scope.check(status)
        handles.append(handle)
    origin = halos + iterations - 1 if reverse else halos
    step = -1 if reverse else 1
    plane = [([0, 0, origin], [4, 5, origin + 1])]
    read_plane = [([0, 0, origin - 1], [4, 5, origin + 2])] if immutable_halo else plane
    full = [([0, 0, halos], [4, 5, halos + iterations])]
    first = [access(read=read_plane), access(write=plane, overwrite=plane), access(read=plane),
             access(write=plane, overwrite=plane)]
    bindings_a = [BatchBinding(handles[0], 2, step, first[0]), BatchBinding(handles[1], 2, step, first[1])]
    if metadata:
        bindings_a.append(BatchBinding(handles[2], 2**32 - 1, 0, access()))
    bindings_a = (BatchBinding * len(bindings_a))(*bindings_a)
    bindings_b = (BatchBinding * 2)(BatchBinding(handles[1], 2, step, first[2]), BatchBinding(handles[2], 2, step, first[3]))
    units = [Unit(1, 101, bindings_a, len(bindings_a), 1e8, size * 8),
             Unit(1, 102, bindings_b, len(bindings_b), 1e8, size * 8)]
    references = [*arrays, *first, bindings_a, bindings_b]
    if forget:
        event = (BatchBinding * 1)(BatchBinding(handles[1], 2**32 - 1, 0, access()))
        units.insert(0, Unit(2, 0, event, 1, 0, 0))
        references.append(event)
    packed = (Unit * len(units))(*units)
    exported = (Binding * 1)(Binding(handles[2], access(read=full))) if export else None
    result = Batch(1, execution_mode, iterations, packed, len(packed), exported, 1 if export else 0)
    result.references = [*references, packed]
    result.full = full
    result.handles = handles
    result.arrays = arrays
    result.shape = shape
    result.halos = halos
    result.reverse = reverse
    result.immutable_halo = immutable_halo
    return result


def numerical(batch, *, failure=False):
    windows = []

    @Worker
    def callback(pointer, _user, launches):
        window = pointer.contents
        captured = {window.views[i].buffer: window.views[i] for i in range(window.view_count)}
        views = [captured[handle.value] for handle in batch.handles]
        assert window.version == 1
        assert [[view.layout.extents[i] for i in range(3)] for view in views] == [batch.shape] * 3
        assert [[view.layout.lower[i] for i in range(3)] for view in views] == [[-2, -4, -7]] * 3
        payloads = [c.cast(view.device, c.POINTER(c.c_double)) for view in views]
        windows.append((window.begin, window.count, window.stream))
        for ordinal in range(window.begin, window.begin + window.count):
            z = batch.halos + (batch.iterations - 1 - ordinal if batch.reverse else ordinal)
            for y in range(5):
                for x in range(4):
                    index = x + 4 * y + 20 * z
                    payloads[1][index] = (payloads[0][index - 20] + payloads[0][index] + payloads[0][index + 20]
                                          if batch.immutable_halo else 2 * payloads[0][index])
                    payloads[2][index] = payloads[1][index] + x - 2 + 10 * (y - 4) + 100 * (z - 7)
        launches[0] = 2
        return 6 if failure else 0

    return callback, windows


def execute(scope, batch, callback=None, *, preview=False, calibration=None, transfers=None):
    result = BatchReport()
    status = scope.lib.fort_scope_batch_execute_v1(
        scope.handle, c.byref(batch), c.byref(costs() if calibration is None else calibration),
        c.byref(transfer_costs() if transfers is None else transfers), -1 if preview else 1,
        c.cast(callback, c.c_void_p) if callback else None, None, c.byref(result))
    return status, result


def record_chain(scope, batch):
    begin(scope)
    held = []
    for unit in batch.units[:batch.unit_count]:
        if unit.kind == 2:
            bindings = (Binding * unit.count)(*[Binding(item.buffer, access()) for item in unit.bindings[:unit.count]])
        else:
            full = []
            for item in unit.bindings[:unit.count]:
                effect = item.access
                full.append(Binding(item.buffer, access(
                    read=batch.full if effect.read_count else (), write=batch.full if effect.write_count else (),
                    overwrite=batch.full if effect.overwrite_count else ())))
            bindings = (Binding * len(full))(*full)
        scope.check(scope.lib.fort_scope_plan_add(scope.handle, unit.kind, unit.unit, bindings, len(bindings),
                                                 unit.flops, unit.memory_bytes, unit.kind == 1))
        held.append(bindings)
    scope.check(scope.lib.fort_scope_plan_validate(scope.handle))
    return held


def verify(batch):
    shape, original, middle, output = batch.shape, *batch.arrays
    for z in range(shape[2]):
        for y in range(5):
            for x in range(4):
                index = x + 4 * y + 20 * z
                if batch.halos <= z < batch.halos + batch.iterations:
                    assert middle[index] == (original[index - 20] + original[index] + original[index + 20]
                                             if batch.immutable_halo else 2 * original[index])
                    assert output[index] == middle[index] + x - 2 + 10 * (y - 4) + 100 * (z - 7)
                else:
                    assert middle[index] == -101
                    assert output[index] == -202


def test_preview_and_missing_costs_leave_cuda_and_schedule_untouched(batching):
    scope = configured(batching)
    batch = chain(scope, execution_mode=2)
    record_chain(scope, batch)
    assert select(scope, costs()).gpu_units == 2
    before = bytes(scope.stats())
    status, first = execute(scope, batch, preview=True)
    scope.check(status)
    status, second = execute(scope, batch, preview=True)
    scope.check(status)
    assert first.available
    assert not first.applied
    assert first.preparation_operations == second.preparation_operations > 0
    assert bytes(scope.stats()) == before
    absent = BatchReport()
    scope.check(batching.fort_scope_batch_execute_v1(scope.handle, c.byref(batch), c.byref(costs()), None, 1,
                                                    None, None, c.byref(absent)))
    assert absent.reason == 1
    assert not absent.applied
    callback, windows = numerical(batch)
    status, applied = execute(scope, batch, callback)
    scope.check(status)
    assert applied.applied
    assert len(windows) == applied.completed_batches == applied.batches >= 2
    scope.check(batching.fort_scope_wait(scope.handle))
    assert not report(scope).owner_available
    scope.close()
    verify(batch)


@pytest.mark.parametrize(("iterations", "forget", "metadata"), [(3, False, True), (10, True, False), (17, False, False)])
def test_complete_chain_preserves_halos_negative_bounds_and_undefined_intermediate(batching, iterations, forget, metadata):
    scope = configured(batching)
    batch = chain(scope, iterations=iterations, forget=forget, metadata=metadata)
    callback, windows = numerical(batch)
    status, applied = execute(scope, batch, callback)
    scope.check(status)
    assert applied.applied
    assert applied.selected_transfers == 2
    assert applied.actual_upload_bytes == iterations * 20 * 8  # Input only; never the undefined intermediate.
    assert applied.actual_download_bytes == 0
    assert len({window[2] for window in windows}) == 2
    before = scope.stats().waits
    scope.check(batching.fort_scope_wait(scope.handle))
    assert scope.stats().waits == before
    assert stats(scope).staging_device_bytes == 0
    scope.close()
    verify(batch)


def test_explicit_pipeline_runs_a_slow_calibrated_control_while_auto_rejects_it(batching):
    slow = transfer_costs()
    slow.pinned_h2d_latency = 2
    for mode, expected in [(2, True), (3, False)]:
        scope = configured(batching, mode)
        batch = chain(scope)
        callback, windows = numerical(batch)
        status, decision = execute(scope, batch, callback, transfers=slow)
        scope.check(status)
        assert bool(decision.applied) == expected
        if expected:
            assert decision.execution_seconds > decision.baseline_seconds
        else:
            assert decision.reason == 4
            assert not windows
            assert scope.stats().allocations == scope.stats().launches == 0
        scope.close()


@pytest.mark.parametrize("fault", ["FORT_SCOPE_TEST_FAIL_BATCH_ALLOC", "FORT_SCOPE_TEST_FAIL_ALLOC"])
def test_resource_failure_precedes_schedule_consumption_and_numerical_work(batching, monkeypatch, fault):
    scope = configured(batching)
    batch = chain(scope, execution_mode=2)
    held = record_chain(scope, batch)
    assert select(scope, costs()).gpu_units == 2
    callback, windows = numerical(batch)
    monkeypatch.setenv(fault, "1")
    status, result = execute(scope, batch, callback)
    scope.check(status)
    assert not result.applied
    assert result.reason == 6
    assert not windows
    monkeypatch.delenv(fault)
    for unit, bindings in zip(batch.units[:batch.unit_count], held, strict=True):
        chosen = c.c_int()
        scope.check(batching.fort_scope_plan_next(scope.handle, unit.unit, bindings, len(bindings), c.byref(chosen)))
        assert chosen.value == 1
    scope.close()


@pytest.mark.parametrize("fault", [None, "FORT_SCOPE_TEST_FAIL_BATCH_RECORD", "FORT_SCOPE_TEST_FAIL_BATCH_WAIT"])
def test_started_failure_is_poisoned_and_cannot_replay(batching, monkeypatch, fault):
    scope = configured(batching)
    batch = chain(scope)
    callback, windows = numerical(batch, failure=fault is None)
    if fault:
        monkeypatch.setenv(fault, "1")
    status, result = execute(scope, batch, callback)
    assert status == 6
    assert result.applied
    assert batching.fort_scope_host_begin(scope.handle, batch.handles[2], c.byref(access(flags=3))) == 6
    assert batching.fort_scope_close(scope.handle) == 6
    scope.check(batching.fort_scope_abandon(scope.handle))
    assert list(batch.arrays[2]) == [-202] * len(batch.arrays[2])
    assert len(windows) <= 2


def test_cross_chunk_producer_consumer_read_is_rejected_before_cuda(batching):
    scope = configured(batching)
    batch = chain(scope)
    crossed = access(read=[([0, 0, 0], [4, 5, 2])])
    batch.units[1].bindings[0].access = crossed
    batch.references.append(crossed)
    status, result = execute(scope, batch, preview=True)
    scope.check(status)
    assert not result.applied
    assert result.reason == 2
    assert scope.stats().allocations == scope.stats().uploads == scope.stats().launches == 0
    scope.close()


@pytest.mark.parametrize("reverse", [False, True])
def test_immutable_halo_union_precedes_kernels_and_exports_preserve_other_host_sections(batching, reverse):
    scope = configured(batching)
    batch = chain(scope, iterations=11, reverse=reverse, immutable_halo=True, export=True)
    output = batch.arrays[2]
    for z in (0, batch.shape[2] - 1):
        for index in range(20 * z, 20 * (z + 1)):
            output[index] = 42
    callback, windows = numerical(batch)
    status, result = execute(scope, batch, callback)
    scope.check(status)
    assert result.applied
    assert result.completed_batches >= 2
    assert result.prefix_upload_bytes == result.actual_upload_bytes == len(batch.arrays[0]) * 8
    assert result.actual_download_bytes == 11 * 160
    for z in range(batch.shape[2]):
        for index in range(20 * z, 20 * (z + 1)):
            if z in (0, batch.shape[2] - 1):
                assert output[index] == 42
            else:
                assert output[index] != -202
    assert len({window[2] for window in windows}) == 2
    scope.close()
    assert list(output[:20]) == list(output[-20:]) == [42] * 20


def test_current_device_input_is_not_replaced_by_stale_host_bytes(batching):
    scope = configured(batching)
    batch = chain(scope, reverse=True, export=True)
    device = scope.gpu(batch.handles[0], access(flags=3))
    for index in range(len(batch.arrays[0])):
        device[index] += 700
    scope.gpu_end(batch.handles[0])
    callback, _windows = numerical(batch)
    status, result = execute(scope, batch, callback)
    scope.check(status)
    assert result.applied
    assert result.actual_upload_bytes == result.prefix_upload_bytes == 0
    assert result.actual_download_bytes == batch.iterations * 160
    assert batch.arrays[0][0] == 1  # The registered host mirror remains stale until its boundary.
    scope.close()
    assert batch.arrays[0][0] == 701
    verify(batch)


def test_changed_definition_and_all_native_schedule_cannot_reuse_batch_approval(batching):
    scope = configured(batching)
    batch = chain(scope, execution_mode=2)
    held = record_chain(scope, batch)
    assert select(scope, costs()).gpu_units == 2
    scope.check(batching.fort_scope_forget_definition(scope.handle, batch.handles[0]))
    callback, windows = numerical(batch)
    status, result = execute(scope, batch, callback)
    assert status == 7
    assert not result.applied
    assert not windows
    assert scope.stats().allocations == 0
    scope.cpu_begin(batch.handles[0], access(flags=6))
    scope.cpu_end(batch.handles[0])
    status, result = execute(scope, batch, callback)
    scope.check(status)
    assert result.applied
    scope.close()
    verify(batch)
    scope = configured(batching)
    batch = chain(scope, execution_mode=2)
    held = record_chain(scope, batch)
    callback, windows = numerical(batch)
    assert select(scope, costs(), compatible=0).gpu_units == 0
    status, result = execute(scope, batch, callback)
    scope.check(status)
    assert result.reason == 3
    assert not result.applied
    assert not windows
    for unit, bindings in zip(batch.units[:batch.unit_count], held, strict=True):
        chosen = c.c_int(99)
        scope.check(batching.fort_scope_plan_next(scope.handle, unit.unit, bindings, len(bindings), c.byref(chosen)))
        assert chosen.value == 0
    scope.close()


def test_nonfinite_transfer_estimate_declines_before_cuda(batching):
    scope = configured(batching)
    batch = chain(scope)
    rates = transfer_costs()
    rates.pinned_h2d_latency = float.fromhex("0x1.fffffffffffffp+1023")
    status, result = execute(scope, batch, preview=True, transfers=rates)
    scope.check(status)
    assert not result.applied
    assert result.reason == 8
    assert scope.stats().allocations == scope.stats().launches == 0
    scope.close()


def test_unselected_candidate_overflow_never_enters_public_diagnostics(batching, monkeypatch, capfd):
    scope = configured(batching, 3)
    batch = chain(scope)
    status, first = execute(scope, batch, preview=True)
    scope.check(status)
    rates = transfer_costs()
    maximum = float.fromhex("0x1.fffffffffffffp+1023")
    for index in range(4):
        rates.staging_cold_seconds[index] = .6 * maximum
    rates.preparation_operation_seconds = .5 * maximum / first.preparation_operations
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    status, result = execute(scope, batch, transfers=rates)
    scope.check(status)
    assert result.reason == 8
    assert not result.available
    assert not result.applied
    assert scope.stats().allocations == scope.stats().launches == 0
    scope.close()
    captured = capfd.readouterr().err
    assert "Infinity" not in captured
    assert "NaN" not in captured


@pytest.mark.parametrize("produced", [False, True])
def test_sparse_reads_are_not_replaced_by_an_enclosing_rectangle(batching, produced):
    scope = configured(batching)
    batch = chain(scope, iterations=10)
    batch.iterations = 5
    wide = [([0, 0, 1], [4, 5, 3])]
    for unit in batch.units[:batch.unit_count]:
        for binding in unit.bindings[:unit.count]:
            binding.step = 2
            if binding.access.write_count:
                binding.access = access(write=wide, overwrite=wide)
            elif binding.access.read_count:
                binding.access = access(read=wide)
    sparse = access(read=[([0, 0, 1], [4, 5, 2])])
    batch.units[1 if produced else 0].bindings[0].access = sparse
    batch.references.append(sparse)
    status, result = execute(scope, batch, preview=True)
    scope.check(status)
    assert result.reason == 2
    assert not result.applied
    assert scope.stats().allocations == scope.stats().uploads == 0
    scope.close()


def test_pinned_staging_cold_cost_is_charged_once_before_reuse(batching):
    observations = []
    for cold in (.01, .1):
        rates = transfer_costs()
        for index in range(4):
            rates.staging_cold_seconds[index] = cold
            rates.staging_reuse_seconds[index] = .0001
        scope = configured(batching, 1, rates)
        batch = chain(scope)
        held = record_chain(scope, batch)
        assert select(scope, costs()).gpu_units == 2
        decision = report(scope)
        observations.append((decision.execution_seconds, decision.terminal.seconds))
        for unit, bindings in zip(batch.units[:batch.unit_count], held, strict=True):
            chosen = c.c_int()
            scope.check(batching.fort_scope_plan_next(scope.handle, unit.unit, bindings, len(bindings), c.byref(chosen)))
        scope.close()
    assert observations[1][0] - observations[0][0] == pytest.approx(.09, abs=1e-8)
    assert observations[1][1] == pytest.approx(observations[0][1])


def test_pinned_geometry_has_complete_costs_only_after_metadata_setter(batching):
    scope = configured(batching, 1, transfer_costs())
    batch = chain(scope)
    record_chain(scope, batch)
    choice = select(scope, costs())
    assert choice.available
    assert choice.gpu_units == 2
    assert not stats(scope).fallbacks
    assert batching.fort_scope_set_transfer_costs_v1(scope.handle, c.byref(transfer_costs()), 1) == 8
    # Consume the installed schedule without claiming numerical work.
    for unit in batch.units[:batch.unit_count]:
        full = (Binding * unit.count)(*[Binding(item.buffer, access(
            read=batch.full if item.access.read_count else (), write=batch.full if item.access.write_count else (),
            overwrite=batch.full if item.access.overwrite_count else ())) for item in unit.bindings[:unit.count]])
        chosen = c.c_int()
        scope.check(batching.fort_scope_plan_next(scope.handle, unit.unit, full, len(full), c.byref(chosen)))
    scope.close()


def test_batch_evidence_keeps_actual_and_hypothetical_terminal_costs_separate(batching, monkeypatch, capfd):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    scope = configured(batching)
    batch = chain(scope)
    callback, _windows = numerical(batch)
    status, result = execute(scope, batch, callback)
    scope.check(status)
    latest = BatchReport()
    scope.check(batching.fort_scope_batch_report_get_v1(scope.handle, c.byref(latest)))
    assert bytes(latest) == bytes(result)
    assert result.estimated_seconds == pytest.approx(result.execution_seconds + result.terminal_delta_seconds)
    assert scope.stats().download_bytes == 0
    scope.close()
    rows = [json.loads(line.removeprefix("FORT_SCOPED evidence ")) for line in capfd.readouterr().err.splitlines()
            if line.startswith("FORT_SCOPED evidence ")]
    batch_rows = [row for row in rows if row["event"] == "batch_statistics"]
    assert len(batch_rows) == 1
    assert batch_rows[0]["applied"]
    assert batch_rows[0]["actual_download_bytes"] == 0
    assert batch_rows[0]["owner_cost_reason"] == "complete_owner_batch_reprice_unavailable"
    assert [row for row in rows if row["event"] == "transfer_statistics"][0]["pinned_upload_bytes"] == batch.iterations * 160


def test_fortran_additive_batch_layout_matches_c(tmp_path):
    compiler = shutil.which("gfortran")
    if not compiler:
        pytest.skip("gfortran unavailable")
    source = tmp_path / "batch.f90"
    source.write_text("""program check
use iso_c_binding
use fort_scoped_memory
implicit none
type(fort_scope_batch_costs) :: costs
type(fort_scope_batch_binding) :: binding
type(fort_scope_batch_unit) :: unit
type(fort_scope_batch) :: batch
type(fort_scope_batch_view) :: view
type(fort_scope_batch_window) :: window
type(fort_scope_batch_report) :: report
print *, c_sizeof(costs), c_sizeof(binding), c_sizeof(unit), c_sizeof(batch), &
         c_sizeof(view), c_sizeof(window), c_sizeof(report), FORT_SCOPE_BATCH_FIXED_AXIS
end program
""")
    target = tmp_path / "check"
    subprocess.run([compiler, str(Path(__file__).resolve().parents[1] / "runtime/scoped_memory.f90"),
                    str(source), "-o", str(target)], cwd=tmp_path, check=True, capture_output=True, timeout=30)
    observed = [int(value) for value in subprocess.check_output([str(target)], text=True).split()]
    assert observed == [c.sizeof(kind) for kind in (BatchCosts, BatchBinding, Unit, Batch, View, Window, BatchReport)] + [-1]
