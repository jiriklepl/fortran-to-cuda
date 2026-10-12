"""Dense physical uploads must preserve exact logical freshness and copyback."""
# ruff: noqa: F811 - imported pytest fixtures.
from __future__ import annotations

import ctypes as c

import pytest

from compiler.tests.test_scoped_continuation_runtime import consume
from compiler.tests.test_scoped_pinned_runtime import configured, pinned, stats  # noqa: F401
from compiler.tests.test_scoped_planning_runtime import costs, planning, record, select  # noqa: F401
from compiler.tests.test_scoped_runtime import access, runtime  # noqa: F401

DENSE = [([0, 0], [16, 4]), ([0, 4], [17, 8])]
GAP = [([16, 0], [17, 1])]


@pytest.mark.parametrize("generation", [1, 7])
def test_span_preserves_logical_coverage_negative_bounds_and_exact_copyback(pinned, generation):
    scope = configured(pinned, 4)
    values = (c.c_double * 136)(*(generation * 1000 + i for i in range(136)))
    expected = list(values)
    status, handle = scope.register(values, [17, 8], lower=[-9, -3])
    scope.check(status)
    pointer = scope.gpu(handle, access(read=DENSE))
    assert list(pointer[:136]) == expected  # Includes only source-proven overfetch.
    scope.gpu_end(handle)
    assert scope.stats().uploads == 1
    assert scope.stats().upload_bytes == 136 * 8
    assert stats(scope).effective_mode == 4
    # Physical overfetch does not claim logical freshness: an exact later read
    # must still upload the extra cell, even though its physical value matches.
    scope.gpu(handle, access(read=GAP))
    scope.gpu_end(handle)
    assert scope.stats().uploads == 2
    assert scope.stats().upload_bytes == 137 * 8
    pointer = scope.gpu(handle, access(write=DENSE, overwrite=DENSE))
    for y in range(8):
        for x in range(16 if y < 4 else 17):
            i = x + 17 * y
            pointer[i] += 23
            expected[i] += 23
    scope.gpu_end(handle)
    scope.close()
    assert list(values) == expected  # Host-only holes remain unchanged.


def test_stale_host_gap_cannot_overwrite_newer_device_value(pinned):
    scope = configured(pinned, 4)
    values = (c.c_double * 136)(*range(136))
    status, handle = scope.register(values, [17, 8], lower=[-9, -3])
    scope.check(status)
    pointer = scope.gpu(handle, access(write=GAP, overwrite=GAP))
    pointer[16] = 999
    scope.gpu_end(handle)
    assert values[16] == 16  # The host gap is now stale.
    pointer = scope.gpu(handle, access(read=DENSE))
    scope.gpu_end(handle)
    assert pointer[16] == 999
    assert scope.stats().uploads == 2
    assert scope.stats().upload_bytes == 132 * 8
    assert stats(scope).effective_mode == 0
    assert stats(scope).fallback_reason == 5
    scope.close()
    assert list(values) == [999 if i == 16 else i for i in range(136)]


def test_undefined_holes_are_not_uploaded_or_declared_defined(pinned):
    scope = configured(pinned, 4)
    values = (c.c_double * 136)(*range(136))
    status, handle = scope.register(values, [17, 8], initialized=False, defined=DENSE)
    scope.check(status)
    scope.gpu(handle, access(read=DENSE))
    scope.gpu_end(handle)
    assert scope.stats().uploads == 2
    assert scope.stats().upload_bytes == 132 * 8
    output = c.c_void_p()
    assert pinned.fort_scope_device_begin(scope.handle, handle, c.byref(access(read=GAP)), c.byref(output)) == 7
    assert stats(scope).effective_mode == 0
    scope.close()
    assert list(values) == list(range(136))


@pytest.mark.parametrize(("sections", "bytes_expected", "calls"), [
    ([([0], [4]), ([5], [9])], 9 * 8, 1),  # Exactly 12.5% extra.
    ([([0], [4]), ([6], [10])], 8 * 8, 2),  # Above the fixed cap.
    ([([0], [10])], 10 * 8, 1),  # Already one exact physical operation.
])
def test_fixed_density_threshold_and_already_contiguous_fallback(pinned, sections, bytes_expected, calls):
    scope = configured(pinned, 4)
    values = (c.c_double * 10)(*range(10))
    status, handle = scope.register(values, [10], lower=[-4])
    scope.check(status)
    scope.gpu(handle, access(read=sections))
    scope.gpu_end(handle)
    assert scope.stats().uploads == calls
    assert scope.stats().upload_bytes == bytes_expected
    scope.close()
    assert list(values) == list(range(10))


@pytest.mark.parametrize("mode", [0, 4])
def test_thin_opposite_faces_never_amplify_to_a_volume(pinned, mode):
    scope = configured(pinned, mode)
    values = (c.c_double * 136)(*range(136))
    status, handle = scope.register(values, [17, 8])
    scope.check(status)
    faces = [([0, 0], [1, 8]), ([16, 0], [17, 8])]
    scope.gpu(handle, access(read=faces))
    scope.gpu_end(handle)
    assert scope.stats().uploads == 2
    assert scope.stats().upload_bytes == 16 * 8
    scope.close()


def test_default_direct_does_not_widen_dense_sections(pinned):
    scope = configured(pinned, 0)
    values = (c.c_double * 136)(*range(136))
    status, handle = scope.register(values, [17, 8])
    scope.check(status)
    scope.gpu(handle, access(read=DENSE))
    scope.gpu_end(handle)
    assert scope.stats().uploads == 2
    assert scope.stats().upload_bytes == 132 * 8
    scope.close()


def test_nonzero_physical_origin_retains_original_pitches(pinned):
    scope = configured(pinned, 4)
    values = (c.c_double * 170)(*range(170))
    status, handle = scope.register(values, [17, 10], lower=[-9, -3])
    scope.check(status)
    shifted = [([0, 1], [16, 5]), ([0, 5], [17, 9])]
    pointer = scope.gpu(handle, access(read=shifted))
    assert list(pointer[17:153]) == list(range(17, 153))
    scope.gpu_end(handle)
    assert scope.stats().uploads == 1
    assert scope.stats().upload_bytes == 136 * 8
    scope.close()
    assert list(values) == list(range(170))


def test_cost_selected_schedule_retains_calibrated_exact_transfers(pinned):
    scope = configured(pinned, 4)
    values = (c.c_double * 136)(*range(136))
    status, handle = scope.register(values, [17, 8])
    scope.check(status)
    effects = access(read=DENSE)
    scope.check(pinned.fort_scope_plan_reset(scope.handle))
    bindings = record(scope, 92, handle, effects)
    decision = select(scope, costs())
    assert decision.available
    assert decision.gpu_units == 1
    consume(scope, 92, bindings, 1)
    scope.gpu(handle, effects)
    scope.gpu_end(handle)
    assert scope.stats().uploads == decision.uploads == 2
    assert scope.stats().upload_bytes == decision.upload_bytes == 132 * 8
    assert stats(scope).effective_mode == 0
    assert stats(scope).fallback_reason == 1
    scope.close()


def test_empty_and_overflow_never_allocate(pinned):
    scope = configured(pinned, 4)
    maximum = c.c_size_t(-1).value
    status, empty = scope.register(None, [maximum, 0], lower=[-2, -3])
    scope.check(status)
    scope.gpu(empty, access(flags=1))
    scope.gpu_end(empty)
    value = (c.c_double * 1)(0)
    status, _ = scope.register(value, [maximum], identity=2)
    assert status == 1
    assert scope.stats().allocations == scope.stats().upload_bytes == 0
    scope.close()
