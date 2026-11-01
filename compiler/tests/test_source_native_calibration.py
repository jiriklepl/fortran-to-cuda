"""Native-live management has its own bounded independent evidence."""

from copy import deepcopy

import pytest

from compiler.offload.source_protocol import (
    BINDINGS,
    NATIVE_RAW_NAMES,
    profile_with_source_measurements,
    source_native_costs,
)
from compiler.offload.tile_protocol_calibration import source_scratch_benchmark
from compiler.tests.test_offload_profile import RUNTIME_ID, scoped_profile
from compiler.tests.test_source_protocol_calibration import MAX_BYTES, observations, teams


def native_rows():
    return [{"kind": "source_protocol_observation", "name": name, "bytes": 0, "bindings": count,
             "launches": 0, "rounds": 128, "seconds": [count*1e-6]*5}
            for name in NATIVE_RAW_NAMES for count in BINDINGS]


def fitted(rows):
    return profile_with_source_measurements(scoped_profile(), observations()+rows,
        runtime_id=RUNTIME_ID, max_bytes=MAX_BYTES, team_records=teams())


def test_independent_native_grid_prices_complete_management_without_body_or_copy_claim():
    result = fitted(native_rows())
    native = source_native_costs(result, RUNTIME_ID)
    assert native["prepare_start_complete_seconds"] == 8e-6
    assert native["max_bindings"] == 8
    assert set(native) == {"version", "protocol_id", "max_bindings", "prepare_start_complete_seconds"}
    assert result["source_tasks"]["native_transactions"]["fit"]["failed_holdout_rows"] == []


def test_native_failed_holdout_does_not_expand_fit_or_disable_independent_gpu_protocol():
    rows = native_rows()
    bad = next(row for row in rows if row["bindings"] == 6)
    bad["seconds"] = [1.0]*5
    result = fitted(rows)
    assert result["source_tasks"]["available"] is True
    assert result["source_tasks"]["native_transactions"]["prepare_start_complete_seconds"] == 8e-6
    assert result["source_tasks"]["native_transactions"]["available"] is False
    with pytest.raises(ValueError, match="native source protocol"):
        source_native_costs(result, RUNTIME_ID)


def test_failed_gpu_holdout_cannot_erase_independently_qualified_native_management():
    result = fitted(native_rows())
    result["source_tasks"]["available"] = False
    assert source_native_costs(result, RUNTIME_ID)["prepare_start_complete_seconds"] == 8e-6


def test_historical_full_close_profile_never_fabricates_missing_native_live_cost():
    result = fitted([])
    assert "native_transactions" not in result["source_tasks"]
    with pytest.raises(ValueError, match="native source protocol"):
        source_native_costs(result, RUNTIME_ID)


def test_resident_native_measurement_setup_creates_a_clean_mirror_without_gpu_write_credit():
    source = source_scratch_benchmark(precision=64).split("void native_source(", 1)[1]
    setup = source.split("fort_scope_stats before", 1)[0]
    assert "resident_read{FORT_SCOPE_READ_ALL,0,nullptr,0,nullptr,0,nullptr}" in setup
    assert "fort_scope_device_begin(current.context,buffer,&resident_read,&pointer)" in setup
    assert "fort_scope_device_begin(current.context,buffer,&access,&pointer)" not in setup
    # The measured CPU declaration remains READWRITE for writer rows; only
    # the setup upload stops pretending that an absent GPU body wrote fields.
    assert "writer?FORT_SCOPE_WRITE_ALL:0" in setup


@pytest.mark.parametrize("fault", ["missing_row", "duplicate", "rounds", "copies", "nonfinite", "false_type"])
def test_native_protocol_rejects_incomplete_or_foreign_measurement_rows(fault):
    rows = deepcopy(native_rows())
    if fault == "missing_row":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[-1])
    elif fault == "rounds":
        rows[0]["rounds"] = 32
    elif fault == "copies":
        rows[0]["bytes"] = 8
    elif fault == "nonfinite":
        rows[0]["seconds"][0] = float("nan")
    else:
        rows[0]["bindings"] = True
    with pytest.raises(ValueError, match="native source|source protocol"):
        fitted(rows)
