"""Both host protocols retain observations and contribute their median cost."""
from copy import deepcopy

import pytest

from compiler.offload.calibrate import CalibrationError, parse_scoped_measurements, profile_with_scoped_measurements
from compiler.tests.test_offload_profile import RUNTIME_ID, profile, scoped_observations


def records():
    result = [row for row in scoped_observations() if row.get("name") != "host_access_seconds"]
    return result + [{"kind": "scoped_cost", "name": "host_access_seconds", "protocol": protocol,
                      "seconds": samples}
                     for protocol, samples in (("strict", [10e-6, 1e-6, 1e-6, 1e-6, 1e-6]),
                                               ("original_host", [2e-6, 5e-6, 5e-6, 5e-6, 5e-6]))]


def build(rows):
    return profile_with_scoped_measurements(profile(), rows, runtime_id=RUNTIME_ID,
                                          cold_startup_seconds=[0.1] * 5)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("swap", [False, True])
def test_maximum_median_is_independent_of_first_sample_protocol_or_record_order(reverse, swap):
    rows = records()
    if swap:
        rows[-1]["seconds"], rows[-2]["seconds"] = rows[-2]["seconds"], rows[-1]["seconds"]
    if reverse:
        rows.reverse()
    before = deepcopy(rows)
    result = build(rows)
    assert result["scoped"]["costs"]["host_access_seconds"] == 5e-6
    assert result["scoped"]["measurements"] == before
    assert rows == before
    assert "maximum median" in result["scoped"]["measurement_method"]["host_access"]


@pytest.mark.parametrize("fault", ["missing_strict", "missing_original", "duplicate", "mixed_before", "mixed_after",
                                  "unknown_protocol", "invalid_protocol", "other_cost", "extra_field", "bad_sample"])
def test_host_protocol_records_cannot_supply_missing_or_ambiguous_measurements(fault):
    rows = records()
    if fault.startswith("missing_"):
        rows.pop(-2 if fault == "missing_strict" else -1)
    elif fault == "duplicate":
        rows.append(deepcopy(rows[-1]))
    elif fault.startswith("mixed_"):
        legacy = {"kind": "scoped_cost", "name": "host_access_seconds", "seconds": [1e-6] * 5}
        rows.insert(0 if fault == "mixed_before" else len(rows), legacy)
    elif fault in {"unknown_protocol", "invalid_protocol"}:
        rows[-1]["protocol"] = "other" if fault == "unknown_protocol" else []
    elif fault == "other_cost":
        rows[-1]["name"] = "wait_seconds"
    elif fault == "extra_field":
        rows[-1]["unmeasured"] = True
    else:
        rows[-1]["seconds"][0] = 0
    with pytest.raises(CalibrationError, match="host acquisition|duration"):
        build(rows)


def test_native_record_parser_preserves_both_protocol_sample_vectors():
    import json

    rows = records()
    parsed = parse_scoped_measurements("\n".join(json.dumps(row) for row in rows))
    assert parsed == rows
    assert build(parsed)["scoped"]["costs"]["host_access_seconds"] == 5e-6
