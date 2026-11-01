"""Complete-path calibration preserves uncertainty and independently checked units."""
import json
from copy import deepcopy

import pytest

from compiler.emission.cuda.batch import physical_copy_cost_lines
from compiler.offload.calibrate import CalibrationError, parse_scoped_measurements, profile_with_scoped_measurements
from compiler.offload.physical_copy_costs import COEFFICIENTS, PhysicalCopyCostError
from compiler.offload.physical_copy_fit import PhysicalCopyFitUnavailable, _fit, physical_copies_from_measurements
from compiler.offload.profile import ProfileError, scoped_physical_copy_costs, validate_profile
from compiler.tests.test_offload_profile import RUNTIME_ID, profile, scoped_observations, scoped_profile


def geometry(width, height, depth, count, word, mixed=False):
    operations = []
    at = word
    for index in range(count):
        kind = 1 + index % 2 if mixed else 2
        h, d = (height, 1) if kind == 1 else (height, depth)
        pitch = width + 16 * word
        physical_height = h + 1
        operations.append([at, width, h, d, pitch, physical_height])
        end = at + (d - 1) * pitch * physical_height + (h - 1) * pitch + width
        at = (end + 31) // 32 * 32 + word
    return {"operations": operations, "physical_bytes": at,
            "payload_bytes": sum(op[1] * op[2] * op[3] for op in operations),
            "rows": sum(op[2] * op[3] for op in operations), "word_bytes": word,
            "layout_mask": 6 if mixed and count > 1 else 4,
            "descriptor_work": count * (count + 1) // 2, "kernel_chunks": (count + 47) // 48}


def observations():
    result = []
    shapes = [(1, 2, 2, 2), (4, 5, 3, 3), (16, 2, 4, 49), (1, 8, 8, 64),
              (32, 4, 2, 4), (1, 16, 16, 16), (128, 2, 2, 7)]
    for direction in ("h2d", "d2h"):
        for word in (1, 4, 8):
            for index, (width, height, depth, count) in enumerate(shapes):
                g = geometry(width * word, height, depth, count, word, mixed=index == 4)
                for path in ("direct", "compact_cold", "compact_reuse"):
                    submissions = count if path == "direct" else g["kernel_chunks"]
                    batch = 2e-6 if path == "direct" else 30e-6 if path == "compact_cold" else 4e-6
                    duration = (batch + submissions * 5e-7 + g["payload_bytes"] * 8e-11
                                + g["rows"] * 3e-9 + g["descriptor_work"] * 2e-10)
                    result.append({"kind": "scoped_copy", "version": 2,
                        "name": "direct" if path == "direct" else "compact", "direction": direction,
                        "pool_state": "none" if path == "direct" else path.removeprefix("compact_"),
                        **g, "capacity_bytes": 0 if path == "direct" else 256 * 1024, "pending": False,
                        "seconds": [duration * factor for factor in (.95, 1.02, 1, 1, 1.1, .9, 1)]})
    return result


def fitted_profile():
    value = scoped_profile()
    value["scoped"]["physical_copies"] = physical_copies_from_measurements(observations(), value)
    return value


def test_complete_paths_fit_without_independent_pack_dma_or_descriptor_costs():
    original = observations()
    before = deepcopy(original)
    p = scoped_profile()  # No legacy transfer calibration is needed.
    record = physical_copies_from_measurements(original, p)
    assert original == before
    assert record["measurements"] == original
    assert record["schema_version"] == 2
    assert record["minimum_relative_saving"] == .5
    for path, rates in record["costs"].items():
        batch = 2e-6 if path == "direct" else 30e-6 if path == "compact_cold" else 4e-6
        for rate in rates:
            for name, expected in zip(COEFFICIENTS, (batch, 5e-7, 8e-11, 3e-9, 2e-10), strict=True):
                assert rate[name] == pytest.approx(expected, rel=1e-6, abs=1e-16)
            assert rate["median_relative_error"] == pytest.approx(1 - 1 / 1.02)
            assert rate["p90_relative_error"] > 0
            assert rate["maximum_relative_error"] == pytest.approx(1 / .9 - 1)
            bounds = rate["coverage"]
            assert bounds["word_bytes_mask"] == 7
            assert bounds["layout_combinations_mask"] == (1 << 4) | (1 << 6)
            assert bounds["maximum_operations"] == 64
            assert bounds["maximum_kernel_chunks"] == 2
    p["scoped"]["physical_copies"] = record
    assert validate_profile(p) is p
    assert scoped_physical_copy_costs(p, RUNTIME_ID) is record


def test_public_parse_attachment_and_runtime_identity_use_complete_observations():
    rows = observations()
    assert parse_scoped_measurements("\n".join(json.dumps(row) for row in rows)) == rows
    p = profile_with_scoped_measurements(profile(), scoped_observations() + rows,
                                         runtime_id=RUNTIME_ID, cold_startup_seconds=[.1] * 5)
    assert scoped_physical_copy_costs(p, RUNTIME_ID)["measurements"] == rows
    with pytest.raises(ProfileError, match="runtime_id mismatch"):
        scoped_physical_copy_costs(p, "f" * 64)
    with pytest.raises(ProfileError, match="no physical copy calibration"):
        scoped_physical_copy_costs(scoped_profile(), RUNTIME_ID)


@pytest.mark.parametrize("mutation", ["missing_cold", "too_few"])
def test_incomplete_valid_data_stays_explicitly_unavailable(mutation):
    rows = observations()
    rows = [row for row in rows if row["pool_state"] != "cold"] if mutation == "missing_cold" else rows[:3]
    with pytest.raises(PhysicalCopyFitUnavailable):
        physical_copies_from_measurements(rows, scoped_profile())
    p = profile_with_scoped_measurements(profile(), scoped_observations() + rows,
                                         runtime_id=RUNTIME_ID, cold_startup_seconds=[.1] * 5)
    assert "physical_copies" not in p["scoped"]
    assert not p["scoped"]["physical_copy_fit"]["available"]
    assert all(row in p["scoped"]["measurements"] for row in rows)


@pytest.mark.parametrize("mutation", ["version", "layout", "word", "payload", "chunks", "outside", "bool", "samples", "pending", "duplicate", "pool"])
def test_malformed_measurements_never_silently_supply_a_model(mutation):
    rows = observations()
    row = rows[1]
    if mutation == "version":
        row["version"] = 1
    elif mutation == "layout":
        row["layout_mask"] = 1
    elif mutation == "word":
        row["word_bytes"] = 8
    elif mutation == "payload":
        row["payload_bytes"] += 1
    elif mutation == "chunks":
        row["kernel_chunks"] = 2
    elif mutation == "outside":
        row["physical_bytes"] = 1
    elif mutation == "bool":
        row["descriptor_work"] = True
    elif mutation == "samples":
        row["seconds"] = [1e-5] * 3
    elif mutation == "pending":
        row["pending"] = True
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    else:
        row["pool_state"] = "none"
    with pytest.raises(PhysicalCopyCostError):
        physical_copies_from_measurements(rows, scoped_profile())
    with pytest.raises(CalibrationError):
        profile_with_scoped_measurements(profile(), scoped_observations() + rows,
                                         runtime_id=RUNTIME_ID, cold_startup_seconds=[.1] * 5)


@pytest.mark.parametrize("mutation", ["old", "margin", "zero", "negative", "nan", "word", "layout", "support", "errors"])
def test_profile_rejects_removed_or_malformed_schema(mutation):
    p = fitted_profile()
    r = p["scoped"]["physical_copies"]
    rate = r["costs"]["direct"][0]
    if mutation == "old":
        r["schema_version"] = 1
    elif mutation == "margin":
        r["minimum_relative_saving"] = .25
    elif mutation == "zero":
        rate.update(dict.fromkeys(COEFFICIENTS, 0))
    elif mutation == "negative":
        rate["byte_seconds"] = -1
    elif mutation == "nan":
        rate["batch_seconds"] = float("nan")
    elif mutation == "word":
        rate["coverage"]["word_bytes_mask"] = 8
    elif mutation == "layout":
        rate["coverage"]["layout_combinations_mask"] = 1
    elif mutation == "support":
        rate["coverage"]["minimum_descriptor_work"] = 10**8
    else:
        rate["median_relative_error"] = 100
    with pytest.raises(ProfileError, match="physical_copies"):
        validate_profile(p)


@pytest.mark.parametrize("scale", [1e-120, 1, 1e120])
def test_five_term_fit_predicts_an_independent_geometry_in_scaled_time_units(scale):
    vectors = [(1, 2, 20, 3, 5), (1, 4, 120, 5, 17), (1, 5, 250, 8, 21),
               (1, 3, 300, 13, 31), (1, 7, 90, 2, 12), (1, 1, 70, 9, 8)]
    truth = (2, .4, .03, .1, .05)
    rate = _fit([(x, scale * sum(a * b for a, b in zip(x, truth, strict=True))) for x in vectors])
    probe = (1, 9, 420, 19, 37)
    assert sum(rate[k] * x for k, x in zip(COEFFICIENTS, probe, strict=True)) == pytest.approx(
        scale * sum(a * b for a, b in zip(probe, truth, strict=True)))


def test_diagnostic_uncertainty_does_not_pretend_to_be_a_profitability_bound():
    p = fitted_profile()
    for rates in p["scoped"]["physical_copies"]["costs"].values():
        for rate in rates:
            rate.update(median_relative_error=5, p90_relative_error=10, maximum_relative_error=20)
    validate_profile(p)
    emitted = "\n".join(physical_copy_cost_lines(p["scoped"]["physical_copies"]))
    assert "fort_scope_copy_costs_v2" in emitted
    assert "minimum_relative_saving = 0.5" in emitted
    assert "copy_costs.compact_cold[1].maximum_relative_error = 20.0" in emitted
    assert "copy_costs.compact_reuse[0].layout_combinations_mask = 80U" in emitted
    assert "copy_costs.kernel" not in emitted


@pytest.mark.parametrize("mode", ["auto", "direct"])
@pytest.mark.parametrize("legacy_transfers", [False, True])
def test_normal_emitter_configures_complete_model_independently_of_legacy_packing(tmp_path, mode, legacy_transfers):
    from compiler.tests.test_scoped_batch_sources import numerical, transfer_profile
    p = transfer_profile()
    p["scoped"]["physical_copies"] = fitted_profile()["scoped"]["physical_copies"]
    if not legacy_transfers:
        del p["scoped"]["transfers"]
    emitted = numerical(tmp_path, mode=mode, profile=p)
    assert ("fort_scope_set_copy_costs_v2(" in emitted.cuda) is (mode == "auto")
    assert "fort_scope_set_copy_costs_v1(" not in emitted.cuda
    assert emitted.report["transfer_configuration"]["physical_copy_calibration"]["available"] is (mode == "auto")


@pytest.mark.parametrize("mutation", ["work", "overlap", "slice_overflow"])
def test_independent_geometry_validation_rejects_incorrect_work_and_overlapping_or_wrapped_rows(mutation):
    from compiler.offload.physical_copy_fit import _geometry
    row = deepcopy(observations()[0])
    if mutation == "work":
        row["descriptor_work"] += 1
    elif mutation == "overlap":
        row["operations"][1][0] = row["operations"][0][0]
    else:
        row.update(operations=[[8, 8, 1, 1, 8, 2**63]], physical_bytes=100,
                   payload_bytes=8, rows=1, descriptor_work=1, word_bytes=8, layout_mask=1, kernel_chunks=1)
    with pytest.raises(PhysicalCopyCostError):
        _geometry(row)


def test_zero_byte_coefficient_is_valid_for_a_positive_complete_path():
    p = fitted_profile()
    rate = p["scoped"]["physical_copies"]["costs"]["direct"][0]
    rate["byte_seconds"] = 0
    assert validate_profile(p) is p
