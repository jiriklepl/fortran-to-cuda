"""Predetermined generic protocol fits retain failed evidence and exact bounds."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from compiler.emission.common.resources import read_scoped_runtime
from compiler.offload.calibrate import profile_with_scoped_measurements
from compiler.offload.profile import ProfileError, validate_profile
from compiler.offload.source_protocol import (
    BINDINGS,
    FIT_BINDINGS,
    FIT_LAUNCHES,
    LAUNCHES,
    RAW_NAMES,
    TEAM_UNITS,
    calibrate_source_protocol,
    profile_with_source_measurements,
    source_task_costs,
)
from compiler.tests.test_offload_profile import RUNTIME_ID, scoped_observations, scoped_profile

MAX_BYTES = 8 * 1024**2


def observations():
    profile = scoped_profile()
    records = [{"kind": "source_protocol_identity", "version": 1,
        "gpu_uuid": profile["hardware"]["gpu_uuid"], "device": 0, "cpu_threads": 4, "precision": 64,
        "cuda_runtime_version": 13040, "driver_version": 13040, "schedule_kind": 2, "schedule_chunk": 1}]
    sizes = (4096, 65536, 1048576, 4194304, MAX_BYTES)
    for name, raw in RAW_NAMES.items():
        if name == "source_submit_seconds":
            for bindings in BINDINGS:
                for launches in LAUNCHES:
                    records.append({"kind": "source_protocol_observation", "name": raw, "bytes": 0,
                        "bindings": bindings, "launches": launches, "rounds": 32,
                        "seconds": [1e-6 * (bindings + launches)] * 5})
        else:
            for size in sizes[1:] if name == "scratch_idle_grow_seconds" else sizes:
                records.append({"kind": "source_protocol_observation", "name": raw, "bytes": size,
                    "bindings": 0, "launches": 0, "rounds": 128 if name in (
                        "scratch_reuse_acquire_seconds", "scratch_release_seconds") else 8,
                    "seconds": [size * 1e-12] * 5})
    return records


def teams(rank=None, collapse=None):
    return [{"kind": "original_team_observation", "rank": r, "collapse": c, "units": units,
             "cpu_threads": 4, "rounds": 128, "schedule_kind": 2, "schedule_chunk": 1,
             "seconds": [1e-5 * units] * 5}
            for r in ((rank,) if rank else (1, 2, 3))
            for c in ((collapse,) if collapse else range(1, r + 1)) for units in TEAM_UNITS]


def fitted(records=None, team_records=()):
    return profile_with_source_measurements(scoped_profile(), records or observations(),
        runtime_id=RUNTIME_ID, max_bytes=MAX_BYTES, team_records=team_records)


def test_fixed_grid_fit_preserves_raw_and_exact_native_protocol_scope():
    records, native = observations(), teams()
    before, native_before = deepcopy(records), deepcopy(native)
    result = fitted(records, native)
    section = source_task_costs(result, RUNTIME_ID)
    assert section["costs"]["source_submit_seconds"] == pytest.approx(264e-6)
    assert section["max_launches"] == 256
    assert section["max_bindings"] == 8
    assert section["original_teams"]["scope"].endswith("excludes body and PRIVATE allocation")
    assert len(section["original_teams"]["costs"]) == 30
    assert records == before
    assert native == native_before
    assert section["measurements"] == records
    assert validate_profile(result) is result


def test_independent_failed_holdout_is_not_refitted_or_erased():
    records = observations()
    holdout = next(row for row in records[1:] if row["bindings"] == 6 and row["launches"] == 128)
    holdout["seconds"] = [1.] * 5
    result = fitted(records)
    assert result["source_tasks"]["available"] is False
    assert result["source_tasks"]["costs"]["source_submit_seconds"] == pytest.approx(264e-6)
    assert result["source_tasks"]["measurements"] == records
    assert len(result["source_tasks"]["fit"]["failed_holdout_rows"]) == 1
    assert validate_profile(result) is result  # Other hardware costs stay readable.
    with pytest.raises(ValueError, match="holdout"):
        source_task_costs(result, RUNTIME_ID)


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "identity", "unknown", "bad_rounds", "nan",
                                     "fake_precision", "bad_grid", "boolean", "empty"])
def test_incomplete_or_incompatible_rows_supply_no_default(mutation):
    records = observations()
    if mutation == "duplicate":
        records.append(deepcopy(records[-1]))
    elif mutation == "missing":
        records.pop()
    elif mutation == "identity":
        records[0]["driver_version"] += 1
    elif mutation == "unknown":
        records[-1]["name"] = "guessed_source_overhead"
    elif mutation == "bad_rounds":
        records[-1]["rounds"] += 1
    elif mutation == "nan":
        records[-1]["seconds"][0] = float("nan")
    elif mutation == "fake_precision":
        records[0]["precision"] = 32
    elif mutation == "bad_grid":
        records[-1]["bytes"] += 1
    elif mutation == "boolean":
        records[-1]["bindings"] = False
    elif mutation == "empty":
        records = []
    with pytest.raises(ValueError, match="source protocol"):
        profile_with_source_measurements(scoped_profile(), records, runtime_id=RUNTIME_ID, max_bytes=MAX_BYTES)


@pytest.mark.parametrize("field", ["runtime_id", "precision_bits", "costs", "max_launches", "identity"])
def test_optional_profile_extension_is_checked_before_use(field):
    result = fitted()
    if field == "runtime_id":
        result["source_tasks"][field] = "f" * 64
    elif field == "precision_bits":
        result["source_tasks"][field] = 32
    elif field == "costs":
        result["source_tasks"][field]["scratch_release_seconds"] = 0
    elif field == "max_launches":
        result["source_tasks"][field] = 257
    elif field == "identity":
        result["source_tasks"][field]["cpu_threads"] = 3
    with pytest.raises(ProfileError):
        validate_profile(result)


@pytest.mark.parametrize("mutation", ["schedule", "missing", "geometry", "duplicate"])
def test_native_protocol_requires_exact_units_schedule_and_geometry(mutation):
    records = teams()
    if mutation == "schedule":
        records[-1]["schedule_chunk"] += 1
    elif mutation == "missing":
        records.pop()
    elif mutation == "geometry":
        records[-1]["collapse"] = 4
    elif mutation == "duplicate":
        records.append(deepcopy(records[-1]))
    with pytest.raises(ValueError, match="original-team protocol"):
        fitted(team_records=records)


def test_public_calibration_materializes_identity_complete_runtime_and_caps(tmp_path):
    runtime_id = read_scoped_runtime()[1]["runtime_id"]
    profile = scoped_profile()
    profile["scoped"]["runtime_id"] = runtime_id
    args = SimpleNamespace(precision=64, arch="sm_86", threads=4, max_mib=8, device=0,
                           fortran="/tools/fortran", fortran_flag=[])
    commands = []

    def run(argv, directory, log, *, timeout, env=None):
        commands.append((argv, timeout, env))
        log.write_text("mock command; no hardware measurement\n")
        if argv[0] == "/tools/nvcc":
            assert (directory / "scoped_source_quote.hpp").is_file()
            return ""
        if "--version" in argv:
            return "GNU Fortran 15"
        if argv[0].endswith("source-protocol"):
            return "\n".join(json.dumps(row) for row in observations())
        if argv[0].split("/")[-1].startswith("team-"):
            rank, collapse = map(int, argv[0].split("/")[-1].split("-")[1:])
            return "\n".join(json.dumps(row) for row in teams(rank, collapse))
        return ""

    result = calibrate_source_protocol(profile, args, tmp_path, "/tools/nvcc", "/tools/g++",
        run=run, tool=lambda supplied, _defaults: supplied)
    source = result["source_tasks"]
    assert source["available"]
    assert source["runtime_id"] == runtime_id
    assert len(source["original_teams"]["costs"]) == 30
    assert source["calibration"]["runtime"]["runtime_id"] == runtime_id
    assert max(timeout for _, timeout, _ in commands) <= 240
    assert source["calibration"]["application_profiled"] is False


def test_cli_requires_scoped_measurements_and_keeps_ordinary_default(monkeypatch, tmp_path):
    from compiler.offload import calibrate

    calls = []
    monkeypatch.setattr(calibrate, "calibrate", lambda args: calls.append(args))
    with pytest.raises(SystemExit):
        calibrate.main(["--output", str(tmp_path / "bad.json"), "--source-task-costs"])
    assert calls == []
    assert calibrate.main(["--output", str(tmp_path / "base.json")]) == 0
    assert calls[-1].source_task_costs is False
    assert calibrate.main(["--output", str(tmp_path / "source.json"), "--scoped-costs", "--source-task-costs"]) == 0
    assert calls[-1].source_task_costs is True


def test_every_submit_holdout_is_outside_the_prespecified_fit_grid():
    section = fitted()["source_tasks"]
    assert section["fit"]["rows"]["source_submit_seconds"]["fit_rows"] == 15
    assert len([(bindings, launches) for bindings in BINDINGS for launches in LAUNCHES
                if not (bindings in FIT_BINDINGS and launches in FIT_LAUNCHES)]) == 30


def test_scoped_refresh_does_not_relabel_old_source_protocol_runtime():
    original = fitted()
    refreshed = profile_with_scoped_measurements(original, scoped_observations(), runtime_id="f" * 64,
                                                  cold_startup_seconds=[.1] * 5)
    assert "source_tasks" not in refreshed
    assert original["source_tasks"]["runtime_id"] == RUNTIME_ID
    assert refreshed["rates"] == original["rates"]


def test_c_fortran_schedule_modifier_views_keep_both_raw_identities():
    records, native = observations(), teams()
    records[0]["schedule_kind"] = -(1 << 31) + 1
    for row in native:
        row["schedule_kind"] = 1
    result = fitted(records, native)["source_tasks"]
    assert result["identity"]["schedule_kind"] == -2147483647
    assert result["original_teams"]["measurements"] == native
    assert all(row["schedule"]["raw_kind"] == 1 for row in result["original_teams"]["costs"])
    assert result["original_teams"]["schedule"]["c_identity"]["monotonic_modifier"]
    assert "complete original C raw kind" in result["original_teams"]["schedule"]["dispatch"]


@pytest.mark.parametrize("kind", [0, 5, 0x40000001, 1 << 32, True])
def test_unknown_schedule_kinds_or_modifiers_never_normalize_to_valid(kind):
    records = observations()
    records[0]["schedule_kind"] = kind
    with pytest.raises(ValueError, match="schedule"):
        fitted(records)


def test_normalization_does_not_hide_a_different_native_schedule():
    records, native = observations(), teams()
    records[0]["schedule_kind"] = -(1 << 31) + 1
    native[-1]["schedule_kind"] = 2
    with pytest.raises(ValueError, match="actual schedule differs"):
        fitted(records, native)
