"""Offline calibration uses measured costs and fails closed on bad profiles."""
import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from compiler.numerical_contract import numerical_build_contract

from compiler.offload.calibrate import (
    CalibrationError,
    fit_transfer,
    parse_measurements,
    parse_scoped_measurements,
    profile_from_measurements,
    profile_with_scoped_measurements,
    scoped_transfers_from_measurements,
)
from compiler.offload.profile import (
    LATENCY_RATES,
    SCOPED_BATCH_PAYLOADS,
    SCOPED_COST_NAMES,
    THROUGHPUT_RATES,
    WORKER_RATES,
    ProfileError,
    compiler_identity,
    load_profile,
    scoped_costs,
    scoped_transfer_costs,
    validate_profile,
)


def observations(threads=4):
    records = [{"kind": "device", "gpu_name": "Test device", "gpu_uuid": "GPU-01234567-89ab-cdef-0123-456789abcdef",
                "compute_capability": "8.6", "cuda_runtime_version": 13040, "driver_version": 13040,
                "cpu_threads": threads, "device_ordinal": 0, "async_engine_count": 2}]
    for memory in ("pageable", "pinned"):
        for direction in ("h2d", "d2h"):
            for size in (4096, 65536, 1048576, 67108864):
                time = 5e-6 + size / 12e9
                records.append({"kind": "transfer", "direction": direction, "memory": memory,
                                "bytes": size, "seconds": [time, time, time * 50, time, time]})
    for name in THROUGHPUT_RATES + (WORKER_RATES if threads > 1 else ()):
        records.append({"kind": "rate", "name": name, "work": 1e9, "seconds": [0.02, 0.02, 0.5, 0.02, 0.02]})
    for name in LATENCY_RATES:
        records.append({"kind": "latency", "name": name, "seconds": [4e-6] * 5})
    return records


def profile(threads=4):
    return profile_from_measurements(observations(threads), precision_bits=64, cpu_threads=threads,
                                     cpu_name="Test CPU", nvcc_version="NVCC 13.4", host_cxx_version="GCC 14.4",
                                     numerical_contract=numerical_build_contract())


def test_parse_and_fit_retains_raw_samples_and_uses_medians():
    source = "\n".join(json.dumps(record) for record in observations())
    records = parse_measurements(source)
    assert records == observations()
    result = profile_from_measurements(records, precision_bits=64, cpu_threads=4,
                                       cpu_name="Test CPU", nvcc_version="NVCC 13.4", host_cxx_version="GCC 14.4")
    assert result["rates"]["h2d_pinned"]["latency_seconds"] == pytest.approx(5e-6)
    assert result["rates"]["h2d_pinned"]["bandwidth_bytes_per_second"] == pytest.approx(12e9)
    assert result["rates"]["cpu_flops_per_second"] == pytest.approx(50e9)
    assert result["rates"]["cpu_worker_flops_per_second"] == pytest.approx(50e9)
    assert result["measurements"] == records


def test_negative_fit_latency_is_refitted_through_zero():
    values = [{"bytes": size, "seconds": [size * 1e-9 - 1e-7] * 5} for size in (1000, 10000, 100000)]
    fit = fit_transfer(values)
    assert fit["latency_seconds"] == 0
    assert fit["bandwidth_bytes_per_second"] > 0
    assert fit["fit_max_relative_error"] >= 0


def test_single_thread_has_no_hybrid_cpu_worker_rate():
    result = profile(threads=1)
    assert result["rates"]["cpu_worker_flops_per_second"] == 0
    assert result["rates"]["cpu_worker_memory_bytes_per_second"] == 0
    validate_profile(result, cpu_threads=1, precision_bits=64)
    result["rates"]["cpu_worker_flops_per_second"] = 1
    with pytest.raises(ProfileError, match="no CPU workers"):
        validate_profile(result)


@pytest.mark.parametrize("expected", [
    {"precision_bits": 32}, {"cpu_threads": 3}, {"hardware": {"cpu_name": "Another CPU"}},
    {"hardware": {"gpu_uuid": "another device"}}, {"hardware": {"compute_capability": "9.0"}},
    {"toolchain": {"nvcc_version": "NVCC 14"}}, {"toolchain": {"host_cxx_version": "Clang"}},
    {"toolchain": {"driver_version": 13050}},
])
def test_identity_and_thread_precision_mismatches_fail(expected):
    with pytest.raises(ProfileError, match="mismatch"):
        validate_profile(profile(), **expected)


def test_load_validates_costs_and_accepts_matching_identity(tmp_path):
    path = tmp_path / "hardware.json"
    result = profile()
    path.write_text(json.dumps(result))
    assert load_profile(path, precision_bits=64, cpu_threads=4,
                        hardware={"gpu_uuid": result["hardware"]["gpu_uuid"].upper()},
                        toolchain={"host_cxx_version": "GCC 14.4"}) == result
    with pytest.raises(ProfileError, match="cannot load"):
        load_profile(tmp_path / "absent.json")
    path.write_text("not json")
    with pytest.raises(ProfileError, match="cannot load"):
        load_profile(path)


def test_runtime_compiler_identity_uses_explicit_macros_not_banner_equality():
    result = profile()
    result["toolchain"]["nvcc_version"] = "nvcc: NVIDIA\nCuda compilation tools, release 13.4, V13.4.92"
    result["toolchain"]["host_cxx_version"] = "x86_64-linux-gnu-g++-14 (Debian 14.4.0-2) 14.4.0\nCopyright"
    assert compiler_identity(result) == {
        "cpu_name": "Test CPU", "host_compiler": "14.4.0", "cuda_compiler": "13.4.92",
        "cuda_runtime": 13040, "cuda_driver": 13040,
    }


@pytest.mark.parametrize(("field", "banner"), [
    ("host_cxx_version", "clang version 14.4.0"),
    ("host_cxx_version", "g++ 14.4"),
    ("nvcc_version", "Cuda compilation tools, release 13.4"),
])
def test_unknown_compiler_identity_cannot_enable_automatic_execution(field, banner):
    result = profile()
    result["toolchain"].update(nvcc_version="NVCC V13.4.92", host_cxx_version="GCC 14.4.0")
    result["toolchain"][field] = banner
    with pytest.raises(ProfileError, match="cannot be verified"):
        compiler_identity(result)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "1000", None])
def test_invalid_or_unmeasured_rates_never_get_defaults(value):
    result = profile()
    result["rates"]["gpu_flops_per_second"] = value
    with pytest.raises(ProfileError):
        validate_profile(result)
    result = profile()
    result["rates"]["h2d_pinned"]["bandwidth_bytes_per_second"] = value
    with pytest.raises(ProfileError):
        validate_profile(result)


def test_missing_rates_and_schema_fail():
    result = profile()
    del result["rates"]["pack_bytes_per_second"]
    with pytest.raises(ProfileError):
        validate_profile(result)
    result = profile()
    result["schema_version"] = 2
    with pytest.raises(ProfileError, match="schema"):
        validate_profile(result)
    result = profile()
    result["cpu_threads"] = True
    with pytest.raises(ProfileError, match="cpu_threads"):
        validate_profile(result)


@pytest.mark.parametrize("text", ["", "diagnostic line", "[]", '{"kind":"unknown"}',
                                  '{"kind":"rate","name":"gpu_flops_per_second","seconds":[0,1,1]}'])
def test_measurement_protocol_rejects_invalid_output(text):
    with pytest.raises(CalibrationError):
        parse_measurements(text)


def test_missing_duplicate_and_wrong_team_measurements_fail():
    records = observations()
    mutations = [records[1:], records + [deepcopy(records[0])],
                 [r for r in records if r.get("name") != "gpu_flops_per_second"],
                 records + [deepcopy(records[-1])], records + [deepcopy(records[1])]]
    wrong_team = deepcopy(records)
    wrong_team[0]["cpu_threads"] = 2
    mutations.append(wrong_team)
    for candidate in mutations:
        with pytest.raises(CalibrationError):
            profile_from_measurements(candidate, precision_bits=64, cpu_threads=4,
                                      cpu_name="CPU", nvcc_version="nvcc", host_cxx_version="c++")


def test_nonphysical_transfer_fit_fails():
    records = [{"bytes": count, "seconds": [duration] * 5}
               for count, duration in ((4096, 0.3), (65536, 0.2), (1048576, 0.1))]
    with pytest.raises(CalibrationError, match="seconds per byte"):
        fit_transfer(records)
    with pytest.raises(CalibrationError, match="distinct"):
        fit_transfer(records[:2])


RUNTIME_ID = "1234abcd" * 8


def scoped_observations():
    records = [deepcopy(observations()[0])]
    special = {"allocation_seconds", "release_seconds", "planning_operation_seconds", "cold_driver_startup_seconds"}
    for name in SCOPED_COST_NAMES:
        if name not in special:
            records.append({"kind": "scoped_cost", "name": name, "seconds": [2e-6, 2e-6, 1, 2e-6, 2e-6]})
    for name in ("allocation_seconds", "release_seconds"):
        for index, size in enumerate((4096, 65536, 1048576, 67108864), 1):
            records.append({"kind": "scoped_allocation", "name": name, "bytes": size, "seconds": [index * 1e-5] * 5})
    for work, cost in ((4, 1e-6), (32, 2e-6)):
        records.append({"kind": "scoped_planning", "work": work, "seconds": [work * cost] * 5})
    return records


def scoped_profile():
    return profile_with_scoped_measurements(profile(), scoped_observations(), runtime_id=RUNTIME_ID,
                                             cold_startup_seconds=[0.1, 0.1, 10, 0.1, 0.1])


def scoped_transfer_observations():
    records = []
    for index, size in enumerate(SCOPED_BATCH_PAYLOADS, 1):
        for name, value in (("staging_cold_seconds", index * 1e-4), ("staging_reuse_seconds", index * 1e-6)):
            records.append({"kind": "scoped_transfer", "name": name, "bytes": size, "seconds": [value] * 5})
        for name in ("pack_bytes_per_second", "unpack_bytes_per_second"):
            for geometry, rows in (("contiguous", 1), ("thin_rows", size // 8)):
                duration = size / 1e9 + (rows * 1e-8 if geometry == "thin_rows" else 0)
                records.append({"kind": "scoped_transfer", "name": name, "bytes": size,
                                "rows": rows, "geometry": geometry, "seconds": [duration] * 5})
    for name in ("event_record_seconds", "event_wait_seconds", "ready_event_seconds"):
        records.append({"kind": "scoped_transfer", "name": name, "seconds": [1e-6] * 5})
    records.append({"kind": "scoped_transfer", "name": "preparation_operation_seconds",
                    "work": 10, "seconds": [2e-6] * 5})
    return records


def test_scoped_transfer_calibration_prices_thin_rows_and_each_slot_capacity():
    records = scoped_transfer_observations()
    result = profile_with_scoped_measurements(profile(), scoped_observations() + records,
                                             runtime_id=RUNTIME_ID, cold_startup_seconds=[0.1] * 5)
    costs = scoped_transfer_costs(result, RUNTIME_ID)
    assert costs["staging_cold_seconds"] == pytest.approx([1e-4, 2e-4, 3e-4, 4e-4])
    assert costs["staging_reuse_seconds"] == pytest.approx([1e-6, 2e-6, 3e-6, 4e-6])
    assert costs["pack_bytes_per_second"] == pytest.approx(1e9)
    assert costs["unpack_bytes_per_second"] == pytest.approx(1e9)
    assert costs["pack_row_seconds"] == pytest.approx(1e-8)
    assert costs["unpack_row_seconds"] == pytest.approx(1e-8)
    assert costs["preparation_operation_seconds"] == pytest.approx(2e-7)
    assert result["scoped"]["transfers"]["measurements"] == records
    assert parse_scoped_measurements("\n".join(json.dumps(record) for record in records)) == records
    # Direct costs remain independently usable; missing transfer costs cannot
    # be reconstructed from pinned bandwidth or application measurements.
    validate_profile(scoped_profile())
    with pytest.raises(ProfileError, match="no scoped transfer calibration"):
        scoped_transfer_costs(scoped_profile(), RUNTIME_ID)
    with pytest.raises(ProfileError, match="runtime_id mismatch"):
        scoped_transfer_costs(result, "0" * 64)


@pytest.mark.parametrize("mutation", ["missing_size", "missing_geometry", "duplicate", "no_work", "bad_rows"])
def test_incomplete_scoped_transfer_observations_never_supply_guessed_costs(mutation):
    records = scoped_transfer_observations()
    if mutation == "missing_size":
        records = [record for record in records if record.get("bytes") != SCOPED_BATCH_PAYLOADS[-1]]
    elif mutation == "missing_geometry":
        records = [record for record in records if record.get("geometry") != "thin_rows"]
    elif mutation == "duplicate":
        records.append(deepcopy(records[0]))
    elif mutation == "no_work":
        records[-1]["work"] = 0
    else:
        next(record for record in records if "rows" in record)["rows"] = True
    with pytest.raises(CalibrationError):
        scoped_transfers_from_measurements(records)


@pytest.mark.parametrize(("name", "value"), [
    ("staging_cold_seconds", [1e-4]), ("staging_reuse_seconds", [1e-5, 1e-5, 0, 1e-5]),
    ("pack_bytes_per_second", 0), ("pack_row_seconds", -1),
    ("event_record_seconds", float("nan")), ("preparation_operation_seconds", True),
])
def test_scoped_transfer_cost_validation_rejects_incompatible_models(name, value):
    result = scoped_profile()
    result["scoped"]["transfers"] = scoped_transfers_from_measurements(scoped_transfer_observations())
    result["scoped"]["transfers"]["costs"][name] = value
    with pytest.raises(ProfileError, match="scoped.transfers.costs"):
        validate_profile(result)


def test_scoped_extension_uses_real_measurements_and_separate_cold_setup():
    records = scoped_observations()
    assert parse_scoped_measurements("\n".join(json.dumps(record) for record in records)) == records
    result = scoped_profile()
    costs = scoped_costs(result, RUNTIME_ID)
    assert costs["cold_driver_startup_seconds"] == pytest.approx(0.1)
    assert costs["gpu_setup_seconds"] == pytest.approx(2e-6)
    assert costs["launch_enqueue_seconds"] == pytest.approx(2e-6)
    assert costs["launch_enqueue_seconds"] != result["rates"]["launch_latency_seconds"]
    assert costs["allocation_seconds"] == pytest.approx(4e-5)
    assert costs["release_seconds"] == pytest.approx(4e-5)
    assert costs["planning_operation_seconds"] == pytest.approx(2e-6)
    assert result["scoped"]["max_allocation_bytes"] == 67108864
    assert result["scoped"]["measurements"] == records
    assert result["measurements"] == profile()["measurements"]


def test_old_profile_stays_valid_but_cannot_enable_scoped_automatic():
    base = profile()
    validate_profile(base)
    assert "scoped" not in base
    with pytest.raises(ProfileError, match="no scoped runtime calibration"):
        validate_profile(base, scoped_runtime_id=RUNTIME_ID)
    with pytest.raises(ProfileError, match="runtime_id mismatch"):
        validate_profile(scoped_profile(), scoped_runtime_id="abcd1234" * 8)


@pytest.mark.parametrize("value", [None, True, 0, -1, float("nan"), float("inf"), "1"])
@pytest.mark.parametrize("cost", SCOPED_COST_NAMES)
def test_every_scoped_cost_must_be_measured_positive_finite(cost, value):
    result = scoped_profile()
    result["scoped"]["costs"][cost] = value
    with pytest.raises(ProfileError, match="scoped.costs"):
        validate_profile(result)


@pytest.mark.parametrize(("field", "value"), [
    ("schema_version", True), ("schema_version", 2), ("runtime_id", ""),
    ("runtime_id", RUNTIME_ID.upper()), ("max_allocation_bytes", True),
    ("max_allocation_bytes", 0), ("max_allocation_bytes", 2**64),
    ("max_allocation_bytes", 1.5), ("costs", None),
])
def test_scoped_identity_schema_and_allocation_bound_are_validated(field, value):
    result = scoped_profile()
    result["scoped"][field] = value
    with pytest.raises(ProfileError, match="scoped"):
        validate_profile(result)


def test_missing_or_duplicate_scoped_measurements_do_not_supply_defaults():
    records = scoped_observations()
    candidates = [records[1:], records + [deepcopy(records[0])],
                  [record for record in records if record.get("name") != "wait_seconds"],
                  records + [deepcopy(records[1])],
                  records + [deepcopy(next(record for record in records if record["kind"] == "scoped_allocation"))],
                  [record for record in records if record["kind"] != "scoped_planning"],
                  [record for record in records if record.get("bytes") != 67108864 or record.get("name") != "release_seconds"]]
    wrong_team, wrong_device = deepcopy(records), deepcopy(records)
    wrong_team[0]["cpu_threads"] = 1
    wrong_device[0]["gpu_uuid"] = "GPU-other"
    candidates += [wrong_team, wrong_device]
    for candidate in candidates:
        with pytest.raises(CalibrationError):
            profile_with_scoped_measurements(profile(), candidate, runtime_id=RUNTIME_ID,
                                               cold_startup_seconds=[0.1] * 5)


@pytest.mark.parametrize("value", [0, True, 1.5, "1", None, -1])
def test_planner_cost_requires_actual_integer_work_count(value):
    records = scoped_observations()
    records[-1]["work"] = value
    with pytest.raises(CalibrationError, match="planning work"):
        profile_with_scoped_measurements(profile(), records, runtime_id=RUNTIME_ID,
                                           cold_startup_seconds=[0.1] * 5)


def test_fresh_process_startup_samples_cannot_be_guessed_or_incomplete():
    for values in ([0.1], [0.1, 0, 0.1], [0.1, True, 0.1]):
        with pytest.raises(CalibrationError):
            profile_with_scoped_measurements(profile(), scoped_observations(), runtime_id=RUNTIME_ID,
                                               cold_startup_seconds=values)


@pytest.mark.parametrize("text", ["", "not json", "[]", '{"kind":"other"}',
                                   '{"kind":"scoped_cost","name":"wait_seconds","seconds":[0,1,1]}'])
def test_scoped_native_protocol_rejects_unmeasured_or_bad_records(text):
    with pytest.raises(CalibrationError):
        parse_scoped_measurements(text)


def test_scoped_cost_cli_is_explicit_and_preserves_default(monkeypatch, tmp_path):
    from compiler.offload import calibrate as module
    calls = []
    monkeypatch.setattr(module, "calibrate", lambda args: calls.append(args))
    assert module.main(["--output", str(tmp_path / "ordinary.json")]) == 0
    assert calls[-1].scoped_costs is False
    assert module.main(["--output", str(tmp_path / "scoped.json"), "--scoped-costs"]) == 0
    assert calls[-1].scoped_costs is True


def test_default_calibration_never_builds_or_measures_common_runtime(monkeypatch, tmp_path):
    from compiler.offload import calibrate as module
    commands = []
    monkeypatch.setattr(module, "_tool", lambda path, candidates: candidates[0])

    def run(argv, directory, log, *, timeout):
        commands.append(argv)
        if "--version" in argv:
            return "NVCC V13.4.92" if argv[0] == "nvcc" else "GCC 14.4.0"
        if argv[0].endswith("/calibration"):
            return "\n".join(json.dumps(record) for record in observations())
        return ""

    monkeypatch.setattr(module, "_run", run)
    monkeypatch.setattr(module, "_calibrate_scoped", lambda *args: pytest.fail("ordinary calibration used scoped runtime"))
    destination = tmp_path / "ordinary.json"
    assert module.main(["--output", str(destination)]) == 0
    result = json.loads(destination.read_text())
    assert "scoped" not in result
    assert len(commands) == 4
    assert result["calibration"]["application_profiled"] is False
    assert all("scoped" not in str(value) for command in commands for value in command)


def test_scoped_refresh_preserves_exact_base_rates_and_records_input_identity(monkeypatch, tmp_path):
    import hashlib

    from compiler.offload import calibrate as module
    base = profile()
    base["toolchain"].update(nvcc_version="NVCC V13.4.92", host_cxx_version="GCC 14.4.0")
    source = tmp_path / "base.json"
    source.write_text(json.dumps(base))
    source_bytes = source.read_bytes()
    commands = []
    monkeypatch.setattr(module, "_tool", lambda requested, candidates: candidates[0])
    monkeypatch.setattr(module, "cpu_identity", lambda: base["hardware"]["cpu_name"])

    def run(argv, directory, log, *, timeout, env=None):
        commands.append(argv)
        assert argv[1:] == ["--version"]
        return base["toolchain"]["nvcc_version" if argv[0] == "nvcc" else "host_cxx_version"]

    def refresh(old, *args):
        assert old == base
        # A later source edit must not alter the provenance of rates already
        # loaded, or falsely attribute the refreshed costs to those new bytes.
        source.write_text("changed after loading")
        return {**old, "scoped": {"calibration": {}, "runtime_id": RUNTIME_ID}}

    monkeypatch.setattr(module, "_run", run)
    monkeypatch.setattr(module, "_calibrate_scoped", refresh)
    destination = tmp_path / "refresh.json"
    assert module.main(["--output", str(destination), "--scoped-costs", "--refresh-scoped", str(source)]) == 0
    result = json.loads(destination.read_text())
    assert result["rates"] == base["rates"]
    assert result["measurements"] == base["measurements"]
    assert len(commands) == 2
    provenance = result["scoped"]["calibration"]["base_profile"]
    assert provenance["sha256"] == hashlib.sha256(source_bytes).hexdigest()
    assert provenance["base_rates_remeasured"] is False


@pytest.mark.parametrize("change", [{"cpu_threads": 2}, {"precision_bits": 32},
                                  {"hardware": {"cpu_name": "different"}},
                                  {"toolchain": {"nvcc_version": "other"}}])
def test_scoped_refresh_requires_matching_base_identity_before_measurement(monkeypatch, tmp_path, change):
    from compiler.offload import calibrate as module
    base = profile()
    base["toolchain"].update(nvcc_version="NVCC V13.4.92", host_cxx_version="GCC 14.4.0")
    for key, value in change.items():
        if isinstance(value, dict):
            base[key].update(value)
        else:
            base[key] = value
    source = tmp_path / "base.json"
    source.write_text(json.dumps(base))
    monkeypatch.setattr(module, "_tool", lambda requested, candidates: candidates[0])
    monkeypatch.setattr(module, "cpu_identity", lambda: "Test CPU")
    monkeypatch.setattr(module, "_run", lambda argv, *args, **kwargs:
                        "NVCC V13.4.92" if argv[0] == "nvcc" else "GCC 14.4.0")
    monkeypatch.setattr(module, "_calibrate_scoped", lambda *args: pytest.fail("mismatched calibration was measured"))
    assert module.main(["--output", str(tmp_path / "bad.json"), "--scoped-costs", "--refresh-scoped", str(source)]) == 2
    assert not (tmp_path / "bad.json").exists()


def test_scoped_runner_uses_published_runtime_and_fresh_processes(monkeypatch, tmp_path):
    from compiler.offload import calibrate as module
    commands = []
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    monkeypatch.setenv("FORT_SCOPE_TEST_FAIL_ALLOC", "1")

    def run(argv, directory, log, *, timeout, env=None):
        commands.append(argv)
        if argv[0] == "nvcc":
            return ""
        assert env is not None
        assert "FORT_RUNTIME_TRACE" not in env
        assert "CUDA_LAUNCH_BLOCKING" not in env
        assert "FORT_SCOPE_TEST_FAIL_ALLOC" not in env
        if "--cold-startup" in argv:
            return json.dumps({"kind": "cold_driver_startup", "seconds": 0.05})
        return "\n".join(json.dumps(record) for record in scoped_observations())

    monkeypatch.setattr(module, "_run", run)
    args = SimpleNamespace(arch="sm_86", precision=64, threads=4, max_mib=64, device=0)
    result = module._calibrate_scoped(profile(), args, tmp_path, "nvcc", "g++-14")
    manifest = result["scoped"]["calibration"]["runtime"]
    assert result["scoped"]["runtime_id"] == manifest["runtime_id"]
    sources, _ = module.read_scoped_runtime()
    for name, content in sources.items():
        assert (tmp_path / "scoped" / name).read_text() == content
    assert len(commands) == 7
    assert sum("--cold-startup" in command for command in commands) == 5
    assert result["scoped"]["costs"]["cold_driver_startup_seconds"] == 0.05
