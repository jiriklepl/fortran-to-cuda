"""Offline calibration uses measured costs and fails closed on bad profiles."""
import json
from copy import deepcopy

import pytest

from compiler.offload.calibrate import (
    CalibrationError,
    fit_transfer,
    parse_measurements,
    profile_from_measurements,
)
from compiler.offload.profile import (
    LATENCY_RATES,
    THROUGHPUT_RATES,
    WORKER_RATES,
    ProfileError,
    compiler_identity,
    load_profile,
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
                                     cpu_name="Test CPU", nvcc_version="NVCC 13.4", host_cxx_version="GCC 14.4")


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
