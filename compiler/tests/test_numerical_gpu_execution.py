"""GPU observations remain independently authenticated, never AUTO authority."""

from copy import deepcopy

import pytest

from compiler.numerical_contract import numerical_build_contract
from compiler.offload import numerical_gpu_execution as gpu
from compiler.offload.numerical_execution import _hash
from compiler.offload.numerical_execution_workloads import execution_registry
from compiler.tests.test_offload_profile import profile as base_profile


def observations(precision=64):
    profile = base_profile()
    profile["precision_bits"] = precision
    registry = list(execution_registry(precision))
    source = "e" * 64
    renderer = {"emission/cuda/kernels.py": "a" * 64}
    paths = {"nvcc": "/tools/nvcc", "host": "/tools/g++", "runtime": "/compiler/runtime", "artifacts": "/observations"}
    identity = {"kind": gpu.IDENTITY_KIND, "protocol_id": gpu.PROTOCOL_ID, "backend_id": gpu.BACKEND_ID,
        "precision_bits": precision, "registry_id": _hash(registry), "source_generator_id": source,
        "generator_id": gpu.gpu_generator_identity(source, _hash(registry), renderer), "renderer_sources": renderer,
        **{key: profile["hardware"][key] for key in ("gpu_name", "gpu_uuid", "compute_capability", "cpu_name")},
        **profile["toolchain"], "device_ordinal": 0, "sampling_cpu_affinity": [4],
        "numerical_contract": numerical_build_contract(),
        "emitted_sources": {name: "b" * 64 for name in ("workers.cu", "gpu_recipes.hpp", "driver.cu")},
        "objects": {"workers.o": "c" * 64, "driver.o": "d" * 64}, "binary_sha256": "f" * 64,
        "build_paths": paths, "architecture": "native", "build_commands": gpu.gpu_build_commands(paths, "native")}
    records = [identity]

    def coordinate(recipe, size):
        return {"recipe": recipe["name"], "recipe_id": recipe["recipe_id"], "backend": "gpu", "items": size,
            "role": "fit" if size in recipe["fit_sizes"] else "holdout", "agreement_passed": True,
            "traffic_bytes": size * 3 * (precision // 8), "working_set_bytes": size * 3 * (precision // 8)}

    for recipe in registry:
        for control, sizes in (("tiny", gpu.TINY_SIZES), ("cell", recipe["sizes"])):
            records.extend({"kind": gpu.PREFLIGHT_KIND, "control": control, "launches": int(size > 0),
                            **coordinate(recipe, size)} for size in sizes)
    order = 0
    for batch in range(7):
        for recipe in registry:
            for size in recipe["sizes"]:
                records.append({"kind": gpu.BATCH_KIND, "batch": batch, "global_order": order,
                    "repetitions": 100, "launches": 100, "resident_event_seconds": .19,
                    "enqueue_seconds": .01, "wall_seconds": .25, **coordinate(recipe, size)})
                order += 1
    calibration = {"application_profiled": False, "automatic_placement_authorized": False,
                   "base_profile_identity": _hash(profile), "compatibility_identity": gpu.compatibility_identity(profile),
                   "sampling_cpu_affinity": [4], "artifacts": "/observations", "cpu_evidence_revalidated": False,
                   "environment": {key: "0" if key == "FORT_PHASE_TIMING" else None for key in gpu.ENVIRONMENT_KEYS},
                   "raw_samples": {"path": "/observations/gpu-raw-samples.jsonl", "sha256": "1" * 64, "bytes": 1000},
                   "run_commands": [["taskset", "-c", "4", "/observations/numerical-gpu-execution", "0", mode]
                                    for mode in ("--identity", "--preflight", "--measure")]}
    return profile, records, calibration


@pytest.mark.parametrize("precision", [32, 64])
def test_complete_raw_gpu_evidence_is_not_a_cost_model(precision):
    profile, records, calibration = observations(precision)
    original = deepcopy(profile)
    result = gpu.profile_from_gpu_execution_measurements(profile, records, calibration)
    gpu.validate_gpu_execution_profile(result)
    section = result[gpu.SECTION]
    assert len(records) == gpu.MAX_RECORDS == 845
    assert section["preflight_cases"] == 165
    assert len(section["observations"]) == 97
    assert section["gpu_measured"] is True
    assert not section["cost_model_available"]
    assert not section["fixed_launch_price_available"]
    assert not section["automatic_placement_authorized"]
    assert profile == original
    assert {key: value for key, value in result.items() if key != gpu.SECTION} == original


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "recipe", "role", "order", "bool", "short",
                                     "enqueue", "event", "launches", "tiny_launch", "preflight", "nonfinite", "extra"])
def test_incomplete_or_conflicting_gpu_observations_cannot_authorize(mutation):
    profile, records, calibration = observations()
    if mutation == "missing":
        records.pop()
    elif mutation == "duplicate":
        records[-1] = deepcopy(records[-2])
    elif mutation == "recipe":
        records[-1]["recipe_id"] = "0" * 64
    elif mutation == "role":
        records[-1]["role"] = "fit"
    elif mutation == "order":
        records[-1]["global_order"] = 0
    elif mutation == "bool":
        records[-1]["batch"] = True
    elif mutation == "short":
        records[-1]["wall_seconds"] = .199
    elif mutation == "enqueue":
        records[-1]["enqueue_seconds"] = .3
    elif mutation == "event":
        records[-1]["resident_event_seconds"] = .3
    elif mutation == "launches":
        records[-1]["launches"] = 99
    elif mutation == "tiny_launch":
        records[1]["launches"] = 1
    elif mutation == "preflight":
        records[1]["agreement_passed"] = False
    elif mutation == "nonfinite":
        records[-1]["resident_event_seconds"] = float("nan")
    else:
        records[-1]["application_seconds"] = 1.
    with pytest.raises(gpu.NumericalGpuExecutionError):
        gpu.profile_from_gpu_execution_measurements(profile, records, calibration)


def test_completed_numerical_rejection_is_retained_without_fit_or_retry():
    profile, records, calibration = observations()
    records[-1]["agreement_passed"] = False
    result = gpu.profile_from_gpu_execution_measurements(profile, records, calibration)
    gpu.validate_gpu_execution_profile(result)
    assert result[gpu.SECTION]["observations"][-1]["status"] == "rejected"
    assert result[gpu.SECTION]["measurements"][-1] == records[-1]


@pytest.mark.parametrize("override", [["--fmad=true"], ["-fmad=true"], ["--fmad", "true"],
    ["--use_fast_math"], ["-Xcompiler=-ffast-math"], ["-Xcompiler", "-ffp-contract=fast"], ["--ftz=true"]])
def test_exact_argv_grammar_rejects_contradictory_option_spellings(override):
    profile, records, calibration = observations()
    records[0]["build_commands"][0].extend(override)
    with pytest.raises(gpu.NumericalGpuExecutionError, match="exact strict"):
        gpu.profile_from_gpu_execution_measurements(profile, records, calibration)


def test_historical_reading_is_metadata_only_and_current_use_is_separate(monkeypatch):
    from compiler.offload import numerical_execution_workloads as workloads

    profile, records, calibration = observations()
    result = gpu.profile_from_gpu_execution_measurements(profile, records, calibration)
    monkeypatch.setattr(workloads, "execution_recipes", lambda *_: pytest.fail("reader lowered source"))
    gpu.validate_gpu_execution_profile(result)
    monkeypatch.setattr(gpu, "current_gpu_identities", lambda _: ("0" * 64, "0" * 64, "0" * 64, {}))
    with pytest.raises(gpu.NumericalGpuExecutionError, match="current source"):
        gpu.require_current_gpu_execution(result)
    gpu.validate_gpu_execution_profile(result)


def test_optional_cpu_evidence_can_evolve_without_rebinding_gpu_observations(monkeypatch):
    from compiler.offload import numerical_execution as cpu
    from compiler.tests.test_numerical_execution import observations as cpu_observations

    profile, records, calibration = observations()
    result = gpu.profile_from_gpu_execution_measurements(profile, records, calibration)
    _, cpu_records, registry = cpu_observations()
    cpu_records[0]["cpu_name"] = result["hardware"]["cpu_name"]
    cpu_records[0]["host"]["compiler_version"] = result["toolchain"]["host_cxx_version"]
    monkeypatch.setattr(cpu, "_current_identities", lambda _: (cpu_records[0]["generator_id"], _hash(registry)))
    amended = cpu.profile_from_execution_measurements(result, cpu_records)
    cpu.validate_execution_profile(amended)
    gpu.validate_gpu_execution_profile(amended)
    assert amended[gpu.SECTION] == result[gpu.SECTION]
    with pytest.raises(cpu.NumericalExecutionError, match="CPU-only.*GPU evidence unavailable"):
        cpu.execution_compute_model(amended, {"sqrt": 5}, native_participation="fork_join")
    amended["hardware"]["gpu_uuid"] = "different"
    with pytest.raises(gpu.NumericalGpuExecutionError, match="hardware"):
        gpu.validate_gpu_execution_profile(amended)


def test_mutated_summary_contract_and_base_compatibility_reject():
    profile, records, calibration = observations()
    result = gpu.profile_from_gpu_execution_measurements(profile, records, calibration)
    result[gpu.SECTION]["observations"][0]["median_wall_seconds_per_launch"] *= 2
    with pytest.raises(gpu.NumericalGpuExecutionError, match="reconstruction"):
        gpu.validate_gpu_execution_profile(result)
    result = gpu.profile_from_gpu_execution_measurements(profile, records, calibration)
    result["numerical_contract"]["id"] = "another-contract"
    with pytest.raises(gpu.NumericalGpuExecutionError, match="contract"):
        gpu.validate_gpu_execution_profile(result)


@pytest.mark.parametrize("invalid", [None, 1, []])
@pytest.mark.parametrize("index", [1, -1])
def test_malformed_raw_records_have_controlled_errors(invalid, index):
    profile, records, calibration = observations()
    records[index] = invalid
    with pytest.raises(gpu.NumericalGpuExecutionError):
        gpu.profile_from_gpu_execution_measurements(profile, records, calibration)


def test_missing_base_contract_and_incomplete_source_inventory_reject():
    profile, records, calibration = observations()
    profile.pop("numerical_contract")
    with pytest.raises(gpu.NumericalGpuExecutionError, match="contract"):
        gpu.profile_from_gpu_execution_measurements(profile, records, calibration)
    profile, records, calibration = observations()
    records[0]["emitted_sources"].pop("driver.cu")
    with pytest.raises(gpu.NumericalGpuExecutionError, match="complete.*sources"):
        gpu.profile_from_gpu_execution_measurements(profile, records, calibration)


@pytest.mark.parametrize("key", ["missing", "FORT_PHASE_TIMING", "CUDA_LAUNCH_BLOCKING"])
def test_sampling_environment_requires_exact_unprofiled_provenance(key):
    profile, records, calibration = observations()
    if key == "missing":
        calibration.pop("environment")
    else:
        calibration["environment"][key] = "1"
    with pytest.raises(gpu.NumericalGpuExecutionError, match="unprofiled"):
        gpu.profile_from_gpu_execution_measurements(profile, records, calibration)
