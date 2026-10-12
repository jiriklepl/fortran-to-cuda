"""Independent raw GPU fits, fixed-launch holds and historical/current fences."""

import json
from copy import deepcopy

import pytest

from compiler.offload import numerical_gpu_execution as gpu
from compiler.offload import numerical_gpu_execution_costs as costs
from compiler.offload.numerical_execution import ACCESS_CLASS, _hash
from compiler.offload.numerical_execution_workloads import execution_registry
from compiler.tests.test_numerical_gpu_execution import observations

FIXED = 5e-6
ARITHMETIC = 1e-11
INTRINSIC = 2e-10
MEMORY = 1e-12


def evidence(precision=64):
    profile, rows, calibration = observations(precision)
    registry = {r["name"]: r for r in execution_registry(precision)}
    for row in rows[1 + gpu.PREFLIGHT_RECORDS:]:
        recipe = registry[row["recipe"]]
        compute = recipe["arithmetic"] * ARITHMETIC + sum(recipe["intrinsics"].values()) * INTRINSIC
        variable = row["traffic_bytes"] * MEMORY if recipe["role"] == "memory" else max(
            row["items"] * compute, row["traffic_bytes"] * MEMORY)
        row["resident_event_seconds"] = (FIXED + variable) * row["launches"]
        row["wall_seconds"] = max(.25, row["resident_event_seconds"] + .001)
    return gpu.profile_from_gpu_execution_measurements(profile, rows, calibration)


def launch_observations(profile):
    parent = profile[gpu.SECTION]
    keys = ("kind", "protocol_id", "backend_id", "generator_id", "source_generator_id", "registry_id",
            "precision_bits", "device_ordinal", "gpu_name", "gpu_uuid", "compute_capability",
            "cuda_runtime_version", "driver_version", "sampling_cpu_affinity", "nvcc_version", "host_cxx_version")
    paths = {**parent["identity"]["build_paths"], "artifacts": "/launch-controls"}
    implementation = costs.cost_sources()
    identity = {**{k: parent["identity"][k] for k in keys}, "launch_protocol_id": costs.PROTOCOL_ID,
        "parent_evidence_sha256": _hash(parent), "cost_sources": implementation,
        "emitted_sources": {"parent_driver.cu": parent["identity"]["emitted_sources"]["driver.cu"],
            "gpu_recipes.hpp": parent["identity"]["emitted_sources"]["gpu_recipes.hpp"],
            "launch_driver.cu": implementation["numerical_gpu_launch_driver.cu"]},
        "objects": {"workers.o": parent["identity"]["objects"]["workers.o"], "launch_driver.o": "5" * 64},
        "binary_sha256": "6" * 64, "build_paths": paths, "architecture": parent["identity"]["architecture"],
        "build_commands": costs.launch_build_commands(paths, parent["identity"]["architecture"])}
    recipe = next(r for r in parent["registry"] if r["name"] == "memory")

    def coordinate(size, kind):
        return {"kind": kind, "recipe": "memory", "recipe_id": recipe["recipe_id"], "backend": "gpu",
            "items": size, "role": "holdout", "agreement_passed": True,
            "traffic_bytes": size * 3 * (profile["precision_bits"] // 8),
            "working_set_bytes": size * 3 * (profile["precision_bits"] // 8)}

    rows = [identity, *[{**coordinate(n, costs.PREFLIGHT_KIND), "launches": int(n > 0)} for n in costs.CONTROL_SIZES]]
    for batch in range(7):
        for offset, size in enumerate(costs.MEASURED_SIZES):
            rows.append({**coordinate(size, costs.BATCH_KIND), "batch": batch, "global_order": batch * 3 + offset,
                "repetitions": 50000, "launches": 50000, "resident_event_seconds": FIXED * 50000,
                "enqueue_seconds": .10, "wall_seconds": .3})

    def ref(path, digest="7" * 64):
        return {"path": path, "sha256": digest, "bytes": 123}

    parent_hashes = {**parent["identity"]["emitted_sources"], **parent["identity"]["objects"],
                     "numerical-gpu-execution": parent["identity"]["binary_sha256"]}
    argv = ["taskset", "-c", "4", "/launch-controls/numerical-gpu-launch", "0"]
    calibration = {"application_profiled": False, "parent_evidence_sha256": _hash(parent),
        "run_commands": [[*argv, mode] for mode in ("--identity", "--preflight", "--measure")],
        "environment": {k: "0" if k == "FORT_PHASE_TIMING" else None for k in gpu.ENVIRONMENT_KEYS},
        "raw_samples": ref("/launch-controls/launch-raw-samples.jsonl"),
        "tools": {key: ref(paths[key]) for key in ("nvcc", "host")},
        "parent_artifacts": {name: ref("/observations/" + name, digest) for name, digest in parent_hashes.items()},
        "input_profile": ref("/profiles/parent.json")}
    return rows, calibration


def query(profile, monkeypatch, counts=None, **kwargs):
    identity = profile[gpu.SECTION]["identity"]
    monkeypatch.setattr(gpu, "current_gpu_identities", lambda _: (
        identity["source_generator_id"], identity["registry_id"], identity["generator_id"], identity["renderer_sources"]))
    features = {"schema_version": 2, "classification_complete": True, "private_array_elements": 0}
    return costs.gpu_execution_costs(profile, {} if counts is None else counts,
        workload_class=kwargs.get("workload_class", "scalar_expression_v2"),
        workload_features=kwargs.get("workload_features", features), access_class=kwargs.get("access_class", ACCESS_CLASS))


@pytest.mark.parametrize("precision", [32, 64])
def test_complete_fixed_and_variable_costs_preserve_parent_and_do_not_add_enqueue(precision, monkeypatch):
    parent = evidence(precision)
    original = deepcopy(parent)
    rows, calibration = launch_observations(parent)
    profile = costs.profile_from_gpu_launch_measurements(parent, rows, calibration)
    costs.validate_gpu_execution_costs(profile)
    section = profile[costs.SECTION]
    assert parent == original
    assert profile[gpu.SECTION] == parent[gpu.SECTION]
    assert len(rows) == 26
    assert section["startup"]["fit_items"] == [1]
    assert section["startup"]["holdout_items"] == [8, 17]
    assert all(r["status"] == "accepted" for r in section["workload_validation"].values())
    result = query(profile, monkeypatch, {"sqrt": 2})
    assert result["launch_enqueue_included"] is True
    assert result["fixed_seconds"] == pytest.approx(FIXED)
    assert result["fixed_seconds"] != pytest.approx(FIXED + .1 / 50000)
    assert result["intrinsic_seconds_per_item"] == pytest.approx(2 * INTRINSIC)
    assert result["arithmetic_seconds_per_operation"] == pytest.approx(ARITHMETIC)
    assert result["item_range"] == [65536, 1048576]
    assert section["calibration"]["application_profiled"] is False
    assert "team coordination" in section["unpriced"]


@pytest.mark.parametrize("defect", ["missing", "order", "bool", "zero_launch", "short", "event", "nonfinite",
                                   "worker", "driver", "tools", "parent", "environment", "raw", "extra"])
def test_incomplete_control_or_provenance_cannot_supply_costs(defect):
    parent = evidence()
    rows, calibration = launch_observations(parent)
    if defect == "missing":
        rows.pop()
    elif defect == "order":
        rows[-1]["global_order"] = 0
    elif defect == "bool":
        rows[-1]["batch"] = True
    elif defect == "zero_launch":
        rows[1]["launches"] = 1
    elif defect == "short":
        rows[-1]["wall_seconds"] = .19
    elif defect == "event":
        rows[-1]["resident_event_seconds"] = 1.
    elif defect == "nonfinite":
        rows[-1]["enqueue_seconds"] = float("nan")
    elif defect == "worker":
        rows[0]["objects"]["workers.o"] = "0" * 64
    elif defect == "driver":
        rows[0]["emitted_sources"]["launch_driver.cu"] = "0" * 64
    elif defect == "tools":
        calibration["tools"].pop("host")
    elif defect == "parent":
        calibration["parent_artifacts"]["driver.cu"]["sha256"] = "0" * 64
    elif defect == "environment":
        calibration["environment"]["CUDA_LAUNCH_BLOCKING"] = "1"
    elif defect == "raw":
        calibration["raw_samples"]["bytes"] = True
    else:
        rows[-1]["compute_seconds"] = .1
    with pytest.raises(gpu.NumericalGpuExecutionError):
        costs.profile_from_gpu_launch_measurements(parent, rows, calibration)


def test_tiny_independent_holdout_rejection_is_retained_without_retry(monkeypatch):
    parent = evidence()
    rows, calibration = launch_observations(parent)
    for row in rows[5:]:
        if row["items"] == 17:
            row["resident_event_seconds"] *= .5
    profile = costs.profile_from_gpu_launch_measurements(parent, rows, calibration)
    assert profile[costs.SECTION]["startup"]["reason_codes"] == ["holdout_error"]
    assert profile[costs.SECTION]["measurements"] == rows
    assert profile[costs.SECTION]["cost_model_available"] is False
    with pytest.raises(gpu.NumericalGpuExecutionError, match="class rejected"):
        query(profile, monkeypatch)


@pytest.mark.parametrize("name", ["ordinary_mix", "private_mix", "domain_sqrt"])
def test_independent_mixed_private_and_domain_rows_cannot_be_hidden(name, monkeypatch):
    parent = evidence()
    # Reconstruct the original evidence after a systematic held-out mismatch.
    original = parent[gpu.SECTION]
    rows = deepcopy(original["measurements"])
    for row in rows[1 + gpu.PREFLIGHT_RECORDS:]:
        if row["recipe"] == name:
            row["resident_event_seconds"] *= .5
    base = {k: v for k, v in parent.items() if k != gpu.SECTION}
    parent = gpu.profile_from_gpu_execution_measurements(base, rows, original["calibration"])
    controls, calibration = launch_observations(parent)
    profile = costs.profile_from_gpu_launch_measurements(parent, controls, calibration)
    if name == "domain_sqrt":
        assert profile[costs.SECTION]["domain_validation"]["sqrt"]["status"] == "rejected"
        # Domain controls are also independent scalar-class holds. That stronger
        # class rejection is checked before the separate primitive-domain gate.
        with pytest.raises(gpu.NumericalGpuExecutionError, match="class rejected"):
            query(profile, monkeypatch, {"sqrt": 1})
    else:
        kind = "ordinary_expression_v2" if name == "ordinary_mix" else "fixed_private_array_v2"
        assert profile[costs.SECTION]["workload_validation"][kind]["status"] == "rejected"
        with pytest.raises(gpu.NumericalGpuExecutionError, match="class rejected"):
            query(profile, monkeypatch, workload_class=kind)


def test_historical_reader_and_current_query_are_separate(monkeypatch):
    parent = evidence()
    rows, calibration = launch_observations(parent)
    profile = costs.profile_from_gpu_launch_measurements(parent, rows, calibration)
    monkeypatch.setattr(gpu, "current_gpu_identities", lambda _: pytest.fail("historical reader queried current sources"))
    costs.validate_gpu_execution_costs(profile)
    monkeypatch.setattr(gpu, "current_gpu_identities", lambda _: ("0" * 64, "0" * 64, "0" * 64, {}))
    with pytest.raises(gpu.NumericalGpuExecutionError, match="current source"):
        costs.gpu_execution_costs(profile, {}, workload_class="scalar_expression_v2", workload_features={}, access_class=ACCESS_CLASS)
    costs.validate_gpu_execution_costs(profile)
    profile[costs.SECTION]["startup"]["fixed_seconds"] *= 2
    with pytest.raises(gpu.NumericalGpuExecutionError, match="reconstruction"):
        costs.validate_gpu_execution_costs(profile)


def test_public_profile_loading_validates_archived_costs_without_current_rebinding(tmp_path, monkeypatch):
    from compiler.offload.profile import ProfileError, load_profile

    parent = evidence()
    rows, calibration = launch_observations(parent)
    profile = costs.profile_from_gpu_launch_measurements(parent, rows, calibration)
    path = tmp_path / "archived-gpu-costs.json"
    path.write_text(json.dumps(profile))
    monkeypatch.setattr(gpu, "current_gpu_identities", lambda _: pytest.fail("load rebound archived GPU identity"))
    monkeypatch.setattr(costs, "cost_sources", lambda: pytest.fail("load rebound archived cost identity"))
    assert load_profile(path) == profile
    monkeypatch.setattr(gpu, "current_gpu_identities", lambda _: ("0" * 64, "0" * 64, "0" * 64, {}))
    with pytest.raises(gpu.NumericalGpuExecutionError, match="current source"):
        costs.gpu_execution_costs(profile, {}, workload_class="scalar_expression_v2", workload_features={}, access_class=ACCESS_CLASS)
    profile[costs.SECTION]["startup"]["fixed_seconds"] *= 2
    path.write_text(json.dumps(profile))
    with pytest.raises(ProfileError, match="reconstruction"):
        load_profile(path)


@pytest.mark.parametrize("value", [False, 1, "true", None])
def test_launch_inclusion_marker_requires_exact_boolean(value):
    parent = evidence()
    rows, calibration = launch_observations(parent)
    profile = costs.profile_from_gpu_launch_measurements(parent, rows, calibration)
    profile[costs.SECTION]["launch_enqueue_included"] = value
    with pytest.raises(gpu.NumericalGpuExecutionError):
        costs.validate_gpu_execution_costs(profile)


def test_private_shape_access_and_cost_implementation_are_independent_query_gates(monkeypatch):
    parent = evidence()
    rows, calibration = launch_observations(parent)
    profile = costs.profile_from_gpu_launch_measurements(parent, rows, calibration)
    features = {"schema_version": 2, "classification_complete": True,
        "private_array_groups": 4, "private_array_elements": 64,
        "max_private_array_elements": 16, "max_private_array_rank": 2}
    assert query(profile, monkeypatch, {"sqrt": 1}, workload_class="fixed_private_array_v2",
                 workload_features=features)["launch_enqueue_included"] is True
    features["private_array_elements"] = 65
    with pytest.raises(gpu.NumericalGpuExecutionError, match="outside validated"):
        query(profile, monkeypatch, workload_class="fixed_private_array_v2", workload_features=features)
    with pytest.raises(gpu.NumericalGpuExecutionError, match="access identity"):
        query(profile, monkeypatch, access_class="strided")
    monkeypatch.setattr(costs, "cost_sources", lambda: {"changed": "0" * 64})
    with pytest.raises(gpu.NumericalGpuExecutionError, match="implementation"):
        query(profile, monkeypatch)


@pytest.mark.parametrize("counts", [None, 1, [[[], 1]], {"sqrt": True}, [("sqrt", 1), ("sqrt", 2)], {"unknown": 1}])
def test_malformed_query_counts_have_controlled_errors(counts, monkeypatch):
    parent = evidence()
    rows, calibration = launch_observations(parent)
    profile = costs.profile_from_gpu_launch_measurements(parent, rows, calibration)
    identity = profile[gpu.SECTION]["identity"]
    monkeypatch.setattr(gpu, "current_gpu_identities", lambda _: (
        identity["source_generator_id"], identity["registry_id"], identity["generator_id"], identity["renderer_sources"]))
    with pytest.raises(gpu.NumericalGpuExecutionError, match="counts"):
        costs.gpu_execution_costs(profile, counts, workload_class="scalar_expression_v2", workload_features={}, access_class=ACCESS_CLASS)
