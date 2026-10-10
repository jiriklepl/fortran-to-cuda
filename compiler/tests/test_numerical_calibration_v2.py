"""Independent native/CPU/GPU evidence and conservative v3 applicability."""
import json
import math
from copy import deepcopy
from types import SimpleNamespace

import pytest

from compiler.offload.numerical_calibration import (
    COMPUTE_BACKEND_ID,
    COMPUTE_BACKENDS,
    COMPUTE_CLASSES,
    COMPUTE_FIT_SIZES,
    COMPUTE_HOLDOUT_SIZES,
    COMPUTE_RECIPES,
    COMPUTE_SIZES,
    MEMORY_FIT_SIZES,
    MEMORY_HOLDOUT_SIZES,
    MEMORY_SIZES,
    NumericalCalibrationError,
    calibrate_numerical_v2,
    compute_generator_identity,
    memory_compute_seconds,
    native_fixture_source,
    numerical_compute_model,
    numerical_intrinsic_costs,
    parse_compute_measurements,
    profile_from_compute_measurements,
)
from compiler.offload.profile import ProfileError, validate_profile
from compiler.tests.test_offload_profile import profile


def observations_v2():
    base = profile()
    records = [{"kind": "compute_identity_v2", "backend_id": COMPUTE_BACKEND_ID,
        **{name: base["hardware"][name] for name in ("gpu_name", "gpu_uuid", "compute_capability")},
        **{name: base["toolchain"][name] for name in ("cuda_runtime_version", "driver_version")},
        "cpu_threads": 4, "precision_bits": 64, "cpu_affinity": [2, 3, 4, 5],
        "fortran": {"compiler_version": "GCC version 15.2.0", "compiler_options": "-O3 -fopenmp",
                    "semantic_options": "-O3\x1f-fopenmp"}}]
    for backend, scale in zip(COMPUTE_BACKENDS, (1.0, .3, .35, .05), strict=True):
        arithmetic = 1e-10 * scale
        memory = 1e-11 * scale
        primitives = {"sqrt": 2e-9 * scale, "acos": 8e-9 * scale, "cos": 4e-9 * scale,
                      "divide": 3e-9 * scale}
        for family, recipe in COMPUTE_RECIPES.items():
            slope = max(recipe["arithmetic"] * arithmetic +
                sum(count * primitives[name] for name, count in recipe["intrinsics"].items()),
                recipe.get("memory_arrays", 2) * 8 * memory)
            sizes = MEMORY_SIZES if family == "memory_v2" else COMPUTE_SIZES
            fit_sizes = MEMORY_FIT_SIZES if family == "memory_v2" else COMPUTE_FIT_SIZES
            for size in sizes:
                seconds = (2e-6 if backend in {"native_fork_join", "generated_cpu"} else 0) + size * slope
                repetitions = max(1, math.ceil(.21 / seconds))
                samples = [{"batch": batch, "repetitions": repetitions,
                    "elapsed_seconds": repetitions * seconds, "wall_seconds": repetitions * seconds}
                           for batch in range(7)]
                records.append({"kind": "compute_cost_v2", "family": family, "backend": backend,
                    **({"traffic_bytes":size*24,"working_set_bytes":size*24} if family=="memory_v2" else {}),
                    "items": size, "role": "fit" if size in fit_sizes else "holdout",
                    "samples": samples, "agreement_passed": True})
    return records


def calibrated_v2(records=None):
    return profile_from_compute_measurements(profile(), records or observations_v2(), calibration={})


def private_features(**changes):
    return {"schema_version": 2, "classification_complete": True, "private_array_elements": 64,
            "private_array_groups": 4, "max_private_array_elements": 16, "max_private_array_rank": 2, **changes}


def test_v2_native_total_coefficients_and_raw_evidence_are_independent_of_base_cpp_rates():
    original = profile()
    observations = observations_v2()
    calibrated = profile_from_compute_measurements(original, observations,
                                                   calibration={"application_profiled": True})
    assert "numerical" not in original
    assert calibrated["rates"] == original["rates"]
    assert calibrated["numerical"]["measurements"] == observations
    assert calibrated["numerical"]["calibration"]["application_profiled"] is False
    assert calibrated["numerical"]["generator_id"] == compute_generator_identity()
    validate_profile(calibrated)
    model = numerical_compute_model(calibrated, {"sqrt": 3, "acos": 1, "cos": 2}, native_participation="serial")
    assert model["native_fortran"]["arithmetic_seconds_per_operation"] == pytest.approx(1e-10)
    assert model["native_fortran"]["intrinsic_seconds_per_item"] == pytest.approx(22e-9)
    assert model["generated_cpu"]["arithmetic_seconds_per_operation"] == pytest.approx(.35e-10)
    assert model["gpu"]["arithmetic_seconds_per_operation"] == pytest.approx(.05e-10)
    assert model["item_range"] == [65536, 1048576]
    assert 524288 in COMPUTE_HOLDOUT_SIZES
    calibrated["rates"]["cpu_flops_per_second"] *= 100
    assert numerical_compute_model(calibrated, {"sqrt": 3, "acos": 1, "cos": 2}, native_participation="serial") == model


def test_native_serial_and_fork_join_fixed_execution_costs_are_distinct():
    calibrated = calibrated_v2()
    serial = numerical_compute_model(calibrated, {}, native_participation="serial")
    parallel = numerical_compute_model(calibrated, {}, native_participation="fork_join")
    assert serial["native_fortran"]["fixed_seconds"] == 0
    assert serial["native_fortran"]["backend_identity"] == "native_serial"
    assert parallel["native_fortran"]["fixed_seconds"] == pytest.approx(2e-6)
    assert parallel["native_fortran"]["fixed_cost_coverage"] == "fork_join_execution_only"
    assert parallel["gpu"]["fixed_seconds"] == 0
    assert parallel["generated_cpu"]["fixed_seconds"] == pytest.approx(2e-6)
    with pytest.raises(NumericalCalibrationError, match="participation unavailable"):
        numerical_compute_model(calibrated, {})
    with pytest.raises(NumericalCalibrationError, match="existing teams"):
        numerical_compute_model(calibrated, {}, native_participation="existing_team")


def test_rejected_native_family_retains_every_observation_without_poisoning_other_backends():
    records = observations_v2()
    for row in records:
        if (row.get("family") == "primitive_sqrt_v2" and row.get("backend") == "native_serial"
                and row.get("items") == 524288):
            for sample in row["samples"]:
                sample["elapsed_seconds"] *= 2
                sample["wall_seconds"] *= 2
    calibrated = calibrated_v2(records)
    evidence = calibrated["numerical"]["families"]["primitive_sqrt_v2"]
    assert evidence["native_serial"]["status"] == "rejected"
    assert evidence["native_serial"]["reason_codes"] == ["size_holdout_error"]
    assert len(evidence["native_serial"]["observation_ids"]) == 5
    assert evidence["gpu"]["status"] == "accepted"
    assert evidence["native_fork_join"]["status"] == "accepted"
    assert calibrated["numerical"]["measurements"] == records
    validate_profile(calibrated)
    numerical_compute_model(calibrated, {"sqrt": 1}, native_participation="fork_join")
    # Ordinary arithmetic/memory work has no dependency on failed math models.
    numerical_compute_model(calibrated, {}, native_participation="serial")
    with pytest.raises(NumericalCalibrationError, match="primitive_sqrt_v2/native_serial rejected"):
        numerical_compute_model(calibrated, {"sqrt": 1}, native_participation="serial")


def test_missing_one_backend_observation_is_saved_as_rejection():
    records = observations_v2()
    records = [row for row in records if not (row.get("backend") == "gpu" and
        row.get("family") == "primitive_cos_v2" and row.get("items") == 131072)]
    calibrated = calibrated_v2(records)
    assert calibrated["numerical"]["families"]["primitive_cos_v2"]["gpu"]["reason_codes"] == ["missing_observations"]
    assert calibrated["numerical"]["families"]["primitive_cos_v2"]["generated_cpu"]["status"] == "accepted"
    validate_profile(calibrated)


def test_private_and_scalar_expression_holdouts_cannot_fit_primitive_coefficients():
    observations = observations_v2()
    before = calibrated_v2(observations)
    for row in observations:
        if row.get("family") in COMPUTE_CLASSES["fixed_private_array_v2"]:
            for sample in row["samples"]:
                sample["elapsed_seconds"] *= 2
                sample["wall_seconds"] *= 2
    after = calibrated_v2(observations)
    assert before["numerical"]["families"] == after["numerical"]["families"]
    assert after["numerical"]["workload_validation"]["scalar_expression_v2"]["gpu"]["status"] == "accepted"
    assert after["numerical"]["workload_validation"]["fixed_private_array_v2"]["gpu"]["status"] == "rejected"
    validate_profile(after)
    numerical_compute_model(after, {"sqrt": 1}, native_participation="serial")
    with pytest.raises(NumericalCalibrationError, match="fixed_private_array_v2/native_serial rejected"):
        numerical_compute_model(after, {"sqrt": 1}, workload_class="fixed_private_array_v2",
                                workload_features=private_features(), native_participation="serial")


@pytest.mark.parametrize("changes", [{"private_array_elements": 65}, {"max_private_array_elements": 17},
    {"max_private_array_rank": 3}, {"private_array_groups": 5}, {"classification_complete": False},
    {"private_array_elements": True}, {"schema_version": 1}, {"schema_version": True}])
def test_private_array_applicability_is_explicit_and_bounded(changes):
    with pytest.raises(NumericalCalibrationError, match="private-array"):
        numerical_compute_model(calibrated_v2(), {"sqrt": 1}, workload_class="fixed_private_array_v2",
                                workload_features=private_features(**changes), native_participation="serial")


def test_private_array_class_cannot_be_inferred_from_math_counts_or_names():
    calibrated = calibrated_v2()
    numerical_compute_model(calibrated, {"sqrt": 1}, workload_class="fixed_private_array_v2",
                            workload_features=private_features(), native_participation="serial")
    with pytest.raises(NumericalCalibrationError, match="outside calibrated"):
        numerical_compute_model(calibrated, {"sqrt": 1}, workload_class="fixed_private_array_v2", native_participation="serial")
    with pytest.raises(NumericalCalibrationError, match="scalar workload"):
        numerical_compute_model(calibrated, {}, workload_features=private_features(), native_participation="serial")


@pytest.mark.parametrize("mutation", ["batch_count", "batch_order", "short_batch", "boolean_repetitions", "duplicate", "fit_role", "affinity", "fortran_flags"])
def test_malformed_protocol_does_not_enable_estimates(mutation):
    records = observations_v2()
    sample = records[-1]["samples"][0]
    if mutation == "batch_count":
        records[-1]["samples"].pop()
    elif mutation == "batch_order":
        sample["batch"] = 4
    elif mutation == "short_batch":
        sample["wall_seconds"] = .199
    elif mutation == "boolean_repetitions":
        sample["repetitions"] = True
    elif mutation == "duplicate":
        records.append(deepcopy(records[-1]))
    elif mutation == "fit_role":
        records[-1]["role"] = "fit" if records[-1]["role"] == "holdout" else "holdout"
    elif mutation == "affinity":
        records[0]["cpu_affinity"] = [2, 3, 4]
    elif mutation == "fortran_flags":
        records[0]["fortran"]["semantic_options"] = "-Ofast"
    with pytest.raises(NumericalCalibrationError):
        calibrated_v2(records)


def test_saved_acceptance_rejection_and_runtime_identity_are_rechecked():
    calibrated = calibrated_v2()
    calibrated["numerical"]["families"]["primitive_acos_v2"]["gpu"]["status"] = "rejected"
    with pytest.raises(ProfileError, match="saved compute evidence"):
        validate_profile(calibrated)
    with pytest.raises(NumericalCalibrationError, match="implementation identity"):
        numerical_compute_model(calibrated_v2(), {}, native_participation="serial", generator_id="0" * 64)


def test_legacy_cpp_profiles_cannot_establish_native_fortran_costs():
    from compiler.tests.test_numerical_calibration import calibrated
    with pytest.raises(NumericalCalibrationError, match="requires numerical calibration v2"):
        numerical_compute_model(calibrated(), {"sqrt": 1}, native_participation="serial")
    with pytest.raises(NumericalCalibrationError, match="requires schema_version 1"):
        numerical_intrinsic_costs(calibrated_v2(), {"sqrt": 1}, backend_id="", generator_id="")


def test_v2_build_matches_original_fortran_options_fixed_affinity_and_keeps_rejections(tmp_path, monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda _: {2, 3, 4, 5, 6})
    monkeypatch.setenv("FORT_PHASE_TIMING", "1")
    monkeypatch.setenv("OMP_PLACES", "cores")
    monkeypatch.setenv("GOMP_CPU_AFFINITY", "8-11")
    calls = []
    records = observations_v2()
    records[-1]["agreement_passed"] = False

    def run(command, directory, log, *, timeout, env=None):
        calls.append((command, env))
        if env is not None:
            assert env["OMP_PROC_BIND"] == "false"
            assert not {"FORT_PHASE_TIMING", "OMP_PLACES", "GOMP_CPU_AFFINITY"} & env.keys()
            return "\n".join(json.dumps(row) for row in records)
        return ""

    args = SimpleNamespace(arch="native", precision=64, threads=4, device=0,
                           fortran_flag=["-O3", "-fopenmp"], cpu_affinity="2,3,4,5")
    result = calibrate_numerical_v2(profile(), args, tmp_path, "/nvcc", "/g++", run=run, tool=lambda *a: "/gfortran")
    assert len(calls) == 3
    assert calls[0][0][1:3] == ["-O3", "-fopenmp"]
    assert "-cpp" not in calls[0][0]
    assert "-DCALIBRATION_V2=1" in calls[1][0]
    assert calls[2][0][:3] == ["taskset", "-c", "2,3,4,5"]
    assert result["numerical"]["workload_validation"]["ordinary_expression_v2"]["gpu"]["status"] == "rejected"
    assert (tmp_path / "numerical-v2" / "evidence.json").is_file()
    parsed = parse_compute_measurements("\n".join(json.dumps(row) for row in records))
    assert parsed == records


def test_native_specialization_dispatches_before_original_numerical_loops():
    for precision, kind in ((32, "c_float"), (64, "c_double")):
        source = native_fixture_source(precision)
        assert f"integer, parameter :: rk = {kind}" in source
        assert source.count("select case(family)") == 1
        dispatcher = source.split("subroutine fort_numerical_native_v2(", 1)[1].split("end subroutine", 1)[0]
        assert "do i=" not in dispatcher
        for family in range(len(COMPUTE_RECIPES)):
            assert f"call numerical_worker_{family}(n,threads,fork_join,a,b,output)" in dispatcher
            worker = source.split(f"subroutine numerical_worker_{family}(", 1)[1].split("end subroutine", 1)[0]
            assert "family" not in worker
            assert worker.count(f"numerical_work_{family}(a(i),b(i))") == 2
        assert "primitive(operation" not in source
        assert "numerical_work(family" not in source


def test_independent_ordinary_holdout_failure_cannot_authorize_division_or_minmax():
    records = observations_v2()
    for row in records:
        if row.get("family") == "ordinary_mix_v2" and row.get("backend") == "gpu":
            for sample in row["samples"]:
                sample["elapsed_seconds"] *= 2
                sample["wall_seconds"] *= 2
    calibrated = calibrated_v2(records)
    for counts in ({}, {"divide": 4}):
        with pytest.raises(NumericalCalibrationError, match="ordinary_expression_v2/gpu rejected"):
            numerical_compute_model(calibrated, counts, native_participation="serial")
    # The separate mixed-math holdout still has its own valid observations.
    numerical_compute_model(calibrated, {"sqrt": 1, "divide": 1}, native_participation="serial")


def test_variable_division_increment_is_independent_and_counted_once():
    calibrated = calibrated_v2()
    model = numerical_compute_model(calibrated, {"divide": 7}, native_participation="serial")
    assert model["native_fortran"]["intrinsic_seconds_per_operation"]["divide"] == pytest.approx(3e-9)
    assert model["native_fortran"]["intrinsic_seconds_per_item"] == pytest.approx(21e-9)
    records = observations_v2()
    for row in records:
        if row.get("family") == "primitive_divide_v2" and row.get("backend") == "gpu":
            row["agreement_passed"] = False
    rejected = calibrated_v2(records)
    with pytest.raises(NumericalCalibrationError, match="primitive_divide_v2/gpu rejected"):
        numerical_compute_model(rejected, {"divide": 1}, native_participation="serial")


@pytest.mark.parametrize("backend", ["native_serial", "gpu"])
def test_unemitted_intercept_cannot_validate_an_expression_holdout(backend):
    records = observations_v2()
    for row in records:
        if row.get("backend") == backend and row.get("family") != "memory_v2":
            for sample in row["samples"]:
                sample["elapsed_seconds"] += sample["repetitions"] * .1
                sample["wall_seconds"] += sample["repetitions"] * .1
    calibrated = calibrated_v2(records)
    assert calibrated["numerical"]["families"]["arithmetic_v2"][backend]["status"] == "accepted"
    assert calibrated["numerical"]["workload_validation"]["ordinary_expression_v2"][backend]["status"] == "rejected"
    with pytest.raises(NumericalCalibrationError, match="ordinary_expression_v2/.+ rejected"):
        numerical_compute_model(calibrated, {}, native_participation="serial")


def test_divide_and_ordinary_fixture_counts_match_compiler_real_operation_convention(tmp_path):
    from compiler.tests.test_compute_arithmetic_operations import analyze
    init = "x=b(i)\nother=b(n)\nx1=x\nx2=x*0.5_8\nx3=x*0.25_8\nx4=x*0.125_8\n"
    step = """x1=(x1+0.125_8)/(other+x2*0.25_8+1.0_8)
x2=(x2+0.25_8)/(other+x3*0.125_8+1.0_8)
x3=(x3+0.5_8)/(other+x4*0.0625_8+1.0_8)
x4=(x4+0.75_8)/(other+x1*0.03125_8+1.0_8)
"""
    unit, _ = analyze(tmp_path, init + step * 16 + "a(i)=x1+x2+x3+x4", declarations="real(8)::other")
    assert unit.compute_arithmetic_operations_per_iteration == COMPUTE_RECIPES["primitive_divide_v2"]["arithmetic"]
    assert unit.compute_runtime_divisions_per_iteration == 64

    ordinary = """x=x*1.000001_8+0.000001_8
x=max(-0.9_8,min(-0.1_8,-abs(x)))
x=(x-0.1_8)/(-1.0_8-other)
"""
    unit, _ = analyze(tmp_path, "x=b(i)\nother=b(n)\n" + ordinary * 64 + "a(i)=x", declarations="real(8)::other")
    assert unit.compute_arithmetic_operations_per_iteration == COMPUTE_RECIPES["ordinary_mix_v2"]["arithmetic"]
    assert unit.compute_runtime_divisions_per_iteration == 64


def test_memory_protocol_has_eight_predefined_knots_and_independent_midpoints():
    assert len(MEMORY_FIT_SIZES) == 8
    assert all(2*b == 3*a for a,b in zip(MEMORY_FIT_SIZES, MEMORY_FIT_SIZES[1:], strict=False))
    assert len(MEMORY_HOLDOUT_SIZES) == 9
    assert {131072,524288} <= set(MEMORY_HOLDOUT_SIZES)
    assert not set(MEMORY_HOLDOUT_SIZES) & set(MEMORY_FIT_SIZES)
    assert len(observations_v2()) == 269


def test_piecewise_memory_model_tracks_physical_working_set_separately_from_traffic():
    model = {"kind":"piecewise_bandwidth_v1", "working_set_range":[100,800],
             "knots":[{"working_set_bytes":100*k,"seconds_per_traffic_byte":k*1e-10}
                      for k in range(1,9)]}
    assert memory_compute_seconds(model,400,150) == pytest.approx(400*1.5e-10)
    assert memory_compute_seconds(model,800,150) == pytest.approx(800*1.5e-10)
    assert memory_compute_seconds(model,400,200) == pytest.approx(400*2e-10)
    assert memory_compute_seconds(model,400,800) == pytest.approx(400*8e-10)
    assert memory_compute_seconds(model,0,100) == 0
    for working_set in (99,801):
        with pytest.raises(NumericalCalibrationError, match="outside calibrated"):
            memory_compute_seconds(model,400,working_set)
    assert memory_compute_seconds({"kind":"constant_bandwidth_v1","seconds_per_traffic_byte":1e-10},400,1) == pytest.approx(4e-8)


def test_piecewise_memory_uses_all_holdouts_and_keeps_rejected_backend_observations():
    original = calibrated_v2()
    records = observations_v2()
    for row in records:
        if row.get("family") == "memory_v2" and row.get("backend") == "gpu" and row["items"] == 524288:
            for sample in row["samples"]:
                sample["elapsed_seconds"] *= 2
                sample["wall_seconds"] *= 2
    rejected = calibrated_v2(records)
    family = rejected["numerical"]["families"]["memory_v2"]
    assert family["gpu"]["status"] == "rejected"
    assert family["gpu"]["reason_codes"] == ["size_holdout_error"]
    assert len(family["gpu"]["observation_ids"]) == 17
    assert len(family["gpu"]["holdouts"]) == 9
    assert family["gpu"]["model"] == original["numerical"]["families"]["memory_v2"]["gpu"]["model"]
    assert family["native_serial"]["status"] == "accepted"
    assert rejected["numerical"]["measurements"] == records
    validate_profile(rejected)


def test_piecewise_memory_reader_preserves_shared_fixed_cost_once():
    calibrated = calibrated_v2()
    model = numerical_compute_model(calibrated,{},native_participation="fork_join")
    for role,scale in (("native_fortran",.3),("generated_cpu",.35),("gpu",.05)):
        memory = model[role]["memory_cost_model"]
        assert memory["kind"] == "piecewise_bandwidth_v1"
        assert model[role]["memory_seconds_per_byte"] is None
        assert model[role]["fixed_seconds"] == pytest.approx(0 if role=="gpu" else 2e-6)
        assert memory_compute_seconds(memory,65536*24,65536*24) == pytest.approx(65536*24*1e-11*scale)
    validate_profile(calibrated)


@pytest.mark.parametrize(("field", "value"), [("working_set_bytes", True), ("traffic_bytes",1)])
def test_memory_observation_requires_exact_original_three_array_footprint(field,value):
    records = observations_v2()
    next(row for row in records if row.get("family")=="memory_v2")[field] = value
    with pytest.raises(NumericalCalibrationError, match="physical footprint"):
        calibrated_v2(records)


def test_saved_memory_knots_are_reconstructed_from_raw_fit_observations():
    calibrated = calibrated_v2()
    calibrated["numerical"]["families"]["memory_v2"]["gpu"]["model"]["knots"][0]["seconds_per_traffic_byte"] *= 2
    with pytest.raises(ProfileError, match="saved compute evidence"):
        validate_profile(calibrated)


@pytest.mark.parametrize(("traffic", "working_set"), [(0,0),(1,99),(1,801),(-1,100),
    (float("inf"),100),(float("nan"),100),(1,True),(1,-1),(1,2**64)])
def test_memory_reference_rejects_unavailable_queries_including_zero_outside_range(traffic,working_set):
    model={"kind":"piecewise_bandwidth_v1","working_set_range":[100,800],
           "knots":[{"working_set_bytes":100*k,"seconds_per_traffic_byte":k*1e-10} for k in range(1,9)]}
    with pytest.raises(NumericalCalibrationError):
        memory_compute_seconds(model,traffic,working_set)


@pytest.mark.parametrize(("rate", "traffic"), [(1e300,1e300),(1e-300,1e-300)])
def test_memory_reference_rejects_overflow_and_underflow(rate,traffic):
    model={"kind":"piecewise_bandwidth_v1","working_set_range":[100,200],
           "knots":[{"working_set_bytes":k,"seconds_per_traffic_byte":rate} for k in (100,200)]}
    with pytest.raises(NumericalCalibrationError):
        memory_compute_seconds(model,traffic,100)


def test_memory_reference_preserves_exact_endpoint_rate():
    rates=[1e-250,3.141592653589793e-40]
    model={"kind":"piecewise_bandwidth_v1","working_set_range":[100,200],
           "knots":[{"working_set_bytes":k,"seconds_per_traffic_byte":rate}
                    for k,rate in zip((100,200),rates, strict=True)]}
    assert memory_compute_seconds(model,1,100) == rates[0]
    assert memory_compute_seconds(model,1,200) == rates[1]
