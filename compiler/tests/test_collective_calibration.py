"""Collective costs require actual fixed-team observations and exact identity."""

from copy import deepcopy

import pytest

from compiler.offload.calibrate import profile_with_scoped_measurements
from compiler.offload.collective_calibration import (
    AUTO_OWNER_CALLS,
    AUTO_WORK_CALLS,
    COMPUTE_STEPS,
    MEASURED_SAMPLES,
    OBSERVED_REPETITIONS,
    PROTOCOL_FIT_SIZES,
    PROTOCOL_HOLDOUT_SIZE,
    _fit_protocol_costs,
    fixture_facts,
    fixture_source,
    normalize_fortran_options,
    profile_with_collective_measurements,
)
from compiler.offload.profile import (
    SCOPED_TEAM_COST_NAMES,
    SCOPED_TEAM_PROTOCOL_ID,
    SCOPED_TEAM_RATE_NAMES,
    ProfileError,
    scoped_collective_costs,
    validate_profile,
)
from compiler.tests.test_offload_profile import profile, scoped_observations


def base():
    return profile_with_scoped_measurements(
        profile(), scoped_observations(), runtime_id="a" * 64, cold_startup_seconds=[0.1] * 3
    )


def measurements():
    return [
        {
            "kind": "collective_identity",
            "threads": 4,
            "level": 1,
            "protocol_id": SCOPED_TEAM_PROTOCOL_ID,
            "compiler_version": "GCC version 15.2.0",
            "compiler_options": "-O3 -fopenmp -I /one -J/two -o out.o",
        },
        *[
            {
                "kind": "collective_rate",
                "name": name,
                "work": 100,
                "seconds": [0.01] * 5,
                "baseline_seconds": [0.001] * 5,
            }
            for name in SCOPED_TEAM_RATE_NAMES
        ],
        *[{"kind": "collective_cost", "name": name, "seconds": [1e-6] * 5} for name in SCOPED_TEAM_COST_NAMES],
        {"kind": "collective_validation", "seconds": [8e-6] * 5, "predicted_seconds": 8e-6},
    ]


def protocol_observations():
    observations = {}
    for fixture, factor in (("memory", 1), ("compute", 2)):
        for mode, sizes, stages in (
            (0, (0,), (7,)),
            (1, (1,), (3, 4)),
            (2, (*PROTOCOL_FIT_SIZES, PROTOCOL_HOLDOUT_SIZE), (1, 2, 3, 5, 6)),
            (3, (0,), (1,)),
        ):
            for n in sizes:
                rows = []
                shape = 1 if n == PROTOCOL_FIT_SIZES[0] else 3
                for stage in stages:
                    seconds = stage * shape * factor * 1e-6 if mode == 2 else 5 * factor * 1e-6
                    if mode == 1 and stage == 3:
                        seconds = 100  # CPU-only entry observations must not fit the mixed protocol.
                    if mode == 3:
                        seconds = factor * 0.25e-6
                    for sample in range(1, MEASURED_SAMPLES + 1):
                        rows.append({"stage": stage, "sample": sample,
                                     "seconds": seconds * OBSERVED_REPETITIONS,
                                     "calls": OBSERVED_REPETITIONS * (12 if stage == 2 else 1)})
                observations[fixture, mode, n] = rows
    return observations


def test_protocol_fit_pools_fixed_shapes_and_preserves_sources_without_holdout_leakage():
    observations = protocol_observations()
    records = _fit_protocol_costs(observations)
    costs = {record["name"]: record for record in records}
    assert costs["entry_seconds"]["seconds"] == pytest.approx([6e-6] * 9 + [18e-6] * 9)
    assert costs["cpu_worker_seconds"]["seconds"] == pytest.approx([10e-6] * 9)
    owner = costs["owner_seconds"]
    assert owner["seconds"] == pytest.approx([2.5e-6] * 9 + [6.5e-6] * 9)
    assert owner["components"]["fortran_guard_seconds"] == pytest.approx([0.5e-6] * 18)
    assert costs["entry_seconds"]["fit"]["selected_fixture"] == "compute"
    assert costs["entry_seconds"]["fit"]["sizes"] == list(PROTOCOL_FIT_SIZES)
    assert {source["n"] for source in costs["entry_seconds"]["fit"]["sources"]} == set(PROTOCOL_FIT_SIZES)
    source = costs["entry_seconds"]["fit"]["sources"][0]
    assert source["raw_samples"] == [row for row in observations["memory", 2, PROTOCOL_FIT_SIZES[0]]
                                      if row["stage"] == 3]
    for fixture in ("memory", "compute"):
        # Invalid, extreme held-out samples may fail later validation but can
        # never leak into fitting or its selected-source provenance.
        observations[fixture, 2, PROTOCOL_HOLDOUT_SIZE] = [{"stage": 1, "seconds": 1e100, "calls": 0}]
    assert _fit_protocol_costs(observations) == records


@pytest.mark.parametrize("stage", [1, 2, 3, 5, 6])
def test_protocol_fit_rejects_training_stage_counter_mismatches(stage):
    observations = protocol_observations()
    row = next(row for row in observations["compute", 2, PROTOCOL_FIT_SIZES[1]] if row["stage"] == stage)
    row["calls"] -= 1
    with pytest.raises(ValueError, match="production stage count mismatch"):
        _fit_protocol_costs(observations)


def test_optional_team_profile_preserves_serial_rates_and_raw_observations():
    original = base()
    result = profile_with_collective_measurements(original, measurements(), calibration={})
    assert result["rates"] == original["rates"]
    assert result["scoped"]["costs"] == original["scoped"]["costs"]
    assert "collective" not in original["scoped"]
    assert result["scoped"]["collective"]["measurements"] == measurements()
    assert scoped_collective_costs(result, "a" * 64, cpu_threads=4)["cpu_flops"] == pytest.approx(100 / 0.009)
    assert not result["scoped"]["collective"]["calibration"]["application_profiled"]


def test_failed_calibration_command_preserves_partial_output(tmp_path, monkeypatch):
    import subprocess

    from compiler.offload.calibrate import CalibrationError, _run

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(["control"], 1, output=b"partial-out\n", stderr=b"partial-err\n")

    monkeypatch.setattr(subprocess, "run", timeout)
    log = tmp_path / "failed.log"
    with pytest.raises(CalibrationError, match="cannot execute"):
        _run(["control"], tmp_path, log, timeout=1)
    assert "partial-out\npartial-err\n" in log.read_text()


def test_explicit_fortran_flags_replace_defaults_and_require_openmp(tmp_path):
    from types import SimpleNamespace

    from compiler.offload.collective_calibration import calibrate_collective

    with pytest.raises(ValueError, match="include -fopenmp"):
        calibrate_collective(
            profile(),
            SimpleNamespace(fortran_flag=["-O2"]),
            tmp_path,
            "nvcc",
            "g++",
            run=lambda *a, **k: pytest.fail("no command may run"),
            tool=lambda *a: "gfortran",
        )


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r.pop(),
        lambda r: r.pop(1),
        lambda r: r.append(deepcopy(r[1])),
        lambda r: r[0].update(threads=3),
        lambda r: r[0].update(level=2),
        lambda r: r[0].update(protocol_id=1),
        lambda r: r[0].pop("compiler_options"),
        lambda r: r[1].update(seconds=[0.001] * 5, baseline_seconds=[0.002] * 5),
        lambda r: r[1].update(seconds=[True] * 5),
        lambda r: r[-1].update(predicted_seconds=1),
    ],
)
def test_missing_incompatible_or_unphysical_measurements_fail_closed(change):
    records = measurements()
    change(records)
    with pytest.raises(ValueError, match="collective|unknown"):
        profile_with_collective_measurements(base(), records, calibration={})


def test_missing_collective_calibration_is_not_replaced_with_serial_rates():
    with pytest.raises(ProfileError, match="collective synchronization"):
        scoped_collective_costs(base(), "a" * 64)


def test_changed_protocol_source_preserves_base_profile_but_disables_team_estimates():
    result = profile_with_collective_measurements(base(), measurements(), calibration={})
    result["scoped"]["collective"]["protocol_sources"]["identity"] = "b" * 64
    validate_profile(result)
    with pytest.raises(ProfileError, match="source identity mismatch"):
        scoped_collective_costs(result, "a" * 64)


def test_only_build_paths_are_excluded_from_semantic_options():
    a = normalize_fortran_options("-O3 -fopenmp -I /checkout/a -J/mod -o one.o -fcheck=all -ffp-contract=off")
    b = normalize_fortran_options("-O3 -fopenmp -I/checkout/b -J /other -oother.o -fcheck=all -ffp-contract=off")
    assert a == b
    assert a != normalize_fortran_options("-O2 -fopenmp -fcheck=all -ffp-contract=off")
    assert a != normalize_fortran_options("-fopenmp -O3 -fcheck=all -ffp-contract=off")
    assert normalize_fortran_options("-O3 -include /header -DNAME=one") != normalize_fortran_options(
        "-O3 -include /other -DNAME=one"
    )
    assert (
        normalize_fortran_options("-O3 -openmp -offload=host -opt-report")
        == "-O3\x1f-openmp\x1f-offload=host\x1f-opt-report"
    )


def test_fixed_fixture_facts_bind_every_call_inside_one_original_team(tmp_path):
    path = tmp_path / "fixture.f90"
    path.write_text(fixture_source("compute", 4, 64))
    facts = fixture_facts(path, 4)
    assert len(facts["participation"]["call_sites"]) == 16
    assert path.read_text().count("!$omp parallel") == 1
    assert len({site["team_first_line"] for site in facts["participation"]["call_sites"]}) == 1


def test_compute_fixture_uses_fixed_countable_work_and_reuses_one_leaf():
    source = fixture_source("compute", 4, 64, work_calls=AUTO_WORK_CALLS)
    assert "do j=" not in source
    assert source.count("x1=x1*") == COMPUTE_STEPS
    assert source.count("subroutine work(") == 1
    assert source.count("call work(a,b,c,n)") == AUTO_WORK_CALLS
    assert source.count("call native_lookup(a,c,permutation,n)") == 1


def test_fixed_auto_holdout_uses_existing_analysis_budgets_and_public_work_counts(tmp_path):
    from compiler.driver.options import CompilerOptions
    from compiler.emission.common.resources import read_scoped_runtime
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.source import form_source_scopes

    original = tmp_path / "original.f90"
    original.write_text(fixture_source("compute", 4, 64, work_calls=AUTO_WORK_CALLS, owner_calls=AUTO_OWNER_CALLS))
    profile = base()
    profile["scoped"]["runtime_id"] = read_scoped_runtime()[1]["runtime_id"]
    profile["toolchain"].update(nvcc_version="NVCC V13.4.92", host_cxx_version="GCC 14.4.0")
    profile = profile_with_collective_measurements(profile, measurements(), calibration={})
    _, report = form_source_scopes(
        [original],
        "team_operations::step",
        facts=fixture_facts(original, 4),
        options=CompilerOptions(gpu_policy="auto", memory_model="scoped"),
        config=OffloadConfig("auto", profile, 4, True),
    )
    assert report["scope_count"] == 1, report["boundaries"]
    assert report["automatic_estimate_available"]
    (entry,) = report["scopes"][0]["numerical_entries"]
    assert entry["public"]["planning"]["units"][0]["work_per_iteration"] == 12 * COMPUTE_STEPS + 10


@pytest.mark.parametrize(
    "change",
    [
        {"gpu_applied": 1},
        {"gpu_applied": True, "gpu_workers": 0},
        {"gpu_applied": False, "gpu_workers": 1},
        {"field_agreement": False},
    ],
)
def test_public_auto_holdout_requires_explainable_placement_and_field_agreement(change):
    row = {
        "kind": "collective_auto_holdout",
        "seconds": [8e-6] * 5,
        "predicted_seconds": 8e-6,
        "gpu_applied": True,
        "gpu_workers": 16,
        "field_agreement": True,
    }
    row.update(change)
    with pytest.raises(ValueError, match="public auto holdout"):
        profile_with_collective_measurements(base(), [*measurements(), row], calibration={})
