"""Independent convex controls and fixed, held-out source-host geometry."""

import json
import re
import shutil
import subprocess
from fractions import Fraction

import pytest

from compiler.emission.common.resources import read_scoped_runtime
from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.offload.source_submit_model import (
    ARGUMENT_WORDS,
    BINDINGS,
    LAUNCHES,
    fit_model,
    fit_relative,
    quote,
    validate_model,
)
from compiler.offload.source_task_pricing import source_task_pricing
from compiler.offload.tile_protocol_calibration import source_scratch_benchmark
from compiler.tests.test_scoped_runtime import RUNTIME
from compiler.tests.test_source_native_pricing import live_issued, live_profile
from compiler.tests.test_source_protocol_calibration import observations


def rational_control(points):
    """Unscaled exact arithmetic, independent of the implementation solver."""
    best = None
    for mask in range(1, 8):
        axes = [i for i in range(3) if mask & (1 << i)]
        x = [(tuple(map(Fraction, (1, b, n))), Fraction(y)) for b, n, y in points]
        a = [[sum(row[i]*row[j]/y**2 for row, y in x) for j in axes] +
             [sum(row[i]/y for row, y in x)] for i in axes]
        for k in range(len(axes)):
            pivot = next((i for i in range(k, len(axes)) if a[i][k]), None)
            if pivot is None:
                break
            a[k], a[pivot] = a[pivot], a[k]
            v = a[k][k]
            a[k] = [value/v for value in a[k]]
            for i in range(len(axes)):
                if i != k:
                    v = a[i][k]
                    a[i] = [p-v*q for p, q in zip(a[i], a[k], strict=True)]
        else:
            c = [Fraction(0)]*3
            for i, axis in enumerate(axes):
                c[axis] = a[i][-1]
            if min(c) < 0:
                continue
            objective = sum(((sum(row[i]*c[i] for i in range(3))-y)/y)**2 for row, y in x)
            if best is None or objective < best[0]:
                best = objective, c
    return [float(value) for value in best[1]]


@pytest.mark.parametrize("points", [
    [(b, n, 2+3*b+5*n) for b in (1, 4, 8) for n in (1, 4, 16)],
    [(b, n, b+n-1) for b in (2, 4, 8) for n in (2, 4, 16)],
    [(b, n, 100-2*b-n) for b in (1, 4, 8) for n in (1, 4, 16)],
    [(1, 1, 5), (2, 4, 7), (3, 8, 8), (4, 2, 20), (8, 16, 11)],
])
def test_relative_nonnegative_global_optimum_matches_exact_fraction_controls(points):
    assert list(fit_relative(points).values()) == pytest.approx(rational_control(points), rel=1e-10, abs=1e-11)


@pytest.mark.parametrize("unit", [1e-150, 1e-6, 1e150])
def test_time_unit_scaling_does_not_change_the_relative_fit(unit):
    p = [(b, n, 2+3*b+5*n) for b in (1, 4, 8) for n in (1, 4, 16)]
    assert [value/unit for value in fit_relative([(b, n, y*unit) for b, n, y in p]).values()] == pytest.approx([2, 3, 5])


def model_records(profile):
    identity = dict(observations()[0], kernel_argument_words=ARGUMENT_WORDS, kernel_argument_bytes=8*ARGUMENT_WORDS)
    assert identity["gpu_uuid"] == profile["hardware"]["gpu_uuid"]
    return [identity, *[{"kind": "source_protocol_observation", "name": "source_submit_host_enqueue_included",
        "bytes": 0, "bindings": b, "launches": n, "rounds": 32,
        "seconds": [1e-6*(2+3*b+5*n)]*5} for b in BINDINGS for n in LAUNCHES]]


def test_fresh_holdout_failure_never_refits_or_increases_the_predetermined_margin():
    profile = live_profile()
    runtime = read_scoped_runtime()[1]["runtime_id"]
    rows = model_records(profile)
    good = fit_model(rows, profile, runtime)
    assert good["available"]
    assert good["coverage_factor"] == 4
    assert len(good["measurements"]) == 77
    assert quote(good, 4, 18) == pytest.approx(4e-6*(2+12+90))
    rows[next(i for i, row in enumerate(rows) if row.get("bindings") == 3 and row.get("launches") == 18)]["seconds"] = [1.0]*5
    bad = fit_model(rows, profile, runtime)
    assert bad["available"] is False
    assert bad["failed_holdouts"] == [[3, 18]]
    assert bad["coefficients"] == good["coefficients"]
    assert bad["coverage_factor"] == good["coverage_factor"]
    with pytest.raises(ValueError, match="unavailable"):
        quote(bad, 4, 18)


def test_ignored_large_fit_residual_disables_model_and_edited_availability_is_recomputed():
    profile, runtime = live_profile(), read_scoped_runtime()[1]["runtime_id"]
    rows = model_records(profile)
    row = next(row for row in rows if row.get("bindings") == 4 and row.get("launches") == 16)
    row["seconds"] = [1.0]*5
    model = fit_model(rows, profile, runtime)
    assert model["available"] is False
    assert [4, 16] in model["failed_fit_points"]
    assert len(model["coverage_diagnostics"]) == 77
    assert sum(row["fit_point"] for row in model["coverage_diagnostics"]) == 15
    # Resealing metadata cannot turn observational undercoverage into a fit.
    from compiler.offload.source_submit_model import _digest

    model.update(available=True, failed_fit_points=[])
    model["model_identity"] = _digest({k: v for k, v in model.items() if k != "model_identity"})
    with pytest.raises(ValueError, match="observational coverage"):
        validate_model(model, profile, runtime)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "argument_abi", "nan", "bool", "out_of_range"])
def test_incomplete_foreign_or_unbounded_observations_fail_closed(fault):
    profile = live_profile()
    rows = model_records(profile)
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows.append(rows[-1])
    elif fault == "argument_abi":
        rows[0]["kernel_argument_words"] = 1
    elif fault == "nan":
        rows[1]["seconds"][0] = float("nan")
    elif fault == "bool":
        rows[1]["bindings"] = True
    else:
        rows[1]["launches"] = 257
    with pytest.raises(ValueError, match="source submit model"):
        fit_model(rows, profile, read_scoped_runtime()[1]["runtime_id"])


def test_versioned_provider_uses_actual_abi_and_reached_counts_without_changing_old_sections(tmp_path):
    proof, _ = live_issued(tmp_path)
    profile = live_profile()
    runtime = read_scoped_runtime()[1]["runtime_id"]
    profile["source_submit_model"] = fit_model(model_records(profile), profile, runtime)
    profile["source_tasks"]["available"] = False
    pricing = source_task_pricing(proof.team, profile, runtime, native_live=True)
    assert pricing.public()["version"] == 4
    assert pricing.public()["gpu_protocol_available"] is True
    default = emit_tile_task_provider(proof.team, readable_scalars=proof)
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof, pricing=pricing)
    assert provider.public()["version"] == 5
    assert provider.public()["launch_abi"]["units"][0]["bytes"] > 0
    code = provider.cuda_source
    assert "source.source_submit_seconds=4.0*(" in code
    assert "double(expected)*" in code
    assert "source.valid=0;" not in code
    assert code.index("fort_runtime::HostFloatingEnvironment quote_environment;") < code.index("source.source_submit_seconds=4.0*(")
    assert default == emit_tile_task_provider(proof.team, readable_scalars=proof)
    (tmp_path / "provider.cu").write_text(code)
    (tmp_path / "provider.json").write_text(json.dumps(provider.public(), indent=2))


@pytest.mark.native
def test_structural_priced_provider_compiles_against_actual_cuda_and_fortran_abi(tmp_path):
    nvcc, host, fc = shutil.which("nvcc"), shutil.which("g++-14"), shutil.which("gfortran-15")
    if not all((nvcc, host, fc)):
        pytest.skip("CUDA and GNU toolchains unavailable")
    proof, _ = live_issued(tmp_path)
    profile = live_profile()
    runtime = read_scoped_runtime()[1]["runtime_id"]
    profile["source_submit_model"] = fit_model(model_records(profile), profile, runtime)
    profile["source_tasks"]["available"] = False
    pricing = source_task_pricing(proof.team, profile, runtime, native_live=True)
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof, pricing=pricing)
    (tmp_path / "provider.cu").write_text(provider.cuda_source)
    (tmp_path / "provider.f90").write_text(provider.fortran_source)
    commands = [[nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp", "-I", str(RUNTIME),
                 "-I", str(RUNTIME.parent / "emission/common/templates"), "-c", str(tmp_path / "provider.cu"),
                 "-o", str(tmp_path / "provider.o")],
                [fc, "-fopenmp", "-J", str(tmp_path), "-c", str(RUNTIME / "scoped_memory.f90"), "-o", str(tmp_path / "memory.o")],
                [fc, "-fopenmp", "-J", str(tmp_path), "-I", str(tmp_path), "-c", str(tmp_path / "provider.f90"),
                 "-o", str(tmp_path / "fortran.o")]]
    for index, command in enumerate(commands):
        (tmp_path / f"command-{index}.json").write_text(json.dumps(command))
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (tmp_path / f"command-{index}.stderr").write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.native
def test_fixed_launch_envelope_compiles_to_128_uint64_cuda_parameters(tmp_path):
    nvcc, host = shutil.which("nvcc"), shutil.which("g++-14")
    if not nvcc or not host:
        pytest.skip("CUDA toolchain unavailable")
    source = tmp_path / "source.cu"
    source.write_text(source_scratch_benchmark(structural_submit=True))
    output = tmp_path / "source.ptx"
    command = [nvcc, "-std=c++17", "-arch=sm_86", "-ccbin", host, "-Xcompiler=-fopenmp", "-I", str(RUNTIME),
               "-ptx", str(source), "-o", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    (tmp_path / "command.json").write_text(json.dumps(command))
    (tmp_path / "stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    match, = re.findall(r"\.entry\s+[^ (]*empty_kernel[^ (]*\s*\((.*?)\)", output.read_text(), re.S)
    assert len(re.findall(r"\.param\s+\.u64", match)) == 128


def test_structural_calibration_cli_is_explicit_and_requires_source_protocol(monkeypatch, tmp_path):
    from compiler.offload import calibrate

    called = []
    monkeypatch.setattr(calibrate, "calibrate", lambda args: called.append(args.source_submit_model))
    with pytest.raises(SystemExit):
        calibrate.main(["--output", str(tmp_path / "bad.json"), "--source-submit-model"])
    assert not called
    assert calibrate.main(["--output", str(tmp_path / "valid.json"), "--scoped-costs", "--source-task-costs",
                           "--source-submit-model"]) == 0
    assert called == [True]
