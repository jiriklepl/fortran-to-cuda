"""Resident comparison preserves diagnostic and timing evidence boundaries."""
import importlib.util
import json
import os
from array import array
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE), str(HERE.parent / "elmm-pipeline")]
spec = importlib.util.spec_from_file_location("resident_comparison", HERE / "resident_comparison.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.mark.parametrize(("application", "vkfft", "names"), [
    (True, True, ("elmm", "PoisFFT", "VkFFT")),
    (True, False, ("elmm", "PoisFFT")),
    (False, True, ("VkFFT",)),
])
def test_prepare_dependencies_uses_exact_pins_without_replacing_existing_paths(
        tmp_path, monkeypatch, application, vkfft, names):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "content").write_text("pinned\n")
    subprocess.run(["git", "-C", str(source), "add", "content"], check=True)
    subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "pinned"], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    spec = {"url": str(source), "revision": revision}
    (pipeline / "dependencies.json").write_text(json.dumps({"elmm": spec, "PoisFFT": spec}))
    (pipeline / "pressure-dependencies.json").write_text(json.dumps({"VkFFT": spec}))
    monkeypatch.setattr(runner, "PIPELINE", pipeline)
    destination = tmp_path / "dependencies"
    runner.prepare_dependencies(destination, application=application, vkfft=vkfft)
    assert {path.name for path in destination.iterdir()} == set(names)
    for name in names:
        assert (destination / name / "content").read_text() == "pinned\n"
        assert subprocess.check_output(["git", "-C", str(destination / name), "rev-parse", "HEAD"],
                                       text=True).strip() == revision
    with pytest.raises(FileExistsError):
        runner.prepare_dependencies(destination, application=application, vkfft=vkfft)


def test_missing_pipeline_is_reported_before_resident_imports(tmp_path):
    missing = tmp_path / "pipeline"
    result = subprocess.run([sys.executable, str(HERE / "resident_comparison.py"), "--help"],
                            env={**os.environ, "PIPELINE_ROOT": str(missing)},
                            text=True, capture_output=True, check=False)
    assert result.returncode == 1
    assert str(missing) in result.stderr and "PIPELINE_ROOT" in result.stderr
    assert "Traceback" not in result.stderr


def test_application_environment_strips_external_controls(monkeypatch, tmp_path):
    monkeypatch.setenv("FORT_RUNTIME_TRACE", "1")
    monkeypatch.setenv("POISSON_SOLVER_KERNEL_CACHE_DIR", "/uncontrolled")
    monkeypatch.setenv("ELMM_OTHER", "override")
    monkeypatch.setenv("LD_PRELOAD", "/uncontrolled")
    env = runner.application_environment(4, "GPU-id", tmp_path, traced=False)
    assert env["OMP_NUM_THREADS"] == "4"
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-id"
    assert env["POISSON_SOLVER_KERNEL_CACHE_DIR"] == str(tmp_path)
    assert "FORT_RUNTIME_TRACE" not in env and "ELMM_OTHER" not in env and "LD_PRELOAD" not in env
    assert runner.application_environment(4, "GPU-id", None, traced=True)["FORT_RUNTIME_TRACE"] == "1"


def test_field_failure_exposes_comparison_details():
    reference = {"fields-U.bin": (np.array([1.0, 2.0, 3.0]), None)}
    actual = {"fields-U.bin": (np.array([1.0, 4.0, 3.0]), None)}

    with pytest.raises(ValueError) as failure:
        runner.checked_fields(actual, reference)

    assert "fields-U.bin" in str(failure.value)
    assert "1/3" in str(failure.value)
    comparison = failure.value.comparison["fields-U.bin"]
    assert comparison["failed_values"] == 1
    assert comparison["max_absolute_error"] == 2.0
    assert comparison["failed_flat_indices"] == [1]


def test_natural_channel_l2_comparison_is_componentwise_and_checks_zero_reference():
    reference = np.array([100.0, 0.0])
    within_limit = runner.compare_long_vectors(reference, np.array([100.9, 0.0]))
    outside_limit = runner.compare_long_vectors(reference, np.array([101.1, 0.0]))
    zero_reference = runner.compare_long_vectors(np.zeros(2), np.array([0.0, 2e-6]))

    assert within_limit["passed"] and within_limit["relative_l2"] == pytest.approx(0.009)
    assert not outside_limit["passed"]
    assert not zero_reference["passed"] and zero_reference["relative_l2"] is None
    assert zero_reference["max_absolute_error"] == pytest.approx(2e-6)


def test_long_run_norm_handles_tiny_and_large_nonzero_references():
    tiny = runner.compare_long_vectors(np.array([1e-200]), np.array([2e-200]))
    large = runner.compare_long_vectors(np.array([1e200]), np.array([1.005e200]))
    assert not tiny["passed"] and tiny["relative_l2"] == pytest.approx(1.0)
    assert large["passed"] and large["relative_l2"] == pytest.approx(0.005)
    smallest = np.array([np.nextafter(0.0, 1.0)])
    excessive = runner.compare_long_vectors(smallest, np.array([1.0]))
    pointwise = runner.compare_long_vectors(smallest, np.array([1e-7]), criterion="pointwise")
    assert not excessive["passed"] and excessive["relative_l2"] is None
    assert pointwise["passed"]
    json.dumps([excessive, pointwise], allow_nan=False)


@pytest.mark.parametrize(("case", "config", "steps", "expected"), [
    ("channel", None, None, True),
    ("channel", None, 300, False),
    ("channel", Path("custom"), None, False),
    ("channel-simple", None, None, False),
])
def test_long_run_policy_only_applies_to_default_natural_channel(case, config, steps, expected):
    assert runner.uses_natural_channel_policy(case, config, steps) is expected


def test_long_run_requires_the_approved_natural_endpoint():
    with pytest.raises(ValueError, match="time limit"):
        runner.require_natural_channel({"application_timing": {"completed_steps": 0}}, "Time limit reached.")
    with pytest.raises(ValueError, match="time limit"):
        runner.require_natural_channel({"application_timing": {"completed_steps": 2458}}, "Steady state reached.")
    with pytest.raises(ValueError, match="2513 != 2514"):
        runner.require_natural_channel({"application_timing": {"completed_steps": 2513}}, "Time limit reached.", 2514)
    runner.require_natural_channel({"application_timing": {"completed_steps": 2458}}, "Time limit reached.")
    runner.require_natural_channel({"application_timing": {"completed_steps": 2514}}, "Time limit reached.", 2514)


def test_long_run_fields_retain_strict_failure_and_require_matching_bounds():
    layout = {"bits": 64, "shape": [2], "lower_bounds": [1], "upper_bounds": [2]}
    reference = {"fields-U.bin": (np.array([100.0, 0.0]), layout)}
    actual = {"fields-U.bin": (np.array([100.9, 0.0]), layout)}
    result = runner.compare_long_fields(actual, reference)["fields-U.bin"]
    assert result["passed"] and not result["pointwise"]["passed"]
    actual["fields-U.bin"] = (actual["fields-U.bin"][0], {**layout, "lower_bounds": [0]})
    with pytest.raises(ValueError, match="bounds mismatch"):
        runner.compare_long_fields(actual, reference)


def test_long_run_vtk_checks_vector_components_and_exact_coordinates():
    expected = ((2, 1, 1), {"X_COORDINATES": array("f", [0, 1]),
        "Y_COORDINATES": array("f", [0]), "Z_COORDINATES": array("f", [0]),
        "u": array("f", [100, 1, 0, 100, 1, 0])})
    actual = (expected[0], {**expected[1], "u": array("f", [100.5, 1.2, 0, 100.5, 1.2, 0])})
    result = runner.compare_long_vtk(expected, actual)
    assert result["u[0]"]["passed"] and not result["u[1]"]["passed"]
    shifted = (actual[0], {**actual[1], "X_COORDINATES": array("f", [0, 1.00001])})
    assert not runner.compare_long_vtk(expected, shifted)["X_COORDINATES"]["passed"]
    changed_precision = (actual[0], {**actual[1], "u": array("d", [100, 1, 0, 100, 1, 0])})
    with pytest.raises(ValueError, match="precision differs"):
        runner.compare_long_vtk(expected, changed_precision)


def test_long_run_outputs_check_inventory_and_time_columns(tmp_path):
    reference, actual = tmp_path / "reference", tmp_path / "actual"
    reference.mkdir()
    actual.mkdir()
    for directory, time_value in ((reference, 2.0), (actual, 2.0001)):
        np.array([1.0, time_value], dtype="<f8").tofile(directory / "times.unf")
        np.array([100.0, 100.0], dtype="<f8").tofile(directory / "Retau.unf")
        (directory / "frame-y-times.txt").write_text("0 0.0\n1 1.0\n")
    result = runner.compare_long_outputs(reference, actual)
    assert not result["passed"]
    assert not result["files"]["times.unf"]["time"]["passed"]
    assert result["files"]["Retau.unf"]["Retau"]["passed"]
    (actual / "frame-y-times.txt").unlink()
    with pytest.raises(ValueError, match="output inventory"):
        runner.compare_long_outputs(reference, actual)


def test_long_run_outputs_check_wall_components_and_frame_indices(tmp_path):
    reference, actual = tmp_path / "reference", tmp_path / "actual"
    reference.mkdir()
    actual.mkdir()
    np.array([1000.0, 1.0, 1000.0, 1.0], dtype="<f8").tofile(reference / "Retau.unf")
    np.array([1000.0, 1.02, 1000.0, 1.02], dtype="<f8").tofile(actual / "Retau.unf")
    (reference / "frame-y-times.txt").write_text("0 0.0\n1 1.0\n")
    (actual / "frame-y-times.txt").write_text("0 0.0\n2 1.0\n")
    result = runner.compare_long_outputs(reference, actual)
    assert result["files"]["Retau.unf"]["Retau"]["passed"]
    assert not result["files"]["Retau.unf"]["friction_velocity"]["passed"]
    assert not result["files"]["frame-y-times.txt"]["frame_index"]["passed"]
    assert not result["passed"]


def test_long_run_failure_identifies_output_component():
    row = {"fields": {"fields-U.bin": {"passed": True}},
           "outputs": {"files": {"frame-y-49.vtk": {"p": {"passed": False}}}}}
    with pytest.raises(ValueError, match="resident:.*frame-y-49.vtk:p"):
        runner.require_long_comparison(row, "resident")


def test_field_failure_summary_handles_nonfinite_and_many_fields():
    reference = {f"fields-{name}.bin": (np.array([1.0]), None) for name in "ABCDE"}
    actual = {name: (np.array([np.nan]), None) for name in reference}

    with pytest.raises(runner.FieldComparisonError) as failure:
        runner.checked_fields(actual, reference)

    assert "max_abs=nonfinite" in str(failure.value)
    assert "+2 more fields" in str(failure.value)
    assert failure.value.comparison["fields-A.bin"]["candidate_nonfinite_values"] == 1
    assert failure.value.comparison["fields-A.bin"]["max_absolute_error"] is None
    assert len(failure.value.comparison) == 5


def test_inspect_saved_diagnostics_reports_each_failed_field(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner, "REQUIRED_FIELDS", {"fields-U.bin"})
    for name, values in (("native-reference", [1.0, 2.0, 3.0]),
                         ("native-repeat", [1.0, 2.0, 3.0]),
                         ("resident", [1.0, 4.0, 3.0])):
        directory = tmp_path / name
        directory.mkdir()
        np.array(values, dtype="float64").tofile(directory / "fields-U.bin")
        (directory / "fields-U.shape").write_text("64 3 1 3\n")

    with pytest.raises(SystemExit) as failure:
        runner.main(["--inspect-diagnostics", str(tmp_path)])
    assert failure.value.code == 1
    result = json.loads(capsys.readouterr().out)
    assert result["comparisons"]["native-repeat"]["passed"]
    failed = result["comparisons"]["resident"]["failed_fields"]["fields-U.bin"]
    assert failed["failed_values"] == 1
    assert failed["max_absolute_error"] == 2.0
    assert failed["failed_flat_indices"] == [1]


def test_resident_timing_plots_keep_validation_policy(tmp_path):
    import plot_experiment

    fields = {"fields-U.bin": {"passed": True, "pointwise": {"passed": False}}}
    diagnostics = [{"mode": "native"}, {"mode": "native", "fields": fields, "outputs": {"passed": True}},
                   {"mode": "resident", "fields": fields, "outputs": {"passed": True},
                    "gpu_activity": {"gpu_kernels_by_procedure": {"dynamics::convection": 1,
                                                                  "subgrid::sgs_sigma_uniform": 1}}}]
    runs = [{"mode": mode, "warmup": False, "traced": False, "wall_seconds": seconds,
             "outputs": {"passed": True}}
            for mode, seconds in (("native", 10.0), ("resident", 4.0))]
    value = {"status": "passed", "case": "channel", "release_qualified": False,
             "validation_policy": "natural-channel-componentwise-relative-l2-1-percent",
             "threads": 4, "warmups": 0, "diagnostics": diagnostics, "runs": runs,
             "summary": {"native": {"samples_seconds": [10.0], "median_seconds": 10.0},
                         "resident": {"samples_seconds": [4.0], "median_seconds": 4.0}, "speedup": 2.5}}
    report = tmp_path / "report.json"
    report.write_text(json.dumps(value))
    output = tmp_path / "figures"
    plot_experiment.plot_experiment(report, output)
    assert "1% componentwise L2" in (output / "complete-wall-relative.svg").read_text()
    assert (output / "complete-wall-absolute.png").is_file()
    assert (output / "source.json").is_file()
    value["runs"][0]["outputs"]["passed"] = False
    report.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="untraced and validated"):
        plot_experiment.plot_experiment(report, tmp_path / "invalid")


def test_default_work_directory_follows_output(monkeypatch, tmp_path):
    monkeypatch.delenv("WORK_DIR", raising=False)
    monkeypatch.delenv("DEPS_ROOT", raising=False)
    arguments = []
    monkeypatch.setattr(runner, "run", arguments.append)
    runner.main(["--output-dir", str(tmp_path / "output")])
    assert arguments[0].work_dir == tmp_path / "output/work"
    assert arguments[0].app_root is None and arguments[0].vkfft is None


@pytest.mark.parametrize(("argv", "timeout"), [
    ([], 900),
    (["--case", "channel"], 3600),
    (["--case", "channel", "--steps", "300"], 900),
    (["--case", "channel", "--timeout", "1200"], 1200),
])
def test_case_specific_default_timeout(monkeypatch, tmp_path, argv, timeout):
    arguments = []
    monkeypatch.setattr(runner, "run", arguments.append)
    runner.main(["--output-dir", str(tmp_path), *argv])
    assert arguments[0].timeout == timeout


def test_doctor_checks_explicit_fortran_not_ambient_fc(monkeypatch):
    import workflow
    import elmm_pipeline.gpu_toolchain as gpu_toolchain

    monkeypatch.setenv("FC", "missing-ambient-compiler")
    monkeypatch.setattr(workflow, "selected_gpu", lambda _requested: {
        "uuid": "GPU-id", "compute_capability": "8.6"})
    monkeypatch.setattr(workflow.shutil, "which", lambda name: None if name == "missing-ambient-compiler" else name)
    monkeypatch.setattr(workflow.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout="sm_86"))
    monkeypatch.setattr(gpu_toolchain, "discover_toolchain", lambda **_kwargs: {"nvcc": "nvcc"})
    args = SimpleNamespace(nvcc="nvcc", cuda_host="g++-14", arch="auto", gpu=None)
    result = workflow.doctor(args, require_gpu=True, fortran="explicit-fortran")
    assert result["tools"]["explicit-fortran"]["path"] == "explicit-fortran"


@pytest.mark.parametrize("provision", [False, True])
@pytest.mark.parametrize("failed_diagnostic", [None, "native-repeat", "resident"])
@pytest.mark.parametrize("natural_steps", [None, 2458, 2514])
def test_fresh_comparison_excludes_diagnostics_and_rotates_samples(tmp_path, monkeypatch, provision,
                                                                   failed_diagnostic, natural_steps):
    natural = natural_steps is not None
    args = SimpleNamespace(output_dir=tmp_path / "output", work_dir=tmp_path / "work",
        app_root=None if provision else tmp_path, solver=tmp_path, vkfft=None if provision else tmp_path,
        case="channel" if natural else "channel-simple", config=None, steps=None if natural else 2, threads=4,
        jobs=2, repeats=2, warmups=1, timeout=30, build_timeout=30, fortran=sys.executable,
        cuda_host=None, nvcc=None)
    monkeypatch.setattr(runner, "doctor", lambda *_args, **_kwargs: {
        "gpu": {"uuid": "GPU-id", "compute_capability": "8.6"},
        "toolchain": {"cuda_host_cxx": sys.executable, "nvcc": sys.executable}})
    monkeypatch.setattr(runner, "prepare_application_basis", lambda *_args: tmp_path)
    monkeypatch.setattr(runner, "application_paths", lambda _root: {"elmm": tmp_path, "PoisFFT": tmp_path})
    monkeypatch.setattr(runner, "select_configuration", lambda *_args: (tmp_path, None))
    monkeypatch.setattr(runner, "source_identity", lambda _root: {"files": {}})
    monkeypatch.setattr(runner, "solver_identity", lambda _root: {"revision": "test"})
    provisioned = []
    def dependencies(directory, *, application, vkfft):
        provisioned.append((directory, application, vkfft))
        directory.mkdir()
        return directory
    monkeypatch.setattr(runner, "prepare_dependencies", dependencies)

    def prepare(_source, destination, **_kwargs):
        destination.mkdir()
        return {"runtime_arguments": [], "expected_max_steps": 2}

    def build(argv, *, cwd, env, log, timeout):
        directory = Path(argv[argv.index("--output") + 1])
        directory.mkdir()
        binaries = {}
        for name in ("ELMM", "ELMM-native", "ELMM-full20", "ELMM-native-full20"):
            path = directory / name
            path.write_text(name)
            binaries[name] = {"path": str(path), "sha256": runner.digest(path)}
        provider = directory / "provider"
        provider.write_text("provider")
        (directory / "built.json").write_text(json.dumps({"status": "PASS", "binaries": binaries,
            "provider": {"path": str(provider), "sha256": runner.digest(provider)}}))
        (directory / "packages.json").write_text("[]")
        return 12.0

    calls = []

    def application(binary, config, configuration, directory, env, timeout, expected_steps=None):
        calls.append((Path(binary).name, env.get("FORT_RUNTIME_TRACE"),
                      env.get("POISSON_SOLVER_KERNEL_CACHE_DIR")))
        directory.mkdir(parents=True)
        return {"directory": str(directory), "wall_seconds": 2.0 if "native" in binary else 1.0,
            "application_timing": {"completed_steps": natural_steps if natural else 2}}, "vtk", "Time limit reached."

    monkeypatch.setattr(runner, "prepare_config", prepare)
    monkeypatch.setattr(runner, "command", build)
    monkeypatch.setattr(runner, "run_application", application)
    monkeypatch.setattr(runner, "read_fields", lambda *_args: {"fields-U.bin": "values"})
    monkeypatch.setattr(runner, "NATURAL_CHANNEL_FIELDS", {"fields-U.bin"})
    monkeypatch.setattr(runner, "NATURAL_CHANNEL_OUTPUTS", set())

    def check_fields(*_args):
        if failed_diagnostic and len(calls) == {"native-repeat": 2, "resident": 3}[failed_diagnostic]:
            raise runner.FieldComparisonError({"fields-U.bin": {
                "passed": False, "values": 2, "failed_values": 1,
                "max_absolute_error": 2.0, "failed_flat_indices": [1]}})
        return {"passed": True}

    monkeypatch.setattr(runner, "checked_fields", check_fields)
    monkeypatch.setattr(runner, "compare_vtks", lambda *_args: {"passed": True})
    monkeypatch.setattr(runner, "compare_outputs", lambda *_args: {"passed": False})
    monkeypatch.setattr(runner, "compare_long_outputs", lambda *_args: {"passed": True, "files": {}})

    def check_long_fields(*_args):
        failed = failed_diagnostic and len(calls) == {"native-repeat": 2, "resident": 3}[failed_diagnostic]
        return {"fields-U.bin": {"passed": not failed, "pointwise": {"passed": False},
                                 "failed_flat_indices": [1], "max_absolute_error": 2.0}}

    monkeypatch.setattr(runner, "compare_long_fields", check_long_fields)
    monkeypatch.setattr(runner, "verify_resident_execution", lambda *_args: {"verified": True})
    plots = []
    monkeypatch.setattr(runner, "plot_experiment", lambda report, output: plots.append((report, output)))
    if failed_diagnostic:
        with pytest.raises(ValueError, match=failed_diagnostic):
            runner.run(args)
        report = json.loads((args.output_dir / "resident/report.json").read_text())
        assert report["status"] == "failed"
        assert failed_diagnostic in report["error"]["message"]
        diagnostic = report["diagnostics"][-1]
        assert diagnostic["directory"].endswith(failed_diagnostic)
        assert diagnostic["fields"]["fields-U.bin"]["failed_flat_indices"] == [1]
        assert diagnostic["fields"]["fields-U.bin"]["max_absolute_error"] == 2.0
        return
    runner.run(args)
    report = json.loads((args.output_dir / "resident/report.json").read_text())
    assert report["status"] == "passed" and report["release_qualified"] is False
    assert plots == [(args.output_dir / "resident/report.json", args.output_dir / "resident/figures")]
    assert report["validation_policy"] == ("natural-channel-componentwise-relative-l2-1-percent"
                                            if natural else "pointwise-1e-6-1e-5")
    if natural:
        assert not report["diagnostics"][1]["fields"]["fields-U.bin"]["pointwise"]["passed"]
        assert report["diagnostics"][1]["outputs"]["passed"]
    assert len(provisioned) == int(provision)
    if provision:
        assert provisioned[0][1:] == (True, True)
        assert report["dependencies"]["vkfft"].endswith("dependencies/VkFFT")
    assert len(report["diagnostics"]) == 3 and len(report["runs"]) == 6
    assert report["summary"]["native"]["samples_seconds"] == [2.0, 2.0]
    assert report["summary"]["resident"]["samples_seconds"] == [1.0, 1.0]
    assert report["summary"]["speedup"] == 2.0
    assert [row["mode"] for row in report["runs"]] == ["native", "resident", "resident", "native", "native", "resident"]
    assert calls[2][1] == "1" and all(call[1] is None for call in calls[3:])
    assert calls[2][2] != calls[4][2]
    with pytest.raises(FileExistsError):
        runner.run(args)
