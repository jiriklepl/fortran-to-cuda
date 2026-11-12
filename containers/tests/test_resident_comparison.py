"""Resident comparison preserves diagnostic and timing evidence boundaries."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

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


def test_default_work_directory_follows_output(monkeypatch, tmp_path):
    monkeypatch.delenv("WORK_DIR", raising=False)
    monkeypatch.delenv("DEPS_ROOT", raising=False)
    arguments = []
    monkeypatch.setattr(runner, "run", arguments.append)
    runner.main(["--output-dir", str(tmp_path / "output")])
    assert arguments[0].work_dir == tmp_path / "output/work"
    assert arguments[0].app_root is None and arguments[0].vkfft is None


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
def test_fresh_comparison_excludes_diagnostics_and_rotates_samples(tmp_path, monkeypatch, provision):
    args = SimpleNamespace(output_dir=tmp_path / "output", work_dir=tmp_path / "work",
        app_root=None if provision else tmp_path, solver=tmp_path, vkfft=None if provision else tmp_path,
        case="channel-simple", config=None, steps=2, threads=4,
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
                "application_timing": {"completed_steps": 2}}, "vtk", "trace"

    monkeypatch.setattr(runner, "prepare_config", prepare)
    monkeypatch.setattr(runner, "command", build)
    monkeypatch.setattr(runner, "run_application", application)
    monkeypatch.setattr(runner, "read_fields", lambda *_args: {"fields-U.bin": "values"})
    monkeypatch.setattr(runner, "checked_fields", lambda *_args: {"passed": True})
    monkeypatch.setattr(runner, "compare_vtks", lambda *_args: {"passed": True})
    monkeypatch.setattr(runner, "verify_resident_execution", lambda *_args: {"verified": True})
    runner.run(args)
    report = json.loads((args.output_dir / "resident/report.json").read_text())
    assert report["status"] == "passed" and report["release_qualified"] is False
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
