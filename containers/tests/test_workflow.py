"""Check source isolation and actual CUDA visibility without a device."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('container_workflow', Path(__file__).resolve().parents[1] / 'workflow.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_source_identity_does_not_visit_results_or_caches(tmp_path, monkeypatch):
    (tmp_path / 'pipeline.py').write_text('source')
    for directory in ('results', '.cache'):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / 'inaccessible').symlink_to('/does-not-exist')
    first = workflow.source_identity(tmp_path)
    (tmp_path / '.cache/another').write_text('generated')
    assert workflow.source_identity(tmp_path) == first
    assert set(first['files']) == {'pipeline.py'}


def test_cuda_ordinal_is_not_nvml_physical_index(monkeypatch):
    monkeypatch.setattr(workflow.shutil, 'which', lambda name: '/tools/' + name)
    monkeypatch.setattr(workflow.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=(
        '0, GPU-other, Other, 9.0, 555\n1, GPU-allocated, Allocated, 12.0, 555\n')))
    monkeypatch.setattr(workflow, 'visible_gpu_uuids', lambda: ['GPU-allocated'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    assert workflow.selected_gpu('0')['uuid'] == 'GPU-allocated'
    assert workflow.selected_gpu()['compute_capability'] == '12.0'
    with pytest.raises(RuntimeError, match='absent'):
        workflow.selected_gpu('GPU-other')


def test_unmatched_cuda_device_fails_instead_of_picking_nvml_first(monkeypatch):
    monkeypatch.setattr(workflow.shutil, 'which', lambda name: '/tools/' + name)
    monkeypatch.setattr(workflow.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        stdout='0, GPU-other, Other, 9.0, 555\n'))
    monkeypatch.setattr(workflow, 'visible_gpu_uuids', lambda: ['GPU-missing'])
    with pytest.raises(RuntimeError, match='cannot be matched'):
        workflow.selected_gpu()


@pytest.mark.parametrize("unvalidated", [False, True])
def test_experiment_plots_preserve_validation_status(tmp_path, monkeypatch, unvalidated):
    monkeypatch.delenv("PLOT_PYTHON", raising=False)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from sgs_experiment import summarize
    runs = [{"mode": mode, "warmup": False, "traced": False, "wall_seconds": seconds,
             "validation": {"passed": not unvalidated}}
            for mode, seconds in (("native", 10.), ("current", 7.))]
    value = {"status": "completed_unvalidated" if unvalidated else "passed", "case": "channel",
             "baseline_repeatability": not unvalidated, "correctness_qualified": not unvalidated,
             "target_met": None if unvalidated else True, "summary": summarize(runs), "runs": runs,
             "correctness": [], "gpu_evidence": {"verified": True}, "started_utc": "2026-10-12T00:00:00Z",
             "protocol": {"threads": 4, "warmups": 1, "allow_native_variation": unvalidated},
             "performance_eligible": True, "comparison_budget_eligible": True}
    report = tmp_path / "report.json"
    report.write_text(json.dumps(value))
    output = tmp_path / "figures"
    workflow.plot_experiment(report, output)
    svg = (output / "complete-wall-relative.svg").read_text()
    assert ("UNVALIDATED: native outputs are not repeatable" in svg) is unvalidated
    assert ("no correctness or target claim" in svg) is unvalidated
    assert (output / "source.json").is_file()


def test_unvalidated_plot_requires_explicit_variation_request(tmp_path, monkeypatch):
    monkeypatch.delenv("PLOT_PYTHON", raising=False)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "completed_unvalidated", "baseline_repeatability": False,
                                 "correctness_qualified": False, "target_met": None,
                                 "protocol": {"allow_native_variation": False}}))
    with pytest.raises(subprocess.CalledProcessError):
        workflow.plot_experiment(report, tmp_path / "figures")


def test_run_elmm_dry_run_exposes_explicit_native_variation_option(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["workflow.py", "run-elmm", "--case", "channel", "--dry-run",
                                     "--allow-native-variation", "--output-dir", str(tmp_path / "results")])
    workflow.main()
    value = json.loads(capsys.readouterr().out)
    assert value["allow_native_variation"] is True
    assert value["case"] == "channel"
    assert value["steps"] is None
    assert not value["executes_work"]


def test_source_analysis_runs_in_prepared_parser_python(tmp_path, monkeypatch):
    output = tmp_path / "results with spaces"
    parser_python = tmp_path / "prepared parser/bin/python"
    source_root = tmp_path / "prepared sources"
    calls = []
    monkeypatch.setattr(workflow, "guarded", lambda args: True)
    monkeypatch.setattr(workflow, "doctor", lambda *a, **kw: {
        "memory_headroom_bytes": {"host": 8 * 2**30}, "cpu_affinity": list(range(8)),
        "toolchain": {"nvcc": "/cuda/nvcc", "cuda_host_cxx": "/tools/g++", "architecture": "120"}})
    monkeypatch.setattr(workflow, "plot_experiment", lambda *a: None)
    def execute(command, **kwargs):
        calls.append(command)
        report = output / "experiment/report.json"
        report.parent.mkdir()
        report.write_text(json.dumps({"summary": {"native": {"median": 10.}, "current": {"median": 9.}}}))
    monkeypatch.setattr(workflow.subprocess, "run", execute)
    monkeypatch.setattr(sys, "argv", ["workflow.py", "run-elmm", "--case", "channel",
        "--output-dir", str(output), "--app-root", str(source_root),
        "--compiler-python", str(parser_python), "--allow-native-variation"])
    workflow.main()
    assert len(calls) == 1
    command = calls[0]
    assert command[:2] == [str(parser_python), str(workflow.HERE / "sgs_experiment.py")]
    assert command[command.index("--python") + 1] == str(parser_python)
    assert command[command.index("--app-root") + 1] == str(source_root)
    assert command[command.index("--gpu-arch") + 1] == "120"
    assert "--allow-native-variation" in command
    assert "--steps" not in command


def test_existing_prepared_pipeline_cache_is_reused(tmp_path, monkeypatch, capsys):
    pipeline = tmp_path / "pipeline"
    for name in ("setup.json", "repos/elmm/src/SConstruct", "repos/PoisFFT/src/SConstruct",
                 "envs/local/bin/python", "envs/build/bin/python", "native/lib/ready"):
        path = pipeline / ".cache" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    output = tmp_path / "output"
    monkeypatch.setattr(workflow, "PIPELINE", pipeline)
    monkeypatch.setattr(workflow, "guarded", lambda args: True)
    monkeypatch.setattr(workflow, "doctor", lambda *a, **kw: {
        "memory_headroom_bytes": {"host": 8 * 2**30}, "cpu_affinity": list(range(8)),
        "toolchain": {"nvcc": "/cuda/nvcc", "cuda_host_cxx": "/tools/g++", "architecture": "120"}})
    def unexpected_prepare(args):
        pytest.fail("A prepared cache must not be downloaded or rebuilt")
    monkeypatch.setattr(workflow, "prepare", unexpected_prepare)
    monkeypatch.setattr(workflow, "plot_experiment", lambda *a: None)
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        path = output / "experiment/report.json"
        path.parent.mkdir()
        path.write_text(json.dumps({"summary": {}}))
    monkeypatch.setattr(workflow.subprocess, "run", execute)
    monkeypatch.setattr(sys, "argv", ["workflow.py", "run-elmm", "--output-dir", str(output)])
    workflow.main()
    assert calls[0][0] == str(pipeline / ".cache/envs/local/bin/python")
    assert calls[0][calls[0].index("--app-root") + 1] == str(pipeline / ".cache/repos")
    assert "Using prepared pipeline cache:" in capsys.readouterr().out


def test_plotting_falls_back_without_changing_compiler_python(monkeypatch):
    monkeypatch.delenv("PLOT_PYTHON", raising=False)
    monkeypatch.setenv("PYTHON", "/module/python3.12")
    monkeypatch.setattr(workflow.sys, "executable", "/module/python3.12")
    available = {"/module/python3.12": "/module/python3.12", "python3": "/system/python3.9"}
    monkeypatch.setattr(workflow.shutil, "which", lambda name: available.get(name))
    probes = []
    def probe(argv, **kwargs):
        probes.append(argv[0])
        return SimpleNamespace(returncode=int(argv[0] == "/module/python3.12"), stderr="no matplotlib")
    monkeypatch.setattr(workflow.subprocess, "run", probe)
    assert workflow.plotting_python() == "/system/python3.9"
    assert probes == ["/module/python3.12", "/system/python3.9"]
    assert workflow.os.environ["PYTHON"] == "/module/python3.12"
    assert workflow.sys.executable == "/module/python3.12"


@pytest.mark.parametrize("successful", [True, False])
def test_explicit_plot_python_is_honored_without_fallback(monkeypatch, successful):
    monkeypatch.setenv("PLOT_PYTHON", "/plots/python3.9")
    monkeypatch.setattr(workflow.shutil, "which", lambda name: name)
    probes = []
    def probe(argv, **kwargs):
        probes.append(argv[0])
        return SimpleNamespace(returncode=0 if successful else 1, stderr="No module named matplotlib")
    monkeypatch.setattr(workflow.subprocess, "run", probe)
    if successful:
        assert workflow.plotting_python() == "/plots/python3.9"
    else:
        with pytest.raises(RuntimeError, match="PLOT_PYTHON is unusable"):
            workflow.plotting_python()
    assert probes == ["/plots/python3.9"]


def test_plotting_prefers_current_interpreter_when_dependencies_exist(monkeypatch):
    monkeypatch.delenv("PLOT_PYTHON", raising=False)
    monkeypatch.setattr(workflow.shutil, "which", lambda name: name)
    probes = []
    def probe(argv, **kwargs):
        probes.append(argv[0])
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(workflow.subprocess, "run", probe)
    assert workflow.plotting_python() == sys.executable
    assert probes == [sys.executable]


def test_plotter_parses_as_python39_without_compiler_imports():
    import ast
    plotter = Path(__file__).resolve().parents[1] / "plot_experiment.py"
    tree = ast.parse(plotter.read_text(), feature_version=(3, 9))
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not any(name and (name.startswith("elmm_pipeline") or name.startswith("compiler"))
                   for name in imports)
