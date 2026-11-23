"""Check source isolation and actual CUDA visibility without a device."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

spec = importlib.util.spec_from_file_location('container_workflow', Path(__file__).resolve().parents[1] / 'workflow.py')
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


def test_source_identity_does_not_visit_results_or_caches(tmp_path, monkeypatch):
    (tmp_path / 'source.py').write_text('source')
    for directory in ('results', '.cache'):
        (tmp_path / directory).mkdir()
        (tmp_path / directory / 'inaccessible').symlink_to('/does-not-exist')
    first = workflow.source_identity(tmp_path)
    (tmp_path / '.cache/another').write_text('generated')
    assert workflow.source_identity(tmp_path) == first
    assert set(first['files']) == {'source.py'}


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


def test_only_resident_reports_are_plotted(tmp_path, monkeypatch):
    monkeypatch.delenv("PLOT_PYTHON", raising=False)
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"status": "passed", "summary": {}}))
    with pytest.raises(subprocess.CalledProcessError):
        workflow.plot_experiment(report, tmp_path / "figures")
    assert not (tmp_path / "figures").exists()


def test_reproduce_plots_redraws_only_an_explicit_saved_report(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(workflow, "plot_experiment", lambda report, output: calls.append((report, output)))
    monkeypatch.setattr(sys, "argv", ["workflow.py", "reproduce-plots", "--experiment", str(tmp_path / "report.json"),
                                     "--output-dir", str(tmp_path / "plots")])
    workflow.main()
    assert calls == [(tmp_path / "report.json", tmp_path / "plots/figures")]
    for retired in ("--snapshot", "--campaign"):
        monkeypatch.setattr(sys, "argv", ["workflow.py", "reproduce-plots", retired, str(tmp_path / "saved"),
                                         "--output-dir", str(tmp_path / "plots")])
        with pytest.raises(SystemExit):
            workflow.main()
    assert len(calls) == 1


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
    assert not any(name and name.startswith("compiler")
                   for name in imports)
