"""Check source isolation and actual CUDA visibility without a device."""
import importlib.util
from pathlib import Path
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
