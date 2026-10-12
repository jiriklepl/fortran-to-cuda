"""Ensure Apptainer never stages host results/caches through %files."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_apptainer_build_context_excludes_large_artifacts(tmp_path):
    source = Path(__file__).resolve().parents[1]
    workspace = tmp_path / 'workspace with spaces'
    (workspace / 'containers').mkdir(parents=True)
    shutil.copy2(source / 'build-apptainer.sh', workspace / 'containers/build-apptainer.sh')
    (workspace / 'containers/Apptainer.def').write_text('definition')
    (workspace / 'requirements.txt').write_text('requirements')
    for name in ('compiler', 'benchmarks', 'elmm-pipeline'):
        (workspace / name).mkdir()
        (workspace / name / 'source.py').write_text('source')
    for directory in ('benchmarks/results', 'elmm-pipeline/.cache', 'compiler/__pycache__'):
        (workspace / directory).mkdir(parents=True)
        (workspace / directory / 'do-not-copy').write_text('cached result')
    (workspace / 'compiler/CODE_MAP.md').write_text('private plan')
    commands = tmp_path / 'commands'
    commands.mkdir()
    capture = tmp_path / 'capture.json'
    engine = commands / 'apptainer'
    engine.write_text(f'#!{sys.executable}\n'
        'import json,sys\nfrom pathlib import Path\n'
        f'Path({str(capture)!r}).write_text(json.dumps({{"argv":sys.argv[1:],'
        '"files":sorted(str(p.relative_to(Path.cwd())) for p in Path.cwd().rglob("*") if p.is_file())}))\n')
    engine.chmod(0o755)
    output = tmp_path / 'output image.sif'
    result = subprocess.run(['bash', str(workspace / 'containers/build-apptainer.sh'), '-o', str(output)],
        env={**os.environ, 'PATH': str(commands) + os.pathsep + os.environ['PATH']},
        cwd=tmp_path, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    observed = json.loads(capture.read_text())
    assert observed['argv'] == ['build', str(output), 'containers/Apptainer.def']
    assert set(observed['files']) == {'requirements.txt', 'containers/Apptainer.def',
        'containers/build-apptainer.sh', 'compiler/source.py', 'benchmarks/source.py', 'elmm-pipeline/source.py'}
