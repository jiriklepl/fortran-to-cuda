"""Preserve exact final pipeline test inputs and output; no application campaign."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PARENT = Path('/home/jirka/research/fortran-abomination')
PIPELINE = PARENT / 'elmm-pipeline'
PYTHON = Path('/home/jirka/research/fortran-abomination/.venv/bin/python')
OUTPUT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    roots = {'compiler': PARENT / 'compiler', 'pipeline': PIPELINE / 'elmm_pipeline',
             'pipeline_tests': PIPELINE / 'tests'}
    files = {}
    for name, root in roots.items():
        for path in sorted(root.rglob('*')):
            if path.is_file() and not any(part in {'__pycache__', '.pytest_cache'} for part in path.parts):
                files[name + '/' + str(path.relative_to(root))] = digest(path)
    for name in ['README.md', 'requirements-local.txt', 'requirements-campaign.txt', 'dependencies.json']:
        path = PIPELINE / name
        if path.is_file():
            files['pipeline_root/' + name] = digest(path)
    return files


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    if (OUTPUT / 'final.json').exists() or (OUTPUT / 'final.log').exists():
        raise SystemExit('Preserved final result exists; choose a new checkpoint directory.')
    before = inventory()
    write('source-files-before.json', before)
    argv = ['taskset', '-c', '0,1,6,7', str(PYTHON), '-m', 'pytest', '-q', 'tests',
            '--basetemp=' + str(OUTPUT / 'pytest')]
    env = {**os.environ, 'ELMM_TEST_COMPILER_ROOT': str(PARENT)}
    receipt = {'schema_version': 1, 'claim': 'pipeline test suite only; no application performance claim',
               'argv': argv, 'cwd': str(PIPELINE), 'python_sha256': digest(PYTHON.resolve()),
               'runner_sha256': digest(Path(__file__)),
               'environment': {key: env.get(key) for key in ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH',
                   'ELMM_TEST_COMPILER_ROOT', 'OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES']},
               'commits': {str(root): subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                                            text=True).strip()
                           for root in [PARENT, PIPELINE]},
               'source_files_before_sha256': digest(OUTPUT / 'source-files-before.json')}
    write('command.json', receipt)
    started = time.monotonic()
    with (OUTPUT / 'final.log').open('w') as log:
        run = subprocess.run(argv, cwd=PIPELINE, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    after = inventory()
    write('source-files-after.json', after)
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    result = {**receipt, 'exit_code': run.returncode, 'elapsed_seconds': time.monotonic() - started,
              'source_files_after_sha256': digest(OUTPUT / 'source-files-after.json'),
              'source_files_changed': changed, 'stable_source_inputs': not changed,
              'log_sha256': digest(OUTPUT / 'final.log')}
    write('final.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    sys.exit(run.returncode if run.returncode else int(bool(changed)))


if __name__ == '__main__':
    main()
