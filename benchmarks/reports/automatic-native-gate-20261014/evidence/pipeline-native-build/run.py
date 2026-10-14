"""Preserve corrected compiler/pipeline integration inputs and output; no application campaign."""
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
    roots = {'compiler': PARENT / 'compiler', 'pipeline': PIPELINE / 'elmm_pipeline'}
    files = {}
    for name, root in roots.items():
        for path in sorted(root.rglob('*')):
            if path.is_file() and not any(part in {'__pycache__', '.pytest_cache', '.ruff_cache', 'tests'} for part in path.parts):
                files[name + '/' + str(path.relative_to(root))] = digest(path)
    for name in ['README.md', 'requirements-local.txt', 'requirements-campaign.txt', 'dependencies.json']:
        path = PIPELINE / name
        if path.is_file():
            files['pipeline_root/' + name] = digest(path)
    for name in ['test_local_scopes.py', 'test_scoped_adaptation.py']:
        files['pipeline_tests/' + name] = digest(PIPELINE / 'tests' / name)
    return files


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    if (OUTPUT / 'final.json').exists() or (OUTPUT / 'final.log').exists():
        raise SystemExit('Preserved final result exists; choose a new checkpoint directory.')
    before = inventory()
    write('source-files-before.json', before)
    argv = ['taskset', '-c', '0,1,6,7', str(PYTHON), '-m', 'pytest', '-q', 'tests/test_local_scopes.py::test_automatic_missing_profile_composes_authenticated_native_package',
            'tests/test_scoped_adaptation.py::test_normal_adaptation_uses_public_source_scope_path',
            'tests/test_scoped_adaptation.py::test_real_normal_scopes_build_and_execute_full_fields[auto]',
            '--basetemp=' + str(OUTPUT / 'pytest')]
    env = {**os.environ, 'ELMM_TEST_COMPILER_ROOT': str(PARENT)}
    env['ELMM_SCOPED_NATIVE_TESTS'] = '1'
    env.pop('ELMM_TEST_SCOPED_PROFILE', None)
    receipt = {'schema_version': 1, 'claim': 'public native fallback composition plus real native fixture build and complete fields; no application performance claim',
               'argv': argv, 'cwd': str(PIPELINE), 'python_sha256': digest(PYTHON.resolve()),
               'runner_sha256': digest(Path(__file__)),
               'environment': {key: env.get(key) for key in ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH',
                   'ELMM_TEST_COMPILER_ROOT', 'ELMM_SCOPED_NATIVE_TESTS', 'ELMM_TEST_SCOPED_PROFILE', 'OMP_NUM_THREADS', 'OMP_PROC_BIND', 'OMP_PLACES']},
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
