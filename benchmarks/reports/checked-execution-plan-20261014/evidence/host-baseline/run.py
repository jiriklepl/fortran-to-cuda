"""Preserve exact compiler host test inputs and output; no application campaign."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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
    for name in ['advection.f90', 'filters.f90']:
        source = PIPELINE / '.cache/repos/elmm/src' / name
        files['upstream_probe/' + name] = digest(source)
    return files


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    if (OUTPUT / 'final.json').exists() or (OUTPUT / 'collection.log').exists():
        raise SystemExit('Checkpoint already exists; use a new directory.')
    env = {**os.environ, 'ELMM_TEST_COMPILER_ROOT': str(PARENT),
           'FORT_TEST_ELMM_SOURCE': str(PIPELINE / '.cache/repos/elmm/src')}
    before = inventory()
    write('source-files-before.json', before)
    collect = [str(PYTHON), '-m', 'pytest', '--collect-only', '-q', '-m',
               'not cuda and not native', 'compiler/tests']
    started = time.monotonic()
    with (OUTPUT / 'collection.log').open('w') as log:
        result = subprocess.run(collect, cwd=PARENT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit('Collection failed: inspect collection.log')
    nodes = ['compiler/' + line for line in (OUTPUT / 'collection.log').read_text().splitlines()
             if line.startswith('tests/') and '::' in line]
    if not nodes or len(nodes) != len(set(nodes)):
        raise SystemExit('Empty or duplicate host collection')
    write('collected-tests.json', nodes)
    groups = [nodes[index::3] for index in range(3)]
    commands = [['taskset', '-c', '0,1,6,7', str(PYTHON), '-m', 'pytest', '-q',
                 '--basetemp=' + str(OUTPUT / ('pytest-' + str(index))), *group]
                for index, group in enumerate(groups)]
    receipt = {'schema_version': 1, 'claim': 'complete compiler host collection in three isolated workers; no performance claim',
        'collection_argv': collect, 'worker_argv': commands, 'cwd': str(PARENT),
        'test_count': len(nodes), 'worker_test_counts': list(map(len, groups)),
        'runner_sha256': digest(Path(__file__)), 'python_sha256': digest(PYTHON.resolve()),
        'environment': {key: env.get(key) for key in ['PATH', 'PYTHONPATH', 'LD_LIBRARY_PATH',
            'ELMM_TEST_COMPILER_ROOT', 'FORT_TEST_ELMM_SOURCE', 'OMP_NUM_THREADS', 'OMP_PLACES', 'OMP_PROC_BIND']},
        'commits': {str(root): subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
                    for root in [PARENT, PIPELINE]},
        'source_files_before_sha256': digest(OUTPUT / 'source-files-before.json')}
    write('command.json', receipt)
    def run(index):
        began = time.monotonic()
        path = OUTPUT / ('worker-' + str(index) + '.log')
        with path.open('w') as log:
            result = subprocess.run(commands[index], cwd=PARENT, env=env, stdout=log, stderr=subprocess.STDOUT)
        item = {'worker': index, 'exit_code': result.returncode, 'elapsed_seconds': time.monotonic() - began,
                'log_sha256': digest(path), 'test_count': len(groups[index])}
        write('worker-' + str(index) + '.json', item)
        print(json.dumps(item), flush=True)
        return item
    with ThreadPoolExecutor(max_workers=3) as pool:
        workers = list(pool.map(run, range(3)))
    after = inventory()
    write('source-files-after.json', after)
    changed = sorted(path for path in before.keys() | after.keys() if before.get(path) != after.get(path))
    result = {**receipt, 'workers': workers, 'elapsed_seconds': time.monotonic() - started,
        'source_files_after_sha256': digest(OUTPUT / 'source-files-after.json'),
        'source_files_changed': changed, 'stable_source_inputs': not changed}
    write('final.json', result)
    print(json.dumps({key: result[key] for key in ['test_count', 'elapsed_seconds', 'stable_source_inputs']}), flush=True)
    sys.exit(int(bool(changed) or any(worker['exit_code'] for worker in workers)))


if __name__ == '__main__':
    main()
