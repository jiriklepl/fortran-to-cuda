"""Freeze the corrected compiler/pipeline pair's complete integration result."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path('/home/jirka/research/fortran-abomination')
PIPELINE = ROOT / 'elmm-pipeline'
RUN = PIPELINE / 'work/checked-plan-corrected-validation-20261014'
OUTPUT = Path(__file__).resolve().parent
COMMITS = {str(ROOT): 'ff9565f8ac941f9d9fad19e20027840e83f953b5',
           str(PIPELINE): '494df4e9221b61d3c8f28c7f86ab7dd07cd6908f'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    if (OUTPUT / 'evidence.json').exists():
        raise ValueError('Curated evidence already exists; choose a new checkpoint.')
    receipt = json.loads((RUN / 'final.json').read_text())
    assert receipt['commits'] == COMMITS
    for name, field in [('final.log', 'log_sha256'), ('run.py', 'runner_sha256'),
                        ('source-files-before.json', 'source_files_before_sha256'),
                        ('source-files-after.json', 'source_files_after_sha256')]:
        assert sha(RUN / name) == receipt[field], name
    log = (RUN / 'final.log').read_text()
    summaries = [line.strip() for line in log.splitlines()
                 if re.search(r'\b(?:passed|failed|errors?)\b.*\bin [0-9.]+s\b', line)]
    result = 'passed' if receipt['exit_code'] == 0 and receipt['stable_source_inputs'] else 'failed'
    copied = {}
    destination = OUTPUT / result
    destination.mkdir()
    for name in ['final.json', 'final.log', 'command.json', 'run.py',
                 'source-files-before.json', 'source-files-after.json']:
        digest = sha(RUN / name)
        shutil.copy2(RUN / name, destination / name)
        assert sha(destination / name) == sha(RUN / name) == digest
        copied[result + '/' + name] = {'original_path': str(RUN / name), 'sha256': digest}

    inventories = json.loads((RUN / 'source-files-before.json').read_text())
    groups = {ROOT: {}, PIPELINE: {}}
    for name, digest in inventories.items():
        prefix, relative = name.split('/', 1)
        if prefix == 'compiler':
            repo, path = ROOT, 'compiler/' + relative
        elif prefix == 'pipeline':
            repo, path = PIPELINE, 'elmm_pipeline/' + relative
        elif prefix == 'pipeline_tests':
            repo, path = PIPELINE, 'tests/' + relative
        elif prefix == 'pipeline_root':
            repo, path = PIPELINE, relative
        else:
            raise ValueError(name)
        groups[repo][path] = digest
    verified = {}
    for repo, expected in groups.items():
        commit = COMMITS[str(repo)]
        tracked_paths = set(subprocess.check_output(
            ['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', commit], text=True).splitlines())
        selected = sorted(tracked_paths & expected.keys())
        data = subprocess.check_output(['git', '-C', str(repo), 'archive', commit, '--', *selected])
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            tracked = {row.name: hashlib.sha256(archive.extractfile(row).read()).hexdigest()
                       for row in archive.getmembers() if row.isfile()}
        assert set(tracked) == set(selected)
        mismatches = [name for name, digest in tracked.items() if expected[name] != digest]
        verified[str(repo)] = {'commit': commit, 'tracked_inventory_files_verified': len(tracked),
                               'tracked_hash_mismatches': mismatches,
                               'untracked_inventory_entries': sorted(set(expected) - set(tracked))}
    write('commit-verification.json', verified)
    write('evidence.json', {
        'schema_version': 1,
        'claim': 'Complete pipeline integration suite for the recorded corrected compiler/pipeline pair.',
        'application_performance_qualification': False,
        'full_compiler_suite_qualification': False,
        'status': result, 'commits': COMMITS, 'exit_code': receipt['exit_code'],
        'pytest_summary': summaries[-1] if summaries else None,
        'source_inventory_unchanged': receipt['stable_source_inputs'],
        'source_inventory_sha256': receipt['source_files_before_sha256'],
        'source_files_changed': receipt['source_files_changed'],
        'tracked_commit_mismatches': sum(len(item['tracked_hash_mismatches']) for item in verified.values()),
        'cpu_affinity': '0,1,6,7', 'test_selection': 'pytest -q tests',
        'receipt': result + '/final.json', 'log': result + '/final.log',
        'commit_verification': 'commit-verification.json', 'copied_artifacts': copied,
        'scope_note': 'Concurrent correctness jobs shared hardware; elapsed test time is not application performance evidence.'})
    artifacts = {str(path.relative_to(OUTPUT)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(OUTPUT.rglob('*')) if path.is_file() and path.name != 'artifact-inventory.json'}
    inventory = {'schema_version': 1, 'files': artifacts}
    inventory['identity'] = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write('artifact-inventory.json', inventory)
    print(json.dumps({'status': result, 'files': len(artifacts),
                      'bytes': sum(item['bytes'] for item in artifacts.values()),
                      'inventory_identity': inventory['identity'], 'output': str(OUTPUT)}, sort_keys=True))


if __name__ == '__main__':
    main()
