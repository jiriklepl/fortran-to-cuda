"""Preserve pair-specific pipeline validation without claiming a later release."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('/home/jirka/research/fortran-abomination')
PIPELINE = ROOT / 'elmm-pipeline'
OUTPUT = Path(__file__).resolve().parent
SUCCESS = PIPELINE / 'work/checked-plan-portable-validation-20261014-v2'
COLLECTION = PIPELINE / 'work/checked-plan-portable-validation-20261014'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    if (OUTPUT / 'evidence.json').exists():
        raise ValueError('Curated evidence already exists; do not overwrite it.')
    success = json.loads((SUCCESS / 'final.json').read_text())
    collection = json.loads((COLLECTION / 'final.json').read_text())
    assert success['exit_code'] == 0 and success['stable_source_inputs'] is True
    assert collection['exit_code'] == 2 and collection['stable_source_inputs'] is True
    assert success['commits'] == {
        str(ROOT): 'c16d01a0e2028c0f610f1291277f37df413344c9',
        str(PIPELINE): '494df4e9221b61d3c8f28c7f86ab7dd07cd6908f'}
    assert '616 passed, 22 skipped, 265 subtests passed in 98.14s' in (SUCCESS / 'final.log').read_text()
    assert '2396 errors during collection' in (COLLECTION / 'final.log').read_text()
    for source, receipt in [(SUCCESS, success), (COLLECTION, collection)]:
        for name, field in [('final.log', 'log_sha256'), ('run.py', 'runner_sha256'),
                            ('source-files-before.json', 'source_files_before_sha256'),
                            ('source-files-after.json', 'source_files_after_sha256')]:
            assert sha(source / name) == receipt[field], str(source / name)
        assert (source / 'source-files-before.json').read_bytes() == (source / 'source-files-after.json').read_bytes()

    copied = {}
    for directory, source, names in [
        ('successful', SUCCESS, ['final.json', 'final.log', 'command.json', 'run.py',
                                'source-files-before.json', 'source-files-after.json']),
        ('collection-only', COLLECTION, ['final.json', 'command.json', 'run.py', 'collection-only-note.json']),
    ]:
        destination = OUTPUT / directory
        destination.mkdir()
        for name in names:
            before = sha(source / name)
            shutil.copy2(source / name, destination / name)
            assert sha(destination / name) == sha(source / name) == before
            copied[directory + '/' + name] = {'original_path': str(source / name), 'sha256': before}

    inventories = json.loads((SUCCESS / 'source-files-before.json').read_text())
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
    commits = {}
    for repo, expected in groups.items():
        commit = success['commits'][str(repo)]
        data = subprocess.check_output(['git', '-C', str(repo), 'archive', commit])
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            tracked = {row.name: hashlib.sha256(archive.extractfile(row).read()).hexdigest()
                       for row in archive.getmembers() if row.isfile() and row.name in expected}
        mismatches = [name for name, digest in tracked.items() if expected[name] != digest]
        assert not mismatches, mismatches
        commits[str(repo)] = {'commit': commit, 'tracked_inventory_files_verified': len(tracked),
                              'tracked_hash_mismatches': mismatches,
                              'untracked_inventory_entries': sorted(set(expected) - set(tracked))}
    write('commit-verification.json', commits)
    write('evidence.json', {
        'schema_version': 1,
        'claim': 'Pipeline integration for the recorded commit pair; subsequent compiler corrections are pending.',
        'corrected_release_qualification': False,
        'application_performance_qualification': False,
        'successful_run': {'parent_commit': success['commits'][str(ROOT)],
            'pipeline_commit': success['commits'][str(PIPELINE)],
            'tests_passed': 616, 'tests_skipped': 22, 'subtests_passed': 265, 'pytest_seconds': 98.14,
            'source_inventory_unchanged': True, 'source_inventory_sha256': success['source_files_before_sha256'],
            'receipt': 'successful/final.json', 'log': 'successful/final.log'},
        'collection_failure': {'classification': 'diagnostic runner collection only; no test items executed',
            'collection_errors': 2396, 'warnings': 65, 'tests_executed': 0,
            'cause': 'Implicit pytest discovery traversed retained independent compiler copies in pipeline/work.',
            'correction': 'New runner explicitly selected the complete pipeline tests directory.',
            'production_source_changed': False, 'receipt': 'collection-only/final.json',
            'note': 'collection-only/collection-only-note.json',
            'full_failed_log_included': False, 'original_full_log': str(COLLECTION / 'final.log'),
            'original_full_log_sha256': collection['log_sha256'],
            'source_inventory_sha256': collection['source_files_before_sha256'],
            'source_inventory_equal_to_successful_run': collection['source_files_before_sha256'] == success['source_files_before_sha256']},
        'copied_artifacts': copied,
        'commit_verification': 'commit-verification.json',
        'later_compiler_issues': ['hash-seed portability', 'overbroad proof borrowing'],
        'scope_note': 'This receipt does not claim these later issues are fixed; no later source revision is covered.'})
    artifacts = {str(path.relative_to(OUTPUT)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(OUTPUT.rglob('*')) if path.is_file() and path.name != 'artifact-inventory.json'}
    inventory = {'schema_version': 1, 'files': artifacts}
    inventory['identity'] = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write('artifact-inventory.json', inventory)
    print(json.dumps({'files': len(artifacts), 'bytes': sum(item['bytes'] for item in artifacts.values()),
                      'inventory_identity': inventory['identity'], 'output': str(OUTPUT)}, sort_keys=True))


if __name__ == '__main__':
    main()
