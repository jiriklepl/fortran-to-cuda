"""Preserve validation receipts without duplicating large command inventories."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

ROOT = Path('/home/jirka/research/fortran-abomination')
PIPELINE = ROOT / 'elmm-pipeline'
COMMITS = {str(ROOT): '0eeae487bc68a0a88090a8304e3dceaac1df45a5',
           str(PIPELINE): '8ebaa348bbaa8525435a727560d217a4a15a1bc2'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def verify_commits(inventory):
    groups = {ROOT: {}, PIPELINE: {}}
    external = {}
    for name, digest in inventory.items():
        prefix, relative = name.split('/', 1)
        if prefix == 'compiler':
            repo, path = ROOT, 'compiler/' + relative
        elif prefix == 'pipeline':
            repo, path = PIPELINE, 'elmm_pipeline/' + relative
        elif prefix == 'pipeline_tests':
            repo, path = PIPELINE, 'tests/' + relative
        elif prefix == 'pipeline_root':
            repo, path = PIPELINE, relative
        elif prefix == 'upstream_probe':
            external[name] = digest
            continue
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
        assert not mismatches, (repo, mismatches)
        verified[str(repo)] = {'commit': commit, 'tracked_files_verified': len(tracked),
                               'hash_mismatches': mismatches,
                               'untracked_inventory_entries': sorted(set(expected) - set(tracked))}
    return {'repositories': verified, 'external_before_after_only': external}


def counts(path):
    lines = [line for line in path.read_text().splitlines()
             if re.search(r'\b(?:passed|failed|skipped|errors?)\b.*\bin [0-9.]+s\b', line)]
    assert len(lines) == 1, path
    values = {name: int(number) for number, name in re.findall(
        r'(\d+) (passed|failed|skipped|errors?|deselected)', lines[0])}
    return {'summary': lines[0], **values}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('kind', choices=('host', 'cuda'))
    args = parser.parse_args()
    origin = Path('/tmp') / f'automatic-native-gate-{args.kind}-20261014'
    output = Path(__file__).resolve().parent.parent / args.kind
    output.mkdir(exist_ok=True)
    if (output / 'evidence.json').exists():
        raise ValueError('Evidence already frozen; use a new checkpoint.')
    final = read(origin / 'final.json')
    command = read(origin / 'command.json')
    assert final['commits'] == command['commits'] == COMMITS
    assert all(final[key] == value for key, value in command.items())
    for name, field in [('run.py', 'runner_sha256'), ('source-files-before.json', 'source_files_before_sha256'),
                        ('source-files-after.json', 'source_files_after_sha256')]:
        assert sha(origin / name) == final[field], name
    assert final['stable_source_inputs'] and not final['source_files_changed']
    assert (origin / 'source-files-before.json').read_bytes() == (origin / 'source-files-after.json').read_bytes()
    final_fields = {key: value for key, value in final.items() if key not in command}
    assert encoded({**command, **final_fields}) == (origin / 'final.json').read_bytes()

    names = ['command.json', 'run.py']
    results = []
    if args.kind == 'host':
        nodes = read(origin / 'collected-tests.json')
        assert encoded(nodes) == (origin / 'collected-tests.json').read_bytes()
        assert len(nodes) == len(set(nodes)) == final['test_count']
        collected = ['compiler/' + line for line in (origin / 'collection.log').read_text().splitlines()
                     if line.startswith('tests/') and '::' in line]
        assert collected == nodes
        names.append('collection.log')
        for index, worker in enumerate(final['workers']):
            assert worker == read(origin / f'worker-{index}.json')
            assert worker['worker'] == index
            assert sha(origin / f'worker-{index}.log') == worker['log_sha256']
            argv = command['worker_argv'][index]
            assert argv[:3] == ['taskset', '-c', ['0,1,6,7', '2,3,8,9', '0,1,2,3'][index]]
            assert argv[8:] == nodes[index::3]
            result = counts(origin / f'worker-{index}.log')
            assert sum(result.get(key, 0) for key in ('passed', 'failed', 'skipped', 'error', 'errors')) == worker['test_count']
            results.append({'worker': index, 'cpu_affinity': argv[2], 'exit_code': worker['exit_code'], **result})
            names.extend([f'worker-{index}.json', f'worker-{index}.log'])
    else:
        assert sha(origin / 'final.log') == final['log_sha256']
        results.append({'cpu_affinity': command['argv'][2], 'exit_code': final['exit_code'],
                        **counts(origin / 'final.log')})
        names.append('final.log')

    origins = {}
    for name in names:
        shutil.copy2(origin / name, output / name)
        assert sha(output / name) == sha(origin / name)
        origins[name] = {'original_path': str(origin / name), 'sha256': sha(origin / name)}
    shutil.copy2(origin / 'source-files-before.json', output / 'source-files.json')
    for name in ('source-files-before.json', 'source-files-after.json'):
        origins[name] = {'original_path': str(origin / name), 'sha256': sha(origin / name),
                         'stored_identically_as': 'source-files.json'}
    origins['final.json'] = {'original_path': str(origin / 'final.json'), 'sha256': sha(origin / 'final.json'),
                             'reconstruction': 'Merge command.json with outcome.json.final_fields; json.dumps(sort_keys=True, indent=2) plus newline reproduces original bytes.'}
    if args.kind == 'host':
        origins['collected-tests.json'] = {'original_path': str(origin / 'collected-tests.json'),
            'sha256': sha(origin / 'collected-tests.json'),
            'reconstruction': 'Round-robin interleave command.json.worker_argv[*][8:]; same ordered node IDs are retained in collection.log.'}

    def write(name, value):
        (output / name).write_bytes(encoded(value))

    write('outcome.json', {'schema_version': 1, 'command_receipt': 'command.json', 'final_fields': final_fields})
    write('commit-verification.json', verify_commits(read(output / 'source-files.json')))
    write('evidence.json', {'schema_version': 1, 'commits': COMMITS,
        'status': 'passed' if all(item['exit_code'] == 0 for item in results) else 'failed',
        'claim': 'Complete selected compiler host suite' if args.kind == 'host' else 'Affected CUDA/native regressions only',
        'results': results, 'totals': {key: sum(item.get(key, 0) for item in results)
                                     for key in ('passed', 'failed', 'skipped', 'deselected', 'error', 'errors')},
        'source_inventory_unchanged': True, 'original_artifacts': origins,
        'application_execution_or_performance_evidence': False,
        'scope_note': 'Correctness jobs shared hardware. Test duration is not application performance. Pytest work trees, generated binaries and cache contents are excluded.'})
    artifacts = {str(path.relative_to(output)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'artifact-inventory.json'}
    write('artifact-inventory.json', {'schema_version': 1, 'files': artifacts})
    print(json.dumps({'kind': args.kind, 'results': results, 'files': len(artifacts),
                      'bytes': sum(item['bytes'] for item in artifacts.values())}, sort_keys=True))


if __name__ == '__main__':
    main()
