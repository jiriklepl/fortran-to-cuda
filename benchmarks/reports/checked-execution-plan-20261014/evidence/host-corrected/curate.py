"""Freeze complete host validation and its required four-CPU supplement."""
from __future__ import annotations

import ast
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
HOST = Path('/tmp/checked-plan-host-corrected-20261014')
SUPPLEMENT = Path('/tmp/checked-plan-four-cpu-validation-20261014')
OUTPUT = Path(__file__).resolve().parent
COMMITS = {str(ROOT): 'ff9565f8ac941f9d9fad19e20027840e83f953b5',
           str(PIPELINE): '494df4e9221b61d3c8f28c7f86ab7dd07cd6908f'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def summary(path):
    matches = re.findall(r'^(\d+) passed(?:, (\d+) skipped)? in ([0-9.]+)s.*$',
                         path.read_text(), re.MULTILINE)
    assert len(matches) == 1, path
    passed, skipped, seconds = matches[0]
    return {'passed': int(passed), 'skipped': int(skipped or 0), 'pytest_seconds': float(seconds)}


def verify_receipt(root):
    receipt = read(root / 'final.json')
    assert receipt['commits'] == COMMITS
    assert receipt['stable_source_inputs'] and not receipt['source_files_changed']
    for name, field in [('run.py', 'runner_sha256'),
                        ('source-files-before.json', 'source_files_before_sha256'),
                        ('source-files-after.json', 'source_files_after_sha256')]:
        assert sha(root / name) == receipt[field], (root, name)
    assert read(root / 'source-files-before.json') == read(root / 'source-files-after.json')
    command = read(root / 'command.json')
    assert all(receipt[key] == value for key, value in command.items())
    return receipt


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
            external[name] = {'sha256': digest, 'repository_commit_verified': False,
                              'note': 'External cached source probe; covered by before/after inventory only.'}
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
        verified[str(repo)] = {'commit': commit, 'tracked_inventory_files_verified': len(tracked),
                               'tracked_hash_mismatches': mismatches,
                               'untracked_inventory_entries': sorted(set(expected) - set(tracked))}
    return {'repositories': verified, 'external_inventory_entries': external}


def main():
    if (OUTPUT / 'evidence.json').exists():
        raise ValueError('Curated evidence already exists; choose a new checkpoint.')
    host = verify_receipt(HOST)
    supplement = verify_receipt(SUPPLEMENT)
    assert supplement['exit_code'] == 0
    assert sha(SUPPLEMENT / 'final.log') == supplement['log_sha256']
    assert host['source_files_before_sha256'] == supplement['source_files_before_sha256']
    nodes = read(HOST / 'collected-tests.json')
    assert len(nodes) == len(set(nodes)) == host['test_count']
    collected = ['compiler/' + line for line in (HOST / 'collection.log').read_text().splitlines()
                 if line.startswith('tests/') and '::' in line]
    assert collected == nodes
    dependency_nodes = [node for node in nodes if node.startswith('compiler/tests/test_dependency_source.py::')]
    skipped = []
    worker_results = []
    dependency_path = 'compiler/tests/test_dependency_source.py'
    dependency_text = subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', COMMITS[str(ROOT)] + ':' + dependency_path], text=True)
    functions = {node.lineno: node.name for node in ast.parse(dependency_text).body
                 if isinstance(node, ast.FunctionDef)}
    for index, worker in enumerate(host['workers']):
        assert worker == read(HOST / f'worker-{index}.json')
        assert worker['worker'] == index and worker['exit_code'] == 0
        log = HOST / f'worker-{index}.log'
        assert sha(log) == worker['log_sha256']
        argv = host['worker_argv'][index]
        assert argv[:3] == ['taskset', '-c', '0,1,2,3,6,7,8,9']
        assert argv[8:] == nodes[index::3]
        result = summary(log)
        assert result['passed'] + result['skipped'] == worker['test_count']
        worker_results.append({'worker': index, **result, 'exit_code': worker['exit_code']})
        skips = re.findall(r'^SKIPPED \[1\] compiler/tests/test_dependency_source.py:(\d+): (.*)$',
                           log.read_text(), re.MULTILINE)
        assert len(skips) == result['skipped']
        for line, reason in skips:
            assert reason == 'run CPU-only source query tests on exactly four CPUs'
            node = dependency_path + '::' + functions[int(line)]
            assert node in nodes[index::3]
            skipped.append({'nodeid': node, 'worker': index, 'source_line': int(line), 'reason': reason,
                            'nodeid_basis': 'raw skip source line mapped to the verified committed function'})
    supplementary_counts = summary(SUPPLEMENT / 'final.log')
    assert supplementary_counts['passed'] == len(dependency_nodes)
    assert supplementary_counts['skipped'] == 0
    assert supplement['argv'][0:3] == ['taskset', '-c', '0,1,6,7']
    assert dependency_path in supplement['argv']
    assert all(item['nodeid'] in dependency_nodes for item in skipped)
    verification = verify_commits(read(HOST / 'source-files-before.json'))

    copied = {}
    host_names = ['run.py', 'command.json', 'final.json', 'collection.log', 'collected-tests.json',
                  'source-files-before.json', 'source-files-after.json',
                  *[f'worker-{index}.{suffix}' for index in range(3) for suffix in ('json', 'log')]]
    supplement_names = ['run.py', 'command.json', 'final.json', 'final.log',
                        'source-files-before.json', 'source-files-after.json']
    for origin, prefix, names in [(HOST, 'full-host', host_names),
                                  (SUPPLEMENT, 'four-cpu-supplement', supplement_names)]:
        destination = OUTPUT / prefix
        destination.mkdir()
        for name in names:
            digest = sha(origin / name)
            shutil.copy2(origin / name, destination / name)
            assert sha(destination / name) == sha(origin / name) == digest
            copied[prefix + '/' + name] = {'original_path': str(origin / name), 'sha256': digest}
    write('commit-verification.json', verification)
    write('evidence.json', {
        'schema_version': 1, 'status': 'passed', 'commits': COMMITS,
        'claim': 'Complete compiler host selection, with all four affinity-dependent skips passed in a separate four-CPU whole-file run.',
        'application_performance_qualification': False, 'cuda_native_suite_qualification': False,
        'source_inventory_entries': len(read(HOST / 'source-files-before.json')),
        'source_inventory_unchanged': True, 'source_inventory_sha256': host['source_files_before_sha256'],
        'tracked_commit_mismatches': 0, 'commit_verification': 'commit-verification.json',
        'full_host': {'selected': len(nodes), 'passed': sum(item['passed'] for item in worker_results),
                      'skipped': len(skipped), 'failed': 0, 'workers': worker_results,
                      'cpu_affinity': '0,1,2,3,6,7,8,9', 'selection': 'not cuda and not native',
                      'receipt': 'full-host/final.json', 'skipped_tests': skipped},
        'four_cpu_supplement': {**supplementary_counts, 'failed': 0, 'cpu_affinity': '0,1,6,7',
                               'nodeids': dependency_nodes,
                               'nodeid_basis': 'same-source full host collection filtered to the entire rerun file',
                               'receipt': 'four-cpu-supplement/final.json', 'log': 'four-cpu-supplement/final.log'},
        'combined_coverage': {'unique_selected_host_tests_passed': len(nodes),
                              'repeated_already_passed_tests': len(dependency_nodes) - len(skipped),
                              'remaining_skipped_tests': 0,
                              'note': 'Counts combine separate affinity-appropriate runs; this is not one 5153-pass invocation.'},
        'scope_note': 'Correctness jobs shared hardware. Elapsed test time is not application performance evidence. Pytest work trees, generated binaries and caches are excluded.',
        'copied_artifacts': copied})
    artifacts = {str(path.relative_to(OUTPUT)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(OUTPUT.rglob('*')) if path.is_file() and path.name != 'artifact-inventory.json'}
    inventory = {'schema_version': 1, 'files': artifacts}
    inventory['identity'] = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write('artifact-inventory.json', inventory)
    print(json.dumps({'files': len(artifacts), 'bytes': sum(item['bytes'] for item in artifacts.values()),
                      'host_passed': sum(item['passed'] for item in worker_results), 'host_skipped': len(skipped),
                      'supplement_passed': supplementary_counts['passed'], 'output': str(OUTPUT)}, sort_keys=True))


if __name__ == '__main__':
    main()
