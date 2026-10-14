"""Freeze the diagnostic fixture correction separately from full validation."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path('/home/jirka/research/fortran-abomination')
OUTPUT = Path(__file__).resolve().parent
BASE = '0eeae487bc68a0a88090a8304e3dceaac1df45a5'
FINAL = 'af7584edcec5037b1f3bbc0ab3cc128d3611fd4c'
PIPELINE = '8ebaa348bbaa8525435a727560d217a4a15a1bc2'
TEST = 'compiler/tests/test_lexical_owner_diagnostics.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


def read(path):
    return json.loads(path.read_text())


def main():
    if (OUTPUT / 'evidence.json').exists():
        raise ValueError('Evidence already frozen.')
    changed = subprocess.check_output(['git', '-C', str(ROOT), 'diff', '--name-only', BASE, FINAL], text=True).splitlines()
    assert changed == [TEST]
    final_test = subprocess.check_output(['git', '-C', str(ROOT), 'show', FINAL + ':' + TEST])
    final_digest = hashlib.sha256(final_test).hexdigest()
    baseline_path = OUTPUT.parent / 'host/source-files.json'
    baseline = read(baseline_path)
    patch = subprocess.check_output(['git', '-C', str(ROOT), 'diff', BASE, FINAL, '--', TEST])
    (OUTPUT / 'committed-test-correction.patch').write_bytes(patch)
    runs = []
    for label, suffix in [('initial', ''), ('corrected-working-tree', '-corrected'), ('committed', '-committed')]:
        origin = Path('/tmp') / f'automatic-native-gate-diagnostics{suffix}-20261014'
        receipt = read(origin / 'final.json')
        expected_head = FINAL if label == 'committed' else BASE
        assert receipt['commits'] == {str(ROOT): expected_head, str(ROOT / 'elmm-pipeline'): PIPELINE}
        assert receipt['stable_source_inputs'] and not receipt['source_files_changed']
        for name, field in [('run.py', 'runner_sha256'), ('final.log', 'log_sha256'),
                            ('source-files-before.json', 'source_files_before_sha256'),
                            ('source-files-after.json', 'source_files_after_sha256')]:
            assert sha(origin / name) == receipt[field], (label, name)
        before = (origin / 'source-files-before.json').read_bytes()
        assert before == (origin / 'source-files-after.json').read_bytes()
        inventory = json.loads(before)
        delta = {name: inventory.get(name) for name in baseline.keys() | inventory.keys()
                 if baseline.get(name) != inventory.get(name)}
        incidental = sorted(name for name in delta if name != TEST)
        assert TEST in delta
        assert all(name.startswith('compiler/.ruff_cache/') for name in incidental)
        for name in incidental:
            assert not subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', FINAL, '--', name])
        assert encoded({**baseline, **delta}) == before
        if label != 'initial':
            assert delta[TEST] == final_digest
        else:
            assert delta[TEST] != final_digest
        destination = OUTPUT / label
        destination.mkdir(exist_ok=True)
        copied = {}
        for name in ('command.json', 'final.json', 'final.log'):
            shutil.copy2(origin / name, destination / name)
            assert sha(destination / name) == sha(origin / name)
            copied[name] = {'original_path': str(origin / name), 'sha256': sha(origin / name)}
        if not (OUTPUT / 'run.py').exists():
            shutil.copy2(origin / 'run.py', OUTPUT / 'run.py')
        assert sha(OUTPUT / 'run.py') == receipt['runner_sha256']
        (destination / 'source-inventory-delta.json').write_bytes(encoded(delta))
        log = (origin / 'final.log').read_text()
        summary, = [line for line in log.splitlines() if re.search(r'\bpassed\b.*\bin [0-9.]+s\b', line)]
        counts = {key: int(value) for value, key in re.findall(r'(\d+) (passed|failed|skipped)', summary)}
        runs.append({'label': label, 'origin': str(origin), 'recorded_head': expected_head,
            'pipeline_commit': PIPELINE, 'exit_code': receipt['exit_code'], 'pytest_summary': summary,
            'counts': counts, 'source_inventory_unchanged': True,
            'source_inventory_sha256': receipt['source_files_before_sha256'],
            'inventory_base': '../host/source-files.json', 'inventory_base_sha256': sha(baseline_path),
            'inventory_delta': label + '/source-inventory-delta.json',
            'inventory_reconstruction': 'Merge base with delta, then json.dumps(sort_keys=True, indent=2) plus newline.',
            'tested_file_sha256': delta[TEST], 'tracked_mismatch_with_recorded_head': [] if label == 'committed' else [TEST],
            'tracked_inventory_matches_final_correction_commit': label != 'initial',
            'incidental_untracked_cache_delta': incidental,
            'intermediate_source_contents_preserved': label != 'initial',
            'intermediate_note': 'Initial WIP source content is not archived; its exact hash and failing raw outcome are preserved.' if label == 'initial' else None,
            'runner': 'run.py', 'runner_sha256': receipt['runner_sha256'], 'raw_artifacts': copied})
    evidence = {'schema_version': 1, 'claim': 'Separate diagnostic fixture correction supplement; full host baseline remains failed.',
        'baseline_compiler_commit': BASE, 'corrected_compiler_commit': FINAL, 'pipeline_commit': PIPELINE,
        'committed_changed_paths': changed, 'production_changed': False,
        'final_test_sha256': final_digest, 'final_patch': 'committed-test-correction.patch',
        'full_suite_rerun_on_corrected_commit': False, 'application_performance_evidence': False, 'runs': runs}
    (OUTPUT / 'evidence.json').write_bytes(encoded(evidence))
    artifacts = {str(path.relative_to(OUTPUT)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(OUTPUT.rglob('*')) if path.is_file() and path.name != 'artifact-inventory.json'}
    (OUTPUT / 'artifact-inventory.json').write_bytes(encoded({'schema_version': 1, 'files': artifacts}))
    print(json.dumps({'files': len(artifacts), 'bytes': sum(row['bytes'] for row in artifacts.values()),
                      'runs': [{key: row[key] for key in ('label', 'recorded_head', 'counts', 'exit_code')} for row in runs]}))


if __name__ == '__main__':
    main()
