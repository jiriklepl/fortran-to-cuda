"""Preserve the intermediate native-build fixture and bind it to final sources."""
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
RUN = PIPELINE / 'work/missing-cost-native-build-validation-20261014'
OUTPUT = Path(__file__).resolve().parent
FINAL_COMMITS = {ROOT: '0eeae487bc68a0a88090a8304e3dceaac1df45a5',
                 PIPELINE: '8ebaa348bbaa8525435a727560d217a4a15a1bc2'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    if (OUTPUT / 'evidence.json').exists():
        raise ValueError('Evidence is already frozen; choose another checkpoint.')
    receipt = json.loads((RUN / 'final.json').read_text())
    assert receipt['exit_code'] == 0 and receipt['stable_source_inputs'] is True
    assert (RUN / 'source-files-before.json').read_bytes() == (RUN / 'source-files-after.json').read_bytes()
    for name, field in [('final.log', 'log_sha256'), ('run.py', 'runner_sha256'),
                        ('source-files-before.json', 'source_files_before_sha256'),
                        ('source-files-after.json', 'source_files_after_sha256')]:
        assert sha(RUN / name) == receipt[field]
    assert '5 passed in 4.47s' in (RUN / 'final.log').read_text()
    binding = json.loads((RUN / 'working-tree-binding.json').read_text())
    for name, repo in [('compiler', ROOT), ('pipeline', PIPELINE)]:
        change = binding['changes'][name]
        assert change['base_commit'] == receipt['commits'][str(repo)]
        assert sha(RUN / change['patch']) == change['patch_sha256']
        final_patch = subprocess.check_output(['git', '-C', str(repo), 'diff', '--binary',
            change['base_commit'], FINAL_COMMITS[repo], '--', *change['paths']])
        assert final_patch == (RUN / change['patch']).read_bytes()

    groups = {ROOT: {}, PIPELINE: {}}
    for name, digest in json.loads((RUN / 'source-files-before.json').read_text()).items():
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
    verification = {}
    for repo, expected in groups.items():
        commit = FINAL_COMMITS[repo]
        tracked = set(subprocess.check_output(
            ['git', '-C', str(repo), 'ls-tree', '-r', '--name-only', commit], text=True).splitlines())
        selected = sorted(tracked & expected.keys())
        data = subprocess.check_output(['git', '-C', str(repo), 'archive', commit, '--', *selected])
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            files = {row.name: hashlib.sha256(archive.extractfile(row).read()).hexdigest()
                     for row in archive.getmembers() if row.isfile()}
        assert set(files) == set(selected)
        differences = [path for path, digest in files.items() if expected[path] != digest]
        assert differences == []
        verification[str(repo)] = {'final_commit': commit,
            'recorded_working_tree_base': receipt['commits'][str(repo)],
            'tracked_inventory_files_verified': len(files), 'final_commit_differences': differences,
            'untracked_inventory_entries': sorted(expected.keys() - files.keys()),
            'working_patch_equals_base_to_final_diff': True}
    write('final-source-verification.json', verification)

    fixture, = RUN.glob('pytest/**/integration.json')
    fixture_root = fixture.parent
    integration = json.loads(fixture.read_text())
    adapter = integration['adapter']
    assert adapter['scope_manifest']['scope_count'] == 0 and adapter['native_success'] is True
    assert adapter['gpu_sources'] == adapter['extra_sources'] == adapter['changed_files'] == []
    assert 'gpu' not in integration['build']
    assert integration['build']['extra_sources'] == integration['build']['patched_sources'] == []
    assert (fixture_root / 'integration-runtime.log').read_text().strip() == 'FULL_FIELDS_OK'
    build_receipts = json.loads((fixture_root / 'build/build-receipts.json').read_text())
    assert build_receipts['status'] == 'complete' and len(build_receipts['units']) == 2
    for path, digest in build_receipts['files_sha256'].items():
        assert sha(path) == digest, path

    copied = {}
    paths = [(name, RUN / name) for name in [
        'final.json', 'final.log', 'command.json', 'run.py', 'source-files-before.json',
        'working-tree-binding.json', 'compiler-working.patch', 'pipeline-working.patch']]
    paths += [('fixture/' + name, fixture_root / name) for name in [
        'integration.json', 'integration-runtime.log', 'integration-build.log',
        'build/build-receipts.json', 'build/build-command.json', 'build/build.log']]
    for name, source in paths:
        destination = OUTPUT / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(source)
        shutil.copy2(source, destination)
        assert sha(destination) == digest
        copied[name] = {'original_path': str(source), 'sha256': digest}
    write('evidence.json', {'schema_version': 1,
        'claim': 'Five focused pipeline integrations, including one real native fixture build and full-field execution.',
        'classification': 'Intermediate working-tree test run; exact recorded production and selected test files match the later final commits.',
        'recorded_base_commits': receipt['commits'],
        'final_matching_commits': {str(repo): commit for repo, commit in FINAL_COMMITS.items()},
        'tests_passed': 5, 'pytest_seconds': 4.47, 'stable_source_inputs': True,
        'source_inventory_sha256': receipt['source_files_before_sha256'],
        'coverage_limit': 'Inventory includes compiler files except compiler tests/caches, pipeline implementation, selected root metadata, and the two invoked pipeline test modules. It is not a complete compiler-test inventory.',
        'test_or_documentation_differences_in_recorded_inventory': [],
        'source_inventory_note': 'One identical before/after inventory is included; both hashes remain in final.json.',
        'actual_native_builds': 1, 'native_object_count': 2, 'scope_count': 0,
        'gpu_build_contract_present': False, 'generated_or_patched_build_sources': 0,
        'fixture_validation': 'FULL_FIELDS_OK: unchanged input and exact producer/native-correction/consumer results for two 1D shapes, with negative host lower bounds.',
        'fixture_artifact_provenance': 'Retained pytest artifacts are hashed at curation. Build receipts bind actual compile/link inputs and outputs; the fixture artifacts themselves were not inventoried before/after execution.',
        'application_build_or_run': False, 'channel_field_or_halo_validation': False,
        'application_performance_qualification': False, 'automatic_offloading_qualified': False,
        'final_source_verification': 'final-source-verification.json', 'copied_artifacts': copied})
    artifacts = {str(path.relative_to(OUTPUT)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(OUTPUT.rglob('*')) if path.is_file() and path.name != 'artifact-inventory.json'}
    inventory = {'schema_version': 1, 'files': artifacts}
    inventory['identity'] = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    write('artifact-inventory.json', inventory)
    print(json.dumps({'files': len(artifacts), 'bytes': sum(row['bytes'] for row in artifacts.values()),
                      'inventory_identity': inventory['identity'], 'output': str(OUTPUT)}, sort_keys=True))


if __name__ == '__main__':
    main()
