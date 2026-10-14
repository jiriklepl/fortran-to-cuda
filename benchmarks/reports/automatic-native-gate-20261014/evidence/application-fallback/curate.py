"""Freeze compact evidence for missing-cost native selection, not offloading."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path('/home/jirka/research/fortran-abomination')
RUN = ROOT / 'benchmarks/results/automatic-native-gate-20261014'
OUTPUT = Path(__file__).resolve().parent
COMMITS = {'compiler': '0eeae487bc68a0a88090a8304e3dceaac1df45a5',
           'pipeline': '8ebaa348bbaa8525435a727560d217a4a15a1bc2'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    if (OUTPUT / 'evidence.json').exists():
        raise ValueError('Evidence is already frozen; choose a new checkpoint.')
    receipt = json.loads((RUN / 'completion.json').read_text())
    worker = json.loads((RUN / 'worker-result.json').read_text())
    manifest_path = RUN / 'generated/scope-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert receipt['exit_code'] == 0 and receipt['changed_inputs'] == []
    assert worker['status'] == 'authenticated_unchanged_native_fallback'
    assert worker['scope_count'] == 0 and worker['changed_inputs'] == []
    for name, field in [('generation.log', 'log_sha256'), ('run.py', 'runner_sha256'),
                        ('worker-result.json', 'worker_result_sha256')]:
        assert sha(RUN / name) == receipt[field]
    assert sha(manifest_path) == worker['manifest_sha256']
    assert len(receipt['input_sha256']) == 178
    prior_path = ROOT / 'benchmarks/results/checked-plan-application-20261014/automatic-generation/completion.json'
    assert sha(prior_path) == receipt['prior_input_receipt_sha256']
    assert receipt['input_sha256'] == json.loads(prior_path.read_text())['input_sha256']
    for path, digest in receipt['input_sha256'].items():
        assert sha(path) == digest, path
    verified = {}
    for name, repo, pathspec in [('compiler', ROOT, 'compiler'),
                                ('pipeline', ROOT / 'elmm-pipeline', 'elmm_pipeline')]:
        identity = receipt['identities'][name]
        assert identity['commit'] == COMMITS[name]
        archive_path = RUN / (name + '.tar')
        assert sha(archive_path) == identity['archive_sha256']
        data = subprocess.check_output(['git', '-C', str(repo), 'archive', COMMITS[name], '--', pathspec])
        assert hashlib.sha256(data).hexdigest() == identity['archive_sha256']
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            files = {row.name: hashlib.sha256(archive.extractfile(row).read()).hexdigest()
                     for row in archive.getmembers() if row.isfile()}
        assert files == identity['files']
        verified[name] = {'commit': COMMITS[name], 'archive_sha256': identity['archive_sha256'],
                          'tracked_archive_files_verified': len(files), 'mismatches': []}
    write('commit-verification.json', verified)

    copied = {}
    for name in ['completion.json', 'worker-result.json', 'generation.log', 'run.py']:
        digest = sha(RUN / name)
        shutil.copy2(RUN / name, OUTPUT / name)
        assert sha(OUTPUT / name) == digest
        copied[name] = {'original_path': str(RUN / name), 'sha256': digest}
    selected = {key: manifest[key] for key in [
        'schema_version', 'entry', 'scope_execution', 'automatic_scope_available', 'scope_count',
        'scopes', 'boundaries', 'source_edits', 'sources', 'runtime', 'automatic_estimate_available',
        'capture_facts_sha256', 'numerical_sources', 'numerical_decisions', 'build_sources', 'artifacts_sha256']}
    selected['native_effects'] = {key: manifest['native_effects'][key] for key in [
        'schema_version', 'entry', 'complete', 'closure_complete_available', 'closure_materialization',
        'summarized_operations', 'procedures', 'materialized_rejections']}
    write('manifest-view.json', {'is_projection': True,
        'full_manifest': {'original_path': str(manifest_path), 'sha256': sha(manifest_path),
                          'bytes': manifest_path.stat().st_size},
        'source_inputs': {'count': len(manifest['source_inputs']), 'canonical_json_sha256': canonical(manifest['source_inputs'])},
        'analysis_sources_canonical_json_sha256': canonical(manifest['analysis_sources']),
        'canonical_hash_encoding': 'UTF-8 json.dumps(sort_keys=True,separators=(comma,colon))',
        'selected_public_fields': selected,
        'omitted': 'Large local structured-source diagnostics and repeated source maps; full raw manifest remains hash-bound.'})
    referenced = {}
    for name in ['launch.json', 'generated/scope-manifest.json', 'generated/scope-transformation.json',
                 'generated/scope-compiler.log', 'compiler.tar', 'pipeline.tar']:
        path = RUN / name
        referenced[name] = {'original_path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}
    write('evidence.json', {'schema_version': 1,
        'claim': 'Authenticated unchanged-native fallback selected before expensive numerical discovery.',
        'status': worker['status'], 'commits': COMMITS, 'scope_count': 0,
        'elapsed_seconds': receipt['elapsed_seconds'], 'cap_seconds': receipt['cap_seconds'],
        'virtual_memory_bytes': receipt['virtual_memory_bytes'], 'cpu_affinity': '4,5,10,11',
        'authenticated_input_files': 178, 'source_input_files': 69,
        'input_inventory_equal_to_prior_checkpoint': True, 'changed_inputs': [],
        'source_edits': 0, 'build_units': 0, 'runtime': None, 'native_build_contract_empty': True,
        'application_built': False, 'application_executed': False, 'performance_measured': False,
        'automatic_offloading_qualified': False, 'numerical_window_admitted': False,
        'cost_estimates_available': False, 'missing_cost_reasons': worker['missing_cost_reasons'],
        'comparison_limit': 'Elapsed time belongs to a native-selection diagnostic. It is not application speedup or faster completion of whole-Convection GPU admission.',
        'manifest_view': 'manifest-view.json', 'copied_artifacts': copied,
        'hash_bound_external_artifacts': referenced, 'commit_verification': 'commit-verification.json'})
    artifacts = {str(path.relative_to(OUTPUT)): {'sha256': sha(path), 'bytes': path.stat().st_size}
                 for path in sorted(OUTPUT.rglob('*')) if path.is_file() and path.name != 'artifact-inventory.json'}
    inventory = {'schema_version': 1, 'files': artifacts}
    inventory['identity'] = canonical(inventory)
    write('artifact-inventory.json', inventory)
    print(json.dumps({'files': len(artifacts), 'bytes': sum(row['bytes'] for row in artifacts.values()),
                      'inventory_identity': inventory['identity'], 'output': str(OUTPUT)}, sort_keys=True))


if __name__ == '__main__':
    main()
