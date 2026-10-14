"""One capped native-fallback diagnostic; no GPU admission or application run."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tarfile
import time

ROOT = Path('/home/jirka/research/fortran-abomination')
INPUTS = ROOT / 'benchmarks/results/checked-plan-application-20261014'
OUTPUT = Path(__file__).resolve().parent
PYTHON = ROOT / '.venv/bin/python'
PRIOR_RECEIPT = INPUTS / 'automatic-generation/completion.json'
PRIOR_RECEIPT_SHA256 = 'd2c5050821cc3c46be9071b64d3178383f5261b65ee59dee6c56bc79dabe7d25'
PIPELINE_COMMIT = '8ebaa348bbaa8525435a727560d217a4a15a1bc2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    (OUTPUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def checked_inputs():
    if digest(PRIOR_RECEIPT) != PRIOR_RECEIPT_SHA256:
        raise ValueError('The preserved input-inventory receipt changed.')
    prior = json.loads(PRIOR_RECEIPT.read_text())
    expected = prior['input_sha256']
    if len(expected) != 178:
        raise ValueError('Expected the same 178 authenticated diagnostic inputs.')
    changed = [path for path, sha in expected.items() if digest(path) != sha]
    if changed:
        raise ValueError('Frozen diagnostic inputs changed: ' + repr(changed))
    request = json.loads((INPUTS / 'request.json').read_text())
    facts = json.loads(Path(request['capture_facts']).read_text())
    analysis = json.loads(Path(request['analysis_sources']).read_text())
    if facts['sources'] != analysis['source_inputs'] or request['source_inputs'] != analysis['source_inputs']:
        raise ValueError('Capture/configured source identities differ.')
    if request['options'] != {'gpu_policy': 'auto', 'host_threads': 4,
                             'scope_execution': 'reached', 'scope_transfers': 'direct'}:
        raise ValueError('Frozen diagnostic policy changed.')
    if request['numerical_sources'] is not None or request['configuration_specialization'] is not None:
        raise ValueError('Diagnostic must retain its unspecialized original request.')
    return request, facts, expected


def worker():
    resource.setrlimit(resource.RLIMIT_AS, (3 * 1024**3, 3 * 1024**3))
    sys.path.insert(0, str(OUTPUT / 'pipeline'))
    from elmm_pipeline.local_adapter import compose_scope_packages, transform_scopes
    request, facts, inputs = checked_inputs()
    started = time.monotonic()
    result = {}
    try:
        metadata = transform_scopes(Path(request['source']), OUTPUT / 'compiler', PYTHON,
            OUTPUT / 'generated', entry=request['entry'],
            source_files=[Path(path) for path in request['source_files']],
            capture_facts=facts, gpu_options=request['options'],
            analysis_sources=Path(request['analysis_sources']), summary_cache=OUTPUT / 'summary-cache')
        manifest = metadata['scope_manifest']
        assert manifest['scope_count'] == 0 and manifest['scopes'] == []
        assert manifest['sources'] == manifest['artifacts_sha256'] == {}
        assert manifest['source_edits'] == manifest['build_sources'] == []
        assert manifest['runtime'] is None and metadata['changed'] is False
        assert metadata['gpu_sources'] == metadata['extra_sources'] == []
        assert metadata['compiler_report']['outputs'] == ['scope-manifest.json']
        assert manifest['automatic_scope_available'] is manifest['automatic_estimate_available'] is False
        assert manifest['numerical_decisions'] == []
        assert manifest['native_effects']['closure_materialization'] == 'not_requested'
        assert 'reached_window_discovery' not in manifest
        reasons = [row['reason'] for row in manifest['boundaries']]
        assert any('calibration' in reason and 'native' in reason for reason in reasons)
        # The public composition/build contract must remain empty. With no edits,
        # checked original-path destinations are read-only identity checks.
        composed = compose_scope_packages([metadata], {path: Path(path) for path in manifest['source_inputs']})
        assert composed['runtime'] is None
        assert composed['patched_sources'] == composed['build_artifacts'] == []
        assert composed['gpu_sources'] == composed['extra_sources'] == []
        result = {'status': 'authenticated_unchanged_native_fallback', 'scope_count': 0,
                  'source_edits': 0, 'build_units': 0, 'runtime': None,
                  'automatic_scope_available': False, 'automatic_estimate_available': False,
                  'native_build_contract_empty': True, 'missing_cost_reasons': reasons,
                  'closure_materialization': 'not_requested',
                  'source_input_count': len(manifest['source_inputs']),
                  'manifest_sha256': digest(OUTPUT / 'generated/scope-manifest.json')}
    except BaseException as error:
        result = {'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)}
        raise
    finally:
        result.update(elapsed_seconds=time.monotonic() - started,
            changed_inputs=[path for path, sha in inputs.items() if digest(path) != sha],
            application_built=False, application_executed=False, performance_measured=False,
            automatic_offloading_qualified=False)
        write('worker-result.json', result)
        print(json.dumps(result, sort_keys=True), flush=True)


def main(compiler_commit, pipeline_commit):
    if (OUTPUT / 'launch.json').exists() or (OUTPUT / 'compiler').exists():
        raise ValueError('Checkpoint already started; preserve it and choose a new directory.')
    request, _, inputs = checked_inputs()
    identities = {}
    for name, repo, pathspec, commit in [
        ('compiler', ROOT, 'compiler', compiler_commit),
        ('pipeline', ROOT / 'elmm-pipeline', 'elmm_pipeline', pipeline_commit),
    ]:
        resolved = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', commit + '^{commit}'], text=True).strip()
        if commit != resolved:
            raise ValueError('A full explicit committed revision is required: ' + commit)
        data = subprocess.check_output(['git', '-C', str(repo), 'archive', commit, '--', pathspec])
        archive = OUTPUT / (name + '.tar')
        archive.write_bytes(data)
        destination = OUTPUT / name
        destination.mkdir()
        with tarfile.open(fileobj=io.BytesIO(data)) as stream:
            stream.extractall(destination, filter='data')
        identities[name] = {'commit': commit, 'archive_sha256': digest(archive),
            'files': {str(path.relative_to(destination)): digest(path)
                      for path in sorted(destination.rglob('*')) if path.is_file()}}
    command = ['timeout', '--kill-after=5', '300', 'taskset', '-c', '4,5,10,11',
               str(PYTHON), str(Path(__file__).resolve()), '--worker']
    record = {'schema_version': 1, 'kind': 'authenticated unchanged-native fallback diagnostic',
        'scope': 'Missing-cost selection before discovery; not a GPU-admission retry or useful-residency qualification.',
        'request_sha256': digest(INPUTS / 'request.json'),
        'input_bundle_sha256': digest(INPUTS / 'input-bundle.json'),
        'prior_input_receipt_sha256': PRIOR_RECEIPT_SHA256,
        'runner_sha256': digest(__file__), 'identities': identities, 'input_sha256': inputs,
        'command': command, 'cwd': str(ROOT), 'cap_seconds': 300, 'virtual_memory_bytes': 3 * 1024**3,
        'policy': request['options']['gpu_policy'], 'profile': None,
        'application_built': False, 'application_executed': False, 'performance_measured': False,
        'automatic_offloading_qualified': False}
    write('launch.json', record)
    started = time.monotonic()
    with (OUTPUT / 'generation.log').open('w') as log:
        completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    record.update(exit_code=completed.returncode, elapsed_seconds=time.monotonic() - started,
        log_sha256=digest(OUTPUT / 'generation.log'),
        status='timeout' if completed.returncode == 124 else 'finished' if completed.returncode == 0 else 'failed',
        manifest_emitted=(OUTPUT / 'generated/scope-manifest.json').exists(),
        changed_inputs=[path for path, sha in inputs.items() if digest(path) != sha])
    if (OUTPUT / 'worker-result.json').exists():
        record['worker_result_sha256'] = digest(OUTPUT / 'worker-result.json')
    write('completion.json', record)
    print(json.dumps({key: record[key] for key in
        ['status', 'exit_code', 'elapsed_seconds', 'manifest_emitted', 'changed_inputs']}, sort_keys=True), flush=True)
    sys.exit(completed.returncode if completed.returncode else int(bool(record['changed_inputs'])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--compiler-commit')
    parser.add_argument('--pipeline-commit', default=PIPELINE_COMMIT)
    args = parser.parse_args()
    if args.worker:
        worker()
    elif not args.compiler_commit:
        parser.error('--compiler-commit is required; launch only after the coordinating task approves the frozen pair')
    else:
        main(args.compiler_commit, args.pipeline_commit)
