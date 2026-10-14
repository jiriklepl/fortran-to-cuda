"""Freeze the audited source repair and regenerate actual configured inputs.

This prepares inputs only. It does not invoke compiler generation, build an
application, run a solver, or specialize runtime case configuration values.
"""
from __future__ import annotations

import difflib
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path('/home/jirka/research/fortran-abomination')
PIPELINE = ROOT / 'elmm-pipeline'
OUTPUT = Path(__file__).resolve().parent
SEED = OUTPUT / 'seed'
CACHE = PIPELINE / '.cache/repos'
OLD_REQUEST = ROOT / 'benchmarks/results/application-path-20261014/request.json'
AUDIT = ROOT / 'benchmarks/reports/native-diagnostics-20261014/evidence/repair-full-rebuild.json'
PATCH = ROOT / 'benchmarks/reports/source-session-20261014/evidence/advection-intent-diagnostic/source.patch'
FC = Path('/usr/bin/gfortran-14')
sys.path.insert(0, str(PIPELINE))

from elmm_pipeline.scope_extraction import export_analysis_sources
from elmm_pipeline.structural import Project


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def tree(root):
    return {str(path.relative_to(root)): sha(path) for path in sorted(Path(root).rglob('*')) if path.is_file()}


def mapped(path):
    path = Path(path)
    for original, target in [('elmm', 'ELMM'), ('PoisFFT', 'PoisFFT')]:
        if path.is_relative_to(CACHE / original):
            return SEED / target / path.relative_to(CACHE / original)
    raise ValueError('Original source is outside the two pinned input trees: ' + str(path))


def main():
    if SEED.exists() or (OUTPUT / 'input-bundle.json').exists():
        raise ValueError('Input checkpoint already exists; preserve it and choose a new directory.')
    started = time.monotonic()
    os.sched_setaffinity(0, {4, 5, 10, 11})
    os.environ['FC'] = str(FC)
    original_request = json.loads(OLD_REQUEST.read_text())
    audit = json.loads(AUDIT.read_text())
    repair, = audit['source_changes']
    assert not audit['changed_executable_objects'] and not audit['changed_nondebug_relocation_objects']
    assert sha(PATCH) == repair['patch_sha256']
    archive_receipts = {}
    for original, target in [('elmm', 'ELMM'), ('PoisFFT', 'PoisFFT')]:
        revision = audit['source_commits'][target]
        argv = ['git', '-C', str(CACHE / original), 'archive', '--format=tar', revision]
        archive = subprocess.check_output(argv)
        destination = SEED / target
        destination.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(destination, filter='data')
        archive_receipts[target] = {'revision': revision, 'argv': argv,
                                    'archive_sha256': hashlib.sha256(archive).hexdigest()}
    before = tree(SEED)
    for source, expected in original_request['source_inputs'].items():
        assert sha(mapped(source)) == expected, source
    source = SEED / repair['path']
    assert sha(source) == repair['before_sha256']
    original_text = source.read_text()
    lines = original_text.splitlines(keepends=True)
    declarations = []
    for procedure, destination in [('CDUdiv', 'U2'), ('CDVdiv', 'V2'), ('CDWdiv', 'W2'),
                                   ('CDUadv', 'U2'), ('CDVadv', 'V2'), ('CDWadv', 'W2')]:
        indexes = [i for i, line in enumerate(lines) if line.strip().lower().startswith('subroutine ' + procedure.lower() + '(')]
        i, = indexes
        expected = f'    real(knd), contiguous, intent(out) :: {destination}(-2:,-2:,-2:)\n'
        assert lines[i + 1] == expected
        lines[i + 1] = expected.replace('intent(out)', 'intent(inout)')
        declarations.append({'procedure': procedure, 'destination': destination, 'line': i + 2})
    source.write_text(''.join(lines))
    assert sha(source) == repair['after_sha256']
    shutil.copy2(PATCH, OUTPUT / 'audited-source.patch')
    (OUTPUT / 'applied-source.patch').write_text(''.join(difflib.unified_diff(
        original_text.splitlines(keepends=True), lines, fromfile='a/' + repair['path'], tofile='b/' + repair['path'])))
    after = tree(SEED)
    assert [name for name in before.keys() | after.keys() if before.get(name) != after.get(name)] == [repair['path']]
    write(OUTPUT / 'seed-before.json', before)
    write(OUTPUT / 'seed-after.json', after)

    cases = {}
    for name, case_source in [('channel-simple', ROOT / 'containers/cases/channel-simple'),
                              ('channel', SEED / 'ELMM/examples/channel')]:
        hashes = tree(case_source)
        destination = OUTPUT / 'cases' / name
        shutil.copytree(case_source, destination)
        assert tree(destination) == tree(case_source) == hashes
        cases[name] = {'original_path': str(case_source), 'path': str(destination), 'files': hashes,
                       'identity': identity(hashes), 'configuration_changed': False,
                       'workload_changed': False, 'compiler_branch_assumptions': []}
    source_inputs = {str(mapped(path)): sha(mapped(path)) for path in original_request['source_inputs']}
    assert len(source_inputs) == 69
    original_facts_path = Path(original_request['original_capture_facts'])
    assert sha(original_facts_path) == original_request['original_capture_sha256']
    facts = json.loads(original_facts_path.read_text())
    assert set(facts) == {'schema_version', 'participation', 'captures', 'sources'}
    assertions = {key: facts[key] for key in ['schema_version', 'participation', 'captures']}
    assert assertions == {key: original_request['capture_facts'][key] for key in assertions}
    reissued = {**assertions, 'sources': source_inputs}
    write(OUTPUT / 'capture-facts.json', reissued)
    write(OUTPUT / 'capture-facts-provenance.json', {
        'schema_version': 1, 'original_path': str(original_facts_path), 'original_sha256': sha(original_facts_path),
        'new_path': str(OUTPUT / 'capture-facts.json'), 'new_sha256': sha(OUTPUT / 'capture-facts.json'),
        'assertions_identity': identity(assertions), 'assertions_changed': False,
        'source_hashes_recomputed_from_actual_repaired_files': True,
        'basis': 'Explicitly reissued original storage, escape, allocation and entry-initialization assertions; '
                 'six INTENT corrections change no allocation or configuration and grant no new initialized sections.',
        'source_repair': repair, 'compiler_effect_or_numerical_proofs_reused': False})

    print('Exporting 69 configured sources from the repaired seed.', flush=True)
    implementation = {str(path): sha(path) for path in [
        PIPELINE / 'elmm_pipeline/scope_extraction.py', PIPELINE / 'elmm_pipeline/structural.py',
        PIPELINE / 'elmm_pipeline/gpu_calls.py']}
    project = Project(SEED / 'ELMM/src', ['DPREC'])
    analysis_path = export_analysis_sources(project, OUTPUT / 'analysis', source_files=list(source_inputs))
    analysis = json.loads(analysis_path.read_text())
    assert analysis['source_inputs'] == source_inputs and len(analysis['entries']) == 69
    assert tree(SEED) == after
    for path, expected in implementation.items():
        assert sha(path) == expected
    for path, expected in {**analysis['source_inputs'], **analysis['dependencies']}.items():
        assert sha(path) == expected
    for row in analysis['entries']:
        assert sha(row['path']) == row['sha256']
    request = {'schema_version': 1, 'entry': original_request['entry'],
               'source': str(mapped(original_request['source'])),
               'source_files': [path for path in source_inputs if path != str(mapped(original_request['source']))],
               'source_inputs': source_inputs, 'capture_facts': str(OUTPUT / 'capture-facts.json'),
               'analysis_sources': str(analysis_path), 'numerical_sources': None,
               'options': {'gpu_policy': 'auto', 'scope_transfers': 'direct', 'scope_execution': 'reached', 'host_threads': 4},
               'configurations': {name: {'path': row['path'], 'identity': row['identity']} for name, row in cases.items()},
               'configuration_specialization': None, 'immutable_inputs': {},
               'generation_executed': False, 'source_seed': str(SEED)}
    write(OUTPUT / 'request.json', request)
    result = {'schema_version': 1, 'status': 'inputs-prepared', 'compiler_generation_executed': False,
              'application_built': False, 'application_executed': False, 'performance_measured': False,
              'source_archives': archive_receipts, 'source_repair': repair, 'declarations': declarations,
              'source_seed_identity': identity(after), 'seed_files': len(after),
              'cases': cases, 'analysis_sources': str(analysis_path), 'analysis_sha256': sha(analysis_path),
              'configured_source_count': len(analysis['entries']), 'include_dependency_count': len(analysis['dependencies']),
              'capture_facts_sha256': sha(OUTPUT / 'capture-facts.json'),
              'original_request_sha256': sha(OLD_REQUEST), 'repair_audit_sha256': sha(AUDIT),
              'preprocessor': {'argv': [str(FC)], 'sha256': sha(FC),
                               'version': subprocess.check_output([str(FC), '--version'], text=True), 'defines': ['DPREC']},
              'implementation_sha256': implementation, 'reproducer_sha256': sha(__file__),
              'elapsed_seconds': time.monotonic() - started, 'affinity': sorted(os.sched_getaffinity(0)),
              'repositories': {str(path): subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
                               for path in [ROOT, PIPELINE]},
              'workflow': 'Copy the same repaired seed to each native/SGS/automatic build; source manifests refer '
                          'to this frozen original seed and must be applied with explicit checked destination mappings.'}
    write(OUTPUT / 'input-bundle.json', result)
    print(json.dumps({key: result[key] for key in ['status', 'configured_source_count', 'include_dependency_count',
                                                'source_seed_identity', 'elapsed_seconds']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
