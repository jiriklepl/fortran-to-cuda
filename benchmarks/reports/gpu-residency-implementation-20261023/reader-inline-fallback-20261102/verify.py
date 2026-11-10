"""Verify saved reader evidence with stdlib only; run no compiler or application."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import tarfile
import xml.etree.ElementTree as ET

D = Path(__file__).resolve().parent
R = '/tmp/fortran-inline-policy-trace-01/'
S = '/tmp/fortran-startup-application-short-06/'
B = '/tmp/fortran-startup-application-build-06/'
H = '/tmp/fortran-startup-application-short-07/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', action='store_true', help='also hash omitted frozen paths; includes large manifests and binaries')
    args = parser.parse_args()
    index = json.loads((D / 'manifest.json').read_text())
    assert file_sha(D / index['archive']['name']) == index['archive']['sha256']
    objects = {}
    with tarfile.open(D / index['archive']['name'], 'r:gz') as archive:
        for item in archive:
            assert item.isfile() and item.mode == 0o444
            assert re.fullmatch(r'objects/[0-9a-f]{64}', item.name)
            data = archive.extractfile(item).read()
            assert sha(data) == item.name[8:] and item.name[8:] not in objects
            objects[item.name[8:]] = data
    assert len(objects) == index['object_count']
    def raw(path):
        row = index['paths'][path]
        value = objects[row['sha256']]
        assert len(value) == row['bytes']
        return value
    def read(path):
        return json.loads(raw(path))
    for path in index['paths']:
        assert '.git' not in Path(path).parts
        raw(path)
    if args.external:
        for path, row in index['external_fingerprints'].items():
            assert file_sha(path) == row['sha256'], path
            if row['bytes'] is not None:
                assert Path(path).stat().st_size == row['bytes'], path
    q = read(R + 'qualification.json')
    assert sha(raw(R + 'qualification.json')) == 'feea9f3a2f7cd7518b9688953b660514da83179152ec27d4df188668580ac377'
    assert q['status'] == 'PASS_SOURCE_ONLY_AUTHENTICATED_INLINE_FALLBACK_READER'
    assert q['tests'] == {'whole_modules': 6, 'passed': 269, 'new_cases': 40, 'failed': 0,
                          'errors': 0, 'skipped': 0, 'subtests_passed': 309}
    assert q['all_input_guards_held'] and q['owned_groups_absent'] and q['guard_count'] == 205
    assert q['applications_executed'] == q['GPU_queries'] == q['source_queries'] == 0
    for row in q['references']:
        assert sha(raw(row['path'])) == row['sha256']
    owned = read(R + 'owned-final.json')
    for row in owned['files']:
        assert sha(raw(owned['postimages_root'] + '/' + row['path'])) == row['postimage_sha256']
        if row['preimage_sha256'] is not None:
            assert sha(raw(R + 'preimage/gpu_validation.py')) == row['preimage_sha256']
    for attempt, status in [('cpu-01', 'FAIL'), ('cpu-02', 'PASS')]:
        terminal = read(R + attempt + '/completion.json')
        assert terminal['status'] == status and terminal['all_inputs_exact'] and not terminal['changed_inputs']
        assert all(row['absent'] for row in terminal['owned_groups'])
        cases = ET.fromstring(raw(R + attempt + '/results.xml')).findall('.//testcase')
        assert len(cases) == 269 and all(case.find('failure') is None and case.find('error') is None
                                         and case.find('skipped') is None for case in cases)
        assert b'269 passed, 309 subtests passed' in raw(R + attempt + '/pytest.stdout')
    lint = read(R + 'cpu-02/ruff-comparison.json')
    assert lint['status'] == 'PASS_NO_NEW_RUFF_DIAGNOSTICS'
    assert [row['code'] for row in lint['candidate']] == ['TRY004']
    assert any(row['code'] == 'TRY004' for row in lint['preimage'])
    assert b'I001' in raw(R + 'cpu-01/ruff.stdout') and b'TRY004' in raw(R + 'cpu-01/ruff.stdout')
    saved = read(R + 'cpu-02/saved-log.json')
    log = raw(saved['log']['path']).decode()
    assert sha(log.encode()) == saved['log']['sha256']
    decisions = saved['offload_decisions']
    rows = decisions['authenticated_inline_fallbacks']
    assert len(rows) == 139 and saved['allocations'] == saved['kernel_launches'] == 0
    assert saved['verified_gpu_execution'] is False and decisions['gpu_units'] == 0
    lines = log.splitlines()
    for row in rows:
        number = row['log_line']
        assert lines[number-1] == 'FORT_OFFLOAD entry=region mode=native-scoped-resource gpu_units=0 cpu_units=1 unit_kind=regions'
        assert lines[number-2].startswith('FORT_SCOPED position ')
        fields = dict(token.split('=', 1) for token in lines[number-2].split()[2:])
        assert fields['provenance_version'] == fields['domain_version'] == '1'
        assert fields['context'] == str(row['context']) and fields['domain'] == str(row['domain'])
        assert fields['domain_role'] == 'execution'
        assert all(fields[key] == row[key] for key in ('owner', 'procedure', 'segment', 'implementation'))
        assert row['raw_entry'] == 'region' and row['entry'].endswith('::region')
        assert row['gpu_execution_confirmed'] is False and row['gpu_units'] == 0 and row['cpu_units'] == 1
    original = read(S + 'qualification.json')
    assert original['status'] == 'FAIL' and original['all_inputs_exact'] and len(original['cases']) == 2
    assert all(row['process_group_absent'] for row in original['cases'])
    for mode in ('traced_gpu', 'hidden_gpu'):
        report = read(S + 'channel-simple-' + mode + '/report.json')
        assert report['status'] == ('PASS' if mode == 'traced_gpu' else 'FAIL')
        assert report['all_inputs_exact'] and report['timing']['completed_steps'] == report['steps'] == 2
        assert len(report['fields']) == 20 and all(row['passed'] for row in report['fields'].values())
        assert report['outputs']['passed'] and len(report['outputs']['files']) == 6
        assert all(item['passed'] for role in report['comparisons'].values() for item in role['fields'].values())
        if mode == 'hidden_gpu':
            assert report['comparisons']['native']['all_field_bytes_equal'] is True
            assert all(row['all_bytes_equal'] for row in report['comparisons']['native']['outputs']['files'].values())
        else:
            assert report['comparisons']['baseline']['all_field_bytes_equal'] is True
        streams = read(S + 'channel-simple-' + mode + '/run.log.streams.json')
        assert streams['application_returncode'] == 0 and streams['process_exited_before_merge']
    offline = read(H + 'recovery-completion.json')
    assert offline['status'] == 'FAIL' and offline['applications'] == 0 and offline['GPU_execution'] is False
    assert b'AssertionError' in raw(H + 'recovery.stderr')
    build = read(B + 'completion.json')
    assert build['status'] == 'PASS' and build['all_inputs_exact'] and not build['GPU_or_application_execution']
    assert read(B + 'composition-result.json')['source_count'] == 69
    assert read(B + 'full20.json')['sha256'] == '5476fe2c5506de17454dec47f2b2c1e12ce7a17dd5a6b3e08ef546872d9c3fba'
    print(json.dumps({'status': 'PASS', 'embedded_paths': len(index['paths']),
        'external_fingerprints': len(index['external_fingerprints']), 'external_checked': args.external,
        'reader_cases': 269, 'subtests': 309, 'saved_native_fallbacks': 139,
        'saved_GPU_allocations': 0, 'saved_GPU_launches': 0, 'short06_gate': 'FAIL preserved',
        'short07_offline_harness': 'FAIL preserved; zero apps', 'promotion_or_full_application_claim': False}, indent=2))


if __name__ == '__main__':
    main()
