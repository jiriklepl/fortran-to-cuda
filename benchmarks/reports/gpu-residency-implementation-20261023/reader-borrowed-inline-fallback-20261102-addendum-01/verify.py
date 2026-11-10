"""Check immutable saved evidence only; execute no source, tests, applications or GPU work."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import tarfile
import xml.etree.ElementTree as ET

D = Path(__file__).resolve().parent
R = '/tmp/fortran-inline-policy-trace-02/'
S = '/tmp/fortran-startup-application-short-10/'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', action='store_true', help='also hash omitted frozen paths, including large manifests/binaries')
    args = parser.parse_args()
    manifest = json.loads((D / 'manifest.json').read_text())
    assert file_sha(D / 'evidence.tar.gz') == manifest['archive']['sha256']
    objects = {}
    with tarfile.open(D / 'evidence.tar.gz', 'r:gz') as archive:
        for item in archive:
            assert item.isfile() and item.mode == 0o444
            assert re.fullmatch(r'objects/[0-9a-f]{64}', item.name)
            data = archive.extractfile(item).read()
            assert sha(data) == item.name[8:] and item.name[8:] not in objects
            objects[item.name[8:]] = data
    assert len(objects) == manifest['object_count']

    def raw(path):
        row = manifest['paths'][path]
        value = objects[row['sha256']]
        assert len(value) == row['bytes']
        return value

    def read(path):
        return json.loads(raw(path))

    def reference(row):
        table = manifest['paths'] if row['path'] in manifest['paths'] else manifest['external_fingerprints']
        assert table[row['path']]['sha256'] == row['sha256'], row['path']
        if 'bytes' in row and table[row['path']]['bytes'] is not None:
            assert table[row['path']]['bytes'] == row['bytes']

    for path in manifest['paths']:
        assert '.git' not in Path(path).parts and not raw(path).startswith(b'\x7fELF')
    if args.external:
        for path, row in manifest['external_fingerprints'].items():
            assert file_sha(path) == row['sha256'], path
            if row['bytes'] is not None:
                assert Path(path).stat().st_size == row['bytes'], path
    q = read(R + 'qualification.json')
    assert sha(raw(R + 'qualification.json')) == 'abd4e016c68587a883c7b2d8cada62f38131ffd1cf2ef07d55fcb4bfd5b850ce'
    assert q['status'] == 'PASS_SOURCE_ONLY_AUTHENTICATED_INLINE_FALLBACK_READER'
    assert q['tests']['passed'] == 293 and q['tests']['subtests_passed'] == 309 and q['tests']['whole_modules'] == 6
    assert q['all_input_guards_held'] and q['owned_groups_absent'] and q['guard_count'] == 208
    assert q['applications_executed'] == q['GPU_queries'] == q['source_queries'] == 0
    for row in q['references']:
        reference(row)
    for attempt, status, count in [('cpu-01', 'FAIL', 290), ('cpu-02', 'FAIL', 291), ('cpu-03', 'FAIL', 293), ('cpu-04', 'PASS', 293)]:
        terminal = read(R + attempt + '/completion.json')
        assert terminal['status'] == status and terminal['all_inputs_exact'] and not terminal['changed_inputs']
        assert all(row['absent'] for row in terminal['owned_groups'])
        cases = ET.fromstring(raw(R + attempt + '/results.xml')).findall('.//testcase')
        assert len(cases) == count
        assert all(case.find('failure') is None and case.find('error') is None and case.find('skipped') is None for case in cases)
    assert b'I001' in raw(R + 'cpu-03/candidate-ruff.stdout')
    lint = read(R + 'cpu-04/ruff-comparison.json')
    assert lint['status'] == 'PASS_NO_NEW_RUFF_DIAGNOSTICS'
    assert [row['code'] for row in lint['candidate']] == ['TRY004']
    assert any(row['code'] == 'TRY004' for row in lint['preimage'])
    promotion = read(R + 'promotable-owned.json')
    older = read('/tmp/fortran-inline-policy-trace-01/owned-final.json')
    final = read(R + 'owned-final.json')
    for original, prior, current in zip(promotion['files'], older['files'], final['files']):
        assert original['path'] == prior['path'] == current['path']
        assert original['preimage_sha256'] == prior['preimage_sha256']
        assert original['qualified_intermediate_sha256'] == prior['postimage_sha256'] == current['preimage_sha256']
        assert original['postimage_sha256'] == current['postimage_sha256']
        assert sha(raw(final['postimages_root'] + '/' + current['path'])) == current['postimage_sha256']
        assert sha(raw(older['postimages_root'] + '/' + prior['path'])) == current['preimage_sha256']
    saved = read(R + 'cpu-04/saved-logs.json')
    assert saved['applications_executed'] == saved['GPU_queries'] == 0 and saved['old_artifacts_modified'] is False
    for name, expected, borrowed in [('prior_simple', 139, 0), ('new_channel', 103, 30)]:
        value = saved['cases'][name]
        log = raw(value['log']['path'])
        assert sha(log) == value['log']['sha256']
        lines = log.decode().splitlines()
        rows = value['offload_decisions']['authenticated_inline_fallbacks']
        assert len(rows) == expected and value['borrowed_fallback_count'] == borrowed
        assert value['allocations'] == value['kernel_launches'] == 0 and value['verified_gpu_execution'] is False
        assert sum('borrowed_owner_route' in row for row in rows) == borrowed
        for row in rows:
            assert lines[row['log_line'] - 1] == 'FORT_OFFLOAD entry=region mode=native-scoped-resource gpu_units=0 cpu_units=1 unit_kind=regions'
            fields = dict(token.split('=', 1) for token in lines[row['log_line'] - 2].split()[2:])
            assert fields['provenance_version'] == fields['domain_version'] == '1'
            assert fields['context'] == str(row['context']) and fields['domain'] == str(row['domain'])
            assert fields['domain_role'] == 'execution'
            assert all(fields[key] == row[key] for key in ('owner', 'procedure', 'segment', 'implementation'))
            assert row['gpu_execution_confirmed'] is False and row['gpu_units'] == 0 and row['cpu_units'] == 1
            if 'borrowed_owner_route' in row:
                route = row['borrowed_owner_route']
                assert route['kind'] == 'checked borrowed helper storage epoch'
                edges = route['root_to_owner'] + route['owner_to_caller']
                assert tuple((edge['caller'], edge['callee'], edge['first_line'], edge['last_line']) for edge in edges) in {
                    (('dynamics::convection', 'momentumadvection::cd4divu', 369, 369),
                     ('momentumadvection::cd4divu', 'arrayutilities::multiply_3d', 489, 489)),
                    (('dynamics::convection', 'momentumadvection::cd4divv', 370, 370),
                     ('momentumadvection::cd4divv', 'arrayutilities::multiply_3d', 658, 658)),
                    (('dynamics::convection', 'momentumadvection::cd4divv', 370, 370),
                     ('momentumadvection::cd4divv', 'arrayutilities::set_3d', 522, 522)),
                    (('dynamics::convection', 'momentumadvection::cd4divw', 371, 371),
                     ('momentumadvection::cd4divw', 'arrayutilities::multiply_3d', 832, 832)),
                    (('dynamics::convection', 'momentumadvection::cd4divw', 371, 371),
                     ('momentumadvection::cd4divw', 'arrayutilities::set_3d', 696, 696))}
                for edge in edges:
                    assert all(re.fullmatch('[0-9a-f]{64}', edge[key]) for key in (
                        'caller_graph', 'callee_graph', 'caller_source_sha256', 'checked_call_identity', 'coordinator_identity'))
                    assert edge['handle_bindings'] and edge['resource_mappings']
    short = read(S + 'qualification.json')
    assert sha(raw(S + 'qualification.json')) == 'aff732fa0c4845825feea908499214ffdf1d23187d6bfd09c53c885e7a82fb35'
    assert short['status'] == 'PASS' and short['kind'] == 'build06_short_distinct_case_qualification'
    assert short['aggregate_recipe'] == 'short10_four_original_no_new'
    assert (short['original_applications'], short['new_applications'], short['offline_recovery_applications'], short['saved_observer_recoveries']) == (4, 0, 0, 3)
    assert short['distinct_application_cases'] == 4 and short['all_inputs_exact'] and short['all_process_groups_absent']
    assert all(short[key] == 'FAIL' for key in ('original_short06_status', 'original_short07_recovery_status', 'original_short08_status', 'original_short09_status'))
    assert short['original_reports_rewritten'] is False
    assert short['pressure_backend'] is None and short['pressure_implementation'] == 'original CPU PoisFFT'
    assert short['external_GPU_pressure_backend'] is False and short['FFTW_plans_per_application'] == 4
    for row in short['references']:
        reference(row)
    assert {(row['case'], row['mode']) for row in short['case_results']} == {
        (case, mode) for case in ('channel-simple', 'channel') for mode in ('traced_gpu', 'hidden_gpu')}
    for row in short['case_results']:
        assert row['status'] == 'PASS' and row['steps'] == 2 and row['fields'] == 20 and row['bounds_halos']
        assert row['application_execution_count'] == 1 and row['FFTW_plans'] == 4
        assert row['outputs'] == (6 if row['case'] == 'channel-simple' else 10)
        report = read(row['report']['path'])
        assert report['all_inputs_exact'] and len(report['fields']) == 20
        assert all(field['passed'] for field in report['fields'].values())
        assert report['outputs']['passed'] and len(report['outputs']['files']) == row['outputs']
        if row['saved_observer_recovery']:
            assert report['status'].startswith('PASS_SAVED_') and report['application_executions'] == 0
            assert report['raw_reports_rewritten'] is False and report['numeric_comparisons_recomputed'] is False
            assert report['original_application_returncode'] == 0 and report['original_completed_steps'] == 2
            assert read(report['original_report']['path'])['status'] == 'FAIL'
        else:
            assert report['status'] == 'PASS'
        streams = read(row['completed_streams']['path'])
        assert streams['application_returncode'] == 0 and streams['process_exited_before_merge']
    assert read(S + 'recovery-completion.json')['applications'] == 0
    assert read(S + 'recovery-completion.json')['GPU_execution'] is False
    assert read(S + 'recovery-completion.json')['status'] == 'PASS_OFFLINE_RECOVERY'
    assert read('/tmp/fortran-startup-application-build-06/completion.json')['status'] == 'PASS'
    promotion_result = read(R + 'root-live-promotion.json')
    assert promotion_result['status'] == 'PASS_EXACT_TWO_FILE_PROMOTION'
    assert promotion_result['files'] == promotion['files']
    assert promotion_result['runtime_changed'] is False and promotion_result['numerical_work_changed'] is False
    assert read(R + 'root-promotion-commit.json')['pipeline_commit'] == 'aeb07f4fbaf55809ad3ae5b4c17ca52e97bd5b72'
    live_cases = ET.fromstring(raw(R + 'root-live-reader-results.xml')).findall('.//testcase')
    assert len(live_cases) == 293 and all(case.find('failure') is None and case.find('error') is None for case in live_cases)
    for name, row in manifest['prior_packet_files'].items():
        reference({'path': row['path'], 'sha256': row['sha256']})
    print(json.dumps({'status': 'PASS', 'embedded_paths': len(manifest['paths']),
        'external_fingerprints': len(manifest['external_fingerprints']), 'external_checked': args.external,
        'reader_tests': 293, 'subtests': 309, 'saved_native_fallbacks': [139, 103], 'borrowed': 30,
        'short_original_apps': 4, 'new_apps': 0, 'saved_observer_recoveries': 3,
        'historical_failures_preserved': True, 'natural_or_performance_claim': False}, indent=2))


if __name__ == '__main__':
    main()
