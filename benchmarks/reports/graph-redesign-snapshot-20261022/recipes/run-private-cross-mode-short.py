"""Held private2/10 deterministic-TOP cross-mode field guards. Never launches natural runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import build_private as build

ROOT = Path(__file__).resolve().parent


def reference_bindings(plan):
    result = {}
    for row in plan['existing_control']['references']:
        if row['stage'] != 'prefix':
            continue
        directory = Path(row['directory'])
        for name, expected in row['field_files_sha256'].items():
            result[str(directory / name)] = expected
        for name, expected in row['vtk_sha256'].items():
            result[str(directory / 'output' / name)] = expected
        result[str(directory / 'run.log')] = row['run_log_sha256']
        result[str(directory / 'plans.jsonl')] = row['plan_log_sha256']
    return result


def assert_originals(plan, protected, public_inputs=None):
    build.verify_hashes(protected)
    current = build.public_original_guard(plan)
    if public_inputs is not None and current != public_inputs:
        raise RuntimeError('original full public closure differs from preflight')
    return current


def attest_plans(path, expected):
    observed = []
    keys = ('function', 'precision', 'original_thread_init_observed', 'threads', 'original_flags',
            'rank', 'dimensions', 'sign', 'kinds', 'plan_sha256')
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get('schema_version') != 1 or row.get('event') != 'plan' or not row.get('original_thread_init_observed'):
            raise RuntimeError('unattested FFTW plan record')
        if row['applied_flags'] != row['original_flags'] | 2097152:
            raise RuntimeError('FFTW plan is not originalflags plus WISDOM_ONLY')
        row['plan_sha256'] = hashlib.sha256(row['plan'].encode()).hexdigest()
        observed.append({key: row[key] for key in keys})
    if observed != expected:
        raise RuntimeError('four exact native-control FFTW plan identities changed')
    return observed


def compare_run(directory, references, policy, fields, compare_fields, read_outputs, read_vtk, compare_outputs, compare_vtk, np):
    candidate = fields(directory, {**policy, 'field_count': 20})
    if len(candidate) != 20:
        raise RuntimeError('missing full20 field observer output')
    comparisons = []
    for reference in references:
        reference_dir = Path(reference['directory'])
        original = fields(reference_dir, {**policy, 'field_count': 20})
        comparison = compare_fields(candidate, original, policy)
        for name, (values, layout) in candidate.items():
            expected, _ = original[name]
            if values.shape == expected.shape:
                different = np.flatnonzero(values.view(np.uint64) != expected.view(np.uint64))
                comparison[name]['bitwise_different_values'] = int(different.size)
                comparison[name]['first_bitwise_different_flat_indices'] = different[:8].tolist()
                comparison[name]['first_bitwise_different_fortran_indices'] = [
                    [int(index) + lower for index, lower in zip(
                        np.unravel_index(int(flat), layout['shape'], order='F'), layout['lower_bounds'], strict=True)]
                    for flat in different[:8]]
        vtk = compare_outputs(read_outputs(reference_dir / 'output', read_vtk),
                              read_outputs(directory / 'output', read_vtk), compare_vtk)
        comparisons.append({'reference': reference['name'], 'directory': str(reference_dir),
                            'strict_full20_passed': all(row['passed'] for row in comparison.values()),
                            'fields': comparison, 'vtk_validation': vtk,
                            'cross_backend_bitwise_equality_required': False})
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('TEXT ONLY: no validator/build/application executed. Natural2458 remains separately held.')
        return
    start = time.monotonic()
    plan, stage = build.checked_plan()
    receipt_path = ROOT / 'build-receipt.json'
    receipt = build.read(receipt_path)
    if (receipt.get('status') != 'built_private_diagnostic_only' or receipt.get('plan_sha256') != build.PLAN_SHA256
            or receipt.get('script_stage_sha256') != build.digest(build.STAGE)
            or not receipt.get('original_inputs_before_after_verified')
            or not receipt.get('new_pressure_modules_identical')
            or receipt.get('actual_compile_count') != 2 or receipt.get('actual_diagnostic_link_count') != 2):
        raise RuntimeError('required reviewed private two-compile/two-link build has not passed')
    output = ROOT / 'short-01'
    if output.exists():
        raise RuntimeError('preserve earlier short-run evidence; recovery requires review/new output')
    output.mkdir()
    report_path = output / 'report.json'
    report = {'schema_version': 1, 'kind': 'private_deterministic_TOP_prefix_cross_mode',
              'status': 'preflight_running', 'official_qualification': False, 'performance_sample': False,
              'plan_sha256': build.PLAN_SHA256, 'script_stage_sha256': build.digest(build.STAGE),
              'driver_sha256': build.digest(__file__), 'build_receipt_sha256': build.digest(receipt_path),
              'field_policy': plan['validation']['field_policy'], 'runs': [],
              'natural_execution_authorized': False, 'natural_execution_count': 0}
    build.write(report_path, report)
    protected = {}
    try:
        protected = build.protected_inputs(plan, stage)
        build.add_bindings(protected, receipt['protected_inputs_sha256'])
        build.add_bindings(protected, receipt['outputs_sha256'])
        build.add_bindings(protected, reference_bindings(plan))
        protected[str(receipt_path)] = build.digest(receipt_path)
        build.verify_hashes(protected)
        public_inputs = assert_originals(plan, protected)
        build.add_bindings(protected, public_inputs)
        report['original_public_guard_before'] = {'passed': True, 'authenticated_files': len(public_inputs),
                                                  'scope': 'original prepared artifacts; altered binaries never retagged'}
        sys.path[:0] = [str(build.REPO / 'containers'), str(build.REPO / 'elmm-pipeline')]
        import numpy as np
        from elmm_pipeline.campaigns.halos import compare_fields, fields
        from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log
        from probe_prepared_e2e import probe_environment
        from sgs_experiment import compare_outputs, read_outputs

        prefix_plan = build.read('/tmp/fortran-native-top-reference-serial-01/prefix-01/plan.json')
        expected_affinity = plan['application_environment']['cpu_affinity']
        if sorted(os.sched_getaffinity(0)) != expected_affinity:
            raise RuntimeError('current process affinity differs from reviewed0..11 control')
        launch_records = []
        for steps in [2, 10]:
            references = [row for row in plan['existing_control']['references']
                          if row['stage'] == 'prefix' and row['steps'] == steps]
            if len(references) != 2:
                raise RuntimeError('both native fixedTOP references are required')
            first = references[0]
            config_source = Path(first['directory']).parent / 'config'
            expected_configs = prefix_plan['configuration_sha256'][first['name']]
            for mode in ['be58', 'candidate20']:
                directory = output / (mode + '-steps-' + str(steps).zfill(2))
                directory.mkdir()
                config = directory / 'config'
                shutil.copytree(config_source, config)
                actual_configs = {str(path.relative_to(config)): build.digest(path)
                                  for path in config.rglob('*') if path.is_file()}
                if actual_configs != expected_configs:
                    raise RuntimeError('copied prefix configuration differs from original fixedTOP reference')
                run_dir = directory / 'run'
                shutil.copytree(config, run_dir)
                if (run_dir / 'output').exists():
                    raise RuntimeError('fresh run dir unexpectedly includes copied output')
                env = probe_environment(4)
                for name in list(env):
                    if name.startswith('ELMM_'):
                        env.pop(name)
                for name, value in plan['application_environment']['required_overlay'].items():
                    if value is None:
                        env.pop(name, None)
                    else:
                        env[name] = value
                env.pop('CUDA_VISIBLE_DEVICES', None)
                plan_log = run_dir / 'plans.jsonl'
                env['FORT_FFTW_CONTROL_LOG'] = str(plan_log)
                binary = ROOT / 'build' / mode / 'ELMM-fields'
                if receipt['outputs_sha256'].get(str(binary)) != build.digest(binary):
                    raise RuntimeError('new private diagnostic executable changed')
                runtime_args = prefix_plan['configurations'][first['name']]['runtime_arguments']
                argv = [str(binary), *runtime_args]
                stdout = run_dir / 'stdout.raw'
                stderr = run_dir / 'stderr.raw'
                launch = {'mode': mode, 'steps': steps, 'argv': argv, 'cwd': str(run_dir),
                          'environment_sha256': build.value_digest(env),
                          'environment': {name: env.get(name) for name in [*plan['application_environment']['required_overlay'], 'FORT_FFTW_CONTROL_LOG', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH']},
                          'cpu_affinity': sorted(os.sched_getaffinity(0)), 'binary_sha256': build.digest(binary),
                          'configuration_sha256': actual_configs,
                          'timeout_seconds': build.remaining(start, 240), 'returncode': None,
                          'stdout': str(stdout), 'stderr': str(stderr), 'performance_sample': False}
                launch_records.append(launch)
                report['runs'] = launch_records
                report['status'] = 'short_runs_active'
                build.write(report_path, report)
                build.verify_hashes(protected)
                try:
                    with stdout.open('xb') as out, stderr.open('xb') as err:
                        child = subprocess.run(argv, cwd=run_dir, env=env, stdout=out, stderr=err,
                                               timeout=launch['timeout_seconds'], check=False)
                    launch['returncode'] = child.returncode
                finally:
                    for label, path in [('stdout', stdout), ('stderr', stderr)]:
                        if path.exists():
                            launch[label + '_sha256'] = build.digest(path)
                    build.write(report_path, report)
                if launch['returncode'] != 0:
                    raise RuntimeError('private short application failed; preserve raw output and stop')
                text = stdout.read_text(errors='strict') + '\n' + stderr.read_text(errors='strict')
                launch['validation_text_sha256'] = hashlib.sha256(text.encode()).hexdigest()
                # Raw streams remain intact; the derived log is concatenated only after process completion.
                (run_dir / 'run.log').write_text(text)
                launch['timing'] = validate_log(text, steps, expected_completed_steps=steps)
                launch['plans'] = attest_plans(plan_log, plan['application_environment']['expected_exact_plan_records'])
                launch['plan_log_sha256'] = build.digest(plan_log)
                launch['comparisons'] = compare_run(run_dir, references, report['field_policy'], fields,
                                                     compare_fields, read_outputs, read_vtk, compare_outputs, compare_vtk, np)
                launch['field_files_sha256'] = {path.name: build.digest(path) for path in run_dir.glob('fields-*')}
                launch['vtk_sha256'] = {path.name: build.digest(path) for path in (run_dir / 'output').glob('*.vtk')}
                launch['strict_passed'] = all(row['strict_full20_passed'] and row['vtk_validation']['passed'] for row in launch['comparisons'])
                build.write(report_path, report)
                build.verify_hashes(protected)
                if not launch['strict_passed']:
                    raise RuntimeError('private prefix numerical guard failed; preserve all comparisons and keep natural runs held')
        after = assert_originals(plan, protected, public_inputs)
        build.remaining(start, 240)
        report.update(status='completed_private_short_diagnostic', all_four_runs_strict20_VTK_and_four_plans_passed=True,
                      original_public_guard_after={'passed': True, 'authenticated_files': len(after)},
                      original_and_reference_inputs_before_after_verified=True)
    except BaseException as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        if protected:
            try:
                build.verify_hashes(protected)
                report['protected_inputs_unchanged_after_failure'] = True
            except BaseException as guard_error:
                report['post_failure_guard_error'] = {'type': type(guard_error).__name__, 'message': str(guard_error)}
        raise
    finally:
        report['nonperformance_phase_elapsed_seconds'] = time.monotonic() - start
        build.write(report_path, report)
    print('PRIVATE SHORT GUARDS CLOSED', report_path, build.digest(report_path))


if __name__ == '__main__':
    main()
