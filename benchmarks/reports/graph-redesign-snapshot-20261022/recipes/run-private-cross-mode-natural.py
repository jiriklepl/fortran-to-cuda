"""Held private fixed-TOP cross-mode diagnostic; no official qualification or timing."""
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

ROOT = Path(__file__).resolve().parent
EXISTING = Path('/tmp/fortran-deterministic-top-cross-mode-20')
STAGE = ROOT / 'natural-stage.json'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    if temporary.exists():
        raise RuntimeError('preserve existing partial receipt: ' + str(temporary))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def bind_reference_files(references):
    result = {}
    for row in references:
        directory = Path(row['directory'])
        for name, expected in row['field_files_sha256'].items():
            result[str(directory / name)] = expected
        for name, expected in row['vtk_sha256'].items():
            result[str(directory / 'output' / name)] = expected
        result[str(directory / 'run.log')] = row['run_log_sha256']
        result[str(directory / 'plans.jsonl')] = row['plan_log_sha256']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--authorization', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print('TEXT ONLY. No validator, build, application or GPU execution. Root authorization remains required.')
        return
    if args.authorization is None:
        raise ValueError('explicit root authorization sidecar is required')
    stage = read(STAGE)
    if stage['status'] != 'sealed_private_natural_runner_held_for_root_review':
        raise RuntimeError('natural stage is not sealed')
    authorization = read(args.authorization)
    if (authorization.get('authorized_execution') != 'private_natural2458_cross_mode_only'
            or authorization.get('natural_stage_sha256') != digest(STAGE)):
        raise RuntimeError('root authorization does not match exact natural stage')
    for path, expected in stage['bound_inputs_sha256'].items():
        if digest(path) != expected:
            raise RuntimeError('changed natural stage input: ' + path)

    # Import only after source/sidecar authorization checks; existing phase files are untouched.
    sys.path.insert(0, str(EXISTING))
    import build_private as build
    import run_short as short
    plan, existing_stage = build.checked_plan()
    receipt_path = EXISTING / 'build-receipt.json'
    short_path = EXISTING / 'short-01/report.json'
    receipt = read(receipt_path)
    prefix = read(short_path)
    if (receipt.get('status') != 'built_private_diagnostic_only'
            or not receipt.get('original_inputs_before_after_verified')
            or not receipt.get('new_pressure_modules_identical')
            or receipt.get('actual_compile_count') != 2
            or receipt.get('actual_diagnostic_link_count') != 2):
        raise RuntimeError('required authentic private build did not pass')
    if (prefix.get('status') != 'completed_private_short_diagnostic'
            or not prefix.get('all_four_runs_strict20_VTK_and_four_plans_passed')
            or not prefix.get('original_and_reference_inputs_before_after_verified')
            or prefix.get('build_receipt_sha256') != digest(receipt_path)
            or prefix.get('natural_execution_count') != 0):
        raise RuntimeError('all original short guards must pass before this separate stage')
    references = [row for row in plan['existing_control']['references'] if row['stage'] == 'natural']
    if (len(references) != 2 or any(row['steps'] != 2458 for row in references)
            or references[0]['field_files_sha256'] != references[1]['field_files_sha256']):
        raise RuntimeError('both retained byte-identical fixedTOP native natural references are required')
    natural = plan['future_application_stages'][1]
    if natural['per_application_timeout_seconds'] != 1800:
        raise RuntimeError('unexpected natural application cap')
    if sorted(os.sched_getaffinity(0)) != plan['application_environment']['cpu_affinity']:
        raise RuntimeError('current process affinity differs from reviewed0..11')
    output = ROOT / 'natural-01'
    output.mkdir(exist_ok=False)
    report_path = output / 'report.json'
    started = time.monotonic()
    report = {
        'schema_version': 1,
        'kind': 'private_fixed_TOP_cross_mode_natural2458',
        'status': 'preflight_active',
        'official_qualification': False,
        'performance_sample': False,
        'altered_source_basis': True,
        'original_prepared_binaries_or_source_proofs_retagged': False,
        'natural_stage_sha256': digest(STAGE),
        'driver_sha256': digest(__file__),
        'root_authorization': {'path': str(args.authorization), 'sha256': digest(args.authorization)},
        'original_plan_sha256': build.PLAN_SHA256,
        'original_script_stage_sha256': digest(build.STAGE),
        'build_receipt_sha256': digest(receipt_path),
        'short_report_sha256': digest(short_path),
        'field_policy': plan['validation']['field_policy'],
        'concurrency_qualification': stage['concurrency_qualification'],
        'numerical_failure_policy': stage['numerical_failure_policy'],
        'runs': [],
    }
    write(report_path, report)
    protected = {}
    try:
        protected = build.protected_inputs(plan, existing_stage)
        build.add_bindings(protected, receipt['protected_inputs_sha256'])
        build.add_bindings(protected, receipt['outputs_sha256'])
        build.add_bindings(protected, bind_reference_files(references))
        build.add_bindings(protected, stage['bound_inputs_sha256'])
        protected[str(STAGE)] = digest(STAGE)
        protected[str(args.authorization)] = digest(args.authorization)
        protected[str(receipt_path)] = digest(receipt_path)
        protected[str(short_path)] = digest(short_path)
        public_inputs = short.assert_originals(plan, protected)
        build.add_bindings(protected, public_inputs)
        report['original_public_guard_before'] = {
            'passed': True, 'authenticated_files': len(public_inputs),
            'scope': 'Original prepared artifacts only; private altered binaries never retagged.'}
        sys.path[:0] = [str(build.REPO / 'containers'), str(build.REPO / 'elmm-pipeline')]
        import numpy as np
        from elmm_pipeline.campaigns.halos import compare_fields, fields
        from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log
        from probe_prepared_e2e import probe_environment
        from sgs_experiment import compare_outputs, read_outputs

        config_source = Path(natural['config_source'])
        expected_configs = natural['configuration_sha256']
        policy = report['field_policy']
        for mode in ['be58', 'candidate20']:
            directory = output / mode
            directory.mkdir()
            config = directory / 'config'
            shutil.copytree(config_source, config)
            hashes = {str(path.relative_to(config)): digest(path) for path in config.rglob('*') if path.is_file()}
            if hashes != expected_configs:
                raise RuntimeError('natural config differs from originalfixedTOP native config')
            run_dir = directory / 'run'
            shutil.copytree(config, run_dir)
            if (run_dir / 'output').exists():
                raise RuntimeError('fresh run directory includes existing physical output')
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
            binary = EXISTING / 'build' / mode / 'ELMM-fields'
            if receipt['outputs_sha256'].get(str(binary)) != digest(binary):
                raise RuntimeError('private diagnostic executable changed')
            argv = [str(binary), *natural['runtime_arguments']]
            stdout, stderr = run_dir / 'stdout.raw', run_dir / 'stderr.raw'
            row = {
                'mode': mode, 'steps': 2458, 'expected_natural_end': True,
                'argv': argv, 'cwd': str(run_dir), 'binary_sha256': digest(binary),
                'configuration_sha256': hashes, 'timeout_seconds': 1800,
                'environment': {name: env.get(name) for name in [*plan['application_environment']['required_overlay'], 'FORT_FFTW_CONTROL_LOG', 'CUDA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH', 'LIBRARY_PATH', 'OMP_WAIT_POLICY', 'GOMP_SPINCOUNT', 'FORT_SCOPE_GRAPH_QUEUE', 'FORT_SCOPE_CUDA_GRAPHS']}, 'environment_sha256': build.value_digest(env),
                'cpu_affinity': sorted(os.sched_getaffinity(0)),
                'stdout': str(stdout), 'stderr': str(stderr), 'returncode': None,
                'performance_sample': False,
            }
            report['runs'].append(row)
            report['status'] = 'natural_application_active'
            write(report_path, report)
            build.verify_hashes(protected)
            print('PRIVATE NATURAL START', mode, 'steps2458', flush=True)
            app_start = time.monotonic()
            try:
                with stdout.open('xb') as out, stderr.open('xb') as err:
                    child = subprocess.run(argv, cwd=run_dir, env=env, stdout=out, stderr=err,
                                           timeout=1800, check=False)
                row['returncode'] = child.returncode
            finally:
                row['nonperformance_elapsed_seconds'] = time.monotonic() - app_start
                for label, path in [('stdout', stdout), ('stderr', stderr)]:
                    if path.exists():
                        row[label + '_sha256'] = digest(path)
                write(report_path, report)
            if row['returncode'] != 0:
                raise RuntimeError('private natural process failed; independent next backend held')
            text = stdout.read_text(errors='strict') + '\n' + stderr.read_text(errors='strict')
            row['validation_text_sha256'] = hashlib.sha256(text.encode()).hexdigest()
            (run_dir / 'run.log').write_text(text)
            row['timing'] = validate_log(text, None, expected_completed_steps=2458)
            if not row['timing']['natural_end']:
                raise RuntimeError('natural end missing; independent next backend held')
            if 'FORT_SCOPED ' in text or 'FORT_OFFLOAD ' in text:
                raise RuntimeError('unexpected runtime tracing in TRACE0 diagnostic')
            row['plans'] = short.attest_plans(plan_log, plan['application_environment']['expected_exact_plan_records'])
            row['plan_log_sha256'] = digest(plan_log)
            row['comparisons'] = short.compare_run(run_dir, references, policy, fields, compare_fields,
                                                   read_outputs, read_vtk, compare_outputs, compare_vtk, np)
            row['field_files_sha256'] = {path.name: digest(path) for path in run_dir.glob('fields-*')}
            row['vtk_sha256'] = {path.name: digest(path) for path in (run_dir / 'output').glob('*.vtk')}
            row['execution_IO_layout_FFTW_valid'] = True
            row['strict_passed'] = all(item['strict_full20_passed'] and item['vtk_validation']['passed'] for item in row['comparisons'])
            row['ordinary_numerical_failure_continues_independent_candidate20'] = mode == 'be58' and not row['strict_passed']
            write(report_path, report)
            build.verify_hashes(protected)
            print('PRIVATE NATURAL CLOSED', mode, 'strict_passed', row['strict_passed'], flush=True)
            # A completed strict numerical FAIL is preserved and does not suppress the independent20 diagnosis.
            # Missing/malformed physical output, layout, process, proof or FFTW errors raise and hold it.
        secondary_reference = [{'name': 'private_fixed_TOP_be58_natural2458', 'directory': str(output / 'be58/run')}]
        report['secondary_candidate20_vs_be58'] = short.compare_run(
            output / 'candidate20/run', secondary_reference, policy, fields, compare_fields,
            read_outputs, read_vtk, compare_outputs, compare_vtk, np)
        after = short.assert_originals(plan, protected, public_inputs)
        report.update(
            status='completed_private_natural_cross_mode_pass' if all(row['strict_passed'] for row in report['runs']) else 'completed_private_natural_cross_mode_numerical_fail',
            original_public_guard_after={'passed': True, 'authenticated_files': len(after)},
            original_reference_and_private_inputs_before_after_verified=True,
            natural_applications_completed=2,
            original_source_basis_qualification=False,
            final_performance_qualification=False)
    except BaseException as error:
        report.update(status='held_execution_IO_layout_proof_or_FFTW_failure', error={'type': type(error).__name__, 'message': str(error)})
        if protected:
            try:
                build.verify_hashes(protected)
                report['protected_inputs_unchanged_after_failure'] = True
            except BaseException as guard_error:
                report['post_failure_guard_error'] = {'type': type(guard_error).__name__, 'message': str(guard_error)}
        raise
    finally:
        report['nonperformance_controller_elapsed_seconds'] = time.monotonic() - started
        write(report_path, report)
    print('PRIVATE NATURAL REPORT', report_path, digest(report_path), flush=True)


if __name__ == '__main__':
    main()
