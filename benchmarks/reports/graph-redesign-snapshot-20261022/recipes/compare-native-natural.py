"""Private serialized-TOP reduction natural2458 repeat2 diagnostic; never original qualification or timing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO = Path('/home/jirka/research/fortran-abomination')
sys.path[:0] = [str(REPO / 'containers'), str(REPO / 'elmm-pipeline')]
from reproduce_e2e import FIELD_POLICY, run_directory
from probe_prepared_e2e import probe_environment
from sgs_experiment import compare_outputs, digest, read, read_outputs, write
from elmm_pipeline.common import command
from elmm_pipeline.configuration import prepare_config
from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log
from elmm_pipeline.campaigns.halos import fields, compare_fields

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--execute', action='store_true')
args = parser.parse_args()
CONTROL = Path('/tmp/fortran-fftw-wisdom-diagnostic-01/executed-native-control-01')
control_path = CONTROL / 'report.json'
control = read(control_path)
original_native = control['diagnostic_native']
PRIVATE = Path('/tmp/fortran-native-top-reference-serial-01')
receipt_path = PRIVATE / 'build-receipt.json'
receipt = read(receipt_path)
prefix_path = PRIVATE / 'prefix-01/report.json'
prefix = read(prefix_path)
if prefix['status'] != 'completed_controlled_diagnostic' or not all(
        prefix['comparisons'][str(n)]['strict_passed'] and prefix['comparisons'][str(n)]['all20_fields_byte_identical']
        and prefix['comparisons'][str(n)]['vtk_validation']['passed'] for n in [2, 10]):
    raise RuntimeError('required serializedTOP2/10 exact repeatability gates changed')
if receipt['status'] != 'built_private_diagnostic_only':
    raise RuntimeError('reviewed private one-object build must close first')
binary = PRIVATE / 'build/ELMM-fields'
native = {**original_native, 'binary': str(binary), 'sha256': digest(binary)}
for path, expected in receipt['outputs_sha256'].items():
    if digest(path) != expected: raise RuntimeError('private diagnostic build output changed')
controller = control['controller']
bindings = {str(control_path): digest(control_path), str(Path(__file__)): digest(__file__),
            native['binary']: native['sha256'], native['source']: native['source_sha256'],
            native['original_build_receipt']: native['original_build_receipt_sha256']}
for name in ['source', 'library', 'compiler']:
    item = controller[name]; bindings[item['path']] = item['sha256']
bindings.update(control['runs'][0]['wisdom_sha256'])
bindings.update(receipt['outputs_sha256'])
bindings.update(read(PRIVATE / 'build-plan.json')['original_inputs_sha256'])
for path in [receipt_path, prefix_path, PRIVATE / 'build-plan.json', PRIVATE / 'build-private.py']:
    bindings[str(path)] = digest(path)
for module in list(sys.modules.values()):
    source = getattr(module, '__file__', None)
    if source and Path(source).resolve().is_relative_to(REPO):
        bindings[str(Path(source).resolve())] = digest(source)
def check():
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise RuntimeError('changed diagnostic input: ' + path)
    if sorted(os.sched_getaffinity(0)) != control['cpu_affinity']:
        raise RuntimeError('affinity differs from controlled natural diagnostic')
check()
output = args.output.resolve()
plan_path = output / 'plan.json'
if not output.exists():
    output.mkdir(parents=True)
    configurations = {}
    config_hashes = {}
    for steps in [2458]:
        for repetition in [1, 2]:
            name = f'steps-{steps:02}-repeat-{repetition}'
            destination = output / name / 'config'
            configurations[name] = prepare_config(CONTROL / 'native-verify-1/config', destination, steps=None)
            config_hashes[name] = {str(path.relative_to(destination)): digest(path) for path in destination.rglob('*') if path.is_file()}
    write(plan_path, {'schema_version': 1, 'status': 'prepared_only', 'official_qualification': False,
        'performance_sample': False, 'source_or_binary_changed': True, 'private_change': 'Only reviewed pressure TOP/BOTTOM reference reduction uses original master deterministic sum and barrier; original TOP configuration retained', 'threads': 4,
        'cpu_affinity': sorted(os.sched_getaffinity(0)), 'bound_inputs_sha256': bindings,
        'configurations': configurations, 'configuration_sha256': config_hashes,
        'actual_capture_plans': control['runs'][0]['plans'], 'per_application_timeout_seconds': 1800,
        'qualification': 'Two natural2458 private native pressure-reduction diagnostic runs reuse attested original natural-control wisdom and original TOP case, with no maxsteps override. All original objects except reviewed private pressure.o retain exact original order and bytes. Compare all20 float64 storage fields, shapes/bounds and exact bits without relaxing strict policy.'})
plan = read(plan_path)
if plan['bound_inputs_sha256'] != bindings:
    raise RuntimeError('staged plan bindings changed')
if not args.execute:
    print('PREPARED ONLY', plan_path)
    raise SystemExit()
if (output / 'report.json').exists():
    raise RuntimeError('refuse prior execution output')
environment = probe_environment(4)
for name in list(environment):
    if name.startswith('ELMM_'):
        environment.pop(name)
environment.update(LD_PRELOAD=controller['library']['path'], FORT_RUNTIME_TRACE='0', FORT_OFFLOAD_TRACE='0',
    FORT_FFTW_CONTROL_PREFIX=str(CONTROL / 'wisdom'), FORT_FFTW_CONTROL_MODE='verify')
identity_keys = ('function', 'precision', 'original_thread_init_observed', 'threads', 'original_flags',
                 'rank', 'dimensions', 'sign', 'kinds', 'plan_sha256')
def attest(path):
    observed = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row['schema_version'] != 1 or row['event'] != 'plan' or not row['original_thread_init_observed']:
            raise RuntimeError('unattested plan')
        if row['applied_flags'] != row['original_flags'] | 2097152:
            raise RuntimeError('every plan must require WISDOM_ONLY')
        row['plan_sha256'] = hashlib.sha256(row['plan'].encode()).hexdigest()
        observed.append({key: row[key] for key in identity_keys})
    if observed != plan['actual_capture_plans']:
        raise RuntimeError('exact natural-control plan identities differ')
    return observed
report = {'schema_version': 1, 'kind': 'controlled natural2458 private serializedTOP native bit-level repeatability',
    'status': 'running', 'official_qualification': False, 'performance_sample': False,
    'plan_sha256': digest(plan_path), 'field_policy': FIELD_POLICY, 'runs': [], 'comparisons': {}}
try:
    for steps in [2458]:
        runs = []
        for repetition in [1, 2]:
            check()
            name = f'steps-{steps:02}-repeat-{repetition}'
            for relative, expected in plan['configuration_sha256'][name].items():
                if digest(output / name / 'config' / relative) != expected:
                    raise RuntimeError('changed configuration')
            directory = run_directory(output / name / 'config', output / name / 'run')
            plan_log = directory / 'plans.jsonl'
            launch = {**environment, 'FORT_FFTW_CONTROL_LOG': str(plan_log)}
            argv = [native['binary'], *plan['configurations'][name]['runtime_arguments']]
            text, elapsed = command(argv, cwd=directory, env=launch, log=directory / 'run.log', timeout=1800)
            timing = validate_log(text, None, expected_completed_steps=steps)
            if not timing['natural_end']:
                raise RuntimeError('serializedTOP diagnostic did not reach originalnaturalend')
            plans = attest(plan_log)
            row = {'name': name, 'steps': steps, 'timing': timing, 'plans': plans,
                'argv': argv, 'cwd': str(directory), 'log_sha256': digest(directory / 'run.log'),
                'plan_log_sha256': digest(plan_log), 'diagnostic_wall_seconds': elapsed,
                'performance_sample': False, 'environment': {key: launch.get(key) for key in [
                    'OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_SCHEDULE', 'OMP_PROC_BIND', 'OMP_WAIT_POLICY',
                    'GOMP_SPINCOUNT', 'LD_PRELOAD', 'FORT_FFTW_CONTROL_PREFIX', 'FORT_FFTW_CONTROL_MODE']},
                'field_files_sha256': {path.name: digest(path) for path in directory.glob('fields-*')}, 'vtk_sha256': {path.name: digest(path) for path in (directory / 'output').glob('*.vtk')}}
            report['runs'].append(row); runs.append(directory)
            write(output / 'report.json', report)
            print('NATURAL TOP RUN CLOSED', name, timing['completed_steps'], flush=True)
        first = fields(runs[0], {**FIELD_POLICY, 'field_count': 20})
        second = fields(runs[1], {**FIELD_POLICY, 'field_count': 20})
        comparison = compare_fields(second, first, FIELD_POLICY)
        for name, (values, layout) in second.items():
            expected, _ = first[name]
            different = np.flatnonzero(values.view(np.uint64) != expected.view(np.uint64))
            comparison[name]['bitwise_different_values'] = int(different.size)
            comparison[name]['first_bitwise_different_flat_indices'] = different[:8].tolist()
            comparison[name]['first_bitwise_different_fortran_indices'] = [
                [int(index) + lower for index, lower in zip(np.unravel_index(int(flat), layout['shape'], order='F'), layout['lower_bounds'], strict=True)]
                for flat in different[:8]]
        vtk = compare_outputs(read_outputs(runs[0] / 'output', read_vtk), read_outputs(runs[1] / 'output', read_vtk), compare_vtk)
        report['comparisons'][str(steps)] = {'vtk_validation': vtk, 'strict_passed': all(row['passed'] for row in comparison.values()),
            'all20_fields_byte_identical': not any(row['bitwise_different_values'] for row in comparison.values()),
            'fields': comparison}
        check(); write(output / 'report.json', report)
        print('NATURAL TOP COMPARED', steps, 'different fields', [name for name, row in comparison.items() if row['bitwise_different_values']], flush=True)
    report['status'] = 'completed_controlled_diagnostic'
except BaseException as error:
    report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
    raise
finally:
    write(output / 'report.json', report)
