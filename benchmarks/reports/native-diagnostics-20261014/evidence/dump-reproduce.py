"""Private-main complete-field diagnostic; never a performance sample."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE.parent
WORK = CHECKPOINT.parent
ORIGINAL = Path('/home/jirka/research/fortran-abomination')
PIPELINE = WORK / 'elmm-pipeline'
sys.path.insert(0, str(PIPELINE))
from elmm_pipeline.build_receipts import CommandRecorder, read_receipts, require_build_receipts
from elmm_pipeline.campaigns.driver_link import derive_driver_link
from elmm_pipeline.campaigns.halos import compare_fields, fields
from elmm_pipeline.common import clean_environment
from elmm_pipeline.configuration import prepare_config
from elmm_pipeline.validation import compare_vtk, read_vtk, validate_log

FROZEN = (ORIGINAL / 'benchmarks/results/generic-residency-30pct-20261009/milestones/'
          '20261010T132625Z-three-case-strict-correctness/evidence/campaign/campaign.json')
RECEIPT = CHECKPOINT / 'build/build-receipts.json'
REQUIRED = ['U', 'V', 'W', 'Pr', 'Viscosity', 'U2', 'V2', 'W2', 'Ustar', 'Vstar', 'Wstar']
EXTRA = ['Temperature', 'Moisture', 'Scalar', 'TDiff', 'U_avg', 'V_avg', 'W_avg', 'Pr_avg', 'Scalar_avg']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked(path, expected):
    if digest(path) != expected:
        raise ValueError('Frozen diagnostic input changed: ' + str(path))


def dump_block(specifications, imports=()):
    lines = ['  block', *['    ' + line for line in imports], '    integer :: diagnostic_unit']
    for label, expression in specifications:
        lines.extend([
            f'    if (.not.allocated({expression})) error stop "missing diagnostic field {label}"',
            f"    open(newunit=diagnostic_unit,file='fields-{label}.bin', &",
            "         access='stream',form='unformatted',status='new')",
            f'    write(diagnostic_unit) {expression}',
            '    close(diagnostic_unit)',
            f"    open(newunit=diagnostic_unit,file='fields-{label}.shape',status='new')",
            f'    write(diagnostic_unit,*) storage_size({expression}), shape({expression}), &',
            f'         lbound({expression}), ubound({expression})',
            '    close(diagnostic_unit)',
        ])
    return '\n'.join([*lines, '  end block', ''])


def prepare():
    path = HERE / 'driver.json'
    if path.exists():
        metadata = json.loads(path.read_text())
        require_build_receipts(RECEIPT)
        for name, sha in metadata['files_sha256'].items():
            checked(name, sha)
        read_receipts(HERE / 'command-receipts')
        return metadata
    receipt = require_build_receipts(RECEIPT)
    frozen = json.loads(FROZEN.read_text())
    tolerance = next(row for row in frozen['halo_drivers'] if row.get('application_case') == 'channel-simple')
    if (tolerance['dtype'] != 'float64' or tolerance['field_count'] != 11
            or tolerance['metadata_format'] != 'storage-shape-bounds-v1'):
        raise ValueError('Maintained full-field driver contract differs')
    source = CHECKPOINT / 'ELMM/src/main.f90'
    text = source.read_text()
    intermediates = REQUIRED[5:]
    imports = [f'use Dynamics, only: diagnostic_{name} => {name}' for name in intermediates]
    before = [(name, name) for name in REQUIRED[:5] + EXTRA[:4]]
    before += [(name, 'diagnostic_' + name) for name in intermediates]
    averaged = EXTRA[4:]
    first, second = '  call Dynamics_Deallocate\n', '  call DeallocateGlobals\n'
    if text.count(first) != 1 or text.count(second) != 1:
        raise ValueError('Unique original lifetime boundaries required')
    text = text.replace(first, dump_block(before, imports) + first)
    text = text.replace(second, dump_block([(name, name) for name in averaged],
        ['use OutputData, only: ' + ', '.join(averaged)]) + second)
    main = HERE / 'diagnostic-main.f90'
    main.write_text(text)
    main_unit = next(row for row in receipt['units'] if row['source'] == str(source))
    plan = derive_driver_link(receipt['binary'], fortran=main_unit['execution']['compiler']['path'],
                              source=main, object_path=HERE / 'diagnostic-main.o', output=HERE / 'ELMM-diagnostic')
    # The maintained derivation helper parses logs; actual new receipts are the
    # authority here and must agree byte-for-byte with both selected commands.
    if (plan['provenance']['original_compile_argv'] != main_unit['execution']['argv']
            or plan['provenance']['original_link_argv'] != receipt['link']['argv']):
        raise ValueError('Diagnostic relink differs from actual executed build receipts')
    write(HERE / 'relink-plan.json', plan)
    recorder = CommandRecorder(HERE / 'command-receipts')
    modules = sorted((CHECKPOINT / 'build').rglob('*.mod'))
    with (HERE / 'build.log').open('w') as log:
        for key, original in [('compile_argv', main_unit['execution']), ('link_argv', receipt['link'])]:
            environment = dict(os.environ)
            for name, value in original['environment'].items():
                if value is None:
                    environment.pop(name, None)
                else:
                    environment[name] = value
            result = recorder.run(plan[key], cwd=plan['cwd'], env=environment,
                                  stdout=log, stderr=log, additional_inputs=modules)
            if result.returncode:
                raise RuntimeError('Diagnostic main build failed; retained build.log')
    read_receipts(HERE / 'command-receipts')
    require_build_receipts(RECEIPT)
    files = [Path(__file__), main, HERE / 'ELMM-diagnostic', HERE / 'diagnostic-main.o',
             HERE / 'relink-plan.json', RECEIPT, FROZEN, PIPELINE / 'elmm_pipeline/campaigns/halos.py',
             PIPELINE / 'elmm_pipeline/campaigns/driver_link.py', PIPELINE / 'elmm_pipeline/build_receipts.py',
             WORK / 'containers/sgs_experiment.py']
    metadata = {'schema_version': 1, 'kind': 'private-main-full-field-diagnostic', 'performance_measured': False,
        'original_build_identity': receipt['identity'], 'original_binary': receipt['binary'],
        'binary': str(HERE / 'ELMM-diagnostic'), 'maintained_fields': REQUIRED,
        'additional_fields': EXTRA, 'original_numerical_sources_modified': False,
        'tolerance_provenance': {'campaign': str(FROZEN), 'driver': tolerance['name']},
        'driver': {name: tolerance[name] for name in ['dtype', 'rtol', 'atol', 'metadata_format']},
        'files_sha256': {str(item): digest(item) for item in files}}
    metadata['driver']['field_count'] = len(REQUIRED + EXTRA)
    write(path, metadata)
    return metadata


def run(case):
    metadata = prepare()
    directory = HERE / case
    directory.mkdir(exist_ok=False)
    config_source = (WORK / 'containers/cases/channel-simple' if case == 'channel-simple'
                     else CHECKPOINT / 'ELMM/examples/channel')
    configuration = prepare_config(config_source, directory / 'config')
    config_hashes = {str(path.relative_to(directory / 'config')): digest(path)
                     for path in (directory / 'config').rglob('*') if path.is_file()}
    environment = clean_environment(4)
    for name in list(environment):
        if name.startswith(('FORT_', 'OMP_', 'GOMP_', 'NVCOMPILER_', 'PGI_')):
            environment.pop(name)
    environment.update(OMP_NUM_THREADS='4', OMP_DYNAMIC='FALSE', OMP_SCHEDULE='static', OMP_PROC_BIND='false')
    os.sched_setaffinity(0, {0, 1, 2, 3})
    spec = importlib.util.spec_from_file_location('output_checks', WORK / 'containers/sgs_experiment.py')
    outputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(outputs)
    original_directory = ORIGINAL / 'benchmarks/results/e2e-native-checkpoint-20261014' / case / 'native-0'
    original_vtk = outputs.read_outputs(original_directory / 'output', read_vtk)
    report = {'schema_version': 1, 'status': 'running', 'performance_measured': False,
              'qualification': False, 'case': case, 'driver_sha256': digest(HERE / 'driver.json'),
              'configuration': configuration, 'configuration_sha256': config_hashes,
              'affinity': sorted(os.sched_getaffinity(0)), 'environment': {name: environment[name]
                  for name in ('OMP_NUM_THREADS', 'OMP_DYNAMIC', 'OMP_SCHEDULE', 'OMP_PROC_BIND')},
              'unchanged_original_vtk_reference': str(original_directory),
              'unchanged_original_vtk_sha256': {str(path): digest(path)
                  for path in (original_directory / 'output').rglob('*.vtk')}, 'runs': []}
    write(directory / 'report.json', report)
    reference = None
    try:
        for number in range(2):
            prepare()
            destination = directory / f'native-{number}'
            shutil.copytree(directory / 'config', destination)
            argv = [metadata['binary'], *configuration['runtime_arguments']]
            print('START', case, number, flush=True)
            started = time.perf_counter()
            result = subprocess.run(argv, cwd=destination, env=environment, capture_output=True,
                                    text=True, check=False, timeout=900)
            elapsed = time.perf_counter() - started
            (destination / 'stdout.log').write_text(result.stdout)
            (destination / 'stderr.log').write_text(result.stderr)
            if result.returncode:
                raise RuntimeError('Diagnostic executable failed: ' + str(destination))
            timing = validate_log(result.stdout, configuration['expected_max_steps'],
                                 expected_completed_steps=109 if case == 'channel-simple' else None)
            observed = fields(destination, metadata['driver'])
            if set(observed) != {'fields-' + name + '.bin' for name in REQUIRED + EXTRA}:
                raise ValueError('Complete-field names differ from explicit driver coverage')
            if reference is None:
                reference = observed
            comparison = compare_fields(observed, reference, metadata['driver'])
            vtk = outputs.compare_outputs(original_vtk, outputs.read_outputs(destination / 'output', read_vtk), compare_vtk)
            row = {'number': number, 'argv': argv, 'timing': timing, 'diagnostic_wall_seconds': elapsed,
                   'performance_sample': False, 'original_binary_vtk_agreement': vtk,
                   'comparisons': comparison, 'files_sha256': {str(path.relative_to(destination)): digest(path)
                       for path in destination.rglob('*') if path.is_file()},
                   'passed': vtk['passed'] and all(item['passed'] for item in comparison.values())}
            report['runs'].append(row)
            write(directory / 'report.json', report)
            print('DONE', case, number, 'steps', timing['completed_steps'], 'all_fields_pass', row['passed'], flush=True)
        prepare()
        report['status'] = 'passed' if all(row['passed'] for row in report['runs']) else 'failed'
    except BaseException as error:
        report.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        write(directory / 'report.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['channel-simple', 'channel'])
    arguments = parser.parse_args()
    if arguments.case:
        run(arguments.case)
    else:
        print(json.dumps(prepare(), indent=2))
