"""Private deterministic-TOP two-object build. Default is text-only; natural runs never occur here."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PLAN = ROOT / 'plan.json'
PLAN_SHA256 = '840033fd059760ba0584e35a3223808c4f36fa6ecec2025e16baa9c61cccb017'
STAGE = ROOT / 'script-stage.json'
REPO = Path('/home/jirka/research/fortran-abomination')


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def value_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    if temporary.exists():
        raise RuntimeError('preserve existing partial receipt: ' + str(temporary))
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verify_hashes(bindings):
    for path, expected in bindings.items():
        if digest(path) != expected:
            raise RuntimeError('changed protected input: ' + path)


def add_bindings(target, values):
    for path, expected in values.items():
        if path in target and target[path] != expected:
            raise RuntimeError('conflicting original authority: ' + path)
        target[path] = expected


def checked_plan():
    if digest(PLAN) != PLAN_SHA256:
        raise RuntimeError('reviewed plan changed')
    stage = read(STAGE)
    if stage['plan'] != {'path': str(PLAN), 'sha256': PLAN_SHA256}:
        raise RuntimeError('script stage belongs to another plan')
    verify_hashes(stage['script_files_sha256'])
    verify_hashes(stage['validation_helpers_sha256'])
    return read(PLAN), stage


def protected_inputs(plan, stage):
    bindings = dict(plan['future_pre_and_post_execution_guards']['inputs_sha256'])
    add_bindings(bindings, stage['validation_helpers_sha256'])
    add_bindings(bindings, stage['original_modules_sha256'])
    add_bindings(bindings, stage['script_files_sha256'])
    bindings[str(PLAN)] = PLAN_SHA256
    bindings[str(STAGE)] = digest(STAGE)
    for record in plan['build_choice']['recipes'].values():
        authority = record['original_build_receipt']
        if digest(authority['path']) != authority['sha256']:
            raise RuntimeError('original build receipt changed')
        add_bindings(bindings, read(authority['path'])['files_sha256'])
    prebuild = read(plan['proof_and_profile_scope']['candidate20_prebuild_authority']['path'])
    add_bindings(bindings, prebuild['component_metadata_sha256'])
    composed = Path('/tmp/fortran-convection-graph-probe-01/composed-20')
    bindings[str(composed / 'metadata.json')] = prebuild['composition_metadata_sha256']
    prepared = read(plan['proof_and_profile_scope']['candidate20_prepared_authority']['path'])
    for generated in prepared['generation'].values():
        for package in [generated, *generated['scope_packages']]:
            manifest = package['scope_manifest_reference']
            bindings[manifest['path']] = manifest['sha256']
    return bindings


def public_original_guard(plan):
    """Validate ORIGINAL prepared artifacts only; never pass altered binaries to this validator."""
    sys.path[:0] = [str(REPO / 'containers'), str(REPO / 'elmm-pipeline')]
    import reproduce_e2e as public_runner
    authority = plan['proof_and_profile_scope']['candidate20_prepared_authority']
    if digest(authority['path']) != authority['sha256']:
        raise RuntimeError('original prepared authority changed')
    prepared = read(authority['path'])
    selected, identities = public_runner.validate_prepared_builds(
        prepared, prepared, threads=4, build_args=[])
    del selected
    gc.collect()
    verify_hashes(identities)
    return identities


def environment(overlay):
    env = dict(os.environ)
    for name, value in overlay.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    if env.get('LD_PRELOAD'):
        raise RuntimeError('unexpected compiler/linker preload; stop instead of changing recorded recipe')
    return env


def remaining(start, limit):
    seconds = 240 - (time.monotonic() - start)
    if seconds <= 0:
        raise TimeoutError('private build/short phase240 cap exhausted')
    return min(limit, seconds)


def verify_recipes(plan):
    for mode, record in plan['build_choice']['recipes'].items():
        source = Path(record['original_pressure_unit']['source'])
        if source.read_text().replace(plan['source_basis']['exact_before'], plan['source_basis']['exact_after']) != Path(record['private_source']['path']).read_text():
            raise RuntimeError('private source is not exact reviewed replacement')
        original = record['original_pressure_unit']['execution']['argv']
        expected = list(original)
        output = ROOT / 'build' / mode
        expected[expected.index('-o') + 1] = str(output / 'pressure.o')
        indices = [i for i, value in enumerate(expected) if value.startswith('-J')]
        if len(indices) != 1:
            raise RuntimeError('ambiguous original module output')
        expected[indices[0]] = '-J' + str(output / 'modules')
        indices = [i for i, value in enumerate(expected) if value.endswith('pressure.f90')]
        if len(indices) != 1:
            raise RuntimeError('ambiguous original source operand')
        expected[indices[0]] = record['private_source']['path']
        if expected != record['compile_recipe']['argv']:
            raise RuntimeError('compile delta exceeds private source/output/module directory')
        link = read(record['original_fullfield_link_receipt']['path'])
        expected = list(link['argv'])
        expected[expected.index('-o') + 1] = str(output / 'ELMM-fields')
        old_object = record['original_pressure_unit']['object']
        if expected.count(old_object) != 1:
            raise RuntimeError('original pressure object not unique')
        expected[expected.index(old_object)] = str(output / 'pressure.o')
        if expected != record['diagnostic_link_recipe']['argv']:
            raise RuntimeError('link delta exceeds one pressure object and private output')
        for key, original_record in [('compile_recipe', record['original_pressure_unit']['execution']), ('diagnostic_link_recipe', link)]:
            recipe = record[key]
            if recipe['cwd'] != original_record['cwd'] or recipe['environment_overlay'] != original_record['environment']:
                raise RuntimeError('recorded original compiler/link environment or cwd changed')


def run_command(recipe, directory, label, start):
    env = environment(recipe['environment_overlay'])
    stdout = directory / (label + '.stdout.raw')
    stderr = directory / (label + '.stderr.raw')
    record = {'argv': recipe['argv'], 'cwd': recipe['cwd'],
              'environment_sha256': value_digest(env), 'environment_overlay': recipe['environment_overlay'],
              'timeout_seconds': remaining(start, recipe['timeout_seconds']),
              'stdout': str(stdout), 'stderr': str(stderr), 'returncode': None}
    try:
        with stdout.open('xb') as out, stderr.open('xb') as err:
            child = subprocess.run(recipe['argv'], cwd=recipe['cwd'], env=env, stdout=out, stderr=err,
                                   timeout=record['timeout_seconds'], check=False)
        record['returncode'] = child.returncode
    except BaseException as error:
        record['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        for name, path in [('stdout', stdout), ('stderr', stderr)]:
            if path.exists():
                record[name + '_sha256'] = digest(path)
        write(directory / (label + '-execution.json'), record)
    if record['returncode']:
        raise RuntimeError(label + ' failed; raw output preserved')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('TEXT ONLY: explicit later parent lease is required; no build or validator executed')
        return
    start = time.monotonic()
    plan, stage = checked_plan()
    build = ROOT / 'build'
    if build.exists() or (ROOT / 'build-receipt.json').exists():
        raise RuntimeError('preserve previous build evidence; use review before any recovery')
    build.mkdir()
    receipt = {'schema_version': 1, 'kind': 'private_deterministic_TOP_two_object_build',
               'status': 'preflight_running', 'official_qualification': False, 'performance_sample': False,
               'plan_sha256': PLAN_SHA256, 'script_stage_sha256': digest(STAGE),
               'driver_sha256': digest(__file__), 'commands': [], 'outputs_sha256': {}}
    receipt_path = ROOT / 'build-receipt.json'
    write(receipt_path, receipt)
    inputs = {}
    try:
        inputs = protected_inputs(plan, stage)
        verify_hashes(inputs)
        verify_recipes(plan)
        public_inputs = public_original_guard(plan)
        add_bindings(inputs, public_inputs)
        verify_hashes(inputs)
        receipt['protected_inputs_sha256'] = inputs
        receipt['original_public_guard_before'] = {'passed': True, 'authenticated_files': len(public_inputs),
                                                   'scope': 'original prepared artifacts only; private outputs never retagged'}
        receipt['status'] = 'building_private_objects'
        write(receipt_path, receipt)
        for mode, record in plan['build_choice']['recipes'].items():
            directory = build / mode
            (directory / 'modules').mkdir(parents=True)
            for name in ['compile_recipe', 'diagnostic_link_recipe']:
                # Full generated/proof authorities are checked at phase boundaries;
                # exact command recipes restrict all mutations to fresh private outputs.
                remaining(start, 240)
                try:
                    result = run_command(record[name], directory, name, start)
                except BaseException:
                    failed_receipt = directory / (name + '-execution.json')
                    receipt['commands'].append({'mode': mode, 'recipe': name, 'failed': True,
                                                'execution_receipt': str(failed_receipt),
                                                'execution_receipt_sha256': digest(failed_receipt) if failed_receipt.exists() else None})
                    write(receipt_path, receipt)
                    raise
                receipt['commands'].append({'mode': mode, 'recipe': name, **result})
                write(receipt_path, receipt)
                if name == 'compile_recipe':
                    module = directory / 'modules/pressure.mod'
                    if digest(module) != record['new_pressure_module_gate']['expected_sha256']:
                        raise RuntimeError('new pressure module interface differs: ' + mode)
            for output in directory.rglob('*'):
                if output.is_file():
                    receipt['outputs_sha256'][str(output)] = digest(output)
        after = public_original_guard(plan)
        if after != public_inputs:
            raise RuntimeError('original public artifact closure changed during private build')
        verify_hashes(inputs)
        remaining(start, 240)
        receipt.update(status='built_private_diagnostic_only', original_public_guard_after={'passed': True, 'authenticated_files': len(after)},
                       original_inputs_before_after_verified=True, new_pressure_modules_identical=True,
                       actual_compile_count=2, actual_diagnostic_link_count=2)
    except BaseException as error:
        receipt.update(status='failed', error={'type': type(error).__name__, 'message': str(error)})
        if inputs:
            try:
                verify_hashes(inputs)
                receipt['protected_inputs_unchanged_after_failure'] = True
            except BaseException as guard_error:
                receipt['post_failure_guard_error'] = {'type': type(guard_error).__name__, 'message': str(guard_error)}
        raise
    finally:
        receipt['nonperformance_build_elapsed_seconds'] = time.monotonic() - start
        write(receipt_path, receipt)
    print('PRIVATE BUILD CLOSED', receipt_path, digest(receipt_path))


if __name__ == '__main__':
    main()
