"""Clean private application rebuild and per-object code comparison."""
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile

WORK = Path(__file__).resolve().parent
ORIGINAL = WORK / 'build-receipt-checkpoint-gfortran14'
DESTINATION = WORK / 'repaired-native-checkpoint-gfortran14'
PIPELINE = Path('/home/jirka/research/fortran-abomination/elmm-pipeline')
sys.path.insert(0, str(WORK / 'elmm-pipeline'))
from elmm_pipeline.build import build
from elmm_pipeline.build_receipts import require_build_receipts


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def normalize(argv, root):
    return [value.replace(str(root), '<snapshot>') for value in argv]


def main():
    os.sched_setaffinity(0, {4, 5, 10, 11})
    before = require_build_receipts(ORIGINAL / 'build/build-receipts.json')
    diagnostic = ORIGINAL / 'advection-intent-diagnostic'
    correction = json.loads((diagnostic / 'report.json').read_text())
    for name, sha in correction['files_sha256'].items():
        if digest(name) != sha:
            raise ValueError('Private declaration correction changed: ' + name)
    summary = json.loads((ORIGINAL / 'summary.json').read_text())
    DESTINATION.mkdir(exist_ok=False)
    subprocess.run(['cp', str(__file__), str(DESTINATION / 'reproduce.py')], check=True)
    changed_files = []
    for cached, name in [('elmm', 'ELMM'), ('PoisFFT', 'PoisFFT')]:
        revision = summary['source_commits'][name]
        archive = subprocess.check_output(['git', '-C', str(PIPELINE / '.cache/repos' / cached),
                                          'archive', '--format=tar', revision])
        target = DESTINATION / name
        target.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(target, filter='data')
        for path in (target / 'src').rglob('*'):
            if path.is_file():
                original = ORIGINAL / name / path.relative_to(target)
                if not original.is_file() or digest(path) != digest(original):
                    raise ValueError('Pristine archive differs: ' + str(path))
    source = DESTINATION / 'ELMM/src/advection.f90'
    original_sha = digest(source)
    source.write_bytes((diagnostic / 'advection.f90').read_bytes())
    changed_files.append({'path': 'ELMM/src/advection.f90', 'before_sha256': original_sha,
                          'after_sha256': digest(source), 'patch_sha256': digest(diagnostic / 'source.patch')})
    # Reconstruct semantic environment; build() prepends the native library
    # directory itself, so remove exactly that existing prefix beforehand.
    native = str(PIPELINE / '.cache/native/lib')
    for name, value in before['link']['environment'].items():
        if value is None:
            os.environ.pop(name, None)
        else:
            if name in {'LIBRARY_PATH', 'LD_LIBRARY_PATH'}:
                if value == native:
                    value = ''
                elif value.startswith(native + os.pathsep):
                    value = value[len(native) + 1:]
                else:
                    raise ValueError('Unexpected original native library environment')
            os.environ[name] = value
    binary = build(DESTINATION / 'ELMM', DESTINATION / 'PoisFFT', DESTINATION / 'build',
                   PIPELINE / '.cache/envs/build/bin/python', Path(native), 4, 4, [])
    after = require_build_receipts(DESTINATION / 'build/build-receipts.json')
    require_build_receipts(ORIGINAL / 'build/build-receipts.json')
    sections = runpy.run_path(str(diagnostic / 'reproduce.py'))['sections']
    old_units = {str(Path(row['object']).relative_to(ORIGINAL / 'build')): row for row in before['units']}
    new_units = {str(Path(row['object']).relative_to(DESTINATION / 'build')): row for row in after['units']}
    if old_units.keys() != new_units.keys() or len(old_units) != 72:
        raise ValueError('Full application object closure differs')
    report = {'schema_version': 1, 'kind': 'clean-private-source-repair-rebuild',
        'application_executed': False, 'performance_measured': False, 'qualification': False,
        'original_sources_or_binaries_modified': False, 'source_commits': summary['source_commits'],
        'original_build_identity': before['identity'], 'repaired_build_identity': after['identity'],
        'binary': str(binary), 'affinity': sorted(os.sched_getaffinity(0)), 'source_changes': changed_files,
        'excluded_sections': 'Debug/nonallocated metadata only; every allocated and nondebug relocation section compared.',
        'original_binary_sha256': digest(before['binary']), 'repaired_binary_sha256': digest(binary),
        'objects': [], 'helper_sha256': digest(diagnostic / 'reproduce.py'),
        'reproducer_sha256': digest(__file__)}
    for name in old_units:
        old, new = old_units[name], new_units[name]
        if normalize(old['execution']['argv'], ORIGINAL) != normalize(new['execution']['argv'], DESTINATION):
            raise ValueError('Per-source compiler flags differ beyond snapshot paths: ' + name)
        if old['execution']['compiler'] != new['execution']['compiler']:
            raise ValueError('Compiler identity differs: ' + name)
        a, b = sections(Path(old['object'])), sections(Path(new['object']))
        keys = a.keys() | b.keys()
        differences = {key: {'original': a.get(key), 'repaired': b.get(key)} for key in sorted(keys) if a.get(key) != b.get(key)}
        executable = {key for key in keys if (a.get(key, {}).get('flags', 0) | b.get(key, {}).get('flags', 0)) & 4}
        allocated = {key for key in keys if (a.get(key, {}).get('flags', 0) | b.get(key, {}).get('flags', 0)) & 2}
        relocations = {key for key in keys if (a.get(key, {}).get('type') in {4, 9}
                       or b.get(key, {}).get('type') in {4, 9}) and '.debug' not in key}
        report['objects'].append({'object': name, 'original_sha256': digest(old['object']),
            'repaired_sha256': digest(new['object']), 'original_source_sha256': digest(old['source']),
            'repaired_source_sha256': digest(new['source']), 'semantic_flags_equal': True,
            'compiler': old['execution']['compiler'], 'original_argv': old['execution']['argv'],
            'repaired_argv': new['execution']['argv'], 'different_sections': differences,
            'executable_sections_equal': not bool(executable & differences.keys()),
            'allocated_sections_equal': not bool(allocated & differences.keys()),
            'nondebug_relocations_equal': not bool(relocations & differences.keys())})
    report['changed_executable_objects'] = [row['object'] for row in report['objects'] if not row['executable_sections_equal']]
    report['changed_allocated_objects'] = [row['object'] for row in report['objects'] if not row['allocated_sections_equal']]
    report['changed_nondebug_relocation_objects'] = [row['object'] for row in report['objects'] if not row['nondebug_relocations_equal']]
    report['link_argv_equal_after_snapshot_path_normalization'] = normalize(before['link']['argv'], ORIGINAL) == normalize(after['link']['argv'], DESTINATION)
    write(DESTINATION / 'comparison.json', report)
    print(json.dumps({key: report[key] for key in ['original_build_identity', 'repaired_build_identity',
        'changed_executable_objects', 'changed_allocated_objects', 'changed_nondebug_relocation_objects',
        'link_argv_equal_after_snapshot_path_normalization']}, indent=2))


if __name__ == '__main__':
    main()
