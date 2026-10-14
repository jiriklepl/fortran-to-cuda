"""Inspect one private declaration correction, without replacing application objects."""
from __future__ import annotations

import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
WORK = BASE.parent
sys.path.insert(0, str(WORK / 'elmm-pipeline'))
from elmm_pipeline.build_receipts import CommandRecorder, read_receipts, require_build_receipts


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def sections(path):
    data = path.read_bytes()
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('This inspection supports only ELF64 little-endian objects')
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
    section_offset, section_size, count, names_index = header[6], header[11], header[12], header[13]
    if not count or section_size != 64 or names_index >= count:
        raise ValueError('Unsupported ELF section layout')
    rows = [struct.unpack_from('<IIQQQQIIQQ', data, section_offset + i * section_size) for i in range(count)]
    names_row = rows[names_index]
    names = data[names_row[4]:names_row[4] + names_row[5]]
    result = {}
    for row in rows[1:]:
        name = names[row[0]:names.index(b'\0', row[0])].decode('ascii')
        payload = b'' if row[1] == 8 else data[row[4]:row[4] + row[5]]
        result[name] = {'type': row[1], 'flags': row[2], 'size': row[5], 'sha256': digest(payload)}
    return result


def main():
    os.sched_setaffinity(0, {4, 5, 10, 11})
    receipts = BASE / 'build/build-receipts.json'
    receipt = require_build_receipts(receipts)
    source = BASE / 'ELMM/src/advection.f90'
    unit = next(row for row in receipt['units'] if row['source'] == str(source))
    original_text = source.read_text()
    changed = original_text
    replacements = []
    for component in 'UVW':
        for form in ['div', 'adv']:
            procedure = 'CD' + component + form
            pattern = (rf'(?im)(^  subroutine {procedure}\([^\n]+\)\n'
                       rf'    real\(knd\), contiguous, )intent\(out\)( :: {component}2\(-2:,-2:,-2:\))')
            changed, count = re.subn(pattern, r'\1intent(inout)\2', changed)
            if count != 1:
                raise ValueError('Unique declaration required for ' + procedure)
            replacements.append({'procedure': procedure, 'destination': component + '2',
                                 'before': 'intent(out)', 'after': 'intent(inout)'})
    private_source = HERE / 'advection.f90'
    private_source.write_text(changed)
    (HERE / 'source.patch').write_text(''.join(difflib.unified_diff(
        original_text.splitlines(keepends=True), changed.splitlines(keepends=True),
        fromfile=str(source), tofile=str(private_source))))
    original_modules = {str(path): digest(path.read_bytes()) for path in (BASE / 'build').rglob('*.mod')}
    original = unit['execution']
    argv = list(original['argv'])
    replacements_argv = []
    for index, value in enumerate(argv):
        is_source = not value.startswith('-') and (Path(original['cwd']) / value).resolve() == source
        replacement = (str(private_source) if is_source else
                       str(HERE / 'advection.o') if value == unit['object'] else
                       '-J' + str(HERE) if value.startswith('-J') else value)
        if value != replacement:
            replacements_argv.append({'index': index, 'original': value, 'diagnostic': replacement})
        argv[index] = replacement
    if len(replacements_argv) != 3:
        raise ValueError('Expected only source, object-output and module-output substitutions')
    env = dict(os.environ)
    for name, value in original['environment'].items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value
    if not (HERE / 'advection.o').exists():
        recorder = CommandRecorder(HERE / 'command-receipts')
        with (HERE / 'build.log').open('w') as log:
            result = recorder.run(argv, cwd=original['cwd'], env=env, stdout=log, stderr=log,
                                  additional_inputs=original_modules)
        if result.returncode:
            raise RuntimeError('Private declaration diagnostic did not compile')
    records = read_receipts(HERE / 'command-receipts')
    if len(records) != 1 or records[0]['execution']['argv'] != argv:
        raise ValueError('Existing diagnostic command differs')
    require_build_receipts(receipts)
    if original_modules != {name: digest(Path(name).read_bytes()) for name in original_modules}:
        raise ValueError('Original dependency module changed')
    original_object, private_object = Path(unit['object']), HERE / 'advection.o'
    a, b = sections(original_object), sections(private_object)
    comparison = {name: {'original': a.get(name), 'diagnostic': b.get(name), 'equal': a.get(name) == b.get(name)}
                  for name in sorted(a.keys() | b.keys())}
    executable = [name for name in a if a[name]['flags'] & 4]
    allocated = [name for name in a if a[name]['flags'] & 2]
    relocations = [name for name in a if a[name]['type'] in {4, 9} and '.debug' not in name]
    textual = {}
    for name, options in [('machine-code', ['-d', '-r']), ('relocations', ['-r'])]:
        outputs = []
        for label, path in [('original', original_object), ('diagnostic', private_object)]:
            result = subprocess.run(['/usr/bin/objdump', *options, str(path)], capture_output=True,
                                    text=True, check=True)
            (HERE / f'{label}-{name}.txt').write_text(result.stdout)
            normalized = '\n'.join(line for line in result.stdout.splitlines() if 'file format' not in line)
            outputs.append(normalized)
        textual[name] = outputs[0] == outputs[1]
        (HERE / f'{name}.diff').write_text(''.join(difflib.unified_diff(
            outputs[0].splitlines(keepends=True), outputs[1].splitlines(keepends=True),
            fromfile='original', tofile='diagnostic')))
    report = {'schema_version': 1, 'kind': 'private-advection-declaration-code-comparison',
        'performance_measured': False, 'application_executed': False,
        'original_sources_or_objects_modified': False, 'original_modules_unchanged': True,
        'original_build_identity': receipt['identity'], 'affinity': sorted(os.sched_getaffinity(0)),
        'declaration_changes': replacements, 'argv_path_substitutions': replacements_argv,
        'semantic_compile_flags_changed': False, 'compile_receipts': [row['receipt'] for row in records],
        'executable_sections_equal': all(comparison[name]['equal'] for name in executable),
        'allocated_sections_equal': all(comparison[name]['equal'] for name in allocated),
        'nondebug_relocation_sections_equal': all(comparison[name]['equal'] for name in relocations),
        'executable_sections': executable, 'allocated_sections': allocated,
        'nondebug_relocation_sections': relocations, 'textual_comparisons': textual,
        'full_objects_equal': original_object.read_bytes() == private_object.read_bytes(),
        'original_object_sha256': digest(original_object.read_bytes()),
        'diagnostic_object_sha256': digest(private_object.read_bytes()), 'sections': comparison,
        'original_modules_sha256': original_modules,
        'files_sha256': {str(path): digest(path.read_bytes()) for path in
            [Path(__file__), source, private_source, HERE / 'source.patch',
             Path('/usr/bin/objdump'), Path('/usr/bin/gfortran-14'),
             BASE / 'build/momentumadvection.mod', HERE / 'momentumadvection.mod']}}
    write(HERE / 'report.json', report)
    print(json.dumps({key: report[key] for key in ['executable_sections_equal', 'allocated_sections_equal',
        'nondebug_relocation_sections_equal', 'textual_comparisons', 'full_objects_equal']}, indent=2))


if __name__ == '__main__':
    main()
