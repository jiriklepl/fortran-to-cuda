"""Explain every non-debug difference in the clean rebuilt objects."""
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent / 'build-receipt-checkpoint-gfortran14'


def elf(path):
    data = path.read_bytes()
    if data[:6] != b'\x7fELF\x02\x01':
        raise ValueError('ELF64 little endian required')
    header = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
    entries = [struct.unpack_from('<IIQQQQIIQQ', data, header[6] + i * header[11]) for i in range(header[12])]
    names_entry = entries[header[13]]
    names = data[names_entry[4]:names_entry[4] + names_entry[5]]
    rows = []
    for item in entries:
        name = names[item[0]:names.index(b'\0', item[0])].decode()
        rows.append({'name': name, 'payload': data[item[4]:item[4]+item[5]], 'link': item[6]})
    symtab = next(row for row in rows if row['name'] == '.symtab')
    strings = rows[symtab['link']]['payload']
    symbols = []
    for position in range(0, len(symtab['payload']), 24):
        name, info, other, section, value, size = struct.unpack_from('<IBBHQQ', symtab['payload'], position)
        symbols.append({'name': strings[name:strings.index(b'\0', name)].decode(), 'info': info, 'other': other,
                        'section': rows[section]['name'] if section < len(rows) else section,
                        'value': value, 'size': size})
    return {row['name']: row['payload'] for row in rows}, symbols


def normalized(value, root):
    return value.replace(str(root).encode(), b'<snapshot>')


def main():
    comparison = json.loads((HERE / 'comparison.json').read_text())
    changes = []
    for row in comparison['objects']:
        nondbg = [name for name in row['different_sections'] if '.debug' not in name]
        if not nondbg:
            continue
        if set(nondbg) != {'.rodata.str1.8', '.symtab'}:
            raise ValueError('Unexpected non-debug differences: ' + row['object'])
        old_sections, old_symbols = elf(ORIGINAL / 'build' / row['object'])
        new_sections, new_symbols = elf(HERE / 'build' / row['object'])
        name = '.rodata.str1.8'
        old_strings = [part for part in old_sections[name].split(b'\0') if part]
        new_strings = [part for part in new_sections[name].split(b'\0') if part]
        if ([normalized(part, ORIGINAL) for part in old_strings]
                != [normalized(part, HERE) for part in new_strings]):
            raise ValueError('Allocated string difference is not exclusively snapshot path spelling')
        if len(old_symbols) != len(new_symbols):
            raise ValueError('Symbol count differs')
        symbols = []
        for index, (old, new) in enumerate(zip(old_symbols, new_symbols, strict=True)):
            if old == new:
                continue
            if (old['section'] != name or {k:v for k,v in old.items() if k != 'value'}
                    != {k:v for k,v in new.items() if k != 'value'}):
                raise ValueError('Symbol difference is not an offset into the string section')
            a = old_sections[name][old['value']:].split(b'\0',1)[0]
            b = new_sections[name][new['value']:].split(b'\0',1)[0]
            if normalized(a, ORIGINAL) != normalized(b, HERE):
                raise ValueError('Changed symbol does not resolve to the same normalized string')
            symbols.append({'index': index, 'name': old['name'], 'original_offset': old['value'],
                            'repaired_offset': new['value'], 'normalized_target': normalized(a, ORIGINAL).decode()})
        changes.append({'object': row['object'], 'section': name, 'normalized_strings_equal': True,
                        'changed_symbol_offsets': symbols, 'changed_strings': [
                            {'original': a.decode(), 'repaired': b.decode()}
                            for a,b in zip(old_strings,new_strings,strict=True) if a != b]})
    report = {'schema_version': 1, 'status': 'passed', 'application_executed': False,
              'original_build_identity': comparison['original_build_identity'],
              'repaired_build_identity': comparison['repaired_build_identity'],
              'all_72_object_executable_sections_identical': not comparison['changed_executable_objects'],
              'all_72_nondebug_relocation_sections_identical': not comparison['changed_nondebug_relocation_objects'],
              'remaining_nondebug_differences': 'Only source-root spellings in allocated runtime diagnostic strings, alignment padding, and symbol offsets to those strings.',
              'objects': changes, 'comparison_sha256': hashlib.sha256((HERE/'comparison.json').read_bytes()).hexdigest(),
              'reproducer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'limitations': ['Binary bytes and linked displacements are not claimed identical.',
                              'Corrected source is an explicit application repair; this is not an unchanged-source proof.',
                              'No repaired application execution, numerical validation, or performance qualification.']}
    (HERE / 'path-data-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'objects': [row['object'] for row in changes],
                      'changed_string_symbols':sum(len(row['changed_symbol_offsets']) for row in changes)},indent=2))


if __name__ == '__main__':
    main()
