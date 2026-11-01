"""Read actual C exports for diagnostic typed forwarding; unsupported signatures fail."""
from __future__ import annotations

import re
from pathlib import Path


def exports(path):
    text = Path(path).read_text()
    masked = re.sub(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                    lambda match: ''.join('\n' if c == '\n' else ' ' for c in match[0]), text)
    result = []
    for match in re.finditer(r'\bextern\s+"C"\s+(?P<ret>[^;{}()]+?)\b(?P<name>[A-Za-z_]\w*)\s*\(', text):
        if masked[match.start():match.start() + 6] != 'extern':
            continue
        begin, depth = match.end(), 1
        end = begin
        while depth and end < len(masked):
            depth += (masked[end] == '(') - (masked[end] == ')')
            end += 1
        if depth:
            raise ValueError('unclosed C argument list')
        suffix = masked[end:].lstrip()
        if not suffix.startswith('{'):
            continue
        parameters = text[begin:end - 1]
        arguments = []
        for value in parameters.split(','):
            value = value.strip()
            if not value or value == 'void':
                continue
            if any(token in value for token in ('(', ')', '=', '...')):
                raise ValueError('unsupported diagnostic C argument: ' + value)
            argument = re.search(r'([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$', value)
            if argument is None:
                raise ValueError('unnamed diagnostic C argument')
            arguments.append(argument[1])
        result.append({'name': match['name'], 'return_type': match['ret'].strip(),
                       'parameters': parameters, 'arguments': arguments})
    return result


def wrappers(sources, output):
    records = []
    for path in sources:
        for row in exports(path):
            name = row['name']
            pure_generated = (name.endswith(('_choose', '_plan', '_configure', '_native_begin_v1', '_native_end_v1'))
                              or '_query' in name)
            if name.startswith('fort_scope_') or name.startswith('cpp_fort_') and pure_generated:
                records.append(row)
    if len({row['name'] for row in records}) != len(records):
        raise ValueError('duplicate diagnostic C ABI exports')
    text = ['// Private ABI-forwarding timers; numerical generated entry bodies remain in their source owner.',
            '#include "scoped_runtime.h"', '#include <cerrno>',
            'extern "C" void __cyg_profile_func_enter(void *, void *) noexcept;',
            'extern "C" void __cyg_profile_func_exit(void *, void *) noexcept;',
            'struct Timer { void *function; explicit Timer(void *p):function(p) {',
            '  __cyg_profile_func_enter(function,nullptr); }',
            '  ~Timer() { __cyg_profile_func_exit(function,nullptr); } };']
    for row in records:
        name, ret, parameters = row['name'], row['return_type'], row['parameters']
        text += [f'extern "C" {ret} __real_{name}({parameters});',
                 f'extern "C" {ret} __wrap_{name}({parameters}) {{',
                 f'  Timer timer(reinterpret_cast<void *>(&__wrap_{name}));']
        call = f'__real_{name}({", ".join(row["arguments"])})'
        text += [('  ' if ret == 'void' else '  return ') + call + ';', '}']
    Path(output).write_text('\n'.join(text) + '\n')
    return records
