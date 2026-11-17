"""Reached mixed teams preserve original synchronization and native entries."""

import os
import re
import shutil
import subprocess
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT, assert_uncalibrated_native

PRODUCER = '''!$omp do
do i=-2,n-3
b(i)=2*a(i)+real(i,8)
enddo
!$omp end do
'''
CORRECTION = '''!$omp do
do i=-2,n-3
b(i)=b(i)+sum(a)
enddo
!$omp end do
'''
CONSUMER = '''!$omp do
do i=-2,n-3
out(i)=a(i)+b(i)
enddo
!$omp end do
'''
OPENING = '!$omp parallel private(i) shared(a,b,out,n,flag) default(none)\n'
TEAM = OPENING + PRODUCER + CORRECTION + CONSUMER + '!$omp end parallel\n'
SOURCE = '''module joined_owner
implicit none
integer::visits=0
contains
subroutine step(a,b,out,n,flag)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:),out(-2:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
visits=visits+1
''' + TEAM + '''visits=visits+11
end subroutine
end module
'''


def emit(directory, source=SOURCE, *, configured_include=None, include_marker=False, policy='sections',
         scope_execution='reached'):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'owner.f90'
    path.write_text(source)
    hashes = {str(path): sha256(path.read_bytes()).hexdigest()}
    configured = None
    if configured_include is not None:
        include = directory/'correction.inc'
        include.write_text(configured_include)
        prepared = directory/'configured.f90'
        lines, mapping = [], []
        for number, line in enumerate(source.splitlines(), 1):
            if line == '#include "correction.inc"':
                if include_marker:
                    lines.append('')
                    mapping.append(number)
                included = configured_include.splitlines()
                lines.extend(included)
                mapping.extend([None] * len(included))
            else:
                lines.append(line)
                mapping.append(number)
        prepared.write_text('\n'.join(lines)+'\n')
        configured = {'schema_version': 1, 'source_inputs': hashes,
                      'preserves_source_order': True, 'configuration': {'defines': []},
                      'dependencies': {str(include): sha256(include.read_bytes()).hexdigest()},
                      'entries': [{'source': str(path), 'path': str(prepared),
                                   'sha256': sha256(prepared.read_bytes()).hexdigest(), 'line_map': mapping}]}
    facts = {'schema_version': 1, 'participation': 'serial', 'sources': hashes,
             'captures': {'argument::'+name: FACT for name in ('a', 'b', 'out')}}
    builder = ScopeBuilder([path], 'joined_owner::step', facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy=policy, scope_execution=scope_execution, host_threads=4),
                           analysis_sources=configured)
    outputs, report = builder.run()
    replacement = report['sources'].get(str(path), {}).get('replacement')
    text = outputs[replacement] if replacement else source
    return path, outputs, report, text


def teams(report):
    """Use public owner/segment data, including original-body companions."""
    records = []
    for owner in report['scopes']:
        for coordinator in [owner, *owner.get('module_coordinators', ()),
                            *owner.get('internal_coordinators', ())]:
            for segment in coordinator['planning_segments']:
                operations = segment['operations']
                if operations.get('joined_team'):
                    records.append((operations['joined_team'], operations))
    return records


@pytest.mark.parametrize('count', [33, 48])
def test_graph_team_keeps_all_numerical_units_across_native_work(tmp_path, count):
    increment = '!$omp do\ndo i=-2,n-3\nb(i)=b(i)+a(i)\nenddo\n!$omp end do\n'
    source = SOURCE.replace(TEAM, OPENING + PRODUCER + increment * (count - 2)
                            + CORRECTION + CONSUMER + '!$omp end parallel\n')
    path, _, report, text = emit(tmp_path, source, scope_execution='graph')
    assert path.read_text() == source
    (team, operations), = teams(report)
    assert len(team['numerical_units']) == count
    assert not report['scopes'][0]['boundaries']
    assert team['completion']['retains_original_team_and_directives']
    assert text.count('!$omp do') == text.count('!$omp end do') == count + 1
    assert text.count('b(i)=b(i)+a(i)') == count - 2
    native = native_operation_at(operations, source, 'b(i)=b(i)+sum(a)')
    assert native['completion']['native_effects_authority']
    assert native['sections']['available']


@pytest.mark.parametrize('interleaved_native', [False, True])
def test_graph_team_count_does_not_spend_per_demand_effect_budget(tmp_path, monkeypatch, interleaved_native):
    import compiler.scopes.source as source_module

    original_analysis = source_module.SourceEffects

    def bounded_analysis(*args, **kwargs):
        return original_analysis(*args, **kwargs, operations=32)

    monkeypatch.setattr(source_module, 'SourceEffects', bounded_analysis)
    count = 34
    increment = '!$omp do\ndo i=-2,n-3\nb(i)=b(i)+a(i)\nenddo\n!$omp end do\n'
    middle = ((CORRECTION + increment) if interleaved_native else increment) * (count - 2)
    source = SOURCE.replace(TEAM, OPENING + PRODUCER + middle
                            + CORRECTION + CONSUMER + '!$omp end parallel\n')
    path, _, report, text = emit(tmp_path, source, scope_execution='graph')
    (team, operations), = teams(report)
    assert path.read_text() == source
    assert len(team['numerical_units']) == count
    assert not report['scopes'][0]['boundaries']
    assert team['budgets']['planning_units'] > team['budgets']['reached_effect_limit'] == 32
    assert team['budgets']['largest_reached_native_effects'] <= 32
    assert team['budgets']['procedure_source_operations'] <= team['budgets']['source_operation_limit']
    assert team['budgets']['source_operation_limit'] > 32
    assert 'planning_unit_limit' not in team['budgets']
    native_count = count - 1 if interleaved_native else 1
    assert text.count('!$omp do') == text.count('!$omp end do') == count + native_count
    native = native_operation_at(operations, source, 'b(i)=b(i)+sum(a)')
    assert native['completion']['native_effects_authority']
    assert native['sections']['available']


def native_operation_at(operations, source, statement):
    line = source.splitlines().index(statement) + 1
    operation, = [item for item in operations['native_operations']
                  if item['first_line'] <= line <= item['last_line']]
    return operation


def write_fortran(outputs, report, build):
    build.mkdir()
    for name, contents in outputs.items():
        target = build/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(contents)
    sources = [source for source in report['build_sources'] if source['language'] == 'fortran']
    # The manifest is an inventory, not a Fortran module compile order.
    order = {'common_runtime': 0, 'shared_entry': 1, 'original_source': 2}
    return [build/source['path'] for source in sorted(sources, key=lambda source: order[source['role']])]


def module_source():
    child = SOURCE.replace('module joined_owner', 'module joined_child').replace(
        'subroutine step(', 'subroutine child_step(').replace('integer::visits=0\n', '').replace(
            'visits=visits+1\n', '').replace('visits=visits+11\n', '')
    caller = '''module joined_owner
use joined_child,only:child_step
implicit none
contains
subroutine step(a,b,out,n,flag)
real(8),intent(in)::a(-2:)
real(8),intent(inout)::b(-2:),out(-2:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
do i=-2,n-3
b(i)=a(i)+7
enddo
call child_step(a,b,out,n,flag)
do i=-2,n-3
out(i)=out(i)+5
enddo
end subroutine
end module
'''
    return child+caller
