"""Reached mixed teams preserve original synchronization and native entries."""

from hashlib import sha256
import re
import shutil
import subprocess
import os

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT


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


def emit(directory, source=SOURCE, *, configured_include=None, include_marker=False, policy='sections'):
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
                           config=OffloadConfig(policy=policy, scope_execution='reached', host_threads=4),
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


def test_mixed_team_retains_original_work_once_and_reports_separate_authority(tmp_path):
    path, _, report, text = emit(tmp_path)
    assert path.read_text() == SOURCE
    (team, operations), = teams(report)
    assert team['schema_version'] == 1
    assert team['completion']['retains_original_team_and_directives']
    assert len(team['numerical_units']) == 2
    assert all(not unit['completion']['native_effects_authority']
               for unit in team['numerical_units'])
    native = native_operation_at(operations, SOURCE, 'b(i)=b(i)+sum(a)')
    assert native['completion']['native_effects_authority']
    assert not native['completion']['gpu_legality_established']
    assert native['sections']['available']
    assert not report['scopes'][0]['boundaries']
    assert report['implementation_variants']['generated_count'] <= 4
    assert text.count('!$omp parallel') == text.count('!$omp end parallel') == 1
    assert text.count('!$omp do') == text.count('!$omp end do') == 3
    for statement in ('visits=visits+1\n', 'visits=visits+11\n',
                      'b(i)=2*a(i)+real(i,8)', 'b(i)=b(i)+sum(a)', 'out(i)=a(i)+b(i)'):
        assert text.count(statement) == 1
    assert text.count('subroutine step(a,b,out,n,flag)') == 1
    assert 'run_team' in text
    assert '!$omp master' in text and '!$omp barrier' in text
    assert 'fort_scope_host_begin' in text and 'fort_scope_host_end' in text


def test_reached_team_budgets_local_source_and_each_demand_not_total_expanded_effects(tmp_path):
    # Each branch is a separately reached native demand. Repeated reads of an
    # immutable input increase effect expansion without increasing source work
    # or granting GPU reduction execution. They must not exhaust an eager
    # whole-owner closure budget between the two useful numerical units.
    correction = CORRECTION.replace('sum(a)', '+'.join(['sum(a)'] * 6))
    branches = ('if(flag) then\n' + correction + 'endif\n') * 28
    source = SOURCE.replace(CORRECTION, branches)
    _, _, report, text = emit(tmp_path, source)
    (team, operations), = teams(report)
    assert len(team['numerical_units']) == 2
    assert not report['scopes'][0]['boundaries']
    structure = report['native_effects']['structured_effects']
    assert structure['available']
    assert structure['operation_count'] <= structure['operation_limit'] == 256
    assert operations['structured_tree']['operation_count'] > 256
    corrections = [operation for operation in operations['native_operations']
                   if operation['kind'] == 'original native worksharing'
                   and 'argument::a' in operation['resources']]
    assert len(corrections) >= 28
    assert all(operation['completion']['native_effects_authority'] for operation in corrections)
    assert text.count('if(flag) then') == 28
    assert text.count('b(i)=b(i)+' + '+'.join(['sum(a)'] * 6)) == 28
    assert text.count('!$omp parallel') == text.count('!$omp end parallel') == 1


def test_uniform_branch_stays_inside_original_team_and_before_reached_workers(tmp_path):
    branch = ('if(flag) then\n' + PRODUCER + 'else\n'
              + PRODUCER.replace('2*a(i)', '3*a(i)') + 'endif\n')
    source = SOURCE.replace(PRODUCER, branch)
    _, _, report, text = emit(tmp_path, source)
    (team, _), = teams(report)
    assert len(team['numerical_units']) == 3
    assert not report['scopes'][0]['boundaries']
    assert text.count('if(flag) then') == 1
    parallel = text.index('!$omp parallel')
    condition = text.index('if(flag) then')
    assert parallel < condition < text.index('b(i)=2*a(i)', condition)
    assert text.count('b(i)=3*a(i)+real(i,8)') == 1


@pytest.mark.parametrize('change', [
    lambda source: source.replace('!$omp end do', '!$omp end do nowait', 1),
    lambda source: source.replace(PRODUCER, 'if(i>0) then\n'+PRODUCER+'endif\n'),
])
def test_unfinished_or_nonuniform_team_never_emits_detached_gpu_subunits(tmp_path, change):
    source = change(SOURCE)
    _, _, report, text = emit(tmp_path, source)
    assert not teams(report)
    assert 'run_team' not in text
    assert text.count('b(i)=2*a(i)+real(i,8)') == 1
    assert text.count('b(i)=b(i)+sum(a)') == 1


def test_configured_native_include_retains_original_provenance_and_source(tmp_path):
    source = SOURCE.replace(CORRECTION, '#include "correction.inc"\n')
    path, _, report, text = emit(tmp_path, source, configured_include=CORRECTION)
    (team, operations), = teams(report)
    assert len(team['numerical_units']) == 2
    assert not report['scopes'][0]['boundaries']
    native = native_operation_at(operations, source, '#include "correction.inc"')
    assert native['preserves_original_source']
    assert text.count('#include "correction.inc"') == 1


def test_configured_include_mapped_blank_marker_retains_exact_original_directive(tmp_path):
    source = SOURCE.replace(CORRECTION, '#include "correction.inc"\n')
    path, _, report, text = emit(tmp_path, source, configured_include=CORRECTION, include_marker=True)
    (team, operations), = teams(report)
    assert len(team['numerical_units']) == 2
    assert text.count('#include "correction.inc"') == 1
    line = source.splitlines().index('#include "correction.inc"') + 1
    operation, = (item for item in operations['native_operations'] if item['first_line'] == line)
    assert operation['last_line'] == line
    assert operation['completion']['native_effects_authority']
    assert 'b(i)=b(i)+sum(a)' not in text
    assert text.count('!$omp parallel') == 1
    assert path.read_text() == source
    assert report['analysis_sources']['dependencies']
    # Whole-team extraction cannot edit the included statements, but reached
    # mixed execution still retains the original include inside one owner.
    rejected = [item for item in report['boundaries']
                if item['reason'].startswith('inline numerical boundary:')]
    assert rejected
    assert all(item['kind'] == 'candidate_rejection' and
               item['phase'] == 'inline_numerical_extraction' for item in rejected)
    assert not report['scopes'][0]['boundaries']


@pytest.mark.parametrize('source', [
    SOURCE,
    SOURCE.replace('integer::i\n', 'integer::i,omp_get_level,omp_get_num_threads\n'),
], ids=['ordinary_names', 'original_openmp_name_collisions'])
def test_default_none_generated_control_and_runtime_imports_compile(tmp_path, source):
    compiler = shutil.which('gfortran-15') or shutil.which('gfortran')
    if compiler is None:
        pytest.skip('native Fortran compiler unavailable')
    _, outputs, report, text = emit(tmp_path, source)
    assert teams(report)
    assert 'default(none)' in text
    assert 'fort_team_omp_level => omp_get_level' in text
    assert 'fort_team_omp_threads => omp_get_num_threads' in text
    build = tmp_path/'syntax'
    paths = write_fortran(outputs, report, build)
    result = subprocess.run([compiler, '-cpp', '-fopenmp', '-ffree-line-length-none',
                             '-fsyntax-only', *map(str, paths)], cwd=build, capture_output=True,
                            text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr


def test_every_joined_unit_completes_pending_work_before_reached_validation(tmp_path):
    _, _, report, text = emit(tmp_path)
    assert teams(report)
    # Both the numerical and original-native branches must finish before the
    # next continuation validates definitions. Waiting does not publish fields.
    suffixes = re.findall(
        r'if \(fort_team_native\) then\n(.*?)\nendif\nfort_status = fort_scope_wait\(fort_context\)',
        text, flags=re.S)
    assert len(suffixes) == 3
    assert all('fort_scope_host_end(' in suffix for suffix in suffixes)
    runs = list(re.finditer(r'fort_returned = fort_inline_team_run\(', text))
    assert len(runs) == 2
    for run in runs:
        next_reset = text.find('fort_scope_plan_reset_mode(', run.end())
        end = next_reset if next_reset >= 0 else text.index('!$omp end parallel', run.end())
        between = text[run.end():end]
        assert 'fort_status = fort_scope_wait(fort_context)' in between
        assert between.index('fort_scope_host_end(') < between.index('fort_scope_wait(')
        assert '!$omp barrier' in between


def test_successful_all_native_auto_choice_keeps_original_worksharing_and_hooks(tmp_path):
    _, _, report, text = emit(tmp_path, policy='auto')
    assert teams(report)
    selections = list(re.finditer(r'fort_team_run = fort_decision%gpu_units > 0', text))
    assert len(selections) == 2
    for selection in selections:
        native = text.index('if (fort_native_ready .and. .not. fort_team_run) then', selection.end())
        worker = text.index('if (fort_team_run) then', native)
        assert 'fort_scope_plan_validate(fort_context)' in text[native:worker]
        assert 'fort_scope_host_begin(' in text[native:worker]
        original = text.index('else\n!$omp do', worker)
        assert original > worker
        assert 'fort_scope_host_end(' in text[original:]


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


def test_original_module_child_keeps_native_abi_and_borrowed_control_guard(tmp_path):
    _, _, report, text = emit(tmp_path, module_source())
    (team, _), = teams(report)
    assert len(team['numerical_units']) == 2
    companion, = report['scopes'][0]['module_coordinators']
    assert companion['procedure'] == 'joined_child::child_step'
    assert companion['native_abi'] == 'original entry; compiler-control arguments are never referenced'
    assert text.count('subroutine child_step(a,b,out,n,flag)') == 1
    assert text.count('entry '+companion['entry']+'(') == 1
    child_text = text[:text.index('module joined_owner')]
    guard = re.search(r'logical :: (fort_borrowed_\w+)', child_text).group(1)
    assert child_text.index(guard+' = .false.') < child_text.index('entry '+companion['entry'])
    assert child_text.index(guard+' = .true.') < child_text.index('!$omp parallel')
    assert 'if ('+guard+') then' in child_text


def test_original_native_child_never_calls_compiler_runtime_or_workers(tmp_path):
    compiler = shutil.which('gfortran-15') or shutil.which('gfortran')
    cc = shutil.which('gcc')
    if compiler is None or cc is None:
        pytest.skip('native C and Fortran compilers unavailable')
    _, outputs, report, _ = emit(tmp_path, module_source())
    assert teams(report)
    build = tmp_path/'native-abi'
    paths = write_fortran(outputs, report, build)
    symbols = set()
    pattern = r'\b(?:function|subroutine)\s+(\w+)\s*\([^)]*\)\s*bind\s*\(\s*c(?:\s*,\s*name\s*=\s*[\'"]([^\'"]+)[\'"])?\s*\)'
    for path in paths:
        text = re.sub(r'&\s*\n\s*&?', '', path.read_text())
        symbols.update(name or procedure.lower() for procedure, name in re.findall(pattern, text, re.I))
    assert symbols
    sentinels = build/'sentinels.c'
    sentinels.write_text('#include <stdio.h>\n#include <stdlib.h>\n' + '\n'.join(
        f'void {name}(void) {{ fputs("unexpected native ABI call: {name}\\n", stderr); abort(); }}'
        for name in sorted(symbols)))
    result = subprocess.run([cc, '-c', str(sentinels), '-o', str(build/'sentinels.o')],
                            cwd=build, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    driver = build/'driver.f90'
    driver.write_text('''program native_caller
use joined_child,only:child_step
implicit none
real(8)::a(-2:5),b(-2:5),out(-2:5),total
integer::i,k
do i=-2,5
a(i)=real(i+3,8)*0.25d0
enddo
total=sum(a)
do k=1,4
b=-99
out=-99
call child_step(a,b,out,8,k>2)
do i=-2,5
if(b(i)/=2*a(i)+real(i,8)+total) error stop 'native intermediate'
if(out(i)/=a(i)+b(i)) error stop 'native output'
enddo
enddo
print *, 'NATIVE_ABI_OK'
end program
''')
    result = subprocess.run([compiler, '-cpp', '-fopenmp', '-ffree-line-length-none', '-fcheck=all',
                             '-finit-logical=true', *map(str, paths), str(driver), str(build/'sentinels.o'),
                             '-o', str(build/'native')], cwd=build, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout+result.stderr
    result = subprocess.run([str(build/'native')], cwd=build, capture_output=True, text=True, timeout=30,
                            env={**os.environ, 'OMP_NUM_THREADS': '4', 'OMP_DYNAMIC': 'FALSE'})
    assert result.returncode == 0, result.stdout+result.stderr
    assert 'NATIVE_ABI_OK' in result.stdout


def test_private_state_between_worksharing_units_retains_conservative_native_effects(tmp_path):
    writer = CORRECTION.replace('b(i)=b(i)+sum(a)', 'j=i\nb(i)=b(i)+sum(a)')
    middle = PRODUCER.replace('2*a(i)', '4*a(i)')
    reader = CORRECTION.replace('b(i)', 'b(i+j)')
    source = SOURCE.replace('integer::i\n', 'integer::i,j\n').replace(
        'parallel private(i)', 'parallel private(i,j)').replace(CORRECTION, writer+middle+reader)
    _, _, report, text = emit(tmp_path, source)
    (team, operations), = teams(report)
    assert len(team['numerical_units']) == 3
    native = native_operation_at(operations, source, 'b(i+j)=b(i+j)+sum(a)')
    assert not native['sections']['available']
    assert 'thread-private state' in native['sections']['reason']
    assert not native['indirect_inspector']['available']
    assert text.count('b(i+j)=b(i+j)+sum(a)') == 1
