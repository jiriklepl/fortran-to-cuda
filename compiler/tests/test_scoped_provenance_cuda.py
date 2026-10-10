"""Actual device work resolves to public source identities across borrowed calls."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

import pytest

from compiler.numerical_contract import require_explicit_cuda_environment, require_numerical_build_contract


def provenance_source(precision):
    return f"""module tagged_child
implicit none
integer::child_visits=0
contains
subroutine adjust(x,n,escape)
real({precision}),intent(inout)::x(-4:)
integer,intent(in)::n
logical,intent(in)::escape
integer::j
child_visits=child_visits+1
do j=-4,n-5
x(j)=x(j)+real(j,{precision})
enddo
if(n>0) then
! Runtime-length serial SUM remains the original coherent native operation.
x(-4)=x(-4)+sum(x)
endif
if(escape) call opaque(x,n)
end subroutine
end module
module tagged_owner
use tagged_child,only:adjust
implicit none
integer::visits=0
contains
subroutine step(a,b,out,n,escape)
real({precision}),intent(in)::a(-2:)
real({precision}),intent(inout)::b(-2:),out(-2:)
integer,intent(in)::n
logical,intent(in)::escape
integer::i
visits=visits+1
do i=-2,n-3
b(i)=2*a(i)+real(i,{precision})
enddo
call adjust(b,n,escape)
do i=-2,n-3
out(i)=a(i)+b(i)
enddo
end subroutine
end module
"""


def provenance_driver(precision):
    bits = 'int32' if precision == 4 else 'int64'
    return f"""program verify
use iso_fortran_env,only:{bits},error_unit
use tagged_owner,only:generated_step=>step,generated_visits=>visits
use tagged_child,only:generated_adjust=>adjust,generated_child_visits=>child_visits
use reference_owner,only:native_step=>step,native_visits=>visits
use reference_child,only:native_adjust=>adjust,native_child_visits=>child_visits
use callback_audit,only:opaque_visits
implicit none
integer,parameter::widths(4)=[17,0,5,17]
real({precision})::a(-2:19),b(-2:19),out(-2:19),expected_b(-2:19),expected_out(-2:19)
real({precision})::input_reference(-2:19)
integer::iteration,i,n,unit
logical::escape
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do iteration=1,4
n=widths(iteration)
escape=iteration==3
do i=-2,19
a(i)=real(3*i-7+iteration,{precision})/16._{precision}
enddo
input_reference=a
b=-113
out=-117
expected_b=b
expected_out=out
native_visits=0
native_child_visits=0
generated_visits=0
generated_child_visits=0
opaque_visits=0
call native_step(a,expected_b,expected_out,n,escape)
if(opaque_visits/=merge(1,0,escape)) error stop 'native opaque count'
opaque_visits=0
write(error_unit,*)'BEGIN_PROVENANCE_CALL',iteration,n,escape
call generated_step(a,b,out,n,escape)
write(error_unit,*)'END_PROVENANCE_CALL',iteration
if(native_visits/=1.or.generated_visits/=1) error stop 'owner prefix replayed'
if(native_child_visits/=1.or.generated_child_visits/=1) error stop 'child prefix replayed'
if(opaque_visits/=merge(1,0,escape)) error stop 'native escape replayed'
if(any(transfer(a,[0_{bits}],size(a))/=transfer(input_reference,[0_{bits}],size(a)))) &
 error stop 'input or input halo changed'
if(any(transfer(b,[0_{bits}],size(b))/=transfer(expected_b,[0_{bits}],size(b)))) &
 error stop 'complete intermediate or halo differs'
if(any(transfer(out,[0_{bits}],size(out))/=transfer(expected_out,[0_{bits}],size(out)))) &
 error stop 'complete output or halo differs'
write(unit)a,b,out
enddo
b=-113
expected_b=b
call native_adjust(expected_b,3,.false.)
write(error_unit,*)'BEGIN_ORIGINAL_NATIVE_ENTRY'
call generated_adjust(b,3,.false.)
write(error_unit,*)'END_ORIGINAL_NATIVE_ENTRY'
if(any(transfer(b,[0_{bits}],size(b))/=transfer(expected_b,[0_{bits}],size(b)))) &
 error stop 'original native ABI differs'
write(unit)b
close(unit)
print *,'PROVENANCE_COMPLETE_FIELDS_BITWISE_OK'
end program
"""


def actual_events(stderr, *, include_positions=False):
    """Parse retained actual scoped events, never infer work from eligibility."""
    result = []
    operations = 'upload|download|launch|close' + ('|position|restore' if include_positions else '')
    for line in stderr.splitlines():
        match = re.match(r'^FORT_SCOPED (' + operations + r')\s+(.*)$', line)
        if match:
            result.append({'event': match[1], **dict(re.findall(r'(\w+)=([^\s]+)', match[2]))})
    return result


def finite_fields(payload, precision):
    return all(math.isfinite(value) for (value,) in struct.iter_unpack('=f' if precision == 4 else '=d', payload))


@pytest.mark.cuda
@pytest.mark.parametrize('precision', [4, 8])
def test_actual_scope_work_and_close_resolve_to_public_source_provenance(tmp_path, precision):
    fc = shutil.which('gfortran-15') or shutil.which('gfortran')
    nvcc = shutil.which('nvcc')
    host = shutil.which('g++-14') or shutil.which('g++')
    if not all((fc, nvcc, host)):
        pytest.skip('native Fortran and CUDA toolchain required')
    require_explicit_cuda_environment()
    compiler_root = Path(__file__).resolve().parents[2]
    checkout = tmp_path / 'independent-compiler'
    shutil.copytree(compiler_root / 'compiler', checkout / 'compiler', ignore=shutil.ignore_patterns(
        '__pycache__', '.*cache', 'CODE_MAP.md', '*PLAN.md'))
    source = tmp_path / 'owner.f90'
    source.write_text(provenance_source(precision))
    reference = tmp_path / 'reference.f90'
    reference.write_text(source.read_text().replace('tagged_child', 'reference_child')
                         .replace('tagged_owner', 'reference_owner'))
    callback = tmp_path / 'callback.f90'
    callback.write_text(f"""module callback_audit
integer::opaque_visits=0
end module
subroutine opaque(x,n)
use callback_audit,only:opaque_visits
real({precision})::x(*)
integer::n
opaque_visits=opaque_visits+1
if(n>0) x(1)=x(1)+41
end subroutine
""")
    stable = {'storage': 'stable', 'initialized': 'whole', 'escapes': False, 'allocation_changes': False}
    facts = {'schema_version': 1, 'participation': 'serial',
             'captures': {'argument::' + name: stable for name in ('a', 'b', 'out')},
             'sources': {str(source): sha256(source.read_bytes()).hexdigest()}}
    facts_path = tmp_path / 'facts.json'
    facts_path.write_text(json.dumps(facts, indent=2) + '\n')
    output, build = tmp_path / 'generated', tmp_path / 'build'
    build.mkdir()
    environment = dict(os.environ)
    for name in list(environment):
        if name.startswith(('FORT_', 'ELMM_')) or name in {
            'OMP_THREAD_LIMIT', 'OMP_PLACES', 'GOMP_CPU_AFFINITY', 'LD_PRELOAD'}:
            environment.pop(name)
    environment.update(OMP_NUM_THREADS='4', OMP_DYNAMIC='FALSE', OMP_PROC_BIND='false',
                       OMP_SCHEDULE='static', PYTHONPATH=str(checkout), PYTHONDONTWRITEBYTECODE='1')
    commands = []

    def run(argv, *, trace=False, disabled=False):
        ordinal = len(commands)
        commands.append({'argv': list(map(str, argv)), 'cwd': str(build), 'trace': trace,
                         'device_disabled': disabled})
        (tmp_path / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        result = subprocess.run(argv, cwd=build, env={**environment,
            **({'FORT_RUNTIME_TRACE': '1'} if trace else {}),
            **({'CUDA_VISIBLE_DEVICES': ''} if disabled else {})},
            capture_output=True, text=True, timeout=180, check=False)
        (tmp_path / f'command-{ordinal:02}.stdout').write_text(result.stdout)
        (tmp_path / f'command-{ordinal:02}.stderr').write_text(result.stderr)
        (tmp_path / f'command-{ordinal:02}.json').write_text(json.dumps({'exit_code': result.returncode}) + '\n')
        assert result.returncode == 0, result.stdout + result.stderr
        return result

    report = json.loads(run([sys.executable, '-m', 'compiler', '--form-scopes', '--input', str(source),
        '--kernel', 'tagged_owner::step', '--scope-facts', str(facts_path), '--scope-execution', 'reached',
        '--gpu-policy', 'sections', '--memory-model', 'scoped', '--json', '--output-dir', str(output)]).stdout)
    assert report['supported'], report
    manifest = json.loads((output / 'scope-manifest.json').read_text())
    assert manifest == report['scopes']
    owner, = manifest['scopes']
    assert len(owner['gpu_leaves']) == 3, owner
    child, = owner['module_coordinators']
    assert child['procedure'] == 'tagged_child::adjust'
    assert child['boundaries'], 'the opaque child operation must retain its original boundary'
    provenance = manifest['runtime_provenance']
    assert provenance['schema_version'] == 1
    assert provenance['available']
    records = {item['id']: item for item in provenance['records']}
    assert len(records) == len(provenance['records'])
    assert all(re.fullmatch(r'[a-f0-9]{64}', identity) for identity in records)
    flags = [fc, '-O3', '-fopenmp', '-ffp-contract=off', '-fcheck=all', '-J', str(build), '-I', str(build)]
    objects, compiled = [], set()
    for role in ('common_runtime', 'shared_entry'):
        for descriptor in manifest['build_sources']:
            if descriptor['role'] != role:
                continue
            path = output / descriptor['path']
            assert sha256(path.read_bytes()).hexdigest() == manifest['artifacts_sha256'][descriptor['path']]
            target = build / f'artifact-{len(objects):02}.o'
            selected = flags
            if descriptor['language'] == 'cuda':
                contract = descriptor['numerical_contract']
                require_numerical_build_contract(contract)
                selected = [nvcc, '-O3', '-std=c++17', '-ccbin', host, '-Xcompiler=-fopenmp',
                    '-arch=' + os.environ.get('FORT_TEST_CUDA_ARCH', 'native'),
                    *contract['required_cuda_options'],
                    *('-Xcompiler=' + option for option in contract['required_host_options'])]
            run([*selected, '-I', str(output), '-c', str(path), '-o', str(target)])
            objects.append(str(target))
            compiled.add(descriptor['path'])
    for path in (callback, reference, output / manifest['sources'][str(source)]['replacement']):
        target = build / f'dependency-{len(objects):02}.o'
        run([*flags, '-c', str(path), '-o', str(target)])
        objects.append(str(target))
    assert compiled | {manifest['sources'][str(source)]['replacement']} == {
        descriptor['path'] for descriptor in manifest['build_sources']}
    driver = tmp_path / 'driver.f90'
    driver.write_text(provenance_driver(precision))
    target = build / 'driver.o'
    run([*flags, '-c', str(driver), '-o', str(target)])
    binary = build / 'verify'
    run([nvcc, '-ccbin', host, '-Xcompiler=-fopenmp', *objects, str(target), '-lgfortran', '-o', str(binary)])

    observed = run([str(binary)], trace=True)
    assert 'PROVENANCE_COMPLETE_FIELDS_BITWISE_OK' in observed.stdout
    expected = (build / 'fields.bin').read_bytes()
    assert len(expected) == (4 * 3 + 1) * 22 * precision
    assert finite_fields(expected, precision), 'retained fields must be finite, including every halo'
    (tmp_path / 'traced-fields.bin').write_bytes(expected)
    calls = observed.stderr.split('BEGIN_PROVENANCE_CALL')[1:]
    assert len(calls) == 4
    contexts, buffers, attribution = [], [], []
    for ordinal, call in enumerate(calls):
        trace = call.split('END_PROVENANCE_CALL', 1)[0]
        events = actual_events(trace)
        launches = [event for event in events if event['event'] == 'launch']
        assert len(launches) == (3, 0, 2, 3)[ordinal], trace
        if ordinal == 1:
            # A source owner may close an uninitialized context even when all
            # numerical domains are empty. It must perform no CUDA work.
            assert all(event['event'] == 'close' for event in events), trace
            assert len(events) <= 1, trace
            continue
        current_contexts = {int(event['context']) for event in events}
        assert len(current_contexts) == 1, trace
        assert min(current_contexts) > 0, trace
        contexts.append(current_contexts.pop())
        touched = set()
        for event in events:
            assert event['provenance_version'] == '1', event
            assert event['owner'] in records, event
            assert records[event['owner']]['kind'] == 'owner', event
            assert records[event['owner']]['procedure'] == 'tagged_owner::step', event
            assert event['procedure'] in records, event
            assert records[event['procedure']]['kind'] == 'procedure', event
            if event['event'] != 'close':
                assert event['segment'] in records, event
                assert records[event['segment']]['kind'] == 'segment', event
            else:
                assert event['segment'] == 'unknown' or event['segment'] in records, event
            for field in ('operation', 'implementation', 'boundary'):
                assert event[field] == 'unknown' or event[field] in records, event
            if event['event'] in {'upload', 'download'}:
                assert int(event['bytes']) > 0, event
                assert int(event['buffer_handle']) > 0, event
                touched.add(int(event['buffer_handle']))
            if event['event'] == 'launch':
                assert event['implementation'] in records, event
                assert records[event['implementation']]['kind'] == 'implementation', event
                assert records[event['implementation']]['backend'] == 'cuda', event
                assert event['operation'] in records, event
                assert records[event['operation']]['kind'] != 'native_operation', event
        buffers.append(touched)
        assert touched, trace
        procedures = [records[event['procedure']]['procedure'] for event in launches]
        assert procedures[0] == 'tagged_owner::step'
        assert procedures[1] == 'tagged_child::adjust'
        native_downloads = [event for event in events if event['event'] == 'download' and
                            records[event['procedure']]['procedure'] == 'tagged_child::adjust' and
                            event['boundary'] == 'unknown']
        assert native_downloads, 'native child SUM must receive coherent data: ' + trace
        for event in native_downloads:
            assert records[event['operation']]['kind'] == 'native_operation', event
            assert records[event['implementation']]['backend'] == 'original_native', event
        close, = [event for event in events if event['event'] == 'close']
        assert close['boundary'] in records, close
        assert records[close['boundary']]['kind'] == 'boundary', close
        if ordinal == 2:
            assert records[close['boundary']]['procedure'] == 'tagged_child::adjust'
            assert 'opaque' in records[close['boundary']].get('reason', '').lower(), close
            assert events.index(close) == len(events) - 1, 'work continued after reached close: ' + trace
        else:
            assert procedures[-1] == 'tagged_owner::step', 'child return attribution leaked into consumer'
            assert records[launches[-1]['segment']]['procedure'] == 'tagged_owner::step'
            positioned = actual_events(trace, include_positions=True)
            native_download = positioned.index(native_downloads[-1])
            consumer_launch = positioned.index(launches[-1])
            restored = [event for event in positioned[native_download + 1:consumer_launch]
                        if event['event'] == 'restore' and
                        event['procedure'] == launches[0]['procedure'] and
                        event['segment'] == launches[0]['segment']]
            assert restored, 'borrowed return was hidden by the next segment position: ' + trace
        attribution.append({'ordinal': ordinal + 1, 'context': contexts[-1], 'events': events,
            'actual_launches': len(launches),
            'actual_upload_bytes': sum(int(event['bytes']) for event in events if event['event'] == 'upload'),
            'actual_download_bytes': sum(int(event['bytes']) for event in events if event['event'] == 'download')})
    assert len(set(contexts)) == 3, 'repeated owners reused stale context identity'
    assert all(not left.intersection(right) for i, left in enumerate(buffers) for right in buffers[i + 1:]), \
        'buffer handles were reused across separate owner invocations'
    native_call = observed.stderr.split('BEGIN_ORIGINAL_NATIVE_ENTRY', 1)[1].split('END_ORIGINAL_NATIVE_ENTRY', 1)[0]
    assert not actual_events(native_call), 'original native ABI acquired managed execution'
    (tmp_path / 'actual-provenance.json').write_text(json.dumps({'schema_version': 1,
        'precision_bits': 8 * precision, 'manifest_sha256': sha256((output / 'scope-manifest.json').read_bytes()).hexdigest(),
        'calls': attribution, 'performance_sample': False}, indent=2) + '\n')
    for trace, disabled, label in ((False, False, 'untraced'), (True, True, 'device-disabled')):
        result = run([str(binary)], trace=trace, disabled=disabled)
        assert 'PROVENANCE_COMPLETE_FIELDS_BITWISE_OK' in result.stdout
        assert (build / 'fields.bin').read_bytes() == expected
        assert finite_fields((build / 'fields.bin').read_bytes(), precision)
        (tmp_path / f'{label}-fields.bin').write_bytes(expected)
        if disabled:
            assert not [event for event in actual_events(result.stderr)
                        if event['event'] in {'upload', 'download', 'launch'}]
        else:
            assert 'FORT_SCOPED ' not in result.stderr, 'disabled tracing changed observable diagnostics'
