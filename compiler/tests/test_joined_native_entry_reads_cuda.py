"""Compare reached native initializer retention with its previous retirement."""

import json
import os
import re
from collections import Counter

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as support
from compiler.tests.test_joined_native_entry_reads import emit

DRIVER = '''program verify
use native_graph
use iso_fortran_env,only:error_unit
implicit none
integer::n,stage,i,j,k,unit,shape
integer,parameter::sizes(3)=[0,1,5]
open(newunit=unit,file='fields.bin',access='stream',form='unformatted',status='replace')
do shape=1,size(sizes)
n=sizes(shape)
do stage=2,3
do k=-1,6
do j=-1,6
do i=-1,6
a(i,j,k)=real(i+2*j+3*k+20,8)*0.125d0
enddo
enddo
enddo
b=-11.d0
c=7.d0
d=-13.d0
e=-17.d0
f=-19.d0
write(error_unit,'(a,i0,a,i0)')'INIT_BEGIN ',n,' ',stage
if(n==0)then
call advance(999,n)
else
call advance(stage,n)
endif
write(error_unit,'(a,i0,a,i0)')'INIT_END ',n,' ',stage
write(unit)n,stage,lbound(a),ubound(a),a,b,c,d,e,f
enddo
enddo
close(unit)
end program
'''


@pytest.fixture(scope='module')
def binaries(tmp_path_factory):
    from compiler.frontend import reached_read_requirements as reads
    from compiler.frontend.native_completion import NativeCompletionProof

    original = reads.prove_native_read_requirements
    result = {}
    for enabled in (False, True):
        def issue(*args, _enabled=enabled, **kwargs):
            if not _enabled and type(kwargs.get('completion')) is NativeCompletionProof:
                return None
            return original(*args, **kwargs)

        def build(directory):
            path, _builder, outputs, manifest = emit(directory)
            return path, outputs, manifest

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(reads, 'prove_native_read_requirements', issue)
            patch.setattr(support, 'emit', build)
            patch.setattr(support, 'DRIVER', DRIVER)
            result[enabled] = support.epoch_binaries.__wrapped__(tmp_path_factory)
    return result


@pytest.mark.cuda
@pytest.mark.parametrize('hidden', [False, True], ids=['enabled', 'hidden'])
def test_initializer_keeps_gpu_prefix_replicas_and_original_host_values(binaries, hidden):
    ledgers = {}
    for enabled, (directory, candidate, native) in binaries.items():
        env = {**os.environ, 'OMP_NUM_THREADS': '4', 'OMP_DYNAMIC': 'FALSE', 'FORT_RUNTIME_TRACE': '1'}
        if hidden:
            env['CUDA_VISIBLE_DEVICES'] = ''
        label = ('hidden' if hidden else 'gpu') + ('-retained' if enabled else '-retired')
        support.checked([str(native)], directory, label + '-native', env)
        expected = (directory / 'fields.bin').read_bytes()
        result = support.checked([str(candidate)], directory, label + '-candidate', env)
        assert (directory / 'fields.bin').read_bytes() == expected
        manifest = json.loads((directory / 'scope-manifest.json').read_text())
        identities = {r['resource']: str(r['registration_identity'])
                      for r in manifest['scopes'][0]['checked_execution_plan']['resources']}
        cases = re.findall(r'INIT_BEGIN (\d+) (\d+)\n(.*?)INIT_END \1 \2', result.stderr, re.S)
        assert len(cases) == 6
        ledgers[enabled] = {}
        for n, stage, trace in cases:
            totals = Counter()
            for line in trace.splitlines():
                if line.startswith(('FORT_SCOPED upload ', 'FORT_SCOPED download ')):
                    fields = dict(re.findall(r'(\w+)=([^ ]+)', line))
                    totals[line.split()[1], fields['buffer']] += int(fields['bytes'])
            ledgers[enabled][n, stage] = totals
            if hidden or n == '0':
                assert not totals
                continue
            assert trace.count('FORT_SCOPED launch ') == 2, trace
            if enabled:
                assert trace.count('FORT_SCOPED close ') == 1, trace
                assert 'uninitialized_read' not in trace
                assert totals['upload', identities['native_graph::a']] == int(n)**3 * 8
                assert totals['upload', identities['native_graph::b']] == 0
                between = trace.split('FORT_SCOPED launch ')[1]
                assert 'FORT_SCOPED download ' not in between
            else:
                assert trace.count('FORT_SCOPED close ') == 2, trace
                assert 'uninitialized_read' in trace
    if not hidden:
        for key, after in ledgers[True].items():
            before = ledgers[False][key]
            n = int(key[0])
            # The produced field still publishes at unmanaged return. Only
            # its re-upload and the unchanged input's second upload disappear.
            assert sum(before.values()) - sum(after.values()) == 2 * n**3 * 8
