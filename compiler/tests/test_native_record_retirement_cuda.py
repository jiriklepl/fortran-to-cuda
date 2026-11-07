"""Complete native record calls retain storage, with retirement as fallback."""
import json
import re

import pytest

from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests.test_descriptor_record_calls import decline_complete_record_call
from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle import emit
from compiler.tests.test_joined_boundary_lifecycle_cuda import DRIVER as PREVIOUS_DRIVER
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_native_record_lifetime import hierarchical_source


def source():
    # Explicit halos retain original source ordering and ordinary native CALL.
    # The local record pointer indexes exactly the n interior entries.
    return (hierarchical_source()
            .replace('real(8),intent(out)::b(:)', 'real(8),intent(inout)::b(:)')
            .replace('real(8),intent(out)::a(:)', 'real(8),intent(inout)::a(:)')
            .replace('b(i)=real(i,8)', 'b(i+2)=real(i,8)')
            .replace('a(i)=2.0_8*b(i)', 'a(i+2)=2.0_8*b(i+2)')
            .replace('allocate(left(-3:n-4),right(2:n+1))', 'allocate(left(2:n+5),right(2:n+5))')
            .replace('left=0\nright=0', 'left=-101\nright=-99')
            .replace('right(lbound(right,1)+i-1)', 'right(lbound(right,1)+i+1)')
            .replace('do i=1,size(values)', 'do i=1,size(values)-4')
            .replace('p%flux=values(i)', 'p%flux=values(i+2)')
            .replace('values(i)=p%flux', 'values(i+2)=p%flux'))


def sealed_emission(directory, route='retained'):
    from elmm_pipeline.local_adapter import _scope_package_build
    if route == 'retirement':
        with pytest.MonkeyPatch.context() as diagnostic:
            decline_complete_record_call(diagnostic)
            original, builder, middle, patches, outputs, report = emit(directory, source())
        patch, = patches
        assert patch.validate(middle, require_source=True) is patch
        assert patch.completion.public()['requires_domain_retirement']
        assert patch.lifecycle._native_graph is patch.completion._graph
        records = {'completion': patch.completion.public(), 'storage': patch.lifecycle.public()}
    else:
        original, builder = fixture(directory, source())
        outputs, report = builder.run()
        owner = builder._conditional_return_draft.validate().owner
        middle, = [member for member in owner.members.values() if member.routine.qualified == 'hierarchical::middle']
        operation, = [op for unit in middle.units for op in unit.scope.native if op.call_effects is not None]
        proof = operation.call_effects
        assert proof.validate(operation.proof_analysis, middle.routine.qualified, operation.nodes) is proof
        assert operation.coherence_effects == {'hierarchical::right': {'read', 'write'}}
        assert set(operation.host_metadata) == {'hierarchical::native_points'}
        assert not operation.overwrites
        assert operation.preserve_original
        assert not middle.boundaries
        records = {'completion': proof.reachability.native_graph.public(), 'storage': proof.storage.public(),
                   'call_effects': proof.public()}
    # Preserve the exact first returned package before the second publication
    # check below revisits the builder-owned output dictionary.
    outputs = dict(outputs)
    assert builder.finish()
    scope, = report['scopes']
    assert {name.split('#')[0] for name in scope['gpu_leaves']} == {
        'hierarchical::producer', 'hierarchical::consumer'}
    member, = [m for m in scope['checked_execution_plan']['storage_epoch']['return_contract']['members']
               if m['procedure'] == 'hierarchical::middle']
    assert any(row['kind'] == 'reached_retirement' and row['rearm_allowed'] for row in member['routes']) is (route == 'retirement')
    transformed = outputs[report['sources'][str(original)]['replacement']]
    worker = source()[source().index('subroutine record_worker('):source().index('end module')]
    assert worker in transformed
    assert 'call record_worker(native_points,right,n)' in transformed
    package = directory / 'source-package'
    package.mkdir()
    for name, text in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = json.loads((package / 'scope-manifest.json').read_text())
    _scope_package_build({'artifact_root': str(package), 'scope_manifest': manifest,
                          'tool': 'local', 'memory_model': 'scoped'})
    (directory / 'native-record-proof.json').write_text(json.dumps({**records, 'route': route,
        'original_native_worker': worker}, indent=2) + '\n')
    return original, outputs, manifest


@pytest.mark.parametrize('route', ['retained', 'retirement'])
def test_record_halo_source_retains_exact_call_authority_and_strict_transport(tmp_path, route):
    sealed_emission(tmp_path, route)


DRIVER = (PREVIOUS_DRIVER.replace('JOINED', 'RECORD')
          .replace("write(error_unit,'(A)')'RECORD_ZERO_BEGIN'", "native_points%flux=-701\nnative_points%aux=-702\nwrite(error_unit,'(A)')'RECORD_ZERO_BEGIN'")
          .replace("if(allocated(left).or.allocated(right))error stop 'zero trip inspected storage'", "if(allocated(left).or.allocated(right))error stop 'zero trip inspected storage'\nif(any(native_points%flux/=-701).or.any(native_points%aux/=-702))error stop 'zero trip touched records'")
          .replace("write(error_unit,'(A,1X,I0)')'RECORD_CASE_BEGIN',n", "native_points%flux=-701\nnative_points%aux=-702\nwrite(error_unit,'(A,1X,I0)')'RECORD_CASE_BEGIN',n")
          .replace("write(*,'(5(I0,1X))')n", "do i=1,n\nif(native_points(i)%flux/=real(i+1,8).or.native_points(i)%aux/=1.0_8)error stop 'native record write lost'\nenddo\nif(any(native_points(n+1:)%flux/=-701).or.any(native_points(n+1:)%aux/=-702))error stop 'unused record changed'\nwrite(*,'(5(I0,1X))')n")
          .replace("write(*,'(*(ES26.17E3,1X))')observed,left,right", "write(*,'(*(ES26.17E3,1X))')observed,left,right,native_points%flux,native_points%aux"))


@pytest.fixture(scope='module', params=['retained', 'retirement'])
def binaries(tmp_path_factory, request):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, 'emit', lambda directory: sealed_emission(directory, request.param))
        patch.setattr(base, 'DRIVER', DRIVER)
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((directory / 'scope-manifest.json').read_text())
    provenance = {row['id']: row for row in manifest['runtime_provenance']['records']}
    return directory, candidate, native, provenance, request.param


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['traced', 'untraced', 'hidden'])
def test_original_record_call_preserves_fields_and_expected_residency(binaries, mode):
    directory, candidate, native, provenance, route = binaries
    env = environment(mode)
    reference = base.checked([str(native)], directory, mode+'-native', env)
    result = base.checked([str(candidate)], directory, mode+'-candidate', env)
    assert result.stdout == reference.stdout
    assert 'RECORD_NATIVE_FIELDS_HALOS_AND_BOUNDS_OK' in result.stdout
    if mode == 'untraced':
        assert 'FORT_SCOPED ' not in result.stderr
        return
    zero, = re.findall(r'RECORD_ZERO_BEGIN\n(.*?)RECORD_ZERO_END', result.stderr, re.S)
    assert not any(e['kind'] in {'domain_create', 'register', 'launch'} for e in events_in(zero))
    cases = re.findall(r'RECORD_CASE_BEGIN (\d+)\n(.*?)RECORD_CASE_END \1', result.stderr, re.S)
    assert len(cases) == 3
    ledger = []
    for width, text in cases:
        n = int(width)
        events = events_in(text)
        launches = [i for i, event in enumerate(events) if event['kind'] == 'launch']
        ledger.append({'n': n, 'events': events, 'upload_bytes': copy_bytes(events, 'upload'),
                       'download_bytes': copy_bytes(events, 'download')})
        (directory / (mode+'-ledger.json')).write_text(json.dumps(ledger, indent=2)+'\n')
        if mode == 'hidden':
            assert not launches
            assert copy_bytes(events, 'upload') == copy_bytes(events, 'download') == 0
            continue
        assert [procedure(events[i], provenance) for i in launches] == [
            'hierarchical::producer', 'hierarchical::consumer'] * 2
        for stage in range(2):
            first, last = launches[2*stage:2*stage+2]
            between = events[first+1:last]
            assert (events[first]['domain'] != events[last]['domain']) is (route == 'retirement')
            assert any(e['kind'] == 'domain_retire' for e in between) is (route == 'retirement')
            assert copy_bytes(between, 'download') == 8*n
            assert copy_bytes(between, 'upload') == 8*n
        assert copy_bytes(events, 'upload') == 16*n
        assert copy_bytes(events, 'download') == 32*n
