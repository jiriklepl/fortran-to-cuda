"""Mutable scalar-element backing remains native across same-family retirement."""
import json
import re
import shutil

import pytest

from compiler.scopes.conditional_return import SameFamilyNativeCallLifecycle
from compiler.scopes.lexical import LexicalBoundaryPatch
from compiler.scopes.storage_epoch import resource_inventory
from compiler.tests import test_graph_storage_epochs_cuda as base
from compiler.tests import test_same_family_native_call_cuda as original_fixture
from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_hierarchical_child_regions_cuda import copy_bytes, events_in, procedure
from compiler.tests.test_joined_boundary_lifecycle_cuda import environment
from compiler.tests.test_native_retirement_scalar_formals_cuda import physical_gpu_events


def source():
    text = (original_fixture.source()
        .replace('real(8)::observed', 'real(8)::observed\nreal(8)::ratios(0:1)=[1.0_8,2.0_8]\ninteger::selector=0', 1)
        .replace('subroutine consumer(a,b,n)', 'subroutine consumer(a,b,n,coefficient)')
        .replace('real(8),intent(in)::b(:)', 'real(8),intent(in)::b(:)\nreal(8),intent(in)::coefficient')
        .replace('a(i+2)=2.0_8*b(i+2)', 'a(i+2)=coefficient*b(i+2)')
        .replace('call consumer(left,right,n)', 'call consumer(left,right,n,ratios(selector))'))
    assert original_fixture.NATIVE in text
    assert all(line == line.rstrip() for line in text.splitlines())
    return text


def sealed_emission(directory):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    original, builder = fixture(directory, source())
    outputs, report = builder.run()
    assert report['scope_count'] == 1, report.get('graph_owner_attempt')
    owner = builder._conditional_return_draft.validate().owner
    selected = [(member, patch) for member in owner.members.values() for _first, _last, patch in member.patches
        if isinstance(patch, LexicalBoundaryPatch) and type(patch.lifecycle) is SameFamilyNativeCallLifecycle
        and 'native_worker' in str(patch.originals[0]).lower()]
    (member, patch), = selected
    assert patch.validate(member, require_source=True) is patch
    assert patch.completion is None
    assert patch.conditional_guard is None
    assert patch.lifecycle._view.public()['available']
    managed, reserved, _opaque = resource_inventory(owner)
    assert managed == {'hierarchical::left', 'hierarchical::right'}
    assert reserved == {'hierarchical::ratios'}
    assert set(builder.facts['captures']) == managed
    scalar_calls = [call for call in member.return_calls if call.scalar_element_coordinates is not None]
    call, = scalar_calls
    assert call.validate(member) is call
    assert call.scalar_element_coordinates.scalar_elements[0].binding is owner.scalar_element_resources['hierarchical::ratios']
    scope, = report['scopes']
    rows = scope['checked_execution_plan']['storage_epoch']['resources']
    ratio, = [row for row in rows if row['resource'] == 'hierarchical::ratios']
    assert ratio['storage'] == 'opaque host reservation'
    assert ratio['device_replica_authorized'] is False
    assert ratio['initialization_authorized'] is False
    transformed = outputs[report['sources'][str(original)]['replacement']]
    assert original_fixture.NATIVE in transformed
    assert transformed.count(original_fixture.CALL) == 1
    validate_checked_execution_plans(report)
    package = directory / 'source-package'
    for name, text in outputs.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = json.loads((package / 'scope-manifest.json').read_text())
    _scope_package_build({'artifact_root': str(package), 'scope_manifest': manifest})
    assert builder.finish()
    (directory / 'scalar-backing-retirement-proof.json').write_text(json.dumps({
        'retirement': patch.lifecycle.public(), 'coordinate_receipt': call.scalar_element_coordinates.public(),
        'managed_resources': sorted(managed), 'typed_reservations': sorted(reserved), 'ratio_epoch': ratio}, indent=2) + '\n')
    return original, outputs, manifest


def test_source_scalar_backing_retains_original_call_and_strict_reader(tmp_path):
    sealed_emission(tmp_path)


DRIVER = original_fixture.DRIVER
_start = DRIVER.index('character(8)::argument\n')
_end = DRIVER.index("write(error_unit,'(A)')'FAMILY_COLD_BEGIN'", _start)
DRIVER = DRIVER[:_start] + 'integer::selector_case\nreal(8)::coefficients_before(0:1)\n' + DRIVER[_end:]
DRIVER = (DRIVER.replace('do record_mode=-1,1\ndo case_index=1,3', '''do selector_case=0,1
selector=selector_case
ratios=[3.0_8+selector_case,5.0_8+selector_case]
coefficients_before=ratios
do record_mode=-1,1,2
do case_index=1,2''')
    .replace("write(error_unit,'(A,2(1X,I0))')'FAMILY_CASE_BEGIN',record_mode,n",
        "write(error_unit,'(A,3(1X,I0))')'FAMILY_CASE_BEGIN',selector_case,record_mode,n")
    .replace("write(error_unit,'(A,2(1X,I0))')'FAMILY_CASE_END',record_mode,n",
        "write(error_unit,'(A,3(1X,I0))')'FAMILY_CASE_END',selector_case,record_mode,n")
    .replace('right(i+3)/=2.0_8*expected', 'right(i+3)/=ratios(selector)*expected')
    .replace('deallocate(left,right)\nenddo\nenddo', '''if(any(ratios/=coefficients_before).or.selector/=selector_case)error stop 'mutable backing or selector changed'
write(*,'(*(ES26.17E3,1X))')ratios
deallocate(left,right)
enddo
enddo
enddo''')
    .replace('SAME_FAMILY_NATIVE_CALLBACK_FIELDS_HALOS_AND_BOUNDS_OK', 'SCALAR_BACKING_NATIVE_CALLBACK_FIELDS_HALOS_AND_BOUNDS_OK'))


@pytest.fixture(scope='module')
def binaries(tmp_path_factory):
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(base, 'emit', sealed_emission)
        patch.setattr(base, 'DRIVER', DRIVER)
        directory, candidate, native = base.epoch_binaries.__wrapped__(tmp_path_factory)
    manifest = json.loads((directory / 'scope-manifest.json').read_text())
    provenance = {row['id']: row for row in manifest['runtime_provenance']['records']}
    resources = {row['resource']: row['registration_identity']
        for row in manifest['scopes'][0]['checked_execution_plan']['storage_epoch']['resources']}
    return directory, candidate, native, provenance, resources


@pytest.mark.cuda
@pytest.mark.parametrize('mode', ['traced', 'untraced', 'hidden'])
def test_scalar_backing_native_call_and_gpu_children_share_correct_roles(binaries, mode):
    directory, candidate, native, provenance, resources = binaries
    env = environment(mode)
    reference = base.checked([str(native)], directory, mode + '-native', env)
    result = base.checked([str(candidate)], directory, mode + '-candidate', env)
    assert result.stdout == reference.stdout
    assert 'SCALAR_BACKING_NATIVE_CALLBACK_FIELDS_HALOS_AND_BOUNDS_OK' in result.stdout
    if mode == 'untraced':
        assert 'FORT_SCOPED ' not in result.stderr
        return
    for name in ['COLD', 'ZERO']:
        text, = re.findall('FAMILY_' + name + '_BEGIN\n(.*?)FAMILY_' + name + '_END', result.stderr, re.S)
        assert not any(e['kind'] in {'domain_create', 'register'} for e in events_in(text))
        assert not physical_gpu_events(text)
    if mode == 'hidden':
        assert not physical_gpu_events(result.stderr)
    cases = re.findall(r'FAMILY_CASE_BEGIN (\d+) (-?\d+) (\d+)\n(.*?)FAMILY_CASE_END \1 \2 \3', result.stderr, re.S)
    assert len(cases) == 8
    ledger = []
    for selector_case, record_mode, width, text in cases:
        n = int(width)
        events = events_in(text)
        ledger.append({'selector': int(selector_case), 'record_mode': int(record_mode), 'n': n, 'events': events})
        if mode == 'hidden':
            assert not physical_gpu_events(text)
            assert copy_bytes(events, 'upload') == copy_bytes(events, 'download') == 0
            continue
        launches = [i for i, e in enumerate(events) if e['kind'] == 'launch']
        assert [procedure(events[i], provenance) for i in launches] == ['hierarchical::producer', 'hierarchical::consumer']
        first, last = launches
        between = events[first+1:last]
        assert events[first]['domain'] != events[last]['domain']
        assert sum(e['kind'] == 'domain_retire' for e in between) == 1
        assert copy_bytes(between, 'download') == copy_bytes(between, 'upload') == 8*n
        assert copy_bytes(events, 'upload') == 8*n
        assert copy_bytes(events, 'download') == 2*8*n
        assert not any(e['kind'] in {'upload', 'download', 'allocate'}
            and int(e.get('buffer', -1)) == resources['hierarchical::ratios'] for e in events)
        for direction in ['upload', 'download']:
            selected = [e for e in between if e['kind'] == direction]
            assert {int(e['buffer']) for e in selected} == {resources['hierarchical::right']}
    (directory / (mode + '-ledger.json')).write_text(json.dumps(ledger, indent=2) + '\n')


def test_native_scalar_backing_driver_full_fields_and_mutable_table(tmp_path):
    compiler = shutil.which('gfortran-15') or shutil.which('gfortran')
    assert compiler is not None
    original, _builder = fixture(tmp_path, source())
    driver = tmp_path / 'driver.f90'
    driver.write_text(DRIVER)
    assert all(line == line.rstrip() and len(line) <= 132 for line in DRIVER.splitlines())
    native = tmp_path / 'native'
    env = environment('hidden')
    base.checked([compiler, '-fopenmp', '-fcheck=all', '-ffp-contract=off', '-J', str(tmp_path),
        str(original), str(driver), '-o', str(native)], tmp_path, 'native-compile', env)
    result = base.checked([str(native)], tmp_path, 'native-fields', env)
    assert 'SCALAR_BACKING_NATIVE_CALLBACK_FIELDS_HALOS_AND_BOUNDS_OK' in result.stdout
    assert len(re.findall('FAMILY_CASE_BEGIN ', result.stderr)) == 8
