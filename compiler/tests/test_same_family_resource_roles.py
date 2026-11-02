"""Native reservations keep their issued role across whole-family retirement."""
import json
from dataclasses import replace

import pytest

from compiler.ir import CompilationError
from compiler.scopes.conditional_return import SameFamilyNativeCallLifecycle
from compiler.scopes.storage_epoch import resource_inventory
from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_same_family_native_call import source_fixture


def source(*, opaque=False, future=False):
    text = (source_fixture()
        .replace('real(8)::observed', 'real(8)::observed\nreal(8)::ratios(0:1)=[1.0_8,2.0_8]\ninteger::selector=0', 1)
        .replace('subroutine consumer(a,b,n)', 'subroutine consumer(a,b,n,scale)')
        .replace('real(8),intent(in)::b(:)', 'real(8),intent(in)::b(:)\nreal(8),intent(in)::scale')
        .replace('a(i)=2.0_8*b(i)', 'a(i)=scale*b(i)')
        .replace('call consumer(left,right,n)', 'call consumer(left,right,n,ratios(selector))'))
    if opaque:
        text = text.replace('real(8)::observed', 'real(8)::weights(2)=[3.0_8,4.0_8]\nreal(8)::observed', 1)
        text = text.replace('call native_worker(left,right,n,scale)', '''observed=0
 do i=1,2
 observed=observed+weights(i)
 enddo
 call native_worker(left,right,n,scale)''')
    if future:
        text = text.replace('real(8)::observed', 'real(8),allocatable::tail(:)\nreal(8)::observed', 1)
        text = text.replace('right(2:n+1))', 'right(2:n+1),tail(2:n+1))')
        text = text.replace('call consumer(left,right,n,ratios(selector))',
            'call consumer(left,right,n,ratios(selector))\ncall later(tail,left,n)')
        text = text.replace('end module', '''subroutine later(out,input,n)
integer,intent(in)::n
real(8),intent(out)::out(:)
real(8),intent(in)::input(:)
integer::i
do i=1,n
out(i)=input(i)+1.0_8
enddo
end subroutine
end module''')
    return text


def emit(directory, *, opaque=False, future=False):
    path, builder = fixture(directory, source(opaque=opaque, future=future))
    if future:
        builder.facts['captures']['hierarchical::tail'] = dict(builder.facts['captures']['hierarchical::left'])
    outputs, report = builder.run()
    assert report['scope_count'] == 1, report.get('graph_owner_attempt')
    owner = builder._conditional_return_draft.validate().owner
    member = owner.members['hierarchical::middle']
    proof, = [patch.lifecycle for _first, _last, patch in member.patches
        if type(getattr(patch, 'lifecycle', None)) is SameFamilyNativeCallLifecycle]
    return path, builder, owner, member, proof, outputs, report


@pytest.mark.parametrize('opaque', [False, True])
def test_final_typed_roles_need_facts_only_for_managed_fields(tmp_path, opaque):
    _path, builder, owner, member, proof, _outputs, _report = emit(tmp_path, opaque=opaque)
    managed, reserved, _inventory = resource_inventory(owner)
    assert managed == {'hierarchical::left', 'hierarchical::right'}
    assert 'hierarchical::ratios' in reserved
    assert 'hierarchical::ratios' not in builder.facts['captures']
    assert 'hierarchical::ratios' in owner.scalar_element_resources
    if opaque:
        assert 'hierarchical::weights' in reserved
        assert 'hierarchical::weights' not in builder.facts['captures']
        assert _inventory
    for call in member.return_calls:
        assert call.validate(member) is call
    assert proof.validate() is proof
    assert builder.finish()


def test_real_emitted_package_strict_reader_keeps_opaque_roles(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build

    _path, builder, _owner, _member, _proof, outputs, _report = emit(tmp_path)
    package = tmp_path / 'package'
    for name, content in outputs.items():
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    manifest = json.loads((package / 'scope-manifest.json').read_text())
    validate_checked_execution_plans(manifest)
    _scope_package_build({'artifact_root': str(package), 'scope_manifest': manifest})
    assert builder.finish()


def test_final_managed_set_includes_field_discovered_after_native_call(tmp_path, monkeypatch):
    _path, builder, owner, _member, proof, _outputs, _report = emit(tmp_path, future=True)
    managed, reserved, _inventory = resource_inventory(owner)
    assert managed == {'hierarchical::left', 'hierarchical::right', 'hierarchical::tail'}
    assert reserved == {'hierarchical::ratios'}
    captured = []
    original = builder.capture_fact
    def checked(binding):
        captured.append(binding.root)
        return original(binding)
    monkeypatch.setattr(builder, 'capture_fact', checked)
    proof.validate()
    assert managed <= set(captured)
    assert 'hierarchical::ratios' not in captured
    del builder.facts['captures']['hierarchical::tail']
    with pytest.raises(CompilationError):
        proof.validate()


def test_unclassified_registration_is_not_silently_excluded(tmp_path):
    _path, _builder, owner, _member, proof, _outputs, _report = emit(tmp_path)
    binding, slot = owner.resources['hierarchical::ratios']
    dict.__setitem__(owner.resources, 'unknown::reservation', (binding, slot + 100))
    with pytest.raises(CompilationError, match='unclassified canonical registration'):
        proof.validate()


def test_missing_original_reservation_cannot_use_scalar_role_dictionary_alone(tmp_path):
    _path, _builder, _owner, member, proof, _outputs, _report = emit(tmp_path)
    dict.clear(member.original_scalar_element_reservations)
    with pytest.raises(CompilationError, match='unclassified canonical registration'):
        proof.validate()


def test_scalar_backing_cannot_be_reclassified_as_managed(tmp_path):
    _path, _builder, owner, member, proof, _outputs, _report = emit(tmp_path)
    dict.__setitem__(member.units[-1].arrays, 'hierarchical::ratios', owner.scalar_element_resources['hierarchical::ratios'])
    with pytest.raises(CompilationError, match='scalar element backing storage conflicts'):
        proof.validate()


def test_managed_descriptor_metadata_overlap_remains_allowed(tmp_path):
    _path, _builder, owner, member, _proof, _outputs, _report = emit(tmp_path)
    binding, _slot = owner.resources['hierarchical::left']
    dict.__setitem__(member.units[-1].host_metadata, 'hierarchical::left', binding)
    managed, reservations, _inventory = resource_inventory(owner)
    assert 'hierarchical::left' in managed & reservations


@pytest.mark.parametrize('change', ['missing', 'copy_public'])
def test_scalar_coordinate_receipt_is_still_required_by_full_owner(tmp_path, change):
    _path, _builder, owner, member, _proof, _outputs, _report = emit(tmp_path)
    index, call = next((i, call) for i, call in enumerate(member.return_calls) if call.scalar_element_coordinates)
    receipt = None if change == 'missing' else call.scalar_element_coordinates.public()
    list.__setitem__(member.return_calls, index, replace(call, scalar_element_coordinates=receipt))
    with pytest.raises(CompilationError):
        owner.return_contract.validate()


@pytest.mark.parametrize('change', ['registry_removed', 'permit_public', 'canonical_changed'])
def test_opaque_permit_and_canonical_roles_still_revalidate(tmp_path, change):
    _path, _builder, owner, _member, proof, _outputs, _report = emit(tmp_path, opaque=True)
    native = [(member, operation, root, access) for member in owner.members.values() for unit in member.units
        for operation in unit.scope.native for root, access in operation.native_opaque_accesses.items()]
    member, operation, root, access = native[0]
    if change == 'registry_removed':
        access.permit._analysis._native_opaque_read_permits.pop(access.permit.identity)
    elif change == 'permit_public':
        dict.__setitem__(operation.native_opaque_accesses, root, replace(access, permit=access.permit.public()))
    else:
        dict.__setitem__(member.canonical_bindings, access.binding.root, owner.resources['hierarchical::left'][0])
    with pytest.raises(CompilationError):
        proof.validate()


def test_source_initializer_edit_cannot_publish_old_roles(tmp_path):
    path, builder, _owner, _member, _proof, _outputs, _report = emit(tmp_path)
    path.write_text(path.read_text().replace('[1.0_8,2.0_8]', '[1.0_8,9.0_8]'))
    with pytest.raises(CompilationError, match='changed'):
        builder.finish()


def test_late_capture_callback_cannot_publish_a_removed_native_permit(tmp_path, monkeypatch):
    from compiler.scopes import storage_epoch

    _path, builder, owner, _member, proof, _outputs, _report = emit(tmp_path, opaque=True)
    managed = storage_epoch.managed_resources(owner)
    last = [root for root in owner.resources if root in managed][-1]
    access = next(access for member in owner.members.values() for unit in member.units
        for operation in unit.scope.native for access in operation.native_opaque_accesses.values())
    registry = access.permit._analysis._native_opaque_read_permits
    state = {'armed': False, 'mutated': False}
    original_inventory, original_capture = storage_epoch.resource_inventory, builder.capture_fact
    def inventory(current):
        result = original_inventory(current)
        state['armed'] = True
        return result
    def capture(binding):
        result = original_capture(binding)
        if state['armed'] and binding.root == last and not state['mutated']:
            registry.pop(access.permit.identity)
            state['mutated'] = True
        return result
    with monkeypatch.context() as patch:
        patch.setattr(storage_epoch, 'resource_inventory', inventory)
        patch.setattr(builder, 'capture_fact', capture)
        # A standalone lifecycle is not the final owner/publication authority.
        proof.validate()
    assert state['mutated']
    with pytest.raises(CompilationError, match='registered source permit'):
        owner.return_contract.validate()
    with pytest.raises(CompilationError, match='registered source permit'):
        builder.finish()
