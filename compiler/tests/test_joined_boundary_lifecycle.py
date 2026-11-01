"""Exact synchronous native groups can retire and rearm a checked child domain."""
from copy import copy, deepcopy
from dataclasses import replace

import pytest

from compiler.frontend.resource_reachability import prove_native_effect_graph, prove_resource_reachability
from compiler.ir import CompilationError
from compiler.scopes.lexical import LexicalBoundaryPatch
from compiler.tests.test_hierarchical_child_regions import SOURCE, fixture

GROUP = '''!$omp parallel do private(i) schedule(guided,5)
do i=1,n
if(right(lbound(right,1)+i-1)<0)call fatal("negative input")
call native_step(right(i+1))
enddo
!$omp end parallel do'''
JOINED = SOURCE.replace('implicit none', 'use iso_fortran_env,only:error_unit\nimplicit none', 1).replace(
    'if(change)call relocate(left,right)', GROUP).replace('end module', '''subroutine native_step(value)
real(8),intent(inout)::value
value=value+1
end subroutine
subroutine fatal(message)
character(*),intent(in)::message
write(error_unit,'(a)')message
stop
end subroutine
end module''')


def emit(tmp_path, source=JOINED):
    path, builder = fixture(tmp_path, source)
    outputs, report = builder.run()
    assert report['scope_count'] == 1, report.get('graph_owner_attempt', report.get('boundaries'))
    owner = builder._conditional_return_draft.validate().owner
    middle, = [member for member in owner.members.values() if member.routine.qualified == 'hierarchical::middle']
    patches = [patch for _first, _last, patch in middle.patches
               if isinstance(patch, LexicalBoundaryPatch) and patch.completion is not None]
    assert patches
    return path, builder, middle, patches, outputs, report


def test_authentic_child_group_retirement_preserves_checked_children_and_original_code(tmp_path):
    path, builder, middle, patches, outputs, report = emit(tmp_path)
    patch, = patches
    assert patch.lifecycle.preserves_descriptors()
    assert patch.lifecycle._joined_storage.selected_node_ids == patch.completion.selected_node_ids
    assert patch.completion.public()['completion_scope'] == 'normal return only'
    assert patch.validate(middle, require_source=True) is patch
    member, = [row for row in report['scopes'][0]['checked_execution_plan']['storage_epoch']['return_contract']['members']
               if row['procedure'] == 'hierarchical::middle']
    assert any(row['kind'] == 'reached_retirement' and row['rearm_allowed'] for row in member['routes'])
    assert any(row['kind'] == 'checked_child' and row['procedure'] == 'hierarchical::consumer' for row in member['routes'])
    generated = outputs[report['sources'][str(path)]['replacement']]
    assert GROUP in generated
    assert 'fort_scope_resident_close_v1' in generated
    assert 'write(error_unit' in generated.lower()
    assert builder.finish()


def test_demand_free_api_reuses_exact_native_graph_without_resource_demand_authority(tmp_path):
    _path, builder, middle, patches, _outputs, _report = emit(tmp_path)
    patch, = patches
    analysis, procedure = builder.analysis, middle.routine.qualified
    graph = prove_native_effect_graph(analysis, procedure, patch.originals)
    assert graph is patch.completion._graph
    binding = analysis.modules['hierarchical'].bindings['right']
    reached = prove_resource_reachability(analysis, procedure, patch.originals, (binding,))
    assert reached.native_graph is graph
    assert reached.native_graph.public() == graph.public()
    assert not reached.public()['execution_authorized']
    assert not graph.public()['alias_authority']


@pytest.mark.parametrize('bad', ['missing', 'copy', 'public', 'foreign', 'wrong_selection', 'no_storage'])
def test_joined_patch_requires_exact_independent_composed_proofs(tmp_path, bad):
    _path, builder, middle, patches, _outputs, _report = emit(tmp_path)
    patch, = patches
    if bad == 'missing':
        altered = replace(patch, completion=None)
    elif bad == 'copy':
        altered = replace(patch, completion=copy(patch.completion))
    elif bad == 'public':
        altered = replace(patch, completion=patch.completion.public())
    elif bad == 'foreign':
        other_path = tmp_path / 'other'
        other_path.mkdir()
        foreign = emit(other_path)[3][0].completion
        altered = replace(patch, completion=foreign)
    elif bad == 'wrong_selection':
        altered = replace(patch, originals=patch.originals[1:])
    else:
        altered = replace(patch, lifecycle=None)
    with pytest.raises(CompilationError):
        altered.validate(middle, require_source=True)
    assert patch.validate(middle, require_source=True) is patch
    assert builder._conditional_return_draft.owner.return_contract.validate()


def test_removed_completion_registration_rejects_sealed_owner(tmp_path):
    _path, builder, _middle, patches, _outputs, _report = emit(tmp_path)
    patch, = patches
    builder.analysis._native_graph_completions.pop(patch.completion.identity)
    with pytest.raises(CompilationError):
        builder._conditional_return_draft.owner.return_contract.validate()


@pytest.mark.parametrize('mutation', ['diagnostic', 'stop'])
def test_changed_original_terminal_source_cannot_publish_a_sealed_owner(tmp_path, mutation):
    path, builder, _middle, _patches, _outputs, _report = emit(tmp_path)
    source = path.read_text()
    path.write_text(source.replace('negative input', 'different failure') if mutation == 'diagnostic'
                    else source.replace('\nstop\n', '\nreturn\n'))
    with pytest.raises(CompilationError, match='changed'):
        builder.finish()


@pytest.mark.parametrize('source', [
    JOINED.replace('call native_step(right(i+1))', 'call unknown()'),
    JOINED.replace('call native_step(right(i+1))', '!$omp task\ncall native_step(right(i))\n!$omp end task'),
    JOINED.replace('!$omp end parallel do', '!$omp end parallel do nowait'),
])
def test_unknown_or_asynchronous_native_group_keeps_conditional_owner_closed(tmp_path, source):
    _path, builder = fixture(tmp_path, source)
    _outputs, report = builder.run()
    assert report['scope_count'] == 0


def test_authentic_existing_v2_transport_is_accepted_without_new_authority_fields(tmp_path):
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    _path, _builder, _middle, _patches, _outputs, report = emit(tmp_path)
    original = deepcopy(report)
    validate_checked_execution_plans(report)
    assert report == original
    returned = report['scopes'][0]['checked_execution_plan']['storage_epoch']['return_contract']
    assert returned['version'] == 2
    assert not returned['cross_cut_replica_retention_authorized']
    assert not returned['payload_access_authorized']


def test_other_complete_group_in_same_procedure_cannot_lend_completion(tmp_path):
    _path, _builder, middle, patches, _outputs, _report = emit(tmp_path, JOINED.replace(GROUP, GROUP+'\n'+GROUP))
    first, second = patches
    assert first.completion.selected_node_ids != second.completion.selected_node_ids
    with pytest.raises(CompilationError, match='exact team and completion'):
        replace(first, completion=second.completion).validate(middle, require_source=True)


def test_same_group_storage_team_cannot_be_replaced_by_an_unregistered_copy(tmp_path):
    _path, builder, middle, patches, _outputs, _report = emit(tmp_path)
    patch, = patches
    team = patch.lifecycle._joined_storage
    builder.analysis._descriptor_team_proofs[team.identity] = copy(team)
    with pytest.raises(CompilationError):
        patch.validate(middle, require_source=True)


def ordinary_source():
    group = ('!$omp parallel do private(i)\ndo i=1,n\nright(modulo(i,n)+2)=left(i-4)\n'
             'enddo\n!$omp end parallel do\nif(change)call relocate(left,right)')
    return JOINED.replace(GROUP, group)


def force_ordinary_retirement(monkeypatch):
    """Exercise the real fallback after a diagnostic-only admission decline.

    No source/proof operation is mocked: ordinary completion, native graph,
    team storage, emission, sealing and strict transport run normally.
    """
    from compiler.scopes.lexical import LexicalOwner
    admit = LexicalOwner.admit
    reached = []
    def decline(member, nodes, **options):
        if (member.routine.qualified == 'hierarchical::middle' and len(nodes) == 3
                and 'modulo' in str(nodes).lower() and str(nodes[0]).lower().startswith('!$omp parallel do')):
            reached.append(tuple(nodes))
            member.boundary(tuple(nodes), 'diagnostic ordinary-group unit admission unavailable')
            return None
        return admit(member, nodes, **options)
    monkeypatch.setattr(LexicalOwner, 'admit', decline)
    return reached


def test_ordinary_joined_no_call_group_is_retained_through_authentic_child_emission(tmp_path, monkeypatch):
    reached = force_ordinary_retirement(monkeypatch)
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from fparser.two.utils import walk
    _path, _builder, middle, patches, _outputs, report = emit(tmp_path, ordinary_source())
    patch, = patches
    assert reached
    assert not any(type(node).__name__ == 'Call_Stmt' for original in patch.originals for node in walk(original))
    assert not patch.completion._graph.deferred_completion
    identity = patch.completion.public()['retained_ordinary_completion_identity']
    retained = [proof for template in patch.completion._graph.templates
                for _name, _selected, proof in template.completion]
    assert identity in {proof.identity for proof in retained}
    assert patch.validate(middle, require_source=True) is patch
    assert patch.lifecycle.preserves_descriptors()
    validate_checked_execution_plans(report)


def test_ordinary_retained_join_registration_is_checked_at_sealed_consumption(tmp_path, monkeypatch):
    force_ordinary_retirement(monkeypatch)
    _path, builder, middle, patches, _outputs, _report = emit(tmp_path, ordinary_source())
    patch, = patches
    identity = patch.completion.public()['retained_ordinary_completion_identity']
    builder.analysis._joined_completions.pop(identity)
    with pytest.raises(CompilationError):
        patch.validate(middle, require_source=True)


@pytest.mark.parametrize('kind', ['modular', 'record'])
def test_unforced_ordinary_sources_use_coherent_native_units_without_retirement(tmp_path, kind):
    source = ordinary_source()
    if kind == 'record':
        source = (source.replace('right(modulo(i,n)+2)=left(i-4)', 'right(i+1)=points(1)%value')
                  .replace('real(8)::observed', 'type point\nreal(8)::value\nend type\n'
                           'type(point)::points(1)\nreal(8)::observed')
                  .replace('call producer(right,n)', 'points(1)%value=1.0_8\ncall producer(right,n)'))
    _path, builder = fixture(tmp_path, source)
    _outputs, report = builder.run()
    assert report['scope_count'] == 1
    owner = builder._conditional_return_draft.validate().owner
    middle, = [m for m in owner.members.values() if m.routine.qualified == 'hierarchical::middle']
    units = [unit for unit in middle.units if str(unit.nodes[0]).startswith('!$omp parallel do')]
    unit, = units
    assert unit.scope.native
    assert not unit.scope.calls
    assert all(patch.completion is None for _first, _last, patch in middle.patches
               if isinstance(patch, LexicalBoundaryPatch))


@pytest.mark.parametrize('kind', ['foreign', 'other_group'])
def test_ordinary_forced_retirement_cannot_borrow_another_join(tmp_path, monkeypatch, kind):
    force_ordinary_retirement(monkeypatch)
    source = ordinary_source()
    if kind == 'other_group':
        group = source[source.index('!$omp parallel do private(i)\ndo i=1,n\nright(modulo'):]
        group = group[:group.index('!$omp end parallel do')+len('!$omp end parallel do')]
        source = source.replace(group, group+'\n'+group)
    _path, _builder, middle, patches, _outputs, _report = emit(tmp_path, source)
    patch = patches[0]
    if kind == 'foreign':
        other = tmp_path / 'other'
        other.mkdir()
        substitute = emit(other, ordinary_source())[3][0].completion
    else:
        assert len(patches) == 2
        substitute = patches[1].completion
    with pytest.raises(CompilationError):
        replace(patch, completion=substitute).validate(middle, require_source=True)
