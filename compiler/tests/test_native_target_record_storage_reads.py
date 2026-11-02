"""TARGET record reads are storage-only after exact whole-group retirement."""

import pytest

from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.native_lifetime_context import NativeLifetimeContext
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.tests.test_native_ordinary_join_retirement import SOURCE, fixture, issue

POINTS = '''type point
integer::xi,yj,zk
real(8)::fluxp
end type
type(point),allocatable,target::points(:)
'''
WALL = '''!$omp do
do ip=1,size(points)
i=points(ip)%xi
j=points(ip)%yj
k=points(ip)%zk
a(i,j,k)=a(i,j,k)+points(ip)%fluxp
enddo
!$omp end do
'''


def source(action=''):
    return (SOURCE.replace('contains\n', POINTS+'contains\n', 1)
        .replace('integer::i,j,k\nif(.not.allocated(fl))', 'integer::i,j,k,ip\nif(.not.allocated(fl))')
        .replace('!$omp end parallel', WALL+action+'\n!$omp end parallel'))


def project(parts, *, context=True):
    _path,builder,routine,group=parts
    native,completion=issue(parts)
    team=prove_descriptor_team(builder.analysis,routine.qualified,group.original_nodes)
    storage=prove_storage_mutation_effects(builder.analysis,routine.qualified,selection=group.original_nodes,
        joined_storage=team,native_graph=native if context else None)
    return native,completion,team,storage


def test_exact_joined_target_record_reads_carry_no_managed_authority(tmp_path):
    parts=fixture(tmp_path,source=source())
    native,completion,team,proof=project(parts)
    _path,builder,routine,_group=parts
    public=proof.public()
    assert public['complete'],public['frontiers']
    assert not public['escapes']
    assert not public['mutations']
    role=public['native_lifetime_context']
    assert role['whole_original_group_retirement_required']
    assert role['completion_identity']==completion.identity
    assert role['storage_team_identity']==team.identity
    assert role['retirement_required']
    for flag in ('payload_access_authorized','entry_definition_authorized','alias_authorized',
                 'ownership_authorized','capture_authorized','physical_sections_authorized'):
        assert role[flag] is False
    assert proof._native_completion is completion
    assert proof.validate() is proof
    context=NativeLifetimeContext(builder.analysis,native,joined_storage=team)
    points=builder.analysis.modules['scratch_team'].bindings['points']
    assert context.readonly_target(points)
    assert not context.binding(points)
    assert 'scratch_team::points' not in builder.facts['captures']
    assert 'scratch_team::diffusion::fl' not in builder.facts['captures']


def test_default_joined_storage_cannot_borrow_target_read_role(tmp_path):
    _native,_completion,_team,proof=project(fixture(tmp_path,source=source()),context=False)
    assert not proof.public()['complete']
    assert proof._native_completion is None


@pytest.mark.parametrize('action',[
    'points(1)%fluxp=0',
    'points=points',
    'deallocate(points)',
    'call unknown(points)',
])
def test_full_native_body_still_rejects_target_writes_mutations_and_escapes(tmp_path,action):
    parts=fixture(tmp_path,source=source(action))
    try:
        _native,_completion,_team,proof=project(parts)
    except CompilationError:
        return
    assert not proof.public()['complete'] or proof.public()['escapes']


def test_target_owner_retires_whole_raw_include_before_reentry(tmp_path):
    path,builder,_routine,_group=fixture(tmp_path,source=source(),include=True)
    outputs,report=builder.run()
    assert report['scope_count']==1,report.get('graph_owner_attempt')
    scope,=report['scopes']
    assert {name.split('#')[0] for name in scope['gpu_leaves']}=={'scratch_team::producer','scratch_team::consumer'}
    assert all(row['resource'] not in {'scratch_team::points','scratch_team::diffusion::fl'}
        for row in scope['checked_execution_plan']['storage_epoch']['resources'])
    text=outputs[report['sources'][str(path)]['replacement']]
    assert text.count('#include "divergence.inc"')==1
    assert text.count('!$omp parallel private(i,j,k)')==1
    assert text.count('!$omp end parallel')==1
    import json

    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build
    package=tmp_path/'package'
    for name,content in outputs.items():
        target=package/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(content)
    manifest=json.loads((package/'scope-manifest.json').read_text())
    validate_checked_execution_plans(manifest)
    _scope_package_build({'artifact_root':str(package),'scope_manifest':manifest})


@pytest.mark.parametrize('change',['missing_completion','copied_team','foreign_graph','mutated_team'])
def test_group_context_requires_exact_original_receipts(tmp_path,change):
    from copy import copy

    from compiler.ir import CompilationError
    parts=fixture(tmp_path,source=source())
    native,completion,team,_proof=project(parts)
    _path,builder,_routine,_group=parts
    if change=='missing_completion':
        del builder.analysis._native_graph_completions[completion.identity]
    elif change=='copied_team':
        team=copy(team)
    elif change=='foreign_graph':
        native=project(fixture(tmp_path/'foreign',source=source()))[0]
    else:
        object.__setattr__(team,'_originals',team._originals[:-1])
    with pytest.raises(CompilationError):
        NativeLifetimeContext(builder.analysis,native,joined_storage=team)


@pytest.mark.parametrize('change',['ordinary_record','ordinary_role','completion_record','source','declaration','storage_role'])
def test_cached_storage_revalidates_required_native_group_role(tmp_path,change):
    from compiler.ir import CompilationError
    parts=fixture(tmp_path,source=source())
    _native,completion,_team,proof=project(parts)
    if change=='ordinary_record':
        object.__setattr__(completion._ordinary_join,'_record',{})
    elif change=='ordinary_role':
        object.__setattr__(completion,'_ordinary_join',None)
    elif change=='completion_record':
        object.__setattr__(completion,'_record',{})
    elif change=='source':
        path=parts[0]
        path.write_text(path.read_text()+'! source changed\n')
    elif change=='declaration':
        root=parts[1].analysis.modules['scratch_team'].bindings['points']
        object.__setattr__(root,'attributes',frozenset({'pointer'}))
    else:
        object.__setattr__(proof,'_native_completion',None)
    def revalidate():
        if change=='source':
            parts[1].analysis.verify_publication()
        return proof.validate()
    with pytest.raises(CompilationError):
        revalidate()


@pytest.mark.parametrize('context_kind',['absent','fake','foreign'])
def test_target_declaration_service_does_not_grant_generic_target_permission(tmp_path,context_kind):
    from types import SimpleNamespace

    from compiler.frontend.storage_component_reads import record_array_declaration
    from compiler.ir import CompilationError
    parts=fixture(tmp_path,source=source())
    _native,_completion,_team,_proof=project(parts)
    analysis=parts[1].analysis
    points=analysis.modules['scratch_team'].bindings['points']
    context=None
    if context_kind=='fake':
        context=SimpleNamespace(readonly_target=lambda binding:True)
    elif context_kind=='foreign':
        foreign=fixture(tmp_path/'foreign',source=source())
        native,_completion,team,_proof=project(foreign)
        context=NativeLifetimeContext(foreign[1].analysis,native,joined_storage=team)
    with pytest.raises(CompilationError):
        record_array_declaration(analysis,points,retain_scope=lambda scope:None,
            tick=lambda:None,_storage_context=context)


@pytest.mark.parametrize('change',['ordinary_record','target_declaration','component_schema'])
def test_cached_fact_callback_cannot_change_completion_after_last_dependency_check(tmp_path,monkeypatch,change):
    from compiler.ir import CompilationError
    parts=fixture(tmp_path,source=source())
    analysis=parts[1].analysis
    if getattr(analysis,'_source_session',None) is None:
        analysis.seal_session()
    _native,completion,_team,proof=project(parts)
    fact=proof._source_fact
    assert fact is not None
    original=type(fact).require
    changed=[]
    def late(current,token):
        result=original(current,token)
        if token is proof and not changed:
            if change=='ordinary_record':
                object.__setattr__(completion._ordinary_join,'_record',{})
            elif change=='target_declaration':
                root=analysis.modules['scratch_team'].bindings['points']
                object.__setattr__(root,'attributes',frozenset({'pointer'}))
            else:
                schema=analysis.modules['scratch_team'].component_types['point']
                object.__setattr__(schema,'members',dict(schema.members))
            changed.append(True)
        return result
    monkeypatch.setattr(type(fact),'require',late)
    with pytest.raises(CompilationError):
        proof.validate()
    assert changed


@pytest.mark.parametrize('action',['pointer','c_loc','task'])
def test_association_address_export_and_async_work_do_not_gain_storage_role(tmp_path,action):
    body={'pointer':'alias=>points','c_loc':'address=c_loc(points(1))',
          'task':'!$omp task\na(1,1,1)=0\n!$omp end task'}[action]
    text=source(body)
    text=text.replace('module scratch_team\nimplicit none',
        'module scratch_team\nuse iso_c_binding,only:c_loc,c_ptr\nimplicit none')
    text=text.replace('integer::i,j,k,ip\n','integer::i,j,k,ip\ntype(point),pointer::alias(:)\ntype(c_ptr)::address\n')
    parts=fixture(tmp_path,source=text)
    with pytest.raises(CompilationError):
        project(parts)


@pytest.mark.parametrize('change',['declaration','schema','completion'])
def test_final_storage_source_callback_cannot_replace_retirement_dependencies(tmp_path,monkeypatch,change):
    import inspect
    parts=fixture(tmp_path,source=source())
    _path,builder,routine,group=parts
    native,completion=issue(parts)
    team=prove_descriptor_team(builder.analysis,routine.qualified,group.original_nodes)
    analysis=builder.analysis
    original=type(analysis.inputs).verify_live
    changed=[]
    def late(inputs,*args,**kwargs):
        result=original(inputs,*args,**kwargs)
        if inspect.currentframe().f_back.f_code.co_name=='_publish_derivation' and not changed:
            if change=='declaration':
                root=analysis.modules['scratch_team'].bindings['points']
                object.__setattr__(root,'attributes',frozenset({'pointer'}))
            elif change=='schema':
                schema=analysis.modules['scratch_team'].component_types['point']
                object.__setattr__(schema,'members',dict(schema.members))
            else:
                object.__setattr__(completion,'_ordinary_join',None)
            changed.append(True)
        return result
    monkeypatch.setattr(type(analysis.inputs),'verify_live',late)
    with pytest.raises(CompilationError):
        prove_storage_mutation_effects(analysis,routine.qualified,selection=group.original_nodes,
            joined_storage=team,native_graph=native)
    assert changed


@pytest.mark.parametrize('field',[
    '_native_completion','_native_completion_fields','_native_read_dependencies',
    '_native_graph','_joined_storage','_source_fact',
])
@pytest.mark.parametrize('change',['clear','replace'])
def test_cached_fact_callback_cannot_clear_or_replace_storage_role(tmp_path,monkeypatch,field,change):
    from copy import copy
    parts=fixture(tmp_path,source=source())
    _native,_completion,_team,proof=project(parts)
    fact=proof._source_fact
    assert fact is not None
    original=type(fact).require
    changed=[]
    def late(current,token):
        result=original(current,token)
        if token is proof and not changed:
            old=getattr(proof,field)
            value=None if change=='clear' else tuple(list(old)) if isinstance(old,tuple) else copy(old)
            object.__setattr__(proof,field,value)
            changed.append(True)
        return result
    monkeypatch.setattr(type(fact),'require',late)
    with pytest.raises(CompilationError):
        proof.validate()
    assert changed


@pytest.mark.parametrize('timing',['before','after'])
@pytest.mark.parametrize('change',['originals','selection','registry_entry','registry_object'])
def test_cached_storage_retains_team_fields_and_registry(tmp_path,monkeypatch,timing,change):
    parts=fixture(tmp_path,source=source())
    _native,_completion,team,proof=project(parts)
    analysis=parts[1].analysis
    changed=[]
    def mutate():
        if change=='originals':
            object.__setattr__(team,'_originals',team._originals[:-1])
        elif change=='selection':
            object.__setattr__(team,'selected_node_ids',team.selected_node_ids[:-1])
        elif change=='registry_entry':
            del analysis._descriptor_team_proofs[team.identity]
        else:
            analysis._descriptor_team_proofs=dict(analysis._descriptor_team_proofs)
        changed.append(True)
    if timing=='before':
        mutate()
    else:
        fact=proof._source_fact
        original=type(fact).require
        def late(current,token):
            result=original(current,token)
            if token is proof and not changed:
                mutate()
            return result
        monkeypatch.setattr(type(fact),'require',late)
    with pytest.raises(CompilationError):
        proof.validate()
    assert changed
