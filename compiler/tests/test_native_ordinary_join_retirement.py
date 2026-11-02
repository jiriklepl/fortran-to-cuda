"""A whole original multi-DO join remains independent of GPU scratch facts."""
import inspect
import json
from copy import copy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.native_completion import prove_graph_joined_completion
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.resource_reachability import prove_native_effect_graph
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

SOURCE = 'module scratch_team\nimplicit none\nreal(8),allocatable::u(:,:,:),a(:,:,:),b(:,:,:)\ncontains\nsubroutine advance(n)\ninteger,intent(in)::n\ncall producer(a,u,n)\ncall diffusion(a,u,n)\ncall consumer(b,a,n)\nend subroutine\nsubroutine producer(a,u,n)\ninteger,intent(in)::n\nreal(8),intent(inout)::a(-2:,-2:,-2:),u(-2:,-2:,-2:)\ninteger::i,j,k\ndo k=1,n\ndo j=1,n\ndo i=1,n\na(i,j,k)=real(i+j+k,8)\nu(i,j,k)=real(i+j+k,8)**2\nenddo\nenddo\nenddo\nend subroutine\nsubroutine consumer(b,a,n)\ninteger,intent(in)::n\nreal(8),intent(inout)::b(-2:,-2:,-2:)\nreal(8),intent(in)::a(-2:,-2:,-2:)\ninteger::i,j,k\ndo k=1,n\ndo j=1,n\ndo i=1,n\nb(i,j,k)=2*a(i,j,k)\nenddo\nenddo\nenddo\nend subroutine\nsubroutine diffusion(a,u,n)\ninteger,intent(in)::n\nreal(8),intent(inout)::a(-2:,-2:,-2:)\nreal(8),intent(in)::u(-2:,-2:,-2:)\nreal(8),allocatable::fl(:,:,:)\ninteger::i,j,k\nif(.not.allocated(fl))allocate(fl(-1:n+2,-1:n+2,-1:n+2))\n!$omp parallel private(i,j,k)\n!$omp do schedule(runtime) collapse(3)\ndo k=1,n\ndo j=1,n\ndo i=0,n+1\nfl(i,j,k)=u(i,j,k)-u(i-1,j,k)\nenddo\nenddo\nenddo\n!$omp end do\n!$omp do schedule(runtime) collapse(3)\ndo k=1,n\ndo j=1,n\ndo i=1,n\na(i,j,k)=a(i,j,k)+(fl(i+1,j,k)-fl(i,j,k))\nenddo\nenddo\nenddo\n!$omp end do\n!$omp end parallel\nend subroutine\nend module\n'


def fixture(directory, *, automatic=False, include=False, source=SOURCE):
    directory.mkdir(parents=True, exist_ok=True)
    text=source
    if automatic:
        text=text.replace('real(8),allocatable::fl(:,:,:)','real(8)::fl(-1:n+2,-1:n+2,-1:n+2)')
        text=text.replace('if(.not.allocated(fl))allocate(fl(-1:n+2,-1:n+2,-1:n+2))','')
    original=directory/'team.f90'
    configured=None
    if include:
        part=text[text.index('!$omp do schedule(runtime) collapse(3)',text.index('!$omp end do')+1):text.index('!$omp end parallel')]
        text=text.replace(part,'#include "divergence.inc"\n')
        inc=directory/'divergence.inc'
        inc.write_text(part)
    original.write_text(text)
    hashes={str(original):sha256(original.read_bytes()).hexdigest()}
    if include:
        lines=[]
        mapping=[]
        for number,line in enumerate(text.splitlines(),1):
            if line.startswith('#include'):
                lines.append('')
                mapping.append(number)
                lines.extend(part.splitlines())
                mapping.extend([None]*len(part.splitlines()))
            else:
                lines.append(line)
                mapping.append(number)
        prepared=directory/'configured.f90'
        prepared.write_text('\n'.join(lines)+'\n')
        configured={'schema_version':1,'source_inputs':hashes,'preserves_source_order':True,
            'configuration':{'defines':[]},'dependencies':{str(inc):sha256(inc.read_bytes()).hexdigest()},
            'entries':[{'source':str(original),'path':str(prepared),
                'sha256':sha256(prepared.read_bytes()).hexdigest(),'line_map':mapping}]}
    fact={'storage':'stable','initialized':'whole','escapes':False,'allocation_changes':False}
    facts={'schema_version':1,'participation':'serial','sources':hashes,
        'captures':{'scratch_team::'+name:dict(fact) for name in ('u','a','b')}}
    builder=ScopeBuilder([original],'scratch_team::advance',facts=facts,options=CompilerOptions(),
        config=OffloadConfig(policy='sections',scope_execution='graph'),analysis_sources=configured,
        contracts={'scratch_team::advance':{'native_ancestor':{'version':1,
            'device_roots':['scratch_team::producer','scratch_team::consumer']}}})
    routine=builder.analysis.routines['scratch_team::diffusion']
    group,=[g for g in grouped_original_sequence(routine.execution.content,{}) if isinstance(g,NativeGroupCandidate)]
    return original,builder,routine,group


def issue(parts):
    _path,builder,routine,group=parts
    analysis=builder.analysis
    native=prove_native_effect_graph(analysis,routine.qualified,group.original_nodes)
    proof=prove_graph_joined_completion(analysis,routine.qualified,group.original_nodes,native)
    return native,proof


@pytest.mark.parametrize('automatic',[False,True])
@pytest.mark.parametrize('include',[False,True])
def test_ordinary_multido_retirement_keeps_host_scratch_and_raw_source(tmp_path,automatic,include):
    path,builder,routine,group=fixture(tmp_path,automatic=automatic,include=include)
    native,completion=issue((path,builder,routine,group))
    ordinary=completion._ordinary_join
    assert ordinary is not None
    assert completion.public()['retained_ordinary_completion_identity']==ordinary.identity
    assert completion._call_storage is None
    assert completion._reduction_teams==()
    assert completion.public()['native_only']
    assert completion.public()['internal_cuts_authorized'] is False
    assert completion.public()['gpu_legality_established'] is False
    team=prove_descriptor_team(builder.analysis,routine.qualified,group.original_nodes)
    storage=prove_storage_mutation_effects(builder.analysis,routine.qualified,
        selection=group.original_nodes,joined_storage=team)
    assert storage.public()['complete']
    assert not storage.public()['escapes']
    assert 'scratch_team::diffusion::fl' not in builder.facts['captures']
    assert completion.validate(builder.analysis,routine.qualified,completion.selected_node_ids) is completion
    # Fresh owner: probe issuance is not used to prime normal source admission.
    path,fresh,_routine,_group=fixture(tmp_path/'fresh',automatic=automatic,include=include)
    outputs,report=fresh.run()
    assert report['scope_count']==1,report.get('graph_owner_attempt')
    scope,=report['scopes']
    assert {name.split('#')[0] for name in scope['gpu_leaves']}=={'scratch_team::producer','scratch_team::consumer'}
    assert all(row['resource']!='scratch_team::diffusion::fl'
        for row in scope['checked_execution_plan']['storage_epoch']['resources'])
    transformed=outputs[report['sources'][str(path)]['replacement']]
    assert transformed.count('#include "divergence.inc"')==int(include)
    assert transformed.count('!$omp parallel private(i,j,k)')==1
    assert transformed.count('!$omp end parallel')==1
    from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
    from elmm_pipeline.local_adapter import _scope_package_build
    package=tmp_path/'package'
    for name,text in outputs.items():
        target=package/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(text)
    manifest=json.loads((package/'scope-manifest.json').read_text())
    validate_checked_execution_plans(manifest)
    _scope_package_build({'artifact_root':str(package),'scope_manifest':manifest})


@pytest.mark.parametrize('change',['remove','copy','foreign','record','template','graph','role'])
def test_ordinary_join_cannot_be_replaced_or_borrowed(tmp_path,change):
    parts=fixture(tmp_path)
    native,proof=issue(parts)
    _path,builder,routine,_group=parts
    analysis=builder.analysis
    ordinary=proof._ordinary_join
    if change=='remove':
        del analysis._joined_completions[ordinary.identity]
    elif change=='copy':
        analysis._joined_completions[ordinary.identity]=copy(ordinary)
    elif change=='foreign':
        foreign=issue(fixture(tmp_path/'foreign'))[1]
        object.__setattr__(proof,'_ordinary_join',foreign._ordinary_join)
    elif change=='record':
        object.__setattr__(ordinary,'_record',{})
    elif change=='template':
        object.__setattr__(native.templates[0],'completion',())
    elif change=='graph':
        object.__setattr__(proof,'_graph',issue(fixture(tmp_path/'foreign'))[0])
    else:
        object.__setattr__(proof,'_ordinary_join',None)
    with pytest.raises(CompilationError):
        proof.validate(analysis,routine.qualified,proof.selected_node_ids)


@pytest.mark.parametrize('change',['remove','registry','record','template','role'])
def test_final_publication_callback_cannot_change_ordinary_join(tmp_path,monkeypatch,change):
    _path,builder,routine,group=fixture(tmp_path)
    analysis=builder.analysis
    native=prove_native_effect_graph(analysis,routine.qualified,group.original_nodes)
    ordinary,=[p for template in native.templates for name,_ids,p in template.completion if name==routine.qualified]
    original=analysis.verify_publication
    changed=[]
    def late():
        result=original()
        if inspect.currentframe().f_back.f_code.co_name=='_publish_graph_completion' and not changed:
            if change=='remove':
                del analysis._joined_completions[ordinary.identity]
            elif change=='registry':
                analysis._joined_completions=dict(analysis._joined_completions)
            elif change=='record':
                object.__setattr__(ordinary,'_record',{})
            elif change=='template':
                object.__setattr__(native.templates[0],'completion',())
            else:
                registry=analysis._native_graph_completions
                proof,=registry.values()
                object.__setattr__(proof,'_ordinary_join',None)
            changed.append(True)
        return result
    monkeypatch.setattr(analysis,'verify_publication',late)
    with pytest.raises(CompilationError):
        prove_graph_joined_completion(analysis,routine.qualified,group.original_nodes,native)
    assert changed
