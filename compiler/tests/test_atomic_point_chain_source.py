"""Original GPU/host predicate/GPU sequence retains one canonical point buffer."""
import json
from hashlib import sha256

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_atomic_point_predicates import point_source
from compiler.tests.test_source_scopes import FACT


def chain_source():
    source = point_source().replace('step(a,b,p,n,mode,flag,gate)',
                                   'exercise(a,b,p,n,mode,flag,gate,produce)')
    source = source.replace('integer::i', 'logical,intent(in)::produce\ninteger::i')
    producer = """if(produce)then
!$omp parallel do private(i)
do i=-1,-1
gate(i)=mode
enddo
!$omp end parallel do
endif
"""
    consumer = """!$omp parallel do private(i)
do i=1,n
a(i)=a(i)+real(gate(-1),8)
enddo
!$omp end parallel do
"""
    source = source.replace('!$omp parallel private(i)', producer + '!$omp parallel private(i)')
    return source.replace('!$omp end parallel\n', '!$omp end parallel\n' + consumer)


def emit_chain(tmp_path):
    path = tmp_path / 'point_chain.f90'
    path.write_text(chain_source())
    facts = {'schema_version': 1, 'participation': 'serial',
        'sources': {str(path): sha256(path.read_bytes()).hexdigest()},
        'captures': {**{'argument::' + name: dict(FACT) for name in ('a', 'b', 'p')},
            'argument::gate': {**FACT, 'initialized': 'none', 'lifetime': 'invocation'}}}
    builder = ScopeBuilder([path], 'atomic_owner::exercise', facts=facts,
        options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy='sections', scope_execution='graph', host_threads=4))
    outputs, report = builder.run()
    (tmp_path / "source-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return outputs, report, facts


def test_source_point_chain_uses_one_owner_without_inventing_entry_definitions(tmp_path):
    outputs, report, facts = emit_chain(tmp_path)
    assert report['scope_count'] == 1, report
    scope, = report['scopes']
    task, = scope['source_device_tasks']
    assert task['kind'] == 'original_atomic_conditional_joined_team'
    procedures = scope['checked_execution_plan']['procedures']
    assert [p['procedure'] for p in procedures] == ['atomic_owner::exercise']
    assert len(scope['checked_execution_plan']['gpu_leaves']) == 2
    assert facts['captures']['argument::gate']['initialized'] == 'none'
    assert facts['captures']['argument::gate']['lifetime'] == 'invocation'
    texts = '\n'.join(text for name, text in outputs.items() if name.endswith('.f90'))
    assert "atomic predicate source admission failed fatally" in texts
    assert "atomic predicate acquisition failed fatally" in texts
    assert 'fort_scope_admit_source_reads_v1' in texts
