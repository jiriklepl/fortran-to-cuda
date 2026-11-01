"""Source-bound legality data cannot escape into unconditional execution."""
from dataclasses import replace
import json

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.atomic_worksharing_team import atomic_worksharing_alternatives
from compiler.frontend.conditional_numerical import prove_conditional_numerical_context
from compiler.frontend.numerical_effects import prove_numerical_effects
from compiler.frontend.numerical_preparation import NumericalPreparationProvider
from compiler.frontend.reached_read_requirements import prove_reached_read_requirements
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.tests.test_conditional_sections import ENTRY, SOURCE, issue


def prepared(tmp_path, text=SOURCE):
    analysis, routine, nodes, team = issue(tmp_path, text)
    provider = NumericalPreparationProvider(analysis, options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy='sections', host_threads=4), environment_root=ENTRY)
    context = prove_conditional_numerical_context(provider, team)
    alternative = atomic_worksharing_alternatives(team)[0]
    result = provider.prepare_atomic_alternative(alternative, conditional_context=context)
    return analysis, routine, nodes, team, provider, context, alternative, result


def test_context_prepares_original_shifted_copies_but_does_not_grant_execution(tmp_path):
    analysis, routine, nodes, team, provider, context, alternative, result = prepared(tmp_path)
    record = context.public()
    assert record['model_units'] == 2
    assert record['paths'][0]['original_units'] == [0, 1]
    assert record['paths'][1] == {'original_units': [], 'domain': None}
    assert record['execution_authorized'] is False
    assert record['source_read_admission_authorized'] is False
    assert record['definition_transfer_authorized'] is False
    assert result.validate(analysis, ENTRY, nodes) is result
    assert result.conditional_context is context
    assert result.public()['conditional_numerical_context'] == record
    assert result.completion_proof is alternative
    assert len(result.plan.regions) == 2
    domain = context.domain(alternative)
    assert domain.accepts({name: 8 if symbol.name == 'n' else -2 for name, symbol in domain.parameters})
    effects = prove_numerical_effects(analysis, ENTRY, nodes, result)
    reads = prove_reached_read_requirements(analysis, ENTRY, nodes, result)
    assert reads.public()['read_admission_requires_same_conditional_guard'] is True
    assert reads.public()['conditional_numerical_context_identity'] == context.identity
    assert effects.public()['conditional_numerical_context_identity'] == context.identity
    with pytest.raises(CompilationError, match='no generic definition transfer'):
        effects.transfer()
    from compiler.emission.cuda.atomic_team_provider import emit_atomic_team_provider
    emitted = emit_atomic_team_provider(alternative, result, effects, reads)
    assert emitted.public()['conditional_numerical_context_identity'] == context.identity
    assert emitted.cuda_source.index('if (!conditional_legal)') < emitted.cuda_source.index('fort_scope_admit_source_reads_v1(')
    assert emitted.cuda_source.index('fort_scope_admit_source_reads_v1(') < emitted.cuda_source.index('return fort_scope_source_execute_v1(')


@pytest.mark.parametrize('change', ['registry', 'owner', 'identity', 'domain', 'parameter', 'source', 'provider', 'bank'])
def test_changed_conditional_authority_rejects(tmp_path, change):
    analysis, routine, nodes, team, provider, context, alternative, result = prepared(tmp_path)
    if change == 'registry':
        analysis._conditional_numerical_contexts.pop(context.identity)
    elif change == 'owner':
        analysis._conditional_numerical_owner = object()
    elif change == 'identity':
        old = context.identity
        object.__setattr__(context, 'identity', '1' * 64)
        analysis._conditional_numerical_contexts.pop(old)
        analysis._conditional_numerical_contexts[context.identity] = context
    elif change == 'domain':
        object.__setattr__(context._domains[0], 'clauses', ())
    elif change == 'parameter':
        object.__setattr__(context._function.parameters[0], 'name', 'foreign')
    elif change == 'source':
        object.__setattr__(routine.scope.bindings['n'], 'kind', 8)
    elif change == 'provider':
        object.__setattr__(context, 'provider', object())
    else:
        object.__setattr__(context._bank, 'function', replace(context._function, name='foreign'))
    with pytest.raises(CompilationError):
        context.validate(provider)


def test_context_does_not_accept_foreign_alternative_or_untyped_guard(tmp_path):
    analysis, routine, nodes, team, provider, context, alternative, result = prepared(tmp_path)
    with pytest.raises(CompilationError, match='exact typed source context'):
        provider.prepare_atomic_alternative(alternative, conditional_context=context.public())
    empty = atomic_worksharing_alternatives(team)[1]
    with pytest.raises(CompilationError):
        provider.prepare_atomic_alternative(empty, conditional_context=context)
    with pytest.raises(CompilationError):
        provider.prepare_atomic_alternative(alternative)


@pytest.mark.parametrize('expression', ['abs(a(i-n))', 'merge(a(i-n),a(i),i>0)'])
def test_rhs_source_read_semantics_are_not_broadened(tmp_path, expression):
    with pytest.raises(CompilationError):
        prepared(tmp_path, SOURCE.replace('a(i)=a(i-n)', 'a(i)='+expression))


@pytest.mark.parametrize('key,value', [
    ('execution_authorized', True), ('source_read_admission_authorized', True),
    ('source_graph', '1'*64), ('source_work_budget', 65537),
    ('isl_operation_budget', 1000001), ('payload_identity', '2'*64),
])
def test_resealed_record_cannot_change_source_or_guard_authority(tmp_path, key, value):
    from hashlib import sha256
    from compiler.frontend.conditional_numerical import _encoded

    analysis, _, _, _, provider, context, _, _ = prepared(tmp_path)
    old = context.identity
    record = json.loads(context._record)
    record[key] = value
    encoded = _encoded(record)
    object.__setattr__(context, '_record', encoded)
    object.__setattr__(context, 'identity', sha256(encoded.encode()).hexdigest())
    analysis._conditional_numerical_contexts.pop(old)
    analysis._conditional_numerical_contexts[context.identity] = context
    with pytest.raises(CompilationError):
        context.validate(provider)


def test_all_sixteen_original_paths_share_four_exact_unit_models(tmp_path, monkeypatch):
    import compiler.analysis.conditional_affine as model

    body = ''.join('if(flag)then\n!$omp sections\n!$omp section\n'
                   + f'do i=n+{index*4+1},n+{index*4+3}\na(i)=a(i-n)\nenddo\n'
                   + '!$omp end sections nowait\nendif\n' for index in range(4))
    text = SOURCE[:SOURCE.index('if(flag)then')] + body + SOURCE[SOURCE.index('!$omp end parallel'):]
    original, calls = model._region_model, []
    def counted(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(model, '_region_model', counted)
    analysis, routine, nodes, team = issue(tmp_path, text)
    provider = NumericalPreparationProvider(analysis, options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy='sections', host_threads=4), environment_root=ENTRY)
    context = prove_conditional_numerical_context(provider, team)
    assert len(context.public()['paths']) == 16
    assert len(context._bank._unit_cache) == 4 and len(context._bank._pair_cache) == 6
    assert calls == [0, 1, 2, 3]
    for alternative in atomic_worksharing_alternatives(team):
        if alternative.loops:
            result = provider.prepare_atomic_alternative(alternative, conditional_context=context)
            assert len(result.plan.regions) == len(alternative.loops)
            assert result.validate(analysis, ENTRY, nodes) is result
    assert calls == [0, 1, 2, 3]
    assert prove_conditional_numerical_context(provider, team) is context


def test_shared_symbolic_budget_is_not_source_effect_statement_limit(tmp_path, monkeypatch):
    import compiler.frontend.conditional_numerical as context_module

    analysis, routine, nodes, team = issue(tmp_path)
    provider = NumericalPreparationProvider(analysis, options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy='sections', host_threads=4), environment_root=ENTRY)
    monkeypatch.setattr(context_module, '_SCAN_LIMIT', 1)
    with pytest.raises(CompilationError, match='effort exhausted'):
        prove_conditional_numerical_context(provider, team)
    assert not analysis._conditional_numerical_contexts


def test_normal_builder_seals_source_context_and_emits_guarded_sections(tmp_path):
    from hashlib import sha256
    from compiler.scopes.source import ScopeBuilder
    from compiler.tests.test_source_scopes import FACT

    path = tmp_path/'normal.f90'
    path.write_text(SOURCE)
    builder = ScopeBuilder([path], ENTRY, options=CompilerOptions(opt_level=0),
        config=OffloadConfig(policy='sections', scope_execution='graph', host_threads=4),
        facts={'schema_version':1, 'participation':'serial',
               'sources':{str(path):sha256(path.read_bytes()).hexdigest()},
               'captures':{'argument::a':dict(FACT)}})
    outputs, report = builder.run()
    (tmp_path/'report.json').write_text(json.dumps(report,indent=2))
    package = tmp_path/'package'
    package.mkdir()
    for name, text in outputs.items():
        target = package/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    scope, = report['scopes']
    task, = scope['source_device_tasks']
    assert task['version'] == 3
    assert task['source_proof']['version'] == 3
    alternative, = task['alternatives']
    context = alternative['preparation']['conditional_numerical_context']
    assert context['model_units'] == 2
    assert alternative['source_reads']['conditional_numerical_context_identity'] == context['identity']
    assert alternative['provider']['metadata']['conditional_numerical_context_identity'] == context['identity']
    text = outputs[report['sources'][str(path)]['replacement']]
    assert text.count('!$omp sections') == 1
    assert text.count('!$omp end parallel') == 1
    assert 'fort_scope_admit_source_reads_v1' not in text
    assert 'fort_atomic_status=' in text
    assert 'replay prohibited' in text
    # Native fallback is still the exact source block and no application name
    # or geometry appears in the compiler's capability dispatch.
    assert 'a(i)=a(i-n)' in text


def test_discarded_parent_or_context_cannot_be_reissued_during_validation(tmp_path):
    analysis, routine, nodes, team, provider, context, alternative, result = prepared(tmp_path)
    analysis._joined_completions.clear()
    with pytest.raises(CompilationError):
        context.validate(provider)
    assert not analysis._joined_completions
    with pytest.raises(CompilationError):
        result.validate(analysis, ENTRY, nodes)
    assert not analysis._joined_completions
