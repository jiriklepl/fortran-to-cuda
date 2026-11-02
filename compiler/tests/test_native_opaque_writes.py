"""Native opaque writes compose exact source receipts, never definitions."""
import sys
from copy import copy
from dataclasses import fields
from types import SimpleNamespace

import pytest

from compiler.frontend.native_opaque_writes import NativeOnlyArrayWrite, prove_native_opaque_write
from compiler.frontend.resource_reachability import prove_resource_reachability
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.tests.test_native_formal_array_reductions import CALLER, source


def prepare(tmp_path, text=None, *, site=0, graph_storage=True):
    path = tmp_path/'source.f90'
    path.write_text(source(repeated=False) if text is None else text)
    analysis = SourceEffects([path])
    analysis.seal_session()
    calls = tuple(node for node in analysis.routines[CALLER].execution.content
                  if type(node).__name__ in {'Call_Stmt', 'If_Stmt'})
    selected = (calls[site],)
    name = 'mean' if site == 0 else 'other'
    binding = analysis.modules['formal_reduction'].bindings[name]
    reach = prove_resource_reachability(analysis, CALLER, selected, (binding,))
    teams = tuple({id(team): team for _, _, completion in reach._completion
                   for team in getattr(completion, '_reduction_teams', ())}.values())
    storage = prove_storage_mutation_effects(analysis, CALLER, selection=selected,
        joined_storage=teams, native_graph=reach.native_graph if graph_storage else None)
    return path, analysis, selected, binding, reach, storage


def issue(values):
    _, analysis, selected, binding, reach, storage = values
    return prove_native_opaque_write(analysis, CALLER, selected, binding, reachability=reach, storage=storage)


@pytest.mark.parametrize('guarded', [False, True])
@pytest.mark.parametrize('intent', ['out', 'inout'])
@pytest.mark.parametrize('initialized', [False, True])
def test_complete_original_shape_keeps_no_definition_or_entry_credit(tmp_path, guarded, intent, initialized):
    text = source(guarded=guarded, intent=intent, repeated=False)
    if initialized:
        text = text.replace('mean(-1:1)', 'mean(-1:1)=17')
    values = prepare(tmp_path, text)
    path, analysis, selection, binding, reach, storage = values
    original = path.read_bytes()
    proof = issue(values)
    assert proof.validate(analysis, CALLER, selection, binding) is proof
    assert issue(values) is proof
    record = proof.public()
    assert record['mode'] == 'RW'
    assert record['bounds'] == [-1, 1]
    assert record['storage_bytes'] == 24
    assert record['native_graph_identity'] == reach.native_graph.identity
    assert storage._native_graph is reach.native_graph
    assert record['complete_write_effect_inventory'] is True
    assert record['completion_scope'] == 'normal return only'
    assert len(record['joined_team_identities']) == 2
    assert len(record['association_identities']) == 1
    assert len(record['incoming_instances']) == 1
    assert record['original_effect_inventory'] == reach.public()['accesses']
    assert record['storage_disjointness_obligations']
    assert all(not row['discharged'] for row in record['storage_disjointness_obligations'])
    assert all(not value for key, value in record.items()
               if key.endswith(('authorized', 'authority', 'credit', 'established', 'grant')))
    assert ('undefine' in record['target_actions']) is (intent == 'out')
    assert path.read_bytes() == original


def test_partial_writes_do_not_become_whole_output_definedness(tmp_path):
    text = source(intent='inout', repeated=False).replace('result=0\n', '').replace('result=result/max(1,n)', '')
    proof = issue(prepare(tmp_path, text))
    assert proof.public()['complete_write_effect_inventory']
    assert not proof.public()['whole_output_definedness_authorized']
    assert not proof.public()['managed_definition_authorized']


@pytest.mark.parametrize('dtype', ['real(4)', 'real(8)', 'integer(4)', 'integer(8)'])
def test_primitive_module_storage_is_metadata_not_scalarized_element_work(tmp_path, dtype):
    text = source(formal='result(1025)', repeated=False).replace('mean(-1:1)', 'mean(-1:1023)')
    text = text.replace('real(8)', dtype)
    proof = issue(prepare(tmp_path, text))
    assert proof.public()['element_count'] == 1025
    assert proof.public()['storage_bytes'] == 1025*int(dtype[-2])


def test_distinct_original_calls_do_not_lend_storage_or_binding_receipts(tmp_path):
    values = prepare(tmp_path, source(repeated=True), site=0)
    _, analysis, selected, binding, reach, storage = values
    proof = issue(values)
    second = tuple(node for node in analysis.routines[CALLER].execution.content
                   if type(node).__name__ == 'Call_Stmt')[-1]
    other = analysis.modules['formal_reduction'].bindings['other']
    with pytest.raises(CompilationError):
        proof.validate(analysis, CALLER, (second,), other)
    with pytest.raises(CompilationError):
        prove_native_opaque_write(analysis, CALLER, selected, other, reachability=reach, storage=storage)
    assert proof.binding is binding


@pytest.mark.parametrize('change', ['reach_copy', 'storage_copy', 'graph_copy', 'graphless',
                                   'other_selection', 'teams_removed', 'team_copy', 'reach_duck', 'storage_duck'])
def test_exact_typed_prerequisites_cannot_be_replaced(tmp_path, change):
    values = prepare(tmp_path, graph_storage=change != 'graphless')
    _, analysis, selected, binding, reach, storage = values
    if change == 'reach_copy':
        reach = copy(reach)
    elif change == 'storage_copy':
        storage = copy(storage)
    elif change == 'graph_copy':
        object.__setattr__(storage, '_native_graph', copy(reach.native_graph))
    elif change == 'other_selection':
        selected = (selected[0].items[1],)
    elif change == 'teams_removed':
        object.__setattr__(storage, '_joined_storage', ())
    elif change == 'team_copy':
        object.__setattr__(storage, '_joined_storage', (copy(storage._joined_storage[0]), *storage._joined_storage[1:]))
    elif change == 'reach_duck':
        reach = SimpleNamespace(**vars(reach))
    elif change == 'storage_duck':
        storage = SimpleNamespace(**{part.name: getattr(storage, part.name) for part in fields(type(storage))})
    with pytest.raises(CompilationError):
        prove_native_opaque_write(analysis, CALLER, selected, binding, reachability=reach, storage=storage)
    assert not getattr(analysis, '_native_opaque_write_proofs', {})


@pytest.mark.parametrize('mutation', ['drop_write', 'drop_undefine', 'drop_completion', 'association',
                                     'completion_registry', 'storage_registry', 'association_registry'])
def test_effect_and_source_receipt_tampering_never_becomes_write_authority(tmp_path, mutation):
    values = prepare(tmp_path)
    _, analysis, _, _, reach, storage = values
    completion = reach._completion[0][2]
    association = completion._reduction_teams[0]._fixed_array_associations[0]
    if mutation.startswith('drop_') and mutation != 'drop_completion':
        kind = mutation.removeprefix('drop_')
        object.__setattr__(reach, 'accesses', tuple(item for item in reach.accesses if item.kind != kind))
    elif mutation == 'drop_completion':
        object.__setattr__(reach, '_completion', ())
    elif mutation == 'association':
        object.__setattr__(association, 'actual', association.formal)
    elif mutation == 'completion_registry':
        analysis._native_graph_completions[completion.identity] = copy(completion)
    elif mutation == 'storage_registry':
        analysis._storage_mutation_effects[storage.identity] = copy(storage)
    else:
        analysis._native_fixed_array_associations[association.identity] = copy(association)
    with pytest.raises(CompilationError):
        issue(values)


@pytest.mark.parametrize('text_change', [
    ('real(8)::', 'real(8),target::'),
    ('real(8)::', 'real(8),volatile::'),
    ('real(8)::', 'real(8),asynchronous::'),
    ('mean(-1:1)', 'mean(-1:2)'),
    ('call reduce_mean(mean,', 'call reduce_mean(mean(:),'),
    ('result=result/max(1,n)', 'call unknown(result)'),
    ('!$omp end parallel', '!$omp end parallel nowait'),
    ('result=result/max(1,n)', '!$omp task\nresult=1\n!$omp end task'),
])
def test_unsupported_storage_or_incomplete_original_execution_rejects(tmp_path, text_change):
    before, after = text_change
    text = source(guarded=False, repeated=False).replace(before, after)
    with pytest.raises(CompilationError):
        issue(prepare(tmp_path, text))


@pytest.mark.parametrize('mutation', ['copy', 'registered_copy', 'removed', 'mode', 'record', 'budget', 'source'])
def test_published_write_is_an_exact_owned_handle(tmp_path, mutation):
    values = prepare(tmp_path)
    path, analysis, selected, binding, _, _ = values
    proof = issue(values)
    if mutation in {'copy', 'registered_copy'}:
        proof = copy(proof)
        if mutation == 'registered_copy':
            analysis._native_opaque_write_proofs[proof.identity] = proof
    elif mutation == 'removed':
        del analysis._native_opaque_write_proofs[proof.identity]
    elif mutation == 'mode':
        object.__setattr__(proof, 'mode', 'R')
    elif mutation == 'record':
        object.__setattr__(proof, '_record', '{}')
    elif mutation == 'budget':
        analysis.operation_limit += 1
    else:
        path.write_text(path.read_text().replace('result=0', 'result=1'))
    with pytest.raises(CompilationError):
        proof.validate(analysis, CALLER, selected, binding)


@pytest.mark.parametrize('phase', ['issue', 'validate'])
@pytest.mark.parametrize('attachment', [part.name for part in fields(NativeOnlyArrayWrite)])
def test_final_publication_fences_every_compositor_attachment(tmp_path, monkeypatch, phase, attachment):
    values = prepare(tmp_path)
    _, analysis, selected, binding, _, _ = values
    proof = issue(values) if phase == 'validate' else None
    verify = analysis.verify_publication
    changed = []
    def intercept():
        verify()
        frame = sys._getframe(1)
        registry = getattr(analysis, '_native_opaque_write_proofs', {})
        if frame.f_code.co_name != '_derive' or not registry:
            return
        token, = registry.values()
        changed.append((token.identity, token))
        object.__setattr__(token, attachment, object())
    monkeypatch.setattr(analysis, 'verify_publication', intercept)
    def checked():
        if proof is None:
            return issue(values)
        return proof.validate(analysis, CALLER, selected, binding)
    with pytest.raises(CompilationError):
        checked()
    assert len(changed) == 1
    if phase == 'issue':
        key, token = changed[0]
        assert key not in analysis._native_opaque_write_proofs
        arena = analysis.source_authority_context().derivation_arena(analysis)
        with pytest.raises(CompilationError, match='exact arena-issued handle'):
            arena.require_handle(('native_opaque_write', id(analysis), key), token)


@pytest.mark.parametrize('mutation', ['access', 'template', 'instance', 'argument', 'team', 'association',
                                     'storage', 'completion', 'registry', 'arena'])
def test_final_callback_fences_exact_transitive_receipts(tmp_path, monkeypatch, mutation):
    values = prepare(tmp_path)
    _, analysis, _, _, reach, storage = values
    verify = analysis.verify_publication
    changed = []
    def intercept():
        verify()
        if sys._getframe(1).f_code.co_name != '_derive':
            return
        changed.append(True)
        graph = reach.native_graph
        completion = reach._completion[0][2]
        team = completion._reduction_teams[0]
        if mutation == 'access':
            object.__setattr__(reach.accesses[0], 'kind', 'write')
        elif mutation == 'template':
            object.__setattr__(graph.instances[0].template, 'calls', ())
        elif mutation == 'instance':
            object.__setattr__(graph.instances[1], 'parent', None)
        elif mutation == 'argument':
            item = graph.instances[1].call.edge.resolved.associations[0]
            object.__setattr__(item.actual_type, 'binding', item.formal_binding)
        elif mutation == 'team':
            object.__setattr__(team, '_fixed_array_associations', ())
        elif mutation == 'association':
            item = team._fixed_array_associations[0]
            object.__setattr__(item, 'actual', item.formal)
        elif mutation == 'storage':
            object.__setattr__(storage, '_native_graph', None)
        elif mutation == 'completion':
            object.__setattr__(completion, '_reduction_teams', ())
        elif mutation == 'registry':
            analysis._native_fixed_array_associations = analysis._native_fixed_array_associations.copy()
        else:
            object.__setattr__(analysis, '_source_derivation_arena', None)
    monkeypatch.setattr(analysis, 'verify_publication', intercept)
    with pytest.raises(CompilationError):
        issue(values)
    assert changed == [True]
    assert not getattr(analysis, '_native_opaque_write_proofs', {})


@pytest.mark.parametrize('phase', ['issue', 'validate'])
@pytest.mark.parametrize('mutation', ['dtype', 'rank', 'attributes', 'lower_bounds', 'shape_nodes',
                                     'bound_syntax', 'selection_syntax', 'scope_declaration',
                                     'namespace_binding', 'namespace_module', 'namespace_routine', 'source_graph'])
def test_final_publication_cannot_change_consumed_source_storage(tmp_path, monkeypatch, phase, mutation):
    values = prepare(tmp_path)
    _, analysis, selected, binding, _, _ = values
    proof = issue(values) if phase == 'validate' else None
    verify = analysis.verify_publication
    changed = []
    def intercept():
        verify()
        if (sys._getframe(1).f_code.co_name != '_derive'
                or not getattr(analysis, '_native_opaque_write_proofs', {})):
            return
        changed.append(True)
        if mutation == 'bound_syntax':
            object.__setattr__(binding.shape_nodes[0], 'items', (None, None))
        elif mutation == 'selection_syntax':
            object.__setattr__(selected[0], 'items', (None, selected[0].items[1]))
        elif mutation == 'scope_declaration':
            object.__setattr__(binding.declaring_scope, 'node', selected[0])
        elif mutation == 'namespace_binding':
            dict.__setitem__(binding.declaring_scope.bindings, binding.name, copy(binding))
        elif mutation == 'namespace_module':
            dict.__setitem__(analysis.modules, binding.declaring_scope.module, copy(binding.declaring_scope))
        elif mutation == 'namespace_routine':
            dict.__setitem__(analysis.routines, CALLER, copy(analysis.routines[CALLER]))
        elif mutation == 'source_graph':
            key = next(key for key in analysis._live_structures if key[0] == CALLER)
            analysis._live_structures[key] = copy(analysis._live_structures[key])
        else:
            value = {'dtype': 'logical', 'rank': 2, 'attributes': frozenset({'target'}),
                     'lower_bounds': ('8',), 'shape_nodes': ()}[mutation]
            object.__setattr__(binding, mutation, value)
    monkeypatch.setattr(analysis, 'verify_publication', intercept)
    def checked():
        if proof is None:
            return issue(values)
        return proof.validate(analysis, CALLER, selected, binding)
    with pytest.raises(CompilationError):
        checked()
    assert changed == [True]
    if phase == 'issue':
        assert not getattr(analysis, '_native_opaque_write_proofs', {})


@pytest.mark.parametrize('phase', ['parent_survives', 'child_discarded'])
def test_generation_handles_remain_exact_without_persistent_eligibility(tmp_path, phase):
    values = prepare(tmp_path)
    _, analysis, selection, binding, _, _ = values
    parent = SimpleNamespace(analysis=analysis, entry=analysis.routines[CALLER],
                             _check_reached_publication=analysis.verify_publication)
    proof = issue(values) if phase == 'parent_survives' else None
    with GenerationTransaction(parent):
        current = issue(values)
        assert current.validate(analysis, CALLER, selection, binding) is current
        registry = analysis._native_opaque_write_proofs.copy()
        if proof is not None:
            assert current is proof
    if proof is not None:
        analysis._native_opaque_write_proofs = analysis._native_opaque_write_proofs.copy()
        assert issue(values) is proof
    else:
        analysis._native_opaque_write_proofs = registry
        with pytest.raises(CompilationError, match='discarded generation'):
            current.validate(analysis, CALLER, selection, binding)


@pytest.mark.parametrize('replacement', [False, True])
def test_final_exception_cleanup_uses_original_identity_and_preserves_replacement(tmp_path, monkeypatch, replacement):
    values = prepare(tmp_path)
    analysis = values[1]
    verify = analysis.verify_publication
    retained = []
    def intercept():
        verify()
        registry = getattr(analysis, '_native_opaque_write_proofs', {})
        if sys._getframe(1).f_code.co_name != '_derive' or not registry:
            return
        identity, proof = next(iter(registry.items()))
        retained.append((identity, proof))
        if replacement:
            registry[identity] = copy(proof)
        object.__setattr__(proof, 'identity', 'changed-before-original-exception')
        raise RuntimeError('original publication failure')
    monkeypatch.setattr(analysis, 'verify_publication', intercept)
    with pytest.raises(RuntimeError, match='original publication failure'):
        issue(values)
    assert len(retained) == 1
    identity, proof = retained[0]
    registry = analysis._native_opaque_write_proofs
    if replacement:
        assert registry[identity] is not proof
    else:
        assert identity not in registry
