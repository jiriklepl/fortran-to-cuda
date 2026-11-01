"""A capture batch shares one scan without lending authority across hooks."""
from copy import copy

import pytest

from compiler.ir import CompilationError
from compiler.scopes.call_groups import OwnerLocalCallGroup, _candidates
from compiler.tests.test_owner_local_call_groups import SOURCE, fixture


def issued(tmp_path):
    path, builder = fixture(tmp_path, SOURCE.replace('call unknown_prefix()', 'a(:)=a(:)+1.0_8'),
                            lifetime='invocation', unused=True)
    for fact in builder.facts['captures'].values():
        fact['initialized'] = 'whole'
    owner = OwnerLocalCallGroup(builder, next(_candidates(builder)))
    return path, owner, owner.position_epoch, owner.position_epoch._bindings


def test_capture_batch_keeps_identical_facts_with_two_complete_epoch_checks(tmp_path, monkeypatch):
    _path, owner, epoch, bindings = issued(tmp_path)
    builder = owner.builder
    original = type(epoch).validate
    calls = []
    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(type(epoch), 'validate', counted)
    expected = tuple(epoch.capture(builder, binding, builder.capture_fact(binding)) for binding in bindings)
    assert len(calls) == len(bindings) == 3
    calls.clear()
    assert epoch.capture_many(builder, bindings) == expected
    assert calls == [epoch, epoch]
    assert not hasattr(epoch, '_capture_batch')
    assert not hasattr(builder, '_capture_batch')


@pytest.mark.parametrize('fault', ['fact_copy', 'fact_content', 'facts_map', 'descriptor',
                                  'entry_state', 'descriptor_registry', 'entry_registry'])
def test_capture_batch_rejects_transient_callback_authority_substitution(tmp_path, monkeypatch, fault):
    _path, owner, epoch, bindings = issued(tmp_path)
    builder = owner.builder
    original = builder.capture_fact
    calls = []
    def changed(binding):
        fact = original(binding)
        if not calls:
            calls.append(binding)
            if fault == 'fact_copy':
                return dict(fact)  # Equal content is not the supplied object.
            if fault == 'fact_content':
                fact['initialized'] = 'none'
            elif fault == 'facts_map':
                builder.facts['captures'] = dict(builder.facts['captures'])
            elif fault == 'descriptor':
                object.__setattr__(epoch, '_descriptor', copy(epoch._descriptor))
            elif fault == 'entry_state':
                object.__setattr__(epoch._entry_preservation, 'preserved', tuple(reversed(bindings)))
            elif fault == 'descriptor_registry':
                builder.analysis._descriptor_window_proofs.pop(epoch._descriptor.identity)
            else:
                builder.analysis._entry_preservations.pop(epoch._entry_preservation.identity)
        return fact
    monkeypatch.setattr(builder, 'capture_fact', changed)
    with pytest.raises(CompilationError):
        epoch.capture_many(builder, bindings)
    assert len(calls) == 1  # No later callback can restore a transient forgery.


@pytest.mark.parametrize('fault', ['entry', 'registry', 'facts', 'source', 'descriptor_state'])
def test_capture_batch_has_a_fresh_final_publication_fence(tmp_path, monkeypatch, fault):
    path, owner, epoch, bindings = issued(tmp_path)
    builder = owner.builder
    original = type(epoch).validate
    calls = []
    def late(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        calls.append(self)
        if len(calls) == 2:
            if fault == 'entry':
                object.__setattr__(self._entry_preservation, 'preserved', ())
            elif fault == 'registry':
                builder.analysis._entry_preservations.pop(self._entry_preservation.identity)
            elif fault == 'facts':
                builder.facts['captures']['argument::a']['initialized'] = 'none'
            elif fault == 'descriptor_state':
                object.__setattr__(self._descriptor, '_selected', ())
            else:
                path.write_text(path.read_text()+'! changed after the epoch check\n')
        return result
    monkeypatch.setattr(type(epoch), 'validate', late)
    with pytest.raises(CompilationError):
        epoch.capture_many(builder, bindings)


def test_capture_batch_requires_distinct_original_bindings(tmp_path):
    _path, owner, epoch, bindings = issued(tmp_path)
    with pytest.raises(CompilationError, match='distinct original'):
        epoch.capture_many(owner.builder, (bindings[0], bindings[0]))


def test_builder_batch_preserves_repeated_original_associations(tmp_path, monkeypatch):
    _path, owner, epoch, bindings = issued(tmp_path)
    builder = owner.builder
    builder._graph_storage_epoch = epoch
    original = type(epoch).validate
    calls = []
    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)
    expected = tuple(builder.capture(binding) for binding in (*bindings, bindings[0]))
    monkeypatch.setattr(type(epoch), 'validate', counted)
    assert builder.capture_many((*bindings, bindings[0])) == expected
    assert calls == [epoch, epoch]


@pytest.mark.parametrize('fault', ['root', 'name', 'scope'])
def test_aliased_equal_fact_dicts_cannot_hide_changed_original_binding_identity(tmp_path, monkeypatch, fault):
    _path, owner, epoch, bindings = issued(tmp_path)
    builder = owner.builder
    left, right = bindings[:2]
    # Deliberately equal/aliased facts grant no permission to exchange bindings.
    builder.facts['captures'][right.root] = builder.facts['captures'][left.root]
    original = builder.capture_fact
    calls = []
    def changed(binding):
        fact = original(binding)
        if not calls:
            calls.append(binding)
            field, value = ('declaring_scope', copy(binding.declaring_scope)) if fault == 'scope' else (
                fault, getattr(right, fault))
            object.__setattr__(binding, field, value)
        return fact
    monkeypatch.setattr(builder, 'capture_fact', changed)
    with pytest.raises(CompilationError):
        epoch.capture_many(builder, bindings)
    assert len(calls) == 1


@pytest.mark.parametrize('fault', ['canonical', 'native_actuals', 'argument'])
def test_child_capture_batch_keeps_original_canonical_mapping_through_root_hooks(tmp_path, monkeypatch, fault):
    _path, owner, _epoch, _bindings = issued(tmp_path)
    assert owner.run(owner.closed_selection)
    member = owner.members['local_groups::second']
    builder = owner.builder
    original = builder.capture_fact
    calls = []
    def changed(binding):
        fact = original(binding)
        if not calls:
            calls.append(binding)
            if fault == 'canonical':
                values = dict(member.canonical_bindings)
                values['argument::a'] = values['argument::b']
                object.__setattr__(member, 'canonical_bindings', values)
            elif fault == 'native_actuals':
                member.builder._native_host_array_actuals = dict(member.builder._native_host_array_actuals)
            else:
                formal = member.routine.scope.bindings['a']
                object.__setattr__(formal, 'name', 'b')
        return fact
    monkeypatch.setattr(builder, 'capture_fact', changed)
    builder._graph_storage_epoch = owner.position_epoch
    with pytest.raises(CompilationError):
        member.builder.capture_many(tuple(member.routine.scope.bindings[name] for name in ('a', 'b')))
    assert len(calls) == 1


def test_child_batch_rechecks_mapping_after_a_consuming_callback(tmp_path):
    _path, owner, _epoch, _bindings = issued(tmp_path)
    assert owner.run(owner.closed_selection)
    member = owner.members['local_groups::second']
    bindings = tuple(member.routine.scope.bindings[name] for name in ('a', 'b', 'a'))
    expected = tuple(member.builder.capture(binding) for binding in bindings)
    assert member.builder.capture_many(bindings) == expected

    def changed():
        member.builder._native_host_array_actuals = dict(member.builder._native_host_array_actuals)
    with pytest.raises(CompilationError, match='canonical association'):
        member.builder.capture_many(bindings, _consumer_check=changed)
