"""Native summary policy cannot borrow a foreign child association."""
from copy import copy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.call_groups import OwnerLocalCallGroup, _candidates
from compiler.scopes.segments import _native_analysis
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_owner_local_entry_preservation_runtime import TILED


@pytest.fixture
def issued(tmp_path):
    path = tmp_path/'groups.f90'
    path.write_text(TILED)
    fact = {"storage": "stable", "initialized": "whole", "escapes": False,
            "allocation_changes": False, "lifetime": "invocation"}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::"+name: dict(fact) for name in ("a", "b", "c")}}
    contracts = {"live_groups::run": {"native_ancestor": {"version": 2,
        "lifetime": "owner_local_call_groups", "device_roots": ["live_groups::"+name
            for name in ("first", "second", "third")]}}}
    builder = ScopeBuilder([path], 'live_groups::run', facts=facts, contracts=contracts,
                           options=CompilerOptions(), config=OffloadConfig(policy='sections'))
    owner = OwnerLocalCallGroup(builder, next(_candidates(builder)))
    assert owner.run(owner.closed_selection)
    member = owner.members['live_groups::second']
    analysis = _native_analysis(builder.analysis)
    analysis._native_metadata = True
    selected = (member.routine.execution.content[0],)
    return owner, member, analysis, selected


def descriptor(issued):
    owner, member, analysis, selected = issued
    return owner.position_epoch.descriptor_for(member.builder, analysis, member.routine.qualified, selected)


def test_native_policy_projection_gets_its_own_selected_descriptor(issued):
    owner, member, analysis, selected = issued
    assert analysis.structure(member.routine.qualified).identity != member.graph.identity
    proof = descriptor(issued)
    assert proof._analysis is analysis
    assert proof.validate(analysis, member.routine.qualified,
                          analysis._selected_source(member.routine.qualified, selected)[1]) is proof
    assert proof.public()['execution_authorized'] is False
    assert proof.public()['entry_definition_authority'] is False
    assert all(state['resource_kind'] == 'original_ordinary_array_formal'
               for state in proof.public()['resources'])
    assert owner.position_epoch._descriptor._analysis is owner.builder.analysis


@pytest.mark.parametrize('mutation', ['actuals', 'both_maps', 'call', 'companion', 'source'])
def test_native_policy_projection_rejects_changed_original_association(issued, mutation):
    _owner, member, analysis, _selected = issued
    if mutation in {'actuals', 'both_maps'}:
        actuals = dict(member.builder._native_host_array_actuals)
        actuals['argument::b'], actuals['argument::c'] = actuals['argument::c'], actuals['argument::b']
        member.builder._native_host_array_actuals = actuals
        if mutation == 'both_maps':
            object.__setattr__(member, 'canonical_bindings', actuals)
    elif mutation == 'call':
        object.__setattr__(member, 'original_source_call', copy(member.original_source_call))
    elif mutation == 'companion':
        member.builder._native_host_owner = copy(member)
    else:
        original_registry = analysis.routines
        replacement = dict(original_registry)
        replacement[member.routine.qualified] = copy(member.routine)
        with pytest.raises(CompilationError, match='original source registry is immutable'):
            analysis.routines = replacement
        assert analysis.routines is original_registry
        descriptor(issued)
        # Corrupt the private root explicitly to retain the consumer-side check.
        object.__setattr__(analysis, 'routines', replacement)
    with pytest.raises(CompilationError):
        descriptor(issued)


def test_native_descriptor_publication_rechecks_consumed_call_attachment(issued, monkeypatch):
    from compiler.scopes import graph_epochs

    _owner, member, _analysis, _selected = issued
    original = graph_epochs.prove_descriptor_window
    def detached(*args, **kwargs):
        proof = original(*args, **kwargs)
        object.__setattr__(member, 'original_source_call', copy(member.original_source_call))
        return proof
    monkeypatch.setattr(graph_epochs, 'prove_descriptor_window', detached)
    with pytest.raises(CompilationError):
        descriptor(issued)
