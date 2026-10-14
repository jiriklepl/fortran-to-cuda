"""Ordinary numerical pairs are reused only within one lexical planning pass."""

from collections import Counter
from copy import copy
from dataclasses import replace
from hashlib import sha256

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.numerical_effects import NumericalEffectProof
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.ordinary_consumption import _OrdinaryPairConsumption
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.support.numerical_regions import environment_source, reached_numerical_fixture

ENTRY = "sapphire_chain::advance"
SOURCE = """module sapphire_chain
use,intrinsic::ieee_exceptions,only:ieee_set_halting_mode,ieee_invalid
implicit none
contains
subroutine advance(a,b,n)
real(8),intent(inout)::a(-4:),b(-4:)
integer,intent(in)::n
integer::i
!$omp parallel
call ieee_set_halting_mode(ieee_invalid,.false.)
!$omp end parallel
!$omp parallel do private(i) schedule(static)
do i=-3,n
b(i)=2.0_8*a(i)
enddo
!$omp end parallel do
!$omp parallel do private(i) schedule(static)
do i=-3,n
a(i)=b(i)+1.0_8
enddo
!$omp end parallel do
end subroutine
end module
"""


def subject(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "sapphire.f90"
    path.write_text(SOURCE)
    fact = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {root: dict(fact) for root in ("argument::a", "argument::b")}}
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    selections = tuple(group.original_nodes for group in grouped_original_sequence(builder.entry.execution.content, {})
                       if isinstance(group, NativeGroupCandidate)
                       and any(type(node).__name__ == "Block_Nonlabel_Do_Construct" for node in group.original_nodes))
    assert len(selections) == 2
    return path, builder, selections


def counted_validations(monkeypatch, analysis):
    validate, calls = NumericalEffectProof.validate, []

    def counted(self, current, *args, **kwargs):
        if current is analysis:
            calls.append(self.identity)
        return validate(self, current, *args, **kwargs)

    monkeypatch.setattr(NumericalEffectProof, "validate", counted)
    return calls


def test_lexical_pass_validates_each_new_pair_at_issuance_and_return(tmp_path, monkeypatch):
    _path, builder, _selections = subject(tmp_path)
    calls = counted_validations(monkeypatch, builder.analysis)
    owner = LexicalOwner(builder)
    assert owner.run(tuple(builder.entry.execution.content))
    pairs = tuple(builder.inline.original_numerical_regions.values())
    assert len(pairs) == 2
    assert Counter(calls) == {pair.effects.identity: 2 for pair in pairs}
    assert builder._ordinary_pair_state.active is None
    assert not builder._ordinary_pair_state.failed
    outputs, report = builder.finish()
    assert report["scope_count"] == 1
    assert outputs[report["sources"][str(builder.entry.scope.path)]["replacement"]]
    assert Counter(calls) == {pair.effects.identity: 3 for pair in pairs}


def test_private_lookups_reuse_pair_but_public_lookup_remains_fresh(tmp_path, monkeypatch):
    _path, builder, selections = subject(tmp_path)
    pair = builder.inline.prepare_numerical_region(selections[0])
    calls = counted_validations(monkeypatch, builder.analysis)
    with _OrdinaryPairConsumption(builder):
        for _ in range(3):
            assert builder._consume_numerical_region(builder.analysis, ENTRY, selections[0]) is pair
        assert calls == [pair.effects.identity]
        assert builder.resolved_numerical_region(builder.analysis, ENTRY, selections[0]) is pair
        assert calls == [pair.effects.identity] * 2
        with pytest.raises(CompilationError):
            builder.finish()
        with pytest.raises(CompilationError), GenerationTransaction(builder):
            pytest.fail("an active ordinary pass was copied into speculative generation")
    assert calls == [pair.effects.identity] * 3
    assert builder.resolved_numerical_region(builder.analysis, ENTRY, selections[0]) is pair
    assert calls == [pair.effects.identity] * 4
    binding = copy(builder.entry.scope.bindings["n"])
    binding.attributes |= {"volatile"}
    with pytest.raises(CompilationError):
        builder.entry.scope.bindings["n"] = binding
    assert builder.resolved_numerical_region(builder.analysis, ENTRY, selections[0]) is pair


def test_copied_foreign_and_expired_consumers_cannot_reuse_pair(tmp_path):
    _path, builder, selections = subject(tmp_path / "original")
    _foreign_path, foreign, _foreign_selections = subject(tmp_path / "foreign")
    pair = builder.inline.prepare_numerical_region(selections[0])
    consumer = _OrdinaryPairConsumption(builder)
    with consumer:
        assert consumer.resolve(builder.inline, selections[0]) is pair
        with pytest.raises(CompilationError):
            copy(consumer).resolve(builder.inline, selections[0])
        with pytest.raises(CompilationError):
            copy(consumer).__exit__(None, None, None)
        assert builder._ordinary_pair_state.active is consumer
        assert not builder._ordinary_pair_state.failed
    with pytest.raises(CompilationError):
        consumer.resolve(builder.inline, selections[0])
    with pytest.raises(CompilationError), consumer:
        pytest.fail("an expired consumer was reentered")
    def consume_foreign():
        with _OrdinaryPairConsumption(foreign) as other:
            other.resolve(builder.inline, selections[0])

    with pytest.raises(CompilationError):
        consume_foreign()
    assert foreign._ordinary_pair_state.failed


def test_copied_builder_cannot_borrow_active_ordinary_pair(tmp_path):
    _path, builder, selections = subject(tmp_path)
    builder.inline.prepare_numerical_region(selections[0])

    def consume_copy():
        with _OrdinaryPairConsumption(builder):
            copied = copy(builder)
            copied._consume_numerical_region(copied.analysis, ENTRY, selections[0])

    with pytest.raises(CompilationError):
        consume_copy()
    assert builder._ordinary_pair_state.failed


def test_public_proof_parent_budget_is_hard_inside_private_pass(tmp_path):
    _path, builder, selections = subject(tmp_path)
    pair = builder.inline.prepare_numerical_region(selections[0])
    work = []

    def exhausted(amount):
        work.append(amount)
        raise CompilationError("test parent work budget exhausted")

    with _OrdinaryPairConsumption(builder):
        assert builder._consume_numerical_region(builder.analysis, ENTRY, selections[0]) is pair
        with pytest.raises(CompilationError, match="parent work budget exhausted"):
            pair.effects.validate(builder.analysis, ENTRY, selections[0], pair.prepared, consume=exhausted)
        assert work
        assert builder._consume_numerical_region(builder.analysis, ENTRY, selections[0]) is pair


def test_replacing_enrolled_pair_with_equal_copy_is_hard(tmp_path):
    _path, builder, selections = subject(tmp_path)
    pair = builder.inline.prepare_numerical_region(selections[0])

    def replace_pair():
        with _OrdinaryPairConsumption(builder):
            assert builder._consume_numerical_region(builder.analysis, ENTRY, selections[0]) is pair
            builder.inline.original_numerical_regions[tuple(map(id, selections[0]))] = copy(pair)
            builder._consume_numerical_region(builder.analysis, ENTRY, selections[0])

    with pytest.raises(CompilationError):
        replace_pair()
    assert builder._ordinary_pair_state.failed
    assert builder.inline.original_numerical_regions[tuple(map(id, selections[0]))] is pair
    with pytest.raises(CompilationError):
        builder.finish()


def test_explicit_scope_rollback_discards_only_speculative_pair_enrollment(tmp_path):
    _path, builder, selections = subject(tmp_path)
    retained = builder.inline.prepare_numerical_region(selections[0])
    with _OrdinaryPairConsumption(builder):
        assert builder._consume_numerical_region(builder.analysis, ENTRY, selections[0]) is retained
        checkpoint = builder.scope_checkpoint()
        discarded = builder.inline.prepare_numerical_region(selections[1])
        assert builder._consume_numerical_region(builder.analysis, ENTRY, selections[1]) is discarded
        builder.restore_scope_checkpoint(checkpoint)
        assert builder._consume_numerical_region(builder.analysis, ENTRY, selections[0]) is retained
        assert builder._consume_numerical_region(builder.analysis, ENTRY, selections[1]) is None
        replacement = builder.inline.prepare_numerical_region(selections[1])
        assert replacement is not discarded
        assert builder._consume_numerical_region(builder.analysis, ENTRY, selections[1]) is replacement
    assert not builder._ordinary_pair_state.failed


@pytest.mark.parametrize("mutation", ["syntax", "file", "binding", "configuration", "facts", "exception"])
def test_mid_emission_mutations_cannot_publish_and_preserve_transaction_rollback(tmp_path, monkeypatch, mutation):
    path, parent, _selections = subject(tmp_path)
    assignment = walk(parent.entry.execution, F.Assignment_Stmt)[0]
    items = assignment.items
    binding = parent.entry.scope.bindings["n"]
    replacement = copy(binding)
    replacement.attributes |= {"volatile"}
    original_emit = LexicalOwner.emit_unit
    triggered = False

    def emit(owner, unit, index):
        nonlocal triggered
        result = original_emit(owner, unit, index)
        if not triggered and unit.leaves:
            triggered = True
            if mutation == "syntax":
                assignment.items = (*assignment.items[:2], assignment.items[0])
            elif mutation == "file":
                path.write_text(SOURCE + "! changed during original emission\n")
            elif mutation == "binding":
                owner.builder.entry.scope.bindings["n"] = replacement
            elif mutation == "configuration":
                owner.builder.config = replace(owner.builder.config, host_threads=2)
            elif mutation == "facts":
                owner.builder.facts["captures"]["argument::a"]["initialized"] = "none"
            else:
                raise CompilationError("injected original emission failure")
        return result

    monkeypatch.setattr(LexicalOwner, "emit_unit", emit)
    try:
        with GenerationTransaction(parent) as transaction:
            candidate = transaction.builder
            facts = {root: dict(value) for root, value in candidate.facts["captures"].items()}
            configuration = candidate.config
            if mutation == "file":
                assert LexicalOwner(candidate).run(tuple(candidate.entry.execution.content))
                assert triggered
                with pytest.raises(CompilationError, match="source changed"):
                    candidate.finish()
            else:
                with pytest.raises(CompilationError):
                    LexicalOwner(candidate).run(tuple(candidate.entry.execution.content))
                assert triggered
                assert assignment.items is items
                assert candidate.entry.scope.bindings["n"] is binding
                candidate.config = configuration
                candidate.facts["captures"] = facts
                assert not candidate.outputs
                assert not candidate.edits
                assert not candidate.scopes
                assert not candidate.inline.original_numerical_regions
                with pytest.raises(CompilationError):
                    candidate.finish()
                with pytest.raises(CompilationError):
                    transaction.adopt(candidate)
    finally:
        assert assignment.items is items
        path.write_text(SOURCE)
        assert parent.entry.scope.bindings["n"] is binding
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes
    assert not parent.inline.original_numerical_regions
    assert not parent._ordinary_pair_state.failed


def test_same_original_authority_emits_identical_artifacts_with_fresh_lookup_oracle(tmp_path, monkeypatch):
    _path, parent, _selections = subject(tmp_path)

    def emit():
        with GenerationTransaction(parent) as transaction:
            candidate = transaction.builder
            assert LexicalOwner(candidate).run(tuple(candidate.entry.execution.content))
            outputs, report = candidate.finish()
            return dict(outputs), report

    reused_outputs, reused_report = emit()

    # This oracle uses the existing fresh public demand at every private lookup,
    # with the same original AST/Binding identities and untouched parent state.
    def fresh(builder, analysis, procedure, selection):
        return builder.resolved_numerical_region(analysis, procedure, selection)

    monkeypatch.setattr(ScopeBuilder, "_consume_numerical_region", fresh)
    fresh_outputs, fresh_report = emit()
    assert fresh_outputs == reused_outputs
    assert fresh_report == reused_report
    assert any(name.endswith(".cu") for name in reused_outputs)
    assert not parent.outputs
    assert not parent.edits


def test_inherited_reached_owner_uses_its_own_pass_without_ordinary_dispatch(tmp_path, monkeypatch):
    _paths, builder, proof, originals, _provider = reached_numerical_fixture(
        tmp_path, child_source=environment_source())

    def forbidden(_consumer):
        pytest.fail("a reached owner entered an ordinary pair consumption pass")

    monkeypatch.setattr(_OrdinaryPairConsumption, "__enter__", forbidden)
    owner = ReachedWindowOwner(builder, proof, originals)
    assert owner.run()
    assert owner.builder.validate_reached_window() is proof
    assert owner.builder._ordinary_pair_state.active is None
    assert not owner.builder._ordinary_pair_state.failed
    outputs, report = owner.builder.finish()
    assert report["scope_count"] == 1
    assert any(name.endswith(".cu") for name in outputs)


def fail_after_staged_original_unit(monkeypatch, staged):
    original_emit = LexicalOwner.emit_unit

    def emit(owner, unit, index):
        result = original_emit(owner, unit, index)
        if unit.leaves:
            assert result
            assert owner.builder.inline.original_numerical_regions
            owner.builder.add_edit(owner.routine.scope.path, *result[0])
            staged.append(owner.builder)
            raise CompilationError("injected failure after staged original numerical unit")
        return result

    monkeypatch.setattr(LexicalOwner, "emit_unit", emit)


def test_rejected_root_attempt_discards_failed_candidate_and_preserves_native_fallback(tmp_path, monkeypatch):
    path, parent, _selections = subject(tmp_path)
    staged = []
    fail_after_staged_original_unit(monkeypatch, staged)
    nodes = tuple(parent.entry.execution.content)
    assert parent._attempt_lexical_owner(nodes, finalize=False) is False
    assert len(staged) == 1
    assert staged[0] is not parent
    assert staged[0]._ordinary_pair_state.failed
    assert not parent._ordinary_pair_state.failed
    assert parent._ordinary_pair_state.active is None
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes
    assert not parent.inline.original_numerical_regions
    assert parent.lexical_owner_attempt["outcome"] == "error"
    assert "injected failure after staged" in parent.lexical_owner_attempt["error"]
    parent.scan(nodes)
    outputs, report = parent.finish()
    assert "scope-manifest.json" in outputs
    assert report["lexical_owner_attempt"]["outcome"] == "error"
    assert not parent._ordinary_pair_state.failed
    assert path.read_text() == SOURCE


def test_caught_failure_in_owned_transaction_candidate_cannot_adopt(tmp_path, monkeypatch):
    _path, parent, _selections = subject(tmp_path)
    staged = []
    fail_after_staged_original_unit(monkeypatch, staged)
    with GenerationTransaction(parent) as transaction:
        candidate = transaction.builder
        with pytest.raises(CompilationError, match="injected failure after staged"):
            candidate._attempt_lexical_owner(tuple(candidate.entry.execution.content), finalize=False)
        assert staged == [candidate]
        assert candidate._ordinary_pair_state.failed
        assert not candidate.outputs
        assert not candidate.edits
        assert not candidate.scopes
        with pytest.raises(CompilationError):
            candidate.finish()
        with pytest.raises(CompilationError):
            transaction.adopt(candidate)
    assert not parent._ordinary_pair_state.failed
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes
    assert not parent.inline.original_numerical_regions
