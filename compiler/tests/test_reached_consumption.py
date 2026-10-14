"""A private reached-plan pass authenticates before and after generation."""

from contextlib import ExitStack
from copy import copy, deepcopy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend import native_group_structure, source_effects
from compiler.frontend.reached_windows import ReachedWindowProof, prove_reached_window
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.generation_transaction import GenerationTransaction
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.reached_owner import ReachedWindowOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.support.numerical_regions import (
    CHILD,
    ROOTS,
    EmittedNumericalText,
    emit_combined_team_fixture,
    environment_source,
    reached_numerical_fixture,
)

ENTRY = "lapis_owner::advance"
RESOURCES = ("argument::a", "argument::b")
SOURCE = """module lapis_owner
implicit none
contains
subroutine advance(a,b,n)
real(8),intent(inout)::a(:),b(:)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2.0_8*a(i)
enddo
do i=1,n
a(i)=b(i)+1.0_8
enddo
end subroutine
end module
"""


def bound_window(directory):
    """Fresh small source objects for rejection tests, never a shared token."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "lapis.f90"
    path.write_text(SOURCE)
    fact = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {root: dict(fact) for root in RESOURCES}}
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="reached"))
    originals = tuple(node for node in builder.entry.execution.content
                      if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    proof = prove_reached_window(builder.analysis, ENTRY, originals, RESOURCES, facts)
    return path, builder, proof, originals, builder.bind_reached_window(proof, originals)


def test_numerical_consumption_validates_root_at_entry_and_final_return(tmp_path, monkeypatch):
    _paths, builder, proof, originals, _provider = reached_numerical_fixture(
        tmp_path, child_source=environment_source())
    bound = builder.bind_reached_window(proof, originals)
    validate = ReachedWindowProof.validate
    validations = []

    def counted_validate(self, analysis, procedure, selected):
        if self is proof:
            validations.append((analysis, procedure, selected))
        return validate(self, analysis, procedure, selected)

    monkeypatch.setattr(ReachedWindowProof, "validate", counted_validate)
    with bound.consume_reached_plan() as consumption:
        assert len(validations) == 1
        view = consumption.view
        region = next(region for region in view.numerical_regions.values()
                      if region.procedure == CHILD and region.prepared.extraction.numerical_helpers)
        child_path, child = bound.bind_reached_child(originals[0])
        for _ in range(3):
            assert consumption.check(bound) is proof
            assert bound.validate_reached_window() is proof
            assert bound.reached_plan_record()["proof_identity"] == proof.identity
            assert bound.capture(builder.entry.scope.bindings["a"])["initialized"] == "whole"
            assert view.child(consumption, child_path, originals[0]) is child
            resolved = view.numerical_region(consumption, CHILD, region.selection)
            assert resolved.prepared is region.prepared
            assert resolved.effects is region.effects
            assert view.capture(consumption, ROOTS[2])[0]["initialized"] == "none"
        assert len(validations) == 1
    assert len(validations) == 2
    assert all(analysis is builder.analysis for analysis, _procedure, _selected in validations)
    assert all(selected is originals for _analysis, _procedure, selected in validations)


def test_copied_and_foreign_passes_views_and_builders_cannot_borrow_authority(tmp_path):
    _path, parent, _proof, _originals, bound = bound_window(tmp_path / "owner")
    _foreign_path, _foreign_builder, _foreign_proof, _foreign_originals, foreign = bound_window(tmp_path / "foreign")
    with bound.consume_reached_plan() as consumption:
        with pytest.raises(CompilationError):
            copy(consumption).check(bound)
        with pytest.raises(CompilationError):
            copy(consumption.view).record(consumption)
        with pytest.raises(CompilationError):
            consumption.check(copy(bound))
        with pytest.raises(CompilationError):
            consumption.check(foreign)
        with pytest.raises(CompilationError), GenerationTransaction(bound):
            pytest.fail("an active bound pass was copied into speculative generation")
        with pytest.raises(CompilationError), GenerationTransaction(parent):
            pytest.fail("an active pass's original parent was copied into speculative generation")
        with foreign.consume_reached_plan() as other:
            with pytest.raises(CompilationError):
                consumption.view.record(other)
            with pytest.raises(CompilationError):
                other.view.record(consumption)


def test_expired_facade_and_consumption_cannot_be_reentered_or_read(tmp_path):
    _path, _builder, _proof, _originals, bound = bound_window(tmp_path)
    consumption = bound.consume_reached_plan()
    with consumption:
        view = consumption.view
        assert view.capture(consumption, RESOURCES[0])[0]["initialized"] == "whole"
    with pytest.raises(CompilationError):
        view.record(consumption)
    with pytest.raises(CompilationError):
        consumption.check(bound)
    with pytest.raises(CompilationError), consumption:
        pytest.fail("an expired consumption pass was reentered")


def test_adoption_cannot_publish_an_active_consumption(tmp_path):
    _path, parent, proof, originals, _bound = bound_window(tmp_path)

    def publish_active_candidate():
        with GenerationTransaction(parent) as transaction:
            bound = transaction.builder.bind_reached_window(proof, originals)
            active.enter_context(bound.consume_reached_plan())
            transaction.adopt(bound)

    with ExitStack() as active, pytest.raises(CompilationError, match="completed successful reached consumption"):
        publish_active_candidate()
    assert not parent.outputs
    assert not parent.edits


def test_rejected_mutation_before_owner_return_still_blocks_candidate_adoption(tmp_path, monkeypatch):
    _path, parent, proof, originals, _bound = bound_window(tmp_path)
    parent_state = dict(vars(parent))
    registry = parent.analysis._reached_window_proofs
    original_binding = parent.entry.scope.bindings["n"]
    replacement = copy(original_binding)
    replacement.attributes |= {"volatile"}
    emit = LexicalOwner.run
    staged = []

    def mutate_before_return(self, nodes=None):
        result = emit(self, nodes)
        if isinstance(self, ReachedWindowOwner):
            assert result
            assert self.builder.outputs
            staged.append(self)
            parent.entry.scope.bindings["n"] = replacement
        return result

    monkeypatch.setattr(LexicalOwner, "run", mutate_before_return)
    def publish_failed_candidate():
        with GenerationTransaction(parent) as transaction:
            owner = ReachedWindowOwner(transaction.builder, proof, originals)
            with pytest.raises(CompilationError):
                owner.run()
            assert staged == [owner]
            # The rejected write left source metadata intact, but the failed
            # speculative pass still requires a fresh candidate transaction.
            assert parent.entry.scope.bindings["n"] is original_binding
            transaction.adopt(owner.builder)

    with pytest.raises(CompilationError, match="completed successful reached consumption"):
        publish_failed_candidate()
    assert vars(parent).keys() == parent_state.keys()
    assert all(getattr(parent, name) is value for name, value in parent_state.items())
    assert parent.analysis._reached_window_proofs is registry
    assert not parent.outputs
    assert not parent.edits
    assert not parent.scopes


def test_failed_direct_owner_cannot_publish_through_its_original_parent(tmp_path, monkeypatch):
    path, parent, proof, originals, _bound = bound_window(tmp_path)
    sibling = parent.bind_reached_window(proof, originals)
    original_binding = parent.entry.scope.bindings["n"]
    replacement = copy(original_binding)
    replacement.attributes |= {"volatile"}
    emit = LexicalOwner.run
    staged = []

    def mutate_before_return(self, nodes=None):
        result = emit(self, nodes)
        if isinstance(self, ReachedWindowOwner):
            assert result
            staged.append(self)
            with pytest.raises(CompilationError):
                sibling.finish()
            with pytest.raises(CompilationError), GenerationTransaction(sibling):
                pytest.fail("a sibling of an active pass was copied into speculative generation")
            parent.entry.scope.bindings["n"] = replacement
        return result

    monkeypatch.setattr(LexicalOwner, "run", mutate_before_return)
    owner = ReachedWindowOwner(parent, proof, originals)
    with pytest.raises(CompilationError):
        owner.run()
    assert staged == [owner]
    assert parent.entry.scope.bindings["n"] is original_binding
    with pytest.raises(CompilationError):
        parent.finish()
    with pytest.raises(CompilationError):
        owner.builder.finish()
    with pytest.raises(CompilationError):
        sibling.finish()
    with pytest.raises(CompilationError), GenerationTransaction(parent):
        pytest.fail("a failed direct pass was copied into speculative generation")
    with pytest.raises(CompilationError), GenerationTransaction(sibling):
        pytest.fail("a sibling of a failed pass was copied into speculative generation")
    assert path.read_text() == SOURCE
    assert set(path.parent.iterdir()) == {path}


def test_caught_refresh_failure_blocks_publication_before_outer_pass_exit(tmp_path):
    _path, parent, proof, originals, _bound = bound_window(tmp_path)
    rejected = []

    def refresh_then_try_to_publish():
        with GenerationTransaction(parent) as transaction:
            bound = transaction.builder.bind_reached_window(proof, originals)
            with bound.consume_reached_plan() as consumption:
                with pytest.raises(CompilationError):
                    consumption.refresh(copy(proof), ())
                with pytest.raises(CompilationError):
                    bound.finish()
                with pytest.raises(CompilationError):
                    transaction.builder.finish()
                with pytest.raises(CompilationError):
                    transaction.adopt(bound)
                rejected.append("before outer pass exit")

    with pytest.raises(CompilationError):
        refresh_then_try_to_publish()
    assert rejected == ["before outer pass exit"]
    assert not parent.outputs
    assert not parent.edits


def test_public_and_pass_state_copies_cannot_mutate_registered_coverage(tmp_path):
    _path, _builder, proof, _originals, bound = bound_window(tmp_path)
    original = deepcopy(proof.public())
    public = proof.public()
    public["states"][0]["initialized"] = "none"
    public["selected_node_ids"].clear()
    assert proof.public() == original
    with bound.consume_reached_plan() as consumption:
        first, sections = consumption.view.capture(consumption, RESOURCES[0])
        assert sections is None
        first["initialized"] = "none"
        record = consumption.view.record(consumption)
        record["states"].clear()
        second, _sections = consumption.view.capture(consumption, RESOURCES[0])
        assert second["initialized"] == "whole"
        assert consumption.view.record(consumption) == original
        with pytest.raises(TypeError):
            consumption.view.states[RESOURCES[0]] = first
    assert proof.public() == original


def test_session_proof_access_remains_immutable_after_private_pass(tmp_path):
    _path, builder, proof, _originals, bound = bound_window(tmp_path)
    with bound.consume_reached_plan():
        assert bound.capture(builder.entry.scope.bindings["a"])["initialized"] == "whole"
    state = proof.capture_state(RESOURCES[0])
    state["initialized"] = "none"
    assert proof.capture_state(RESOURCES[0])["initialized"] == "whole"
    binding = copy(builder.entry.scope.bindings["n"])
    binding.attributes |= {"volatile"}
    with pytest.raises(CompilationError):
        builder.entry.scope.bindings["n"] = binding
    assert proof.capture_state(RESOURCES[0])["initialized"] == "whole"


@pytest.mark.parametrize("phase", ["entry", "final"])
def test_original_source_budget_failure_is_hard_in_each_validation_phase(tmp_path, phase):
    _path, builder, _proof, _originals, bound = bound_window(tmp_path)

    def exhausted_attempt():
        if phase == "entry":
            builder.analysis.operation_limit = 1
        with bound.consume_reached_plan():
            if phase == "final":
                builder.analysis.operation_limit = 1

    with pytest.raises(CompilationError):
        exhausted_attempt()
    assert not builder.outputs
    assert not builder.edits


def test_private_index_budget_cannot_be_downgraded_to_native_support(tmp_path, monkeypatch):
    _path, _builder, proof, _originals, bound = bound_window(tmp_path)
    validate = ReachedWindowProof.validate

    def exhaust_after_authentication(self, *args):
        result = validate(self, *args)
        if self is proof:
            monkeypatch.setattr(native_group_structure, "NATIVE_GROUP_SCAN_LIMIT", 1)
        return result

    monkeypatch.setattr(ReachedWindowProof, "validate", exhaust_after_authentication)
    with pytest.raises(CompilationError, match="consumption index budget exhausted"), bound.consume_reached_plan():
        pytest.fail("an exhausted private index was consumed")


def test_fresh_source_authority_keeps_identical_numerical_workers_and_private_public_json(tmp_path, monkeypatch):
    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    paths, outputs, report, first_owner = emit_combined_team_fixture(tmp_path)
    first = EmittedNumericalText.capture(paths, outputs, report)
    monkeypatch.setattr(source_effects, "_DEFAULT_SUMMARY_CACHE", SummaryCache())
    paths, outputs, report, second_owner = emit_combined_team_fixture(tmp_path)
    second = EmittedNumericalText.capture(paths, outputs, report)
    assert first_owner.builder.analysis is not second_owner.builder.analysis
    assert first_owner.window_proof is not second_owner.window_proof
    assert first_owner.window_proof.identity != second_owner.window_proof.identity
    assert first.original_sources == second.original_sources
    first_outputs, second_outputs = dict(first.artifacts), dict(second.artifacts)
    assert first_outputs.keys() == second_outputs.keys()
    first_manifest, second_manifest = first.manifest(), second.manifest()
    replacements = {source["replacement"] for source in first_manifest["sources"].values()}
    assert replacements == {source["replacement"] for source in second_manifest["sources"].values()}
    # Exact source-object identities are deliberately different across fresh
    # analyses. They remain visible in owner/trace source and the manifest;
    # every other artifact, including numerical workers, must be byte equal.
    identity_artifacts = {"scope-manifest.json", *replacements}
    for name in first_outputs.keys() - identity_artifacts:
        assert first_outputs[name] == second_outputs[name], name
    workers = {source["path"] for source in first_manifest["build_sources"]
               if source["language"] == "cuda" and source.get("numerical_contract")}
    assert workers
    assert not workers & identity_artifacts
    public = first.manifest()
    public["scopes"].clear()
    assert first.manifest() == first_manifest
    assert second.manifest() == second_manifest
    assert (first_manifest["scopes"][0]["reached_window"]["proof_identity"]
            != second_manifest["scopes"][0]["reached_window"]["proof_identity"])
