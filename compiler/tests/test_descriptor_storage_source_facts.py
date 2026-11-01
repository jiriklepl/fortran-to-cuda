"""Sealed source facts separate semantic issuance from generation consumption."""
from copy import copy

import pytest

from compiler.frontend import descriptor_windows as windows
from compiler.frontend import storage_mutation_effects as storage
from compiler.frontend.descriptor_record_calls import original_record_call
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_record_calls import fixture as record_fixture
from compiler.tests.test_descriptor_window_fresh_derivation import ENTRY, ROOT, fixture, issue


def record_call(tmp_path):
    data = record_fixture(tmp_path, sealed=True)
    analysis, caller = data[:2]
    call, = caller.execution.content
    return analysis, original_record_call(analysis, caller, call,
        (analysis.modules["regions"].bindings["field"],))


def test_descriptor_and_storage_consumption_do_not_repeat_semantic_closure(tmp_path, monkeypatch):
    data = fixture(tmp_path, sealed=True)
    analysis = data[1]
    proof = issue(data)
    effects = storage.prove_storage_mutation_effects(analysis, ENTRY, selection=data[3])
    public, effect_public = proof.public(), effects.public()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("issued immutable fact must not rederive source")

    monkeypatch.setattr(windows, "_derive", unexpected)
    monkeypatch.setattr(storage, "_derive", unexpected)
    assert issue(data) is proof
    assert proof.validate(analysis, ENTRY, data[3]) is proof
    assert proof.resource_state(ROOT)["resource"] == ROOT
    assert storage.prove_storage_mutation_effects(analysis, ENTRY, selection=data[3]) is effects
    assert effects.preserves_descriptors()
    assert proof.public() == public
    assert effects.public() == effect_public


def test_record_call_consumers_reuse_independent_team_and_full_storage_facts(tmp_path, monkeypatch):
    import compiler.frontend.descriptor_teams as teams
    analysis, proof = record_call(tmp_path)
    public = proof.public()

    def unexpected(*_args, **_kwargs):
        raise AssertionError("record-call consumer must not rescan its issued prerequisites")

    monkeypatch.setattr(storage, "_derive", unexpected)
    monkeypatch.setattr(teams, "_derive", unexpected)
    assert proof.validate() is proof
    assert proof._storage.preserves_descriptors()
    assert proof.public() == public
    assert analysis.source_authority_context().derivation_arena(analysis).derivation_count(
        "original_storage_mutation") == 1


@pytest.mark.parametrize("family", ["descriptor", "storage"])
@pytest.mark.parametrize("mutation", ["operation_limit", "contracts", "projection"])
def test_supported_policy_or_projection_replacement_retires_old_fact_context(tmp_path, family, mutation):
    data = fixture(tmp_path, sealed=True)
    analysis = data[1]
    proof = issue(data) if family == "descriptor" else storage.prove_storage_mutation_effects(analysis, ENTRY)
    if mutation == "operation_limit":
        analysis.operation_limit += 1
    elif mutation == "contracts":
        analysis.contracts = {"different_policy": {"opaque": True}}
    else:
        routine = copy(analysis.routines[ENTRY])
        analysis.project_routine(ENTRY, routine)
    with pytest.raises(CompilationError):
        proof.validate(analysis, ENTRY, data[3]) if family == "descriptor" else proof.validate()


@pytest.mark.parametrize("family", ["descriptor", "storage"])
def test_foreign_analysis_and_registered_copy_do_not_acquire_fact_handles(tmp_path, family):
    data = fixture(tmp_path, sealed=True)
    path, analysis = data[:2]
    proof = issue(data) if family == "descriptor" else storage.prove_storage_mutation_effects(analysis, ENTRY)
    other = SourceEffects([path])
    other.seal_session()
    with pytest.raises(CompilationError):
        proof.validate(other, ENTRY, data[3]) if family == "descriptor" else proof.validate(other)
    forged = copy(proof)
    registry = analysis._descriptor_window_proofs if family == "descriptor" else analysis._storage_mutation_effects
    registry[proof.identity] = forged
    with pytest.raises(CompilationError, match="arena-issued handle"):
        forged.validate(analysis, ENTRY, data[3]) if family == "descriptor" else forged.validate()


@pytest.mark.parametrize("mutation", ["remove", "replace", "registry", "owner_policy"])
def test_consumed_record_call_prerequisite_changes_reject(tmp_path, mutation):
    analysis, proof = record_call(tmp_path)
    dependency = proof._storage._source_fact.dependencies[0]
    if mutation == "remove":
        del dependency.registry[dependency.key]
    elif mutation == "replace":
        dependency.registry[dependency.key] = copy(dependency.token)
    elif mutation == "registry":
        setattr(dependency.owner, dependency.name, dict(dependency.registry))
    else:
        dependency.owner.operation_limit += 1
    with pytest.raises(CompilationError):
        proof.validate()


def test_unrelated_registry_growth_preserves_existing_fact(tmp_path):
    data = fixture(tmp_path, sealed=True, statements="a(1)=1\na(2)=2")
    first = issue(data, (data[3][0],))
    second = issue(data, (data[3][1],))
    assert first is not second
    assert first.validate(data[1], ENTRY, first._selected) is first


@pytest.mark.parametrize("family", ["descriptor", "storage"])
def test_changed_live_source_is_rejected_at_publication_after_fact_reuse(tmp_path, family):
    data = fixture(tmp_path, sealed=True)
    proof = issue(data) if family == "descriptor" else storage.prove_storage_mutation_effects(data[1], ENTRY)
    data[0].write_text(data[0].read_text() + "! changed original input\n")
    assert (proof.validate(data[1], ENTRY, data[3]) if family == "descriptor" else proof.validate()) is proof
    with pytest.raises(CompilationError, match="source changed"):
        data[1].verify_publication()


def test_failed_descriptor_issuance_does_not_publish_an_arena_fact(tmp_path):
    data = fixture(tmp_path, sealed=True, statements="call unknown(a)")
    context = data[1].source_authority_context()
    with pytest.raises(CompilationError):
        issue(data)
    assert context.derivation_arena(data[1]).derivation_count("original_descriptor_window") == 0
    assert not data[1]._descriptor_window_proofs


def test_generation_rollback_drops_new_handle_and_preserves_parent_fact(tmp_path):
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.tests.test_graph_storage_epochs import builder_at
    _path, builder = builder_at(tmp_path)
    analysis = builder.analysis
    procedure = "epochs::advance"
    graph = analysis.structure(procedure)
    calls = tuple(node.id for node in graph.nodes.values() if node.kind == "call")
    assert len(calls) == 3
    first = windows.prove_descriptor_window(analysis, procedure, (calls[0],), ("argument::right",))
    with GenerationTransaction(builder):
        assert first.validate(analysis, procedure, first._selected) is first
        temporary = windows.prove_descriptor_window(analysis, procedure, (calls[1],),
                                                    ("argument::left", "argument::right"))
    assert first.validate(analysis, procedure, first._selected) is first
    with pytest.raises(CompilationError):
        temporary.validate(analysis, procedure, temporary._selected)
    replacement = windows.prove_descriptor_window(analysis, procedure, (calls[1],),
                                                  ("argument::left", "argument::right"))
    assert replacement is not temporary
    assert replacement._source_fact.derived is temporary._source_fact.derived
    assert replacement.validate(analysis, procedure, replacement._selected) is replacement


def test_record_dependencies_remain_usable_inside_generation_fork(tmp_path):
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.tests.test_descriptor_record_calls import emitted_source
    from compiler.tests.test_graph_storage_epochs import builder_at
    source = emitted_source().replace("subroutine record_wrapper(a)\nreal(8),intent(inout)::a(:)",
                                      "subroutine record_wrapper(a)\nreal(8),allocatable,intent(inout)::a(:)")
    _path, builder = builder_at(tmp_path, source)
    analysis = builder.analysis
    routine = analysis.routines["epochs::record_wrapper"]
    call, = routine.execution.content
    proof = original_record_call(analysis, routine, call, (routine.scope.bindings["a"],))
    with GenerationTransaction(builder):
        assert proof.validate() is proof
    assert proof.validate() is proof


def test_resource_lookup_copies_only_the_selected_immutable_subtree(tmp_path, monkeypatch):
    data = fixture(tmp_path, sealed=True)
    proof = issue(data)
    original = windows._thaw
    seen = []

    def thaw(value):
        seen.append(value)
        assert value is not proof._record
        return original(value)

    monkeypatch.setattr(windows, "_thaw", thaw)
    state = proof.resource_state(ROOT)
    state["resource"] = "changed returned copy"
    assert proof.resource_state(ROOT)["resource"] == ROOT
    assert seen


def test_discarded_prerequisite_cannot_survive_by_rebinding_its_registry(tmp_path):
    from compiler.frontend.source_facts import token_dependencies
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.tests.test_graph_storage_epochs import builder_at
    _path, builder = builder_at(tmp_path)
    analysis = builder.analysis
    procedure = "epochs::advance"
    call = next(node.id for node in analysis.structure(procedure).nodes.values() if node.kind == "call")
    analysis.source_authority_context()
    with GenerationTransaction(builder):
        proof = windows.prove_descriptor_window(analysis, procedure, (call,), ("argument::right",))
        dependency, = token_dependencies((proof,))
    setattr(analysis, dependency.name, dependency.registry)
    assert dependency.registry[dependency.key] is proof
    with pytest.raises(CompilationError, match="generation branch"):
        dependency.require()


@pytest.mark.parametrize("family", ["descriptor", "storage"])
def test_sealing_requires_a_new_handle_without_retargeting_diagnostic_proof(tmp_path, family):
    data = fixture(tmp_path)
    analysis = data[1]
    old = issue(data) if family == "descriptor" else storage.prove_storage_mutation_effects(analysis, ENTRY)
    analysis.seal_session()
    with pytest.raises(CompilationError, match="mutable diagnostic"):
        old.validate(analysis, ENTRY, data[3]) if family == "descriptor" else old.validate()
    fresh = issue(data) if family == "descriptor" else storage.prove_storage_mutation_effects(analysis, ENTRY)
    assert fresh is not old
    assert old._source_fact is None
    ignored = {"proof_identity", "analysis_identity", "structured_identity"} if family == "descriptor" else set()
    assert {key: value for key, value in fresh.public().items() if key not in ignored} == {
        key: value for key, value in old.public().items() if key not in ignored}


def test_nonrecord_original_call_dependency_keeps_exact_issuer_entry(tmp_path):
    from compiler.tests.test_descriptor_record_calls import scalar_closure_source
    analysis, caller, *_rest = record_fixture(tmp_path, scalar_closure_source(1), sealed=True)
    call, = caller.execution.content
    proof = original_record_call(analysis, caller, call, (analysis.modules["regions"].bindings["field"],))
    token, = proof._storage._calls
    assert not hasattr(token, "_record")
    analysis._original_call_associations[id(analysis), id(token._original)] = copy(token)
    with pytest.raises(CompilationError):
        proof.validate()


def test_existing_arena_prerequisite_keeps_its_generation_handle(tmp_path):
    from compiler.frontend.source_facts import token_dependencies
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.tests.test_descriptor_record_calls import emitted_source
    from compiler.tests.test_graph_storage_epochs import builder_at

    source = emitted_source().replace("subroutine record_wrapper(a)\nreal(8),intent(inout)::a(:)",
                                      "subroutine record_wrapper(a)\nreal(8),allocatable,intent(inout)::a(:)")
    _path, builder = builder_at(tmp_path, source)
    analysis = builder.analysis
    analysis.source_authority_context().derivation_arena(analysis)
    routine = analysis.routines["epochs::record_wrapper"]
    call, = routine.execution.content
    with GenerationTransaction(builder):
        proof = original_record_call(analysis, routine, call, (routine.scope.bindings["a"],))
        dependency, = token_dependencies((proof._region,))
    setattr(analysis, dependency.name, dependency.registry)
    setattr(analysis, dependency.name + "_owner", dependency.marker)
    assert dependency.registry[dependency.key] is proof._region
    with pytest.raises(CompilationError, match="generation"):
        dependency.require()
