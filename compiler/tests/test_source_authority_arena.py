"""Closed source contexts own derivations; generated candidates own handles."""
from copy import copy
from hashlib import sha256

import pytest

from compiler.frontend import local_allocatable_storage as allocation
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_local_allocatable_storage import SOURCE, fixture


def test_complete_allocation_derivation_runs_once_per_immutable_context(tmp_path, monkeypatch):
    parts = fixture(tmp_path, sealed=True)
    _path, analysis, routine, binding = parts
    original = allocation._derive
    calls = []

    def derive(reader, *args, **kwargs):
        assert reader is not analysis
        assert reader._source_authority_reader
        calls.append(reader.source_authority_context())
        return original(reader, *args, **kwargs)

    monkeypatch.setattr(allocation, "_derive", derive)
    proof = allocation.original_local_allocation(analysis, routine, binding)
    record = proof.public()
    for _ in range(5):
        assert proof.validate(analysis, routine, binding) is proof
        assert allocation.original_local_allocation(analysis, routine, binding) is proof
        checkpoint = proof.checkpoint()
        proof.require_checkpoint(checkpoint)
    assert len(calls) == 1
    assert proof.public() == record
    assert proof._arena.derivation_count("original_local_allocation") == 1


def test_sealed_allocation_never_takes_a_broad_raw_namespace_snapshot(tmp_path, monkeypatch):
    _path, analysis, routine, binding = fixture(tmp_path, sealed=True)

    def unexpected(*args, **kwargs):
        pytest.fail("closed source authority was walked again")

    monkeypatch.setattr(allocation, "_raw_authority", unexpected)
    proof = allocation.original_local_allocation(analysis, routine, binding)
    saved = allocation.allocation_source_checkpoint(analysis)
    proof.validate(analysis, routine, binding)
    allocation.require_allocation_source_checkpoint(analysis, saved)


def test_policy_replacement_creates_context_and_invalidates_old_handle(tmp_path):
    _path, analysis, routine, binding = fixture(tmp_path, sealed=True)
    proof = allocation.original_local_allocation(analysis, routine, binding)
    context = proof._context
    analysis.operation_limit += 1
    with pytest.raises(CompilationError, match="stale or foreign"):
        proof.validate(analysis, routine, binding)
    second = allocation.original_local_allocation(analysis, routine, binding)
    assert second is not proof
    assert second._context is not context
    assert second.public() == proof.public()
    assert second.validate(analysis, routine, binding) is second


def test_policy_containers_are_owned_and_immutable(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text(SOURCE)
    contracts = {"test": {"values": [1]}}
    analysis = SourceEffects([path], contracts=contracts)
    analysis.seal_session()
    context = analysis.source_authority_context()
    contracts["test"]["values"].append(2)
    assert list(analysis.contracts["test"]["values"]) == [1]
    with pytest.raises(CompilationError, match="immutable"):
        analysis.contracts["test"]["values"].append(3)
    analysis.contracts = {"test": {"values": [2]}}
    assert analysis.source_authority_context() is not context


def test_read_facade_rejects_authority_mutation_and_detaches_work_caches(tmp_path):
    _path, analysis, routine, _binding = fixture(tmp_path, sealed=True)
    context = analysis.source_authority_context()
    reader = context.reader(analysis)
    with pytest.raises(CompilationError, match="reader is immutable"):
        reader.operation_limit = 10000
    reader._original_call_associations["private"] = {}
    assert "private" not in getattr(analysis, "_original_call_associations", {})
    assert not hasattr(reader, "summaries")
    assert reader.routines is analysis.routines
    assert reader._require_original(routine.qualified) is routine


def test_original_parser_parent_and_namespace_replacement_are_not_supported(tmp_path):
    _path, analysis, routine, _binding = fixture(tmp_path, sealed=True)
    node = routine.execution.content[0]
    original = node.parent
    with pytest.raises(CompilationError, match="immutable"):
        node.parent = None
    assert analysis._source_session.lexical_parent(node) is original
    with pytest.raises(CompilationError, match="immutable"):
        analysis.modules = dict(analysis.modules)


def test_routine_projection_is_persistent_owned_and_has_no_original_authority(tmp_path):
    _path, analysis, routine, binding = fixture(tmp_path, sealed=True)
    proof = allocation.original_local_allocation(analysis, routine, binding)
    projected = analysis.private_projection(routines=True)
    before = projected.routines
    local = copy(routine)
    local.arguments = ()
    projected.project_routine(routine.qualified, local)
    assert before[routine.qualified] is routine
    assert projected.routines[routine.qualified] is local
    assert analysis.routines[routine.qualified] is routine
    with pytest.raises(CompilationError, match="immutable"):
        local.arguments = routine.arguments
    with pytest.raises(CompilationError, match="immutable"):
        projected.routines[routine.qualified] = routine
    with pytest.raises(CompilationError):
        allocation.original_local_allocation(projected, local, binding)
    assert proof.validate(analysis, routine, binding) is proof


def test_equal_public_foreign_session_or_copied_handle_is_not_authority(tmp_path):
    path, analysis, routine, binding = fixture(tmp_path, sealed=True)
    proof = allocation.original_local_allocation(analysis, routine, binding)
    other = SourceEffects([path])
    other.seal_session()
    assert other.session_identity == analysis.session_identity
    with pytest.raises(CompilationError, match="foreign"):
        proof.validate(other, other.routines[routine.qualified],
                       other.routines[routine.qualified].scope.bindings[binding.name])
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        copy(proof).validate(analysis, routine, binding)


def test_unknown_escape_is_still_rejected_by_first_complete_derivation(tmp_path):
    source = SOURCE.replace("call consumer(backing,n)", "call consumer(backing,n)\ncall unknown(backing)")
    _path, analysis, routine, binding = fixture(tmp_path, source, sealed=True)
    context = analysis.source_authority_context()
    for _ in range(2):
        with pytest.raises(CompilationError):
            allocation.original_local_allocation(analysis, routine, binding)
    assert context.derivation_arena(analysis).derivation_count("original_local_allocation") == 0


def test_live_source_is_verified_at_publication_even_after_cached_validation(tmp_path):
    path, analysis, routine, binding = fixture(tmp_path, sealed=True)
    proof = allocation.original_local_allocation(analysis, routine, binding)
    path.write_text(path.read_text() + "! changed after captured source\n")
    assert proof.validate(analysis, routine, binding) is proof
    with pytest.raises(CompilationError, match="source changed"):
        analysis.verify_publication()


def test_transaction_rollback_drops_child_handles_and_keeps_parent_facts(tmp_path):
    from compiler._legacy.config import LegacyCompilerOptions, LegacyOffloadConfig
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.scopes.source import ScopeBuilder

    path = tmp_path / "source.f90"
    path.write_text(SOURCE.replace("integer::stage", "integer::stage\nreal(8),save,allocatable::scratch(:)"))
    facts = {"schema_version": 1, "participation": "serial", "sources": {
        str(path): sha256(path.read_bytes()).hexdigest()}, "captures": {}}
    parent = ScopeBuilder([path], "local_storage::advance", facts=facts,
                          options=LegacyCompilerOptions(),
                          config=LegacyOffloadConfig(policy="sections", scope_execution="graph"))
    analysis, routine = parent.analysis, parent.entry
    backing, scratch = (routine.scope.bindings[name] for name in ("backing", "scratch"))
    first = allocation.original_local_allocation(analysis, routine, backing)
    parent_arena = first._arena
    with GenerationTransaction(parent) as transaction:
        assert transaction.builder.analysis is analysis
        assert first.validate(analysis, routine, backing) is first
        temporary = allocation.original_local_allocation(analysis, routine, scratch)
        assert temporary._arena is not parent_arena
        assert temporary.validate(analysis, routine, scratch) is temporary
    assert first.validate(analysis, routine, backing) is first
    assert analysis._source_derivation_arena is parent_arena
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        temporary.validate(analysis, routine, scratch)
    replacement = allocation.original_local_allocation(analysis, routine, scratch)
    assert replacement is not temporary
    assert replacement._derived is temporary._derived
    assert replacement.validate(analysis, routine, scratch) is replacement
    assert parent_arena.derivation_count("original_local_allocation") == 2


def test_copy_registered_in_mutable_analysis_map_is_not_arena_authority(tmp_path):
    _path, analysis, routine, binding = fixture(tmp_path, sealed=True)
    proof = allocation.original_local_allocation(analysis, routine, binding)
    forged = copy(proof)
    analysis._local_allocation_proofs[proof.identity] = forged
    with pytest.raises(CompilationError, match="exact arena-issued"):
        forged.validate(analysis, routine, binding)


def test_supported_context_replacement_cannot_install_a_foreign_arena(tmp_path):
    _path, analysis, _routine, _binding = fixture(tmp_path, sealed=True)
    context = analysis.source_authority_context()
    for name, value in (("_source_authority_context", context), ("_source_derivation_arena", context.arena)):
        with pytest.raises(CompilationError, match="ownership is immutable"):
            setattr(analysis, name, value)


@pytest.mark.parametrize("field", ["source_graph_operation_limit", "_descriptor_source"])
def test_every_semantic_context_field_participates_in_its_key(tmp_path, field):
    _path, analysis, routine, binding = fixture(tmp_path, sealed=True)
    first = allocation.original_local_allocation(analysis, routine, binding)
    setattr(analysis, field, {routine.qualified: 1} if field == "source_graph_operation_limit" else analysis)
    context = analysis.source_authority_context()
    assert context is not first._context
    with pytest.raises(CompilationError, match="stale or foreign"):
        first.validate(analysis, routine, binding)
    if field == "_descriptor_source":
        with pytest.raises(CompilationError, match="unchanged original declaring routine"):
            allocation.original_local_allocation(analysis, routine, binding)
    else:
        second = allocation.original_local_allocation(analysis, routine, binding)
        assert second._context is context
        assert context.policy[field][routine.qualified] == 1


def test_normal_state_dictionary_access_cannot_change_closed_authority(tmp_path):
    _path, analysis, _routine, _binding = fixture(tmp_path, sealed=True)
    reader = analysis.source_authority_context().reader(analysis)
    for value in (analysis, reader):
        with pytest.raises(TypeError):
            vars(value)["operation_limit"] = 10000
        value.__getstate__()["operation_limit"] = 10000
        assert value.operation_limit != 10000
        with pytest.raises(CompilationError, match="ownership is immutable"):
            value.__dict__ = {**value.__getstate__(), "operation_limit": 10000}
        with pytest.raises(CompilationError, match="cannot be deleted"):
            del value.operation_limit


def test_equivalent_native_forks_share_one_fact_and_issue_distinct_handles(tmp_path):
    from compiler.scopes.segments import _native_analysis
    _path, analysis, routine, binding = fixture(tmp_path, sealed=True)
    first = _native_analysis(analysis).private_projection(routines=True)
    second = _native_analysis(analysis)
    first._native_metadata = second._native_metadata = True
    left = allocation.original_local_allocation(first, routine, binding)
    right = allocation.original_local_allocation(second, routine, binding)
    assert first.routines is not second.routines
    assert left._context is right._context
    assert left is not right
    assert left._derived is right._derived
    assert left._arena.derivation_count("original_local_allocation") == 1
    assert left.validate(first, routine, binding) is left
    assert right.validate(second, routine, binding) is right
    with pytest.raises(CompilationError):
        left.validate(second, routine, binding)
    second.operation_limit += 1
    changed = allocation.original_local_allocation(second, routine, binding)
    assert changed._context is not left._context


def test_unsealed_diagnostic_handle_cannot_enter_production_arena(tmp_path):
    _path, analysis, routine, binding = fixture(tmp_path)
    mutable = allocation.original_local_allocation(analysis, routine, binding)
    analysis.seal_session()
    with pytest.raises(CompilationError, match="mutable diagnostic"):
        mutable.validate(analysis, routine, binding)
    immutable = allocation.original_local_allocation(analysis, routine, binding)
    assert immutable is not mutable
    assert immutable._context is not None
    assert immutable.public() == mutable.public()


def test_process_local_context_keys_do_not_enter_public_proof_or_graph_records(tmp_path):
    path, first, routine, binding = fixture(tmp_path, sealed=True)
    second = SourceEffects([path])
    second.seal_session()
    other = second.routines[routine.qualified]
    left = allocation.original_local_allocation(first, routine, binding)
    right = allocation.original_local_allocation(second, other, other.scope.bindings[binding.name])
    assert left._context is not right._context
    assert left._context._intern_key != right._context._intern_key
    assert left.public() == right.public()
    assert first.structure(routine.qualified).public() == second.structure(routine.qualified).public()


def test_rollback_reuses_one_complete_fact_with_a_new_legal_handle(tmp_path):
    from compiler._legacy.config import LegacyCompilerOptions, LegacyOffloadConfig
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.scopes.source import ScopeBuilder

    path = tmp_path / "source.f90"
    path.write_text(SOURCE)
    facts = {"schema_version": 1, "participation": "serial", "sources": {
        str(path): sha256(path.read_bytes()).hexdigest()}, "captures": {}}
    parent = ScopeBuilder([path], "local_storage::advance", facts=facts,
                          options=LegacyCompilerOptions(),
                          config=LegacyOffloadConfig(policy="sections", scope_execution="graph"))
    analysis, routine = parent.analysis, parent.entry
    binding = routine.scope.bindings["backing"]
    context = analysis.source_authority_context()
    with GenerationTransaction(parent):
        temporary = allocation.original_local_allocation(analysis, routine, binding)
    with pytest.raises(CompilationError):
        temporary.validate(analysis, routine, binding)
    replacement = allocation.original_local_allocation(analysis, routine, binding)
    assert replacement is not temporary
    assert replacement._derived is temporary._derived
    assert replacement.validate(analysis, routine, binding) is replacement
    assert context.arena.derivation_count("original_local_allocation") == 1


def test_recursive_equivalent_fork_cannot_publish_an_incomplete_fact(tmp_path, monkeypatch):
    _path, analysis, routine, binding = fixture(tmp_path, sealed=True)
    context = analysis.source_authority_context()
    original = allocation._derive

    def recursive(reader, *args, **kwargs):
        sibling = analysis.private_projection(routines=True)
        allocation.original_local_allocation(sibling, routine, binding)
        return original(reader, *args, **kwargs)

    monkeypatch.setattr(allocation, "_derive", recursive)
    with pytest.raises(CompilationError, match="recursive source derivation"):
        allocation.original_local_allocation(analysis, routine, binding)
    assert context.arena.derivation_count("original_local_allocation") == 0
    monkeypatch.setattr(allocation, "_derive", original)
    assert allocation.original_local_allocation(analysis, routine, binding).validate(
        analysis, routine, binding)
    assert context.arena.derivation_count("original_local_allocation") == 1


def test_existing_joined_prerequisite_is_fully_imported_once(tmp_path, monkeypatch):
    from compiler.frontend.descriptor_teams import DescriptorTeamProof
    from compiler.tests.test_local_allocatable_storage import joined_source, joined_teams

    parts = fixture(tmp_path, joined_source(), sealed=True)
    _path, analysis, routine, binding = parts
    team, = joined_teams(parts)
    original = DescriptorTeamProof.validate
    calls = []

    def validate(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DescriptorTeamProof, "validate", validate)
    proof = allocation.original_local_allocation(analysis, routine, binding, joined_storage=(team,))
    first = len(calls)
    assert first > 0
    for _ in range(3):
        assert allocation.original_local_allocation(analysis, routine, binding, joined_storage=(team,)) is proof
        proof.validate(analysis, routine, binding)
    assert len(calls) == first
    assert proof._derived.dependencies[0].record == allocation._encoded(team.public())
    assert proof.public()["joined_storage"] == [team.public()]
    assert proof._context.arena.derivation_count("local_allocation_joined_storage") == 1
    assert proof._context.arena.derivation_count("original_local_allocation") == 1


@pytest.mark.parametrize("change", ["copy", "foreign", "unknown_tail"])
def test_imported_joined_capability_does_not_skip_original_authority_or_tail(tmp_path, change):
    from compiler.tests.test_local_allocatable_storage import joined_source, joined_teams

    source = joined_source()
    if change == "unknown_tail":
        source = source.replace("call consumer(backing,n)", "call consumer(backing,n)\ncall unknown(backing)")
    parts = fixture(tmp_path, source, sealed=True)
    _path, analysis, routine, binding = parts
    team, = joined_teams(parts)
    if change == "copy":
        team = copy(team)
    elif change == "foreign":
        other = tmp_path / "other"
        other.mkdir()
        team, = joined_teams(fixture(other, joined_source(), sealed=True))
    with pytest.raises(CompilationError):
        allocation.original_local_allocation(analysis, routine, binding, joined_storage=(team,))
    assert analysis.source_authority_context().arena.derivation_count("original_local_allocation") == 0


@pytest.mark.parametrize("before_seal", [False, True])
def test_mutable_stable_root_input_is_detached_before_context_interning(tmp_path, before_seal):
    _path, analysis, routine, binding = fixture(tmp_path, sealed=not before_seal)
    retained = {"other_module::allocation"}
    analysis._stable_module_allocatables = retained
    if before_seal:
        analysis.seal_session()
    context = analysis.source_authority_context()
    proof = allocation.original_local_allocation(analysis, routine, binding)
    retained.add("other_module::late")
    assert analysis._stable_module_allocatables == frozenset({"other_module::allocation"})
    assert context.policy["_stable_module_allocatables"] is analysis._stable_module_allocatables
    assert analysis.source_authority_context() is context
    assert proof.validate(analysis, routine, binding) is proof


@pytest.mark.parametrize(("name", "value"), [
    ("_native_metadata", []), ("_native_metadata", 1),
    ("_stable_module_allocatables", ["module::allocation"]),
    ("_stable_module_allocatables", {1}),
    ("depth_limit", True), ("procedure_limit", 0),
    ("source_graph_operation_limit", {"local_storage::advance": []}),
    ("_descriptor_source", []), ("_selected_descriptor_window", []),
])
def test_noncanonical_mutable_context_inputs_reject_before_invalidation(tmp_path, name, value):
    _path, analysis, routine, binding = fixture(tmp_path, sealed=True)
    proof = allocation.original_local_allocation(analysis, routine, binding)
    context = proof._context
    with pytest.raises(CompilationError, match="source context|source graph caps"):
        setattr(analysis, name, value)
    assert analysis.source_authority_context() is context
    assert proof.validate(analysis, routine, binding) is proof


def test_mutable_persistent_wrapper_cannot_bypass_projection_ownership(tmp_path):
    from compiler.frontend.source_session import ProjectedRoutines

    _path, analysis, routine, binding = fixture(tmp_path, sealed=True)
    proof = allocation.original_local_allocation(analysis, routine, binding)
    local = copy(routine)
    wrapper = ProjectedRoutines(analysis._source_session.roots["routines"], {routine.qualified: local})
    with pytest.raises(CompilationError, match="source registry is immutable"):
        analysis.routines = wrapper
    local.arguments = ()
    assert proof.validate(analysis, routine, binding) is proof
    projected = analysis.private_projection(routines=True)
    projected.project_routine(routine.qualified, local)
    assert analysis._source_session.owns_projection(local)
    with pytest.raises(CompilationError, match="immutable"):
        local.arguments = routine.arguments
    assert projected.source_authority_context() is not proof._context


def test_two_completed_owners_keep_their_contexts_through_final_publication(tmp_path):
    from compiler.scopes.storage_epoch import require_original_worksharing_publication
    from compiler.tests.test_owner_local_call_groups import SOURCE as GROUP_SOURCE
    from compiler.tests.test_owner_local_call_groups import fixture as group_fixture

    helper = """subroutine native_filter(b,n)
integer,intent(in)::n
real(8),intent(inout)::b(:)
integer::i
!$omp do
do i=1,n
b(i)=real(i,8)
enddo
!$omp end do nowait
end subroutine
subroutine private_scratch(b,n)
integer,intent(in)::n
real(8),intent(in)::b(:)
real(8),allocatable,save::scratch(:)
scratch=b
end subroutine
"""
    source = GROUP_SOURCE.replace("subroutine second(a,b,n)", helper+"subroutine second(a,b,n)")
    source = source.replace("integer::i\ndo i=1,n\nb(i)=real(i,8)",
        "integer::i\nif(n<0)call native_filter(b,n)\nif(n<0)call private_scratch(b,n)\ndo i=1,n\nb(i)=real(i,8)")
    calls = "call first(b,n)\ncall second(a,b,n)\ncall third(a,b,n)"
    source = source.replace("endif\ncall unknown_suffix()", "endif\nif(.not.flag)then\n"+calls+"\nendif\ncall unknown_suffix()")
    _path, builder = group_fixture(tmp_path, source)
    outputs, report = builder.run()
    assert report["scope_count"] == 2, report["boundaries"]
    assert outputs
    assert len(builder._original_worksharing_publications) == 2
    for scope in report["scopes"]:
        epoch = scope["checked_execution_plan"]["storage_epoch"]
        assert epoch["private_allocation_obligations"]
        assert epoch["original_worksharing_storage_obligations"]
    # Both completed owners are rechecked after all later discovery,
    # selected-window projections and shared companion publication.
    require_original_worksharing_publication(builder)
    builder.analysis.verify_publication()
    assert len(builder._original_worksharing_publications) == 2


def test_two_private_allocation_owners_revalidate_both_completed_plans(tmp_path, monkeypatch):
    from compiler.scopes.execution_plan import CheckedExecutionPlan
    from compiler.tests.test_owner_local_call_groups import SOURCE as GROUP_SOURCE
    from compiler.tests.test_owner_local_call_groups import fixture as group_fixture

    helper = """subroutine private_scratch(b,n)
integer,intent(in)::n
real(8),intent(in)::b(:)
real(8),allocatable,save::scratch(:)
scratch=b
end subroutine
"""
    source = GROUP_SOURCE.replace("subroutine second(a,b,n)", helper+"subroutine second(a,b,n)")
    source = source.replace("integer::i\ndo i=1,n\nb(i)=real(i,8)",
        "integer::i\nif(n<0)call private_scratch(b,n)\ndo i=1,n\nb(i)=real(i,8)")
    calls = "call first(b,n)\ncall second(a,b,n)\ncall third(a,b,n)"
    source = source.replace("endif\ncall unknown_suffix()", "endif\nif(.not.flag)then\n"+calls+"\nendif\ncall unknown_suffix()")
    retained = []
    validate = CheckedExecutionPlan.validate

    def observe(plan):
        result = validate(plan)
        if plan._owner.routine.qualified == "local_groups::run" and not any(plan is old for old in retained):
            retained.append(plan)
        return result

    monkeypatch.setattr(CheckedExecutionPlan, "validate", observe)
    _path, builder = group_fixture(tmp_path, source)
    outputs, report = builder.run()
    assert report["scope_count"] == 2, report["boundaries"]
    assert outputs
    assert len(retained) == 2
    epochs = [scope["checked_execution_plan"]["storage_epoch"] for scope in report["scopes"]]
    assert all(epoch["private_allocation_obligations"] for epoch in epochs)
    assert epochs[0]["selected_position_epoch"]["selected_node_ids"] != epochs[1]["selected_position_epoch"]["selected_node_ids"]
    for plan in retained:
        owner = plan._owner
        previous = getattr(owner.builder, "_graph_storage_epoch", None)
        # Re-enter this completed owner's independently proved lifetime
        # context, as GraphOwner._emit_lexical does during publication.
        owner.builder._graph_storage_epoch = owner.position_epoch
        try:
            assert plan.validate() is plan
        finally:
            owner.builder._graph_storage_epoch = previous
    builder.analysis.verify_publication()
    contexts = builder.analysis._source_session._contexts._by_key.values()
    assert any(context.arena.derivation_count("original_local_allocation") for context in contexts)
