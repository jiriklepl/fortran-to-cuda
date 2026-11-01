"""Atomic source participation is derived once per immutable source context."""
from copy import copy

import pytest

from compiler.frontend import atomic_worksharing_team as atomic
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.tests.test_atomic_worksharing_team import fixture


def sealed_fixture(tmp_path):
    path, analysis, routine, nodes = fixture(tmp_path)
    analysis.seal_session()
    return path, analysis, routine, nodes


def test_sealed_consumers_reuse_one_source_derivation_and_identical_public_bytes(tmp_path, monkeypatch):
    import json
    from hashlib import sha256

    path, analysis, routine, nodes = fixture(tmp_path)
    original = atomic._derive
    calls = []

    def counted(*args):
        calls.append(args)
        return original(*args)

    diagnostic = atomic.prove_atomic_worksharing_team(analysis, routine, nodes)
    analysis.seal_session()
    record = original(analysis, routine, nodes)[4]
    before = json.dumps({"identity": sha256(atomic._encoded(record).encode()).hexdigest(), **record}, sort_keys=True)
    monkeypatch.setattr(atomic, "_derive", counted)
    proof = atomic.prove_atomic_worksharing_team(analysis, routine, nodes)
    for _ in range(20):
        assert atomic.prove_atomic_worksharing_team(analysis, routine, nodes) is proof
        for alternative in atomic.atomic_worksharing_alternatives(proof):
            alternative.validate(analysis, routine.qualified, nodes)
        assert proof.predicate_sections(0).available
    assert len(calls) == 1
    assert json.dumps(proof.public(), sort_keys=True) == before
    assert analysis.source_authority_context().derivation_arena(analysis).derivation_count(
        "original_atomic_worksharing_team") == 1
    assert diagnostic._source_fact is None
    assert proof is not diagnostic
    with pytest.raises(CompilationError):
        diagnostic.validate()
    analysis.verify_publication()


@pytest.mark.parametrize("change", ["copy", "registered_copy", "missing_registry", "owner", "foreign_analysis", "foreign_selection"])
def test_original_query_and_exact_issued_handle_cannot_be_substituted(tmp_path, change):
    path, analysis, routine, nodes = sealed_fixture(tmp_path)
    proof = atomic.prove_atomic_worksharing_team(analysis, routine, nodes)
    if change == "copy":
        proof = copy(proof)
    elif change == "registered_copy":
        proof = copy(proof)
        analysis._atomic_worksharing_teams[proof.identity] = proof
    elif change == "missing_registry":
        del analysis._atomic_worksharing_teams[proof.identity]
    elif change == "owner":
        analysis._atomic_worksharing_owner = object()
    elif change == "foreign_analysis":
        analysis = SourceEffects([path])
        analysis.seal_session()
    else:
        nodes = (copy(nodes[0]), *nodes[1:])
    with pytest.raises(CompilationError):
        proof.validate(analysis, routine, nodes)
    if change == "owner":
        with pytest.raises(CompilationError):
            atomic.prove_atomic_worksharing_team(analysis, routine, nodes)


@pytest.mark.parametrize("change", ["operation_limit", "contracts", "projection"])
def test_policy_and_supported_routine_projection_invalidate_issued_fact(tmp_path, change):
    _, analysis, routine, nodes = sealed_fixture(tmp_path)
    proof = atomic.prove_atomic_worksharing_team(analysis, routine, nodes)
    if change == "operation_limit":
        analysis.operation_limit += 1
    elif change == "contracts":
        analysis.contracts = {"changed": {"opaque": True}}
    else:
        analysis.project_routine(routine.qualified, copy(routine))
    with pytest.raises(CompilationError):
        proof.validate()


@pytest.mark.parametrize("change", ["remove", "replace", "registry", "replace_parent"])
def test_exact_parent_join_dependency_is_not_reissued_or_healed(tmp_path, change, monkeypatch):
    _, analysis, routine, nodes = sealed_fixture(tmp_path)
    proof = atomic.prove_atomic_worksharing_team(analysis, routine, nodes)
    parent = proof.parent
    if change == "remove":
        del analysis._joined_completions[parent.identity]
    elif change == "registry":
        analysis._joined_completions = dict(analysis._joined_completions)
    else:
        replacement = copy(parent)
        analysis._joined_completions[parent.identity] = replacement
        if change == "replace_parent":
            object.__setattr__(proof, "parent", replacement)

    def forbidden(*_args):
        raise AssertionError("a consumed parent receipt cannot be healed")

    monkeypatch.setattr(analysis, "joined_completion", forbidden)
    with pytest.raises(CompilationError):
        proof.validate()


def test_changed_disk_source_remains_rejected_at_publication(tmp_path):
    path, analysis, routine, nodes = sealed_fixture(tmp_path)
    proof = atomic.prove_atomic_worksharing_team(analysis, routine, nodes)
    path.write_text(path.read_text().replace("mode==4", "mode==2"))
    assert proof.validate() is proof
    with pytest.raises(CompilationError, match="source changed"):
        analysis.verify_publication()


def test_distinct_valid_registered_parent_cannot_replace_captured_dependency(tmp_path):
    path, _, _, _ = fixture(tmp_path)
    text = path.read_text()
    start, end = text.index("!$omp parallel"), text.index("!$omp end parallel") + len("!$omp end parallel")
    path.write_text(text[:end] + "\n" + text[start:end] + text[end:])
    analysis = SourceEffects([path])
    analysis.seal_session()
    routine = analysis.routines["atomic_fixture::step"]
    first, second = [item.original_nodes for item in grouped_original_sequence(routine.execution.content, {})
                     if isinstance(item, NativeGroupCandidate)]
    proof = atomic.prove_atomic_worksharing_team(analysis, routine, first)
    different = analysis.joined_completion(routine.qualified, second)
    different.validate(analysis, routine.qualified, different.selected_node_ids)
    assert different is not proof.parent
    object.__setattr__(proof, "parent", different)
    with pytest.raises(CompilationError, match="exact consumed parent"):
        proof.validate()


def test_generation_rollback_rejects_discarded_handle_and_reissues_same_fact(tmp_path):
    from compiler.scopes.generation_transaction import GenerationTransaction
    from compiler.tests.test_graph_storage_epochs import SOURCE, builder_at

    extra = """subroutine team(a,b,n,flag)
integer,intent(in)::n
logical,intent(in)::flag
real(8),intent(out)::a(:)
real(8),intent(in)::b(:)
integer::i
!$omp parallel private(i)
if(flag)then
!$omp do
do i=1,n
a(i)=b(i)
enddo
!$omp end do nowait
endif
!$omp end parallel
end subroutine
"""
    _, builder = builder_at(tmp_path, SOURCE.replace("end module", extra + "end module"))
    analysis = builder.analysis
    routine = analysis.routines["epochs::team"]
    nodes, = [item.original_nodes for item in grouped_original_sequence(routine.execution.content, {})
              if isinstance(item, NativeGroupCandidate)]
    analysis.source_authority_context()
    with GenerationTransaction(builder):
        discarded = atomic.prove_atomic_worksharing_team(analysis, routine, nodes)
        assert discarded.validate() is discarded
    with pytest.raises(CompilationError):
        discarded.validate()
    fresh = atomic.prove_atomic_worksharing_team(analysis, routine, nodes)
    assert fresh is not discarded
    assert fresh._source_fact.derived is discarded._source_fact.derived
    with GenerationTransaction(builder):
        assert fresh.validate() is fresh
    assert fresh.validate() is fresh


def test_failed_team_never_publishes_an_arena_fact(tmp_path):
    _, analysis, routine, nodes = fixture(tmp_path, "call unknown(a)\n")
    analysis.seal_session()
    context = analysis.source_authority_context()
    with pytest.raises(CompilationError):
        atomic.prove_atomic_worksharing_team(analysis, routine, nodes)
    assert context.derivation_arena(analysis).derivation_count("original_atomic_worksharing_team") == 0
    assert not analysis._atomic_worksharing_teams


def test_context_change_during_derivation_cannot_be_published(tmp_path, monkeypatch):
    _, analysis, routine, nodes = sealed_fixture(tmp_path)
    context = analysis.source_authority_context()
    original = atomic._derive

    def changed(*args):
        result = original(*args)
        analysis.operation_limit += 1
        return result

    monkeypatch.setattr(atomic, "_derive", changed)
    with pytest.raises(CompilationError, match="stale or foreign"):
        atomic.prove_atomic_worksharing_team(analysis, routine, nodes)
    assert context._arena is None or context._arena.derivation_count("original_atomic_worksharing_team") == 0
    assert not analysis._atomic_worksharing_teams
