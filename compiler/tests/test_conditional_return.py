"""Returned-domain retention is conditional on the actual checked path."""
import shutil
import subprocess
from copy import copy
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes import conditional_return
from compiler.scopes.graph import GraphOwner
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_native_ancestor import ENTRY, ROOTS, SOURCE, fixture


def compile_fixture(tmp_path, source=SOURCE):
    _path, builder = fixture(tmp_path, source=source)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report["graph_owner_attempt"]
    scope, = report["scopes"]
    plan = scope["checked_execution_plan"]
    return outputs, report, scope, plan["storage_epoch"]["return_contract"]


def test_source_contract_covers_children_native_units_and_conditional_retirement(tmp_path):
    outputs, _report, scope, proof = compile_fixture(tmp_path)
    assert scope["checked_execution_plan"]["schema_version"] == 3
    assert {row["procedure"] for row in proof["members"]} == {ENTRY, "ancestor_fixture::middle", *ROOTS}
    cuts = [route for member in proof["members"] for route in member["routes"] if route["kind"] == "reached_retirement"]
    assert any(row["lifetime"] == "proved_lifecycle" and row["rearm_allowed"] is True for row in cuts)
    assert all(proof[key] is False for key in ("escape_completion_assumed", "entry_definition_authority",
        "payload_access_authorized", "motion_authorized", "placement_authorized", "cross_cut_replica_retention_authorized"))
    assert all(row["control_version"] == 5 for row in scope["module_coordinators"])
    changed = "\n".join(value for name, value in outputs.items() if name.startswith("sources/"))
    assert "fort_child_retention_" in changed
    # Original predicate is evaluated once; the reached source-proved
    # synchronous arm retires. The original body is never replayed.
    assert "if (flag) then" in changed.lower()
    assert "intent(inout)" in changed
    assert ".and. fort_child_retention_" in changed


def test_prospective_selected_window_cannot_seed_reviewed_entry_definitions(tmp_path):
    _path, builder = fixture(tmp_path)
    for fact in builder.facts["captures"].values():
        fact.update(lifetime="selected_window", initialized="whole")
    owner = GraphOwner(builder)
    conditional_return.prepare(owner)
    for name in ("a", "b"):
        fact = builder.capture(builder.entry.scope.bindings[name])
        assert fact["initialized"] == "none"
        assert not {"source_sections", "sections", "requires_registered_handle"} & fact.keys()
    assert not hasattr(owner, "return_contract")
    with pytest.raises(CompilationError, match="cannot publish"):
        builder.finish(materialize_native_effects=False)
    builder._conditional_return_draft = None
    with pytest.raises(CompilationError, match="selected-position"):
        builder.capture(builder.entry.scope.bindings["a"])


@pytest.mark.parametrize("mutation", ["native", "call", "condition", "patch"])
def test_missing_or_rebound_execution_route_cannot_seal_return_authority(tmp_path, monkeypatch, mutation):
    seal = conditional_return.seal

    def corrupt(owner):
        middle = owner.members["ancestor_fixture::middle"]
        if mutation == "native":
            middle.graph_native_tasks.clear()
        elif mutation == "call":
            item = owner.return_calls[0]
            owner.return_calls[0] = replace(item, handle_bindings=())
        elif mutation == "condition":
            middle.return_conditions.clear()
        else:
            middle.patches[:] = [row for row in middle.patches if not hasattr(row[2], "conditional_guard")]
        return seal(owner)

    monkeypatch.setattr(conditional_return, "seal", corrupt)
    source = SOURCE.replace("observed=0.0_8", "if(n>0)then\nobserved=0.0_8\nendif") if mutation == "condition" else SOURCE
    _path, builder = fixture(tmp_path, source=source)
    _outputs, report = builder.run()
    assert report["scope_count"] == 0
    assert "returned-domain" in report["graph_owner_attempt"]["error"] or "conditional return" in report["graph_owner_attempt"]["error"]
    assert not report["source_edits"]


@pytest.mark.parametrize("declaration", ["common /alias/ scratch", "namelist /alias/ scratch", "equivalence(scratch,other)"])
def test_child_entry_aliases_cannot_borrow_before_reached_hooks(tmp_path, declaration):
    source = SOURCE.replace("integer::i\ncall producer", "integer::i\nreal(8)::scratch,other\n" + declaration + "\ncall producer", 1)
    source = source.replace("call middle(a,b,n,flag)", "call producer(a,b,n)\ncall middle(a,b,n,flag)", 1)
    _path, builder = fixture(tmp_path, source=source)
    _outputs, report = builder.run()
    assert report["scope_count"] == 0
    assert not report["source_edits"]


def test_whole_assignment_mutation_is_not_a_descriptor_stability_projection(tmp_path):
    source = SOURCE.replace("logical,intent(in)::flag\ninteger::i", "logical,intent(in)::flag\nreal(8),allocatable::local(:)\ninteger::i")
    source = source.replace("call producer(a,b,n)\nobserved", "local=a\ncall producer(a,b,n)\nobserved")
    _outputs, _report, _scope, proof = compile_fixture(tmp_path, source)
    cuts = [route for member in proof["members"] for route in member["routes"] if route["kind"] == "reached_retirement"]
    assert any(route["lifetime"] == "proved_lifecycle" and route["rearm_allowed"] for route in cuts)


def test_selected_storage_projection_includes_hidden_module_mutations(tmp_path):
    source = SOURCE.replace("real(8)::observed", "real(8)::observed\nreal(8),allocatable::hidden(:)")
    source = source.replace("call producer(a,b,n)\nobserved", "hidden=a\ncall producer(a,b,n)\nobserved")
    _outputs, _report, _scope, proof = compile_fixture(tmp_path, source)
    cuts = [route for member in proof["members"] for route in member["routes"] if route["kind"] == "reached_retirement"]
    assert any(route["lifetime"] == "proved_lifecycle" for route in cuts)


def test_selected_module_allocations_use_per_operation_storage_proof_not_inactive_branch_closure(tmp_path):
    source = """module reached_allocations
implicit none
real(8),allocatable::a(:),b(:)
real(8)::observed
contains
subroutine advance(n,flag)
integer,intent(in)::n
logical,intent(in)::flag
call middle(n,flag)
end subroutine
subroutine middle(n,flag)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
call producer(n)
observed=0.0_8
do i=1,n
observed=observed+b(i)
enddo
if(flag) call opaque(b,n)
call consumer(n)
end subroutine
subroutine producer(n)
integer,intent(in)::n
integer::i
do i=1,n
b(i)=2.0_8*a(i)
enddo
end subroutine
subroutine consumer(n)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=b(i)+1.0_8
enddo
end subroutine
subroutine opaque(x,n)
integer,intent(in)::n
real(8),intent(inout)::x(*)
write(*,*) 'ORIGINAL_SYNCHRONOUS_OUTPUT'
x(1:n)=x(1:n)+3.0_8
end subroutine
end module
"""
    path = tmp_path / "module_allocations.f90"
    path.write_text(source)
    entry = "reached_allocations::advance"
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"reached_allocations::" + name: {"storage": "stable", "initialized": "whole",
                 "escapes": False, "allocation_changes": False, "lifetime": "selected_window"} for name in ("a", "b")}}
    builder = ScopeBuilder([path], entry, facts=facts, options=CompilerOptions(), config=OffloadConfig(policy="sections"),
        contracts={entry: {"native_ancestor": {"version": 1,
            "device_roots": ["reached_allocations::producer", "reached_allocations::consumer"]}}})
    _outputs, report = builder.run()
    assert report["scope_count"] == 1, report["graph_owner_attempt"]
    scope, = report["scopes"]
    proof = scope["checked_execution_plan"]["storage_epoch"]["return_contract"]
    middle, = [row for row in proof["members"] if row["procedure"] == "reached_allocations::middle"]
    assert any(route["kind"] == "checked_unit" for route in middle["routes"])
    assert any(route["kind"] == "reached_retirement" and route["rearm_allowed"] for route in middle["routes"])
    assert not proof["entry_definition_authority"]


def test_foreign_selected_syntax_does_not_issue_storage_projection(tmp_path):
    from fparser.two.Fortran2003 import Assignment_Stmt

    from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
    _path, builder = fixture(tmp_path)
    with pytest.raises(CompilationError):
        prove_storage_mutation_effects(builder.analysis, ENTRY, selection=(Assignment_Stmt("a=b"),))


def test_shared_storage_service_exposes_exact_selection_without_reinterpreting_payload(tmp_path):
    _path, builder = fixture(tmp_path)
    routine = builder.analysis.routines[ROOTS[0]]
    proof = builder.analysis.storage_mutation_effects(routine.qualified, selection=tuple(routine.execution.content))
    assert proof.public()["version"] == 3
    assert proof.preserves_descriptors()
    assert proof.public()["entry_exit_lifecycle_authorized"] is False
    assert proof.public()["payload_access_authorized"] is False


def test_descriptor_token_cannot_be_borrowed_by_another_live_owner(tmp_path):
    _path, first = fixture(tmp_path)
    first_owner = GraphOwner(first)
    conditional_return.prepare(first_owner)
    selection = tuple(first.entry.execution.content)
    token = conditional_return.descriptor_for(first, first.analysis, ENTRY, selection)
    assert token is not None
    second = copy(first)
    second_owner = GraphOwner(second)
    conditional_return.prepare(second_owner)
    with pytest.raises(CompilationError, match="another original owner"):
        conditional_return.descriptor_for(second, second.analysis, ENTRY, selection)
    with pytest.raises(CompilationError, match="cannot publish"):
        second.finish(materialize_native_effects=False)


def test_consumer_identity_survives_adoption_without_owner_edges_in_analysis_registry(tmp_path, monkeypatch):
    from compiler.scopes.generation_transaction import GenerationTransaction

    prepare = conditional_return.prepare

    def prepare_with_root_token(owner):
        prepare(owner)
        builder = owner.builder
        conditional_return.descriptor_for(builder, builder.analysis, ENTRY,
                                         tuple(builder.entry.execution.content))

    monkeypatch.setattr(conditional_return, "prepare", prepare_with_root_token)
    _path, parent = fixture(tmp_path)
    _outputs, report = parent.run()
    assert report["scope_count"] == 1, report["graph_owner_attempt"]
    draft = parent._conditional_return_draft
    key = draft.consumer_key(parent)
    owner = draft.owner
    assert owner.execution_plan.validate()._owner is owner
    token = conditional_return.descriptor_for(parent, parent.analysis, ENTRY,
                                             tuple(parent.entry.execution.content))
    assert token._consumer is key
    assert not any(hasattr(token, name) for name in ("_draft", "_builder", "_owner"))

    with GenerationTransaction(parent) as transaction:
        candidate = transaction.builder
        candidate_draft = candidate._conditional_return_draft
        assert candidate_draft is not draft
        assert candidate_draft.owner.builder is candidate
        assert candidate_draft.consumer_key(candidate) is key
        assert candidate_draft.owner.execution_plan.validate()._owner is candidate_draft.owner
        current = conditional_return.descriptor_for(candidate, candidate.analysis, ENTRY,
                                                    tuple(candidate.entry.execution.content))
        assert current._consumer is key
        transaction.adopt()

    adopted = parent._conditional_return_draft
    assert adopted is not candidate_draft
    assert adopted is not draft
    assert adopted.owner.builder is parent
    assert adopted.consumer_key(parent) is key
    assert adopted.owner.execution_plan.validate()._owner is adopted.owner
    parent.finish(materialize_native_effects=False)
    assert conditional_return.descriptor_for(parent, parent.analysis, ENTRY,
                                            tuple(parent.entry.execution.content))._consumer is key


@pytest.mark.parametrize("replacement", ["copied", "fresh"])
def test_descriptor_consumer_key_copy_or_replacement_cannot_validate(tmp_path, replacement):
    _path, builder = fixture(tmp_path)
    owner = GraphOwner(builder)
    conditional_return.prepare(owner)
    selected = tuple(builder.entry.execution.content)
    proof = conditional_return.descriptor_for(builder, builder.analysis, ENTRY, selected)
    original_key = proof._consumer
    other = copy(original_key) if replacement == "copied" else object()
    assert other is not original_key
    forged = replace(proof, _consumer=other)
    builder.analysis._conditional_unit_descriptors[proof.identity] = forged
    with pytest.raises(CompilationError, match="exact registered original selection"):
        forged.validate(builder.analysis, ENTRY, forged._selected)
    with pytest.raises(CompilationError, match="another original owner"):
        conditional_return.descriptor_for(builder, builder.analysis, ENTRY, selected)


def test_another_live_draft_key_cannot_replace_both_token_consumer_fields(tmp_path):
    _path, first = fixture(tmp_path)
    owner = GraphOwner(first)
    conditional_return.prepare(owner)
    selected = tuple(first.entry.execution.content)
    proof = conditional_return.descriptor_for(first, first.analysis, ENTRY, selected)
    original_key = proof._consumer
    second = copy(first)
    second_owner = GraphOwner(second)
    conditional_return.prepare(second_owner)
    other_key = second._conditional_return_draft.consumer_key(second)
    assert other_key is not original_key
    # Even simultaneous mutation of the retained key and its local id receipt
    # cannot rebind the independent, analysis-side issuing-consumer association.
    object.__setattr__(proof, "_consumer", other_key)
    object.__setattr__(proof, "_consumer_identity", id(other_key))
    assert first.analysis._conditional_descriptor_consumers[proof.identity] is original_key
    with pytest.raises(CompilationError, match="exact registered original selection"):
        proof.validate(first.analysis, ENTRY, proof._selected)
    with pytest.raises(CompilationError, match="another original owner"):
        conditional_return.descriptor_for(first, first.analysis, ENTRY, selected)
    with pytest.raises(CompilationError, match="cannot publish"):
        second.finish(materialize_native_effects=False)


def test_twice_called_unresolved_escape_owner_stays_entirely_native(tmp_path):
    source = SOURCE.replace("call opaque(b,n)", "call unknown_escape(b,n)")
    path, builder = fixture(tmp_path, source=source)
    outputs, report = builder.run()
    assert report["scope_count"] == 0
    assert "cannot reenter after unproved lifetime/escape" in report["graph_owner_attempt"]["error"]
    assert not report["source_edits"]
    assert set(outputs) == {"scope-manifest.json"}
    fortran = shutil.which("gfortran-15")
    if fortran is None:
        pytest.skip("GNU Fortran required for independent twice-called original control")
    driver = tmp_path / "twice.f90"
    driver.write_text("""program twice
use ancestor_fixture
real(8)::a(3),b(3)
a=1.0_8
b=0.0_8
call advance(a,b,3,.true.)
call advance(a,b,3,.true.)
if(any(a/=16.0_8).or.opaque_visits/=2)error stop 'native repeat'
end program
subroutine unknown_escape(x,n)
use ancestor_fixture,only:opaque_visits
integer::n
real(8)::x(*)
opaque_visits=opaque_visits+1
x(1:n)=x(1:n)+3.0_8
end subroutine
""")
    binary = tmp_path / "twice"
    command = [fortran, "-fcheck=all", str(path), str(driver), "-J", str(tmp_path), "-o", str(binary)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_committed_root_escape_stops_suffix_discovery_with_exact_source_diagnostic(tmp_path, monkeypatch):
    from compiler.scopes.region_dispatch import InlineRegions

    source = SOURCE.replace("call middle(a,b,n,flag)", "call unknown_escape(a,n)\ncall middle(a,b,n,flag)", 1)
    _path, builder = fixture(tmp_path, source=source)
    monkeypatch.setattr(InlineRegions, "prepare", lambda *_args, **_kwargs: pytest.fail("suffix discovery after root escape"))
    _outputs, report = builder.run()
    assert report["scope_count"] == 0
    assert not report["source_edits"]
    barrier = report["graph_owner_attempt"]["conditional_return_barrier"]
    assert barrier["procedure"] == ENTRY
    assert barrier["original_source"] == ["CALL unknown_escape(a, n)"]
    assert barrier["span"][0] == barrier["span"][1]
    assert barrier["storage"]["classification"] == "unproved_lifetime_or_escape"
    assert "unknown_escape" in barrier["storage"]["first_source_frontier"]["source_excerpt"]
    assert barrier["execution_authority"] is False


def test_known_synchronous_root_prefix_does_not_block_later_checked_children(tmp_path):
    source = SOURCE.replace("call middle(a,b,n,flag)", "call opaque(a,n)\ncall middle(a,b,n,flag)", 1)
    _outputs, _report, _scope, proof = compile_fixture(tmp_path, source)
    root, = [member for member in proof["members"] if member["procedure"] == ENTRY]
    assert any(route["kind"] == "reached_retirement" and route["rearm_allowed"] for route in root["routes"])
    assert any(route["kind"] == "checked_child" for route in root["routes"])


def test_guarded_root_rejection_preserves_one_original_boundary_reason(tmp_path, monkeypatch):
    from compiler.scopes.region_dispatch import InlineRegions

    source = SOURCE.replace("call middle(a,b,n,flag)", "if(flag)call unknown_escape(a,n)\ncall middle(a,b,n,flag)", 1)
    _path, builder = fixture(tmp_path, source=source)
    monkeypatch.setattr(InlineRegions, "prepare", lambda *_args, **_kwargs: pytest.fail("suffix discovery after root escape"))
    _outputs, report = builder.run()
    assert report["scope_count"] == 0
    assert not report["source_edits"]
    attempt = report["graph_owner_attempt"]
    barrier = attempt["conditional_return_barrier"]
    assert barrier["original_source"] == ["IF (flag) CALL unknown_escape(a, n)"]
    assert len(attempt["boundaries"]) == 1
    assert barrier["native_reason"] == attempt["boundaries"][0]["reason"]
    assert "conditional return cannot reenter" not in barrier["native_reason"]


def test_more_than_256_native_units_and_live_tokens_have_source_sized_budget(tmp_path):
    count = 270
    source = SOURCE.replace("observed=0.0_8", "observed=0.0_8\n" + "\n".join(
        "observed=observed+b(1)" for _ in range(count)))
    static = tmp_path / "registry"
    static.mkdir()
    path, base = fixture(static, source=source)
    builder = ScopeBuilder([path], "ancestor_fixture::middle", facts=base.facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections"), contracts={"ancestor_fixture::middle": {
            "native_ancestor": {"version": 1, "device_roots": ROOTS}}})
    routine = builder.analysis.routines["ancestor_fixture::middle"]
    owner = GraphOwner(builder)
    conditional_return.prepare(owner)
    nodes = [node for node in routine.execution.content if type(node).__name__ == "Assignment_Stmt"
             and "+ b(1)" in str(node)]
    assert len(nodes) == count
    tokens = [conditional_return.descriptor_for(builder, builder.analysis, routine.qualified, (node,)) for node in nodes]
    assert all(token is not None for token in tokens)
    assert len({token.identity for token in tokens}) == count
    assert len(builder.analysis._conditional_unit_descriptors) == count
    assert len(builder.analysis._storage_mutation_effects) >= count
    tokens[0].validate(builder.analysis, routine.qualified, tokens[0]._selected)
    generated = tmp_path / "generation"
    generated.mkdir()
    _outputs, _report, scope, proof = compile_fixture(generated, source)
    middle, = [row for row in proof["members"] if row["procedure"] == "ancestor_fixture::middle"]
    assert sum(route["kind"] == "checked_unit" for route in middle["routes"]) >= count
    assert not any("budget" in boundary["reason"] for boundary in scope["boundaries"])
    assert scope["ownership"]["lifetime"] == "one original invocation"
