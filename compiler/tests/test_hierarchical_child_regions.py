"""A retained child prefix does not assert whole-child descriptor stability."""
from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.hierarchical_regions import HierarchicalRegionOwner
from compiler.scopes.source import ScopeBuilder

SOURCE = """module hierarchical
implicit none
real(8),allocatable::left(:),right(:)
real(8)::observed
contains
subroutine producer(b,n)
integer,intent(in)::n
real(8),intent(out)::b(:)
integer::i
do i=1,n
b(i)=real(i,8)
enddo
end subroutine
subroutine consumer(a,b,n)
integer,intent(in)::n
real(8),intent(out)::a(:)
real(8),intent(in)::b(:)
integer::i
do i=1,n
a(i)=2.0_8*b(i)
enddo
end subroutine
subroutine relocate(a,b)
real(8),allocatable,intent(inout)::a(:),b(:)
real(8),allocatable::temporary(:)
call move_alloc(a,temporary)
call move_alloc(b,a)
call move_alloc(temporary,b)
end subroutine
subroutine middle(n,change)
integer,intent(in)::n
logical,intent(in)::change
integer::i
observed=0.0_8
do i=1,n
observed=observed+right(lbound(right,1)+i-1)
enddo
if(change)call relocate(left,right)
call consumer(left,right,n)
end subroutine
subroutine advance(n,steps,change)
integer,intent(in)::n,steps
logical,intent(in)::change
integer::stage
do stage=1,steps
if(.not.allocated(left))then
allocate(left(-3:n-4),right(2:n+1))
left=0
right=0
endif
call producer(right,n)
call middle(n,change)
call relocate(left,right)
enddo
end subroutine
end module
"""
ENTRY = "hierarchical::advance"
READ_FIRST_SOURCE = (SOURCE
    .replace("subroutine producer(b,n)", "subroutine producer(a,b,n)")
    .replace("real(8),intent(out)::b(:)", "real(8),intent(in)::a(:)\nreal(8),intent(inout)::b(:)")
    .replace("do i=1,n\nb(i)=real(i,8)\nenddo",
             "!$omp parallel do private(i)\ndo i=2,n-1\nb(i)=a(i)+real(i,8)\nenddo\n!$omp end parallel do")
    .replace("call producer(right,n)", "call producer(left,right,n)"))


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source)
    fact = {"storage": "stable", "lifetime": "selected_window", "initialized": "whole",
            "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"hierarchical::"+name: dict(fact) for name in ("left", "right")}}
    builder = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    return path, builder


def owner_at(builder):
    from fparser.two.utils import walk
    calls = [node for node in walk(builder.entry.execution) if type(node).__name__ == "Call_Stmt"]
    assert [str(node.items[0]).lower() for node in calls] == ["producer", "middle", "relocate"]
    _graph, cut = builder.analysis._selected_source(ENTRY, (calls[2],))
    return HierarchicalRegionOwner(builder, tuple(calls[:2]), cut[0])


def test_normal_graph_retains_prefix_and_checks_child_retirement(tmp_path):
    path, builder = fixture(tmp_path)
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    scope, = report["scopes"]
    epoch = scope["checked_execution_plan"]["storage_epoch"]
    returned = epoch["return_contract"]
    assert returned["version"] == 2
    assert returned["root_selection"]["selected_node_ids"]
    assert returned["selection_identity"] == returned["root_selection"]["identity"]
    middle, = [row for row in returned["members"] if row["procedure"] == "hierarchical::middle"]
    assert any(row["kind"] == "reached_retirement" and row["rearm_allowed"] for row in middle["routes"])
    assert any(row["kind"] == "checked_child" and row["procedure"] == "hierarchical::consumer"
               for row in middle["routes"])
    assert all(row["lifetime"] == "selected_window" for row in epoch["resources"] if row["device_replica_authorized"])
    replacement = outputs[report["sources"][str(path)]["replacement"]]
    assert "fort_scope_resident_close_v1" in replacement
    assert "fort_scope_buffer_valid_v1" in replacement
    assert "if (change) then" in replacement.lower()
    assert "allocate(left(-3:n-4),right(2:n+1))" in replacement
    assert "call relocate(left,right)" in replacement
    ending = scope["selected_end_retirement"]
    assert ending["insertion_span"] == [scope["last_line"]+1, scope["last_line"]]
    assert ending["context"] == scope["owner"]
    patch, = [row for row in report["source_edits"]
              if row["source"] == str(path)
              and [row["first_line"], row["last_line"]] == ending["insertion_span"]]
    assert sha256(patch["replacement"].encode()).hexdigest() == ending["replacement_sha256"]
    assert "fort_scope_resident_close_v1("+scope["owner"]+")" in patch["replacement"]


@pytest.mark.parametrize("lifetime", ["selected_window", "invocation"])
def test_selected_entry_never_inherits_prefix_definitions(tmp_path, lifetime):
    _path, builder = fixture(tmp_path)
    owner = owner_at(builder)
    for root, fact in builder.facts["captures"].items():
        fact["lifetime"] = lifetime
        binding = builder.visible_binding(builder.entry, root)
        issued = builder.capture(binding)
        assert issued["initialized"] == "none"
        assert issued["lifetime"] == "selected_window"
    with pytest.raises(CompilationError, match="cannot publish"):
        builder.finish(materialize_native_effects=False)
    assert owner.return_calls == []


def test_child_mutation_never_becomes_whole_child_descriptor_proof(tmp_path, monkeypatch):
    from compiler.scopes import conditional_return
    original = conditional_return.prove_storage_mutation_effects
    queries = []

    def tracked(analysis, procedure, **keywords):
        queries.append((procedure, keywords.get("selection")))
        return original(analysis, procedure, **keywords)

    monkeypatch.setattr(conditional_return, "prove_storage_mutation_effects", tracked)
    _path, builder = fixture(tmp_path)
    owner = owner_at(builder)
    nodes = builder.inline.prepare(owner.closed_selection)
    assert owner.run(nodes)
    assert not any(procedure == ENTRY and selection == owner.closed_selection for procedure, selection in queries)
    returned = owner.return_contract.public()
    assert returned["cross_cut_replica_retention_authorized"] is False
    assert returned["root_selection"]["descriptor_stability_authorized"] is False


def test_unknown_nested_escape_cannot_reopen_on_later_invocation(tmp_path):
    source = SOURCE.replace("if(change)call relocate(left,right)", "if(change)call unknown_escape(left)")
    _path, builder = fixture(tmp_path, source)
    owner = owner_at(builder)
    with pytest.raises(CompilationError, match="cannot reenter after unproved lifetime/escape"):
        owner.run(builder.inline.prepare(owner.closed_selection))
    assert not hasattr(owner, "return_contract")
    with pytest.raises(CompilationError, match="cannot publish"):
        builder.finish(materialize_native_effects=False)


def test_exact_root_selection_and_final_frontier_cannot_be_rebound(tmp_path):
    _path, builder = fixture(tmp_path)
    owner = owner_at(builder)
    proof = owner.hierarchical_selection
    with pytest.raises(CompilationError, match="exact original owning region"):
        replace(proof).validate(owner)
    owner.hierarchical_selection = replace(proof, cut_before=proof.selected[0])
    with pytest.raises(CompilationError, match="exact original owning region"):
        owner.hierarchical_selection.validate(owner)


def test_rebound_owner_slot_does_not_reissue_copied_selection(tmp_path):
    _path, builder = fixture(tmp_path)
    owner = owner_at(builder)
    original = owner.hierarchical_selection
    copied = replace(original)
    owner.hierarchical_selection = copied
    builder._conditional_return_draft.selected_root = copied
    assert builder._conditional_return_draft._selected_root_receipt is original
    with pytest.raises(CompilationError, match="original root selection"):
        builder._conditional_return_draft.validate()


def test_selected_root_early_exit_cannot_skip_cleanup(tmp_path):
    source = SOURCE.replace("call middle(n,change)\ncall relocate", "if(change)return\ncall middle(n,change)\ncall relocate")
    _path, builder = fixture(tmp_path, source)
    from fparser.two.utils import walk
    loop, = [node for node in builder.entry.execution.content if type(node).__name__ == "Block_Nonlabel_Do_Construct"]
    middle = tuple(node for node in loop.content if type(node).__name__ in {"Call_Stmt", "If_Stmt"})
    assert any(type(node).__name__ == "Return_Stmt" for original in middle for node in walk(original))
    _graph, cut = builder.analysis._selected_source(ENTRY, (middle[-1],))
    with pytest.raises(CompilationError, match="mandatory cleanup"):
        HierarchicalRegionOwner(builder, middle[:-1], cut[0])


def test_same_callee_after_selected_end_remains_an_original_unborrowed_call(tmp_path):
    source = SOURCE.replace("call middle(n,change)\ncall relocate(left,right)\nenddo",
                            "call middle(n,change)\ncall relocate(left,right)\ncall middle(n,change)\nenddo")
    path, builder = fixture(tmp_path, source)
    outputs, report = builder.run()
    scope, = report["scopes"]
    returned = scope["checked_execution_plan"]["storage_epoch"]["return_contract"]
    root, = [row for row in returned["members"] if row["procedure"] == ENTRY]
    calls = [row for row in root["routes"] if row["kind"] == "checked_child"
             and row["procedure"] == "hierarchical::middle"]
    assert len(calls) == 1
    coordinator, = [row for row in scope["module_coordinators"] if row["procedure"] == "hierarchical::middle"]
    assert len(coordinator["calls"]) == 1
    replacement = outputs[report["sources"][str(path)]["replacement"]]
    second_call_line = max(index for index, line in enumerate(source.splitlines(), 1)
                           if line == "call middle(n,change)")
    assert scope["last_line"] < second_call_line
    assert "call middle(n,change)\nenddo" in replacement
    assert not any(row["source"] == str(path) and row["first_line"] <= second_call_line <= row["last_line"]
                   for row in report["source_edits"])


def test_changed_original_source_still_rejects_artifact_publication(tmp_path):
    path, builder = fixture(tmp_path)
    owner = owner_at(builder)
    assert owner.run(builder.inline.prepare(owner.closed_selection))
    path.write_text(SOURCE.replace("b(i)=real(i,8)", "b(i)=3.0_8*real(i,8)"))
    with pytest.raises(CompilationError, match="source changed"):
        builder.finish(materialize_native_effects=False)


def test_read_first_child_keeps_exact_source_read_authority_separate(tmp_path):
    from compiler.scopes.source_reads import validate_source_reads
    _path, builder = fixture(tmp_path, READ_FIRST_SOURCE)
    owner = owner_at(builder)
    assert owner.run(builder.inline.prepare(owner.closed_selection))
    producer = owner.members["hierarchical::producer"]
    found = []
    for unit in producer.units:
        if not unit.leaves:
            continue
        requirements = getattr(unit.scope, "source_reads", None)
        for call in unit.scope.calls:
            proof = requirements.get(call.procedure) if hasattr(requirements, "get") else requirements
            if proof is not None:
                assert validate_source_reads(producer.builder, call, proof) is proof
                found.append(proof)
    assert found, "read-first GPU unit must retain source reads; NONE is not entry definition authority"
    for proof in found:
        assert proof.public()["proof_role"] == "exact_original_reached_read_preconditions"
    assert all(builder.capture(builder.visible_binding(builder.entry, root))["initialized"] == "none"
               for root in builder.facts["captures"])


@pytest.mark.parametrize("source", [SOURCE, READ_FIRST_SOURCE], ids=["overwrite", "read-first"])
def test_counted_module_read_retains_its_exact_descriptor_and_read_authority(tmp_path, source):
    from compiler.scopes.source_reads import validate_native_source_reads

    _path, builder = fixture(tmp_path, source)
    owner = owner_at(builder)
    assert owner.run(builder.inline.prepare(owner.closed_selection))
    middle = owner.members["hierarchical::middle"]
    operations = [operation for unit in middle.units for operation in unit.scope.native
                  if any(type(node).__name__ == "Block_Nonlabel_Do_Construct" for node in operation.nodes)]
    operation, = operations
    assert operation.source_reads is not None
    assert operation.source_reads._descriptor_window is not None
    sections = validate_native_source_reads(middle.builder, operation)
    resource, = sections.resources
    assert resource.resource == "hierarchical::right"
    assert resource.reads
    assert not resource.writes
    assert not resource.overwrites
    assert operation.source_reads.public()["global_initialization_authority"] is False


@pytest.mark.parametrize("mutation", ["copied_window", "foreign_window", "replaced_registration"])
def test_native_counted_read_rejects_wrong_descriptor_authority(tmp_path, mutation):
    from compiler.frontend.reached_read_requirements import native_read_sections, prove_native_read_requirements

    def operation_at(directory):
        directory.mkdir()
        _path, builder = fixture(directory, READ_FIRST_SOURCE)
        owner = owner_at(builder)
        assert owner.run(builder.inline.prepare(owner.closed_selection))
        member = owner.members["hierarchical::middle"]
        operation, = [operation for unit in member.units for operation in unit.scope.native
                      if any(type(node).__name__ == "Block_Nonlabel_Do_Construct" for node in operation.nodes)]
        return member, operation

    member, operation = operation_at(tmp_path / "original")
    proof = operation.source_reads
    window = proof._descriptor_window
    analysis = operation.proof_analysis
    selected = tuple(member.builder.inline.original_selection(operation.nodes))
    if mutation == "copied_window":
        window = replace(window)
    elif mutation == "foreign_window":
        _other, foreign = operation_at(tmp_path / "foreign")
        window = foreign.source_reads._descriptor_window
    else:
        analysis._conditional_unit_descriptors[window.identity] = replace(window)
    with pytest.raises(CompilationError):
        prove_native_read_requirements(analysis, member.routine.qualified, selected, descriptor_window=window)
    if mutation == "replaced_registration":
        with pytest.raises(CompilationError):
            native_read_sections(analysis, member.routine.qualified, selected, proof)


def test_native_counted_read_rejects_discarded_descriptor_generation(tmp_path):
    from compiler.frontend.descriptor_windows import prove_descriptor_window
    from compiler.frontend.reached_read_requirements import prove_native_read_requirements
    from compiler.scopes.generation_transaction import GenerationTransaction

    path, original = fixture(tmp_path, READ_FIRST_SOURCE)
    entry = "hierarchical::middle"
    builder = ScopeBuilder([path], entry, facts=original.facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    analysis = builder.analysis
    selection = tuple(node for node in builder.entry.execution.content
                      if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    graph, identities = analysis._selected_source(entry, selection)
    assert graph.available
    with GenerationTransaction(builder):
        window = prove_descriptor_window(analysis, entry, identities, (), module_roots=("hierarchical::right",))
        assert prove_native_read_requirements(analysis, entry, selection, descriptor_window=window) is not None
    with pytest.raises(CompilationError):
        prove_native_read_requirements(analysis, entry, selection, descriptor_window=window)


def test_fixed_module_read_needs_no_allocatable_leaf_descriptor_roots(tmp_path):
    from compiler.scopes.source_reads import validate_native_source_reads

    source = SOURCE.replace("real(8)::observed", "real(8)::observed,table(1)=[0.0_8]")
    source = source.replace("observed=0.0_8", "observed=table(1)")
    _path, builder = fixture(tmp_path, source)
    owner = owner_at(builder)
    assert owner.run(builder.inline.prepare(owner.closed_selection))
    member = owner.members["hierarchical::middle"]
    operations = [operation for unit in member.units for operation in unit.scope.native
                  if any("table" in str(node).lower() for node in operation.nodes)]
    operation, = operations
    assert operation.source_reads is not None
    assert operation.source_reads._descriptor_window.public()["resources"] == []
    from compiler.frontend.reached_read_requirements import native_read_sections

    selected = tuple(member.builder.inline.original_selection(operation.nodes))
    resource, = native_read_sections(operation.proof_analysis, member.routine.qualified,
                                    selected, operation.source_reads).resources
    assert resource.resource == "hierarchical::table"
    assert resource.reads
    # The constant module input may remain host-only after coherence filtering.
    assert validate_native_source_reads(member.builder, operation).available
