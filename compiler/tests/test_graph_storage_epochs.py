"""A reached allocation epoch never inherits unknown prefix definitions."""
import json
from copy import copy
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder

SOURCE = """module epochs
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
subroutine relocate(left,right)
real(8),allocatable,intent(inout)::left(:),right(:)
real(8),allocatable::temporary(:)
call move_alloc(left,temporary)
call move_alloc(right,left)
call move_alloc(temporary,right)
end subroutine
subroutine advance(left,right,n,steps)
real(8),allocatable,intent(inout)::left(:),right(:)
integer,intent(in)::n,steps
integer::stage
do stage=1,steps
if(.not.allocated(left))then
allocate(left(-3:n-4),right(2:n+1))
left=0
right=0
endif
call producer(right,n)
call consumer(left,right,n)
call relocate(left,right)
enddo
end subroutine
end module
"""


def builder_at(tmp_path, source=SOURCE):
    path = tmp_path / "source.f90"
    path.write_text(source)
    fact = {"storage": "stable", "lifetime": "selected_window", "initialized": "none",
            "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::left": dict(fact), "argument::right": dict(fact)}}
    builder = ScopeBuilder([path], "epochs::advance", facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"))
    return path, builder


def emit(tmp_path, source=SOURCE):
    path, builder = builder_at(tmp_path, source)
    outputs, report = builder.run()
    return path, outputs, report


def test_original_repeated_pair_retains_epoch_and_none_after_native_allocation(tmp_path):
    path, outputs, report = emit(tmp_path)
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert report["scope_count"] == 1, report.get("graph_owner_attempt", report["boundaries"])
    scope, = report["scopes"]
    epoch = scope["checked_execution_plan"]["storage_epoch"]["selected_position_epoch"]
    assert epoch["entry_definition_authority"] is False
    assert epoch["cross_frontier_retention_authorized"] is False
    assert [call["procedure"] for call in epoch["calls"]] == ["epochs::producer", "epochs::consumer"]
    assert epoch["calls"][0]["canonical_resources"] == {"argument::b": "argument::right"}
    assert epoch["calls"][1]["canonical_resources"] == {
        "argument::a": "argument::left", "argument::b": "argument::right"}
    assert all(resource["lifetime"] == "selected_window"
               for resource in scope["checked_execution_plan"]["storage_epoch"]["resources"])
    replacement = outputs[report["sources"][str(path)]["replacement"]]
    assert "left=0" in replacement
    assert "call relocate(left,right)" in replacement
    assert "fort_scope_resident_close_v1" in replacement
    assert scope["counted_controls"]


def test_epoch_is_exact_owner_issued_before_capture_and_never_seeds_prefix_definitions(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch

    _path, builder = builder_at(tmp_path)
    binding = builder.entry.scope.bindings["left"]
    with pytest.raises(CompilationError, match="registered selected-position"):
        builder.capture(binding)
    owner = GraphOwner(builder)
    proof = discover_graph_storage_epoch(owner)
    assert proof.validate() is proof
    assert proof.capture(builder, binding, {**builder.capture_fact(binding), "initialized": "whole"})["initialized"] == "none"
    with pytest.raises(CompilationError, match="exact issuing source owner"):
        copy(proof).validate()
    with pytest.raises(CompilationError, match="exact issuing source owner"):
        proof.validate(copy(builder))
    bad = {**builder.capture_fact(binding), "lifetime": "invocation"}
    with pytest.raises(CompilationError, match="independent selected-position"):
        proof.capture(builder, binding, bad)
    public = proof.public()
    public["calls"][0]["canonical_resources"] = {}
    assert proof.public()["calls"][0]["canonical_resources"]


def test_epoch_source_and_mandatory_cut_cannot_be_substituted(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch

    path, builder = builder_at(tmp_path)
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    assert epoch is not None
    cut = owner.patches.pop()
    with pytest.raises(CompilationError, match="mandatory unconditional retirement"):
        epoch.validate()
    owner.patches.append(cut)
    path.write_text(path.read_text().replace("call relocate(left,right)", "call relocate(right,left)"))
    # A sealed parser session remains its captured original source. The real
    # generation/publication checkpoint must reject the changed live file.
    with pytest.raises(CompilationError, match="source changed"):
        builder.run()


def test_epoch_does_not_admit_allocatable_out_or_descriptor_mutations_in_the_interval(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch

    source = SOURCE.replace("real(8),intent(out)::b(:)", "real(8),allocatable,intent(out)::b(:)")
    _path, builder = builder_at(tmp_path, source)
    owner = GraphOwner(builder)
    assert discover_graph_storage_epoch(owner) is None
    assert not owner.patches
    assert any("allocatable OUT" in row["reason"] for row in owner.position_epoch_declines)


def test_many_calls_to_same_children_do_not_cut_residency_at_procedure_count(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch

    count = 150
    source = SOURCE.replace("call consumer(left,right,n)",
                            "\n".join(["call consumer(left,right,n)"] * (count-1)))
    _path, builder = builder_at(tmp_path, source)
    assert count > len(builder.analysis.routines)
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    assert epoch is not None
    epoch.validate()
    assert len(epoch.selected) == count
    assert len(epoch.public()["calls"]) == count
    assert {item[1] for item in epoch._calls} == {"epochs::producer", "epochs::consumer"}
    assert owner.graph.nodes[epoch.cut_before].details["procedure"] == "epochs::relocate"
    assert len(builder.analysis._descriptor_window_proofs) == 1


def test_longest_safe_prefix_keeps_cut_before_internal_descriptor_mutation(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch

    source = SOURCE.replace("call relocate(left,right)",
                            "call relocate(left,right)\ncall consumer(left,right,n)\ncall consumer(left,right,n)")
    _path, builder = builder_at(tmp_path, source)
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    assert epoch is not None
    assert len(epoch.selected) == 2
    assert owner.graph.nodes[epoch.cut_before].details["procedure"] == "epochs::relocate"
    assert len(builder.analysis._descriptor_window_proofs) == 1


def local_builder_at(tmp_path, *, save=True):
    """Both arrays may be unallocated until a reached loop iteration."""
    source = SOURCE.replace("subroutine advance(left,right,n,steps)", "subroutine advance(n,steps)")
    source = source.replace("real(8),allocatable,intent(inout)::left(:),right(:)\ninteger,intent(in)::n,steps",
                            "real(8),allocatable" + (",save" if save else "")
                            + "::left(:),right(:)\ninteger,intent(in)::n,steps")
    source = source.replace("call relocate(left,right)", "deallocate(left,right)")
    path = tmp_path / "source.f90"
    path.write_text(source)
    fact = {"storage": "stable", "lifetime": "selected_window", "initialized": "none",
            "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial", "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"epochs::advance::" + name: dict(fact) for name in ("left", "right")}}
    return path, ScopeBuilder([path], "epochs::advance", facts=facts, options=CompilerOptions(),
                              config=OffloadConfig(policy="sections", scope_execution="graph"))


def test_exact_local_save_arrays_use_selected_epoch_with_no_prefix_definition_credit(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch

    _path, builder = local_builder_at(tmp_path)
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    assert epoch is not None
    assert epoch.module_roots == ()
    assert set(epoch.roots) == {"epochs::advance::left", "epochs::advance::right"}
    epoch.validate()
    states = epoch._descriptor.public()["resources"]
    assert {state["resource_kind"] for state in states} == {"original_local_save_allocatable"}
    assert all(state["local_storage_identity"] in builder.analysis._local_allocation_proofs for state in states)
    assert all(state["entry_definition_authority"] is False for state in states)
    assert all(state["requires_original_allocation_presence_and_alias_guards"] is True for state in states)
    assert all(state["requires_original_runtime_bounds"] is True for state in states)
    for name in ("left", "right"):
        binding = builder.entry.scope.bindings[name]
        assert epoch.capture(builder, binding, {**builder.capture_fact(binding), "initialized": "whole"})[
            "initialized"] == "none"
    # This source can have zero iterations. The selected epoch carries no
    # permission to move an allocation probe into its unexecuted prefix.
    assert owner.graph.nodes[epoch.cut_before].kind == "boundary"
    assert "DEALLOCATE" in str(owner.graph.source_nodes(epoch.cut_before)[0])


def test_local_save_epoch_retains_original_allocation_and_retirement_source(tmp_path):
    path, builder = local_builder_at(tmp_path)
    original = path.read_bytes()
    outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt", report["boundaries"])
    scope, = report["scopes"]
    epoch = scope["checked_execution_plan"]["storage_epoch"]["selected_position_epoch"]
    assert epoch["entry_definition_authority"] is False
    assert epoch["cross_frontier_retention_authorized"] is False
    assert all(resource["lifetime"] == "selected_window"
               for resource in scope["checked_execution_plan"]["storage_epoch"]["resources"])
    assert scope["graph_execution"]["epochs"]["entry_coverage"] == "none at selected-position allocation activation"
    replacement = outputs[report["sources"][str(path)]["replacement"]]
    for unchanged in ("if(.not.allocated(left))then", "allocate(left(-3:n-4),right(2:n+1))",
                      "left=0", "right=0", "deallocate(left,right)"):
        assert unchanged in replacement
    assert "fort_scope_resident_close_v1" in replacement
    assert path.read_bytes() == original


def test_ordinary_local_allocatable_never_acquires_save_epoch_authority(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch

    _path, builder = local_builder_at(tmp_path, save=False)
    owner = GraphOwner(builder)
    assert discover_graph_storage_epoch(owner) is None
    assert not owner.patches
    assert any("local SAVE ALLOCATABLE" in row["reason"] for row in owner.position_epoch_declines)


def test_local_save_declaration_proof_does_not_authorize_selected_deallocation(tmp_path):
    from compiler.frontend.descriptor_windows import prove_descriptor_window
    from compiler.frontend.local_allocatable_storage import original_local_allocation
    from compiler.frontend.source_effects import _children

    _path, builder = local_builder_at(tmp_path)
    analysis, routine = builder.analysis, builder.entry
    binding = routine.scope.bindings["left"]
    original_local_allocation(analysis, routine, binding)
    loop, = [node for node in _children(routine.execution) if type(node).__name__ == "Block_Nonlabel_Do_Construct"]
    nodes = tuple(node for node in _children(loop) if type(node).__name__ in {"Call_Stmt", "Deallocate_Stmt"})
    _graph, selected = analysis._selected_source(routine.qualified, nodes)
    with pytest.raises(CompilationError, match="unknown selected storage effects|descriptor"):
        prove_descriptor_window(analysis, routine.qualified, selected, (binding.root,))


def branch_source():
    return SOURCE.replace("call producer(right,n)\ncall consumer(left,right,n)",
        "if(mod(stage,2)==0)then\ncall producer(right,n)\ncall consumer(left,right,n)\n"
        "else\ncall producer(right,n)\ncall consumer(left,right,n)\nendif")


def test_complete_original_if_records_every_guarded_call_without_selecting_a_path(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch

    _path, builder = builder_at(tmp_path, branch_source())
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    assert epoch is not None
    assert len(epoch.selected) == 1
    assert owner.graph.nodes[epoch.selected[0]].kind == "branch"
    assert [row[1] for row in epoch._calls] == ["epochs::producer", "epochs::consumer"] * 2
    assert all(owner.graph.nodes[row[0]].guard for row in epoch._calls)
    assert owner.graph.nodes[epoch.cut_before].details["procedure"] == "epochs::relocate"
    assert epoch.public()["execution_authorized"] is False
    assert epoch.public()["entry_definition_authority"] is False
    assert epoch.public()["cross_frontier_retention_authorized"] is False
    assert len(builder.analysis._descriptor_window_proofs) == 1
    epoch.validate()


def test_structured_epoch_emission_preserves_original_if_and_one_mandatory_retirement(tmp_path):
    path, outputs, report = emit(tmp_path, branch_source())
    assert report["scope_count"] == 1, report.get("graph_owner_attempt", report["boundaries"])
    scope, = report["scopes"]
    epoch = scope["checked_execution_plan"]["storage_epoch"]["selected_position_epoch"]
    assert len(epoch["calls"]) == 4
    assert scope["graph_execution"]["epochs"]["entry_coverage"] == "none at selected-position allocation activation"
    replacement = outputs[report["sources"][str(path)]["replacement"]]
    assert "if(mod(stage,2)==0)then" in replacement
    assert "call relocate(left,right)" in replacement
    assert "fort_scope_resident_close_v1" in replacement


def test_original_single_line_if_call_has_its_exact_mapped_source_identity(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch

    source = SOURCE.replace("call consumer(left,right,n)", "if(stage>0)call consumer(left,right,n)")
    _path, builder = builder_at(tmp_path, source)
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    assert epoch is not None
    assert [owner.graph.nodes[node].kind for node in epoch.selected] == ["call", "branch"]
    assert len(epoch._calls) == 2
    assert owner.graph.nodes[epoch._calls[1][0]].guard
    epoch.validate()


def test_descriptor_change_in_if_arm_requires_its_original_control_local_retirement(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch

    source = SOURCE.replace("call consumer(left,right,n)",
        "if(stage>1)then\ncall relocate(left,right)\nendif\ncall consumer(left,right,n)")
    _path, builder = builder_at(tmp_path, source)
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    assert epoch is not None
    epoch.validate()
    public = epoch.public()
    assert public["cross_frontier_retention_authorized"] is False
    cut, = public["control_local_regions"]["retirement_subtrees"]
    assert owner.graph.nodes[cut["node_id"]].details["procedure"] == "epochs::relocate"
    assert owner.graph.nodes[cut["node_id"]].guard
    assert not epoch.covers(builder.analysis, owner.routine.qualified, (cut["node_id"],))
    assert [call["procedure"] for call in public["calls"]] == ["epochs::producer", "epochs::consumer"]
    assert len(epoch.boundaries()) == 2


def test_new_epoch_does_not_cross_an_unproved_original_nested_loop(tmp_path):
    from compiler.scopes.graph import GraphOwner
    from compiler.scopes.graph_epochs import discover_graph_storage_epoch

    source = SOURCE.replace("integer::stage", "integer::stage,iteration")
    source = source.replace("call consumer(left,right,n)",
                            "do iteration=1,2\ncall consumer(left,right,n)\nenddo")
    _path, builder = builder_at(tmp_path, source)
    owner = GraphOwner(builder)
    assert discover_graph_storage_epoch(owner) is None
    assert not owner.patches
