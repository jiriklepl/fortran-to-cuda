"""Original branch-local cuts keep independently checked storage regions."""
from copy import copy
from dataclasses import replace

import pytest

from compiler.ir import CompilationError
from compiler.scopes.graph import GraphOwner
from compiler.scopes.graph_epochs import discover_graph_storage_epoch
from compiler.tests.test_graph_storage_epochs import SOURCE, builder_at, emit

CONDITIONAL_SOURCE = SOURCE.replace(
    "subroutine advance(left,right,n,steps)",
    "subroutine advance(left,right,n,steps,change)",
).replace(
    "integer,intent(in)::n,steps",
    "integer,intent(in)::n,steps\nlogical,intent(in)::change",
).replace(
    "call producer(right,n)\ncall consumer(left,right,n)",
    "call producer(right,n)\nif(change)then\ncall relocate(left,right)\nendif\ncall consumer(left,right,n)",
)


def test_independent_single_call_regions_survive_an_original_conditional_swap(tmp_path):
    path, output, report = emit(tmp_path, CONDITIONAL_SOURCE)
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    scope, = report["scopes"]
    epoch = scope["checked_execution_plan"]["storage_epoch"]["selected_position_epoch"]
    assert [row["procedure"] for row in epoch["calls"]] == ["epochs::producer", "epochs::consumer"]
    assert epoch["entry_definition_authority"] is False
    assert epoch["cross_frontier_retention_authorized"] is False
    assert path.read_text() == CONDITIONAL_SOURCE
    generated = output[report["sources"][str(path)]["replacement"]]
    assert "if(change)then" in generated
    assert generated.count("call relocate(left,right)") == 2
    branch = generated.split("if(change)then", 1)[1].split("call relocate(left,right)", 1)[0]
    assert "fort_scope_resident_close_v1" in branch


def test_conditional_region_capture_keeps_none_and_excludes_the_swapping_call(tmp_path):
    _path, builder = builder_at(tmp_path, CONDITIONAL_SOURCE)
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    assert epoch is not None
    epoch.validate()
    for binding in (builder.entry.scope.bindings[name] for name in ("left", "right")):
        assert epoch.capture(builder, binding, {**builder.capture_fact(binding), "initialized": "whole"})[
            "initialized"] == "none"
    for identity, node in owner.graph.nodes.items():
        if node.kind == "call" and node.details.get("procedure") == "epochs::relocate":
            assert not epoch.covers(builder.analysis, owner.routine.qualified, (identity,))


def issued(tmp_path, source=CONDITIONAL_SOURCE):
    _path, builder = builder_at(tmp_path, source)
    owner = GraphOwner(builder)
    epoch = discover_graph_storage_epoch(owner)
    assert epoch is not None
    assert epoch.public()["version"] == 2
    return builder, owner, epoch


def test_every_region_and_retirement_keeps_its_original_control_path(tmp_path):
    _builder, owner, epoch = issued(tmp_path)
    record = epoch.public()["control_local_regions"]
    assert record["entry_definition_authority"] is False
    assert record["payload_access_authorized"] is False
    assert record["execution_authorized"] is False
    assert record["cross_cut_retention_authorized"] is False
    cut, = record["retirement_subtrees"]
    assert owner.graph.nodes[cut["node_id"]].guard
    assert owner.graph.nodes[cut["node_id"]].details["procedure"] == "epochs::relocate"
    assert [owner.graph.nodes[row["node_id"]].kind for row in record["regions"]] == [
        "call", "operation", "call"]
    assert len(epoch.boundaries()) == 2
    assert all(row["resource"] in {"argument::left", "argument::right"} for row in record["resources"])


@pytest.mark.parametrize("mutation", ["remove", "copy", "lifecycle"])
def test_retirement_patch_identity_and_lifecycle_are_required(tmp_path, mutation):
    _builder, owner, epoch = issued(tmp_path)
    _identity, patch, _lifecycle = epoch._descriptor._cuts[0]
    index = next(index for index, (_first, _last, current) in enumerate(owner.patches) if current is patch)
    first, last, _original = owner.patches[index]
    if mutation == "remove":
        owner.patches.pop(index)
    else:
        replacement = copy(patch) if mutation == "copy" else replace(patch, lifecycle=None)
        owner.patches[index] = (first, last, replacement)
    with pytest.raises(CompilationError, match="mandatory retirement"):
        epoch.validate()


def test_omitting_a_singleton_consumer_cannot_mint_partial_coverage(tmp_path):
    builder, owner, epoch = issued(tmp_path)
    descriptor = epoch._descriptor
    object.__setattr__(descriptor, "_components", descriptor._components[:-1])
    with pytest.raises(CompilationError, match="uncovered original source"):
        descriptor.validate(builder.analysis, owner.routine.qualified, epoch.selected)


def test_lost_original_component_registration_rejects_composition(tmp_path):
    builder, owner, epoch = issued(tmp_path)
    _identity, proof = epoch._descriptor._components[0]
    del builder.analysis._descriptor_window_proofs[proof.identity]
    with pytest.raises(CompilationError, match="registered original interval|attachment changed"):
        epoch._descriptor.validate(builder.analysis, owner.routine.qualified, epoch.selected)


def test_component_callback_cannot_remove_a_later_retirement(tmp_path, monkeypatch):
    from compiler.frontend.descriptor_windows import DescriptorWindowProof

    builder, owner, epoch = issued(tmp_path)
    original = DescriptorWindowProof.validate
    target = epoch._descriptor._components[0][1]
    cut = epoch._descriptor._cuts[0][1]

    def validate(proof, *args, **kwargs):
        result = original(proof, *args, **kwargs)
        if proof is target:
            owner.patches[:] = [item for item in owner.patches if item[2] is not cut]
        return result

    monkeypatch.setattr(DescriptorWindowProof, "validate", validate)
    with pytest.raises(CompilationError, match="mandatory retirement"):
        epoch._descriptor.validate(builder.analysis, owner.routine.qualified, epoch.selected)


def test_non_scalar_predicate_retires_before_the_complete_original_if(tmp_path):
    source = CONDITIONAL_SOURCE.replace("if(change)then", "if(allocated(left))then")
    _builder, owner, epoch = issued(tmp_path, source)
    identity, patch, _proof = epoch._descriptor._cuts[0]
    assert owner.graph.nodes[identity].kind == "branch"
    assert patch.original_source.lstrip().startswith("if(allocated(left))then")
    assert patch.conditional_guard is None
    assert not epoch.covers(owner.builder.analysis, owner.routine.qualified, (identity,))


def test_unproved_escape_does_not_gain_reopening_from_a_declared_intent(tmp_path):
    source = CONDITIONAL_SOURCE.replace("call relocate(left,right)\nendif", "call unknown_host(left,right)\nendif")
    _path, builder = builder_at(tmp_path, source)
    owner = GraphOwner(builder)
    assert discover_graph_storage_epoch(owner) is None
    assert not owner.patches


def test_hidden_module_allocation_mutation_remains_in_the_cut_lifecycle(tmp_path):
    source = CONDITIONAL_SOURCE.replace("module epochs\ncontains", "module epochs\nreal(8),allocatable::extra(:)\ncontains")
    source = source.replace("call move_alloc(left,temporary)", "if(allocated(extra))deallocate(extra)\ncall move_alloc(left,temporary)")
    _builder, _owner, epoch = issued(tmp_path, source)
    _identity, _patch, lifecycle = epoch._descriptor._cuts[0]
    public = lifecycle.public()
    mutations = [row for record in (public, *public["closure"]) for row in record["mutations"]]
    assert any(row["resource"] == "epochs::extra" for row in mutations)
    assert public["complete"] is True
    assert not public["escapes"]


def test_hard_component_failure_does_not_publish_a_retirement_or_epoch(tmp_path, monkeypatch):
    import compiler.scopes.graph_regions as regions

    _path, builder = builder_at(tmp_path, CONDITIONAL_SOURCE)
    owner = GraphOwner(builder)
    sequence = next(node for node in owner.graph.nodes.values() if node.kind == "sequence"
                    and len(node.children) == 5)
    selected, cut = tuple(sequence.children[1:4]), sequence.children[4]

    def fail(*_args, **_kwargs):
        raise CompilationError("original source attachment changed")

    monkeypatch.setattr(regions, "prove_descriptor_window", fail)
    with pytest.raises(CompilationError, match="source attachment changed"):
        regions.discover_control_local_epoch(owner, selected, cut, ("argument::left", "argument::right"), (),
            tuple(builder.entry.scope.bindings[name] for name in ("left", "right")))
    assert not owner.patches
    assert getattr(owner, "position_epoch", None) is None


def test_later_issuer_cannot_change_an_earlier_component_before_publication(tmp_path, monkeypatch):
    import compiler.scopes.graph_regions as regions

    _path, builder = builder_at(tmp_path, CONDITIONAL_SOURCE)
    owner = GraphOwner(builder)
    sequence = next(node for node in owner.graph.nodes.values() if node.kind == "sequence"
                    and len(node.children) == 5)
    selected, cut = tuple(sequence.children[1:4]), sequence.children[4]
    original, retained = regions.prove_descriptor_window, []

    def issue(*args, **kwargs):
        proof = original(*args, **kwargs)
        if retained:
            object.__setattr__(retained[0], "_record", dict(retained[0]._record))
        retained.append(proof)
        return proof

    monkeypatch.setattr(regions, "prove_descriptor_window", issue)
    with pytest.raises(CompilationError, match="retained proof attachment"):
        regions.discover_control_local_epoch(owner, selected, cut, ("argument::left", "argument::right"), (),
            tuple(builder.entry.scope.bindings[name] for name in ("left", "right")))
    assert not owner.patches
    assert getattr(owner, "position_epoch", None) is None


@pytest.mark.parametrize("association", ["common /flags/gate,other", "equivalence(gate,other)"])
def test_predicate_with_common_or_equivalence_never_gets_an_inner_cut(tmp_path, association):
    from compiler.scopes.graph_regions import _pure_scalar_condition

    source = CONDITIONAL_SOURCE.replace("module epochs\ncontains",
        "module epochs\nlogical::gate,other\n" + association + "\ncontains")
    source = source.replace("if(change)then", "if(gate)then")
    _path, builder = builder_at(tmp_path, source)
    owner = GraphOwner(builder)
    condition = next(identity for identity, node in owner.graph.nodes.items()
                     if node.details.get("evaluation") == "condition"
                     and str(owner.graph.source_nodes(identity)[0]).lower() == "gate")
    assert not _pure_scalar_condition(owner, condition)
    epoch = discover_graph_storage_epoch(owner)
    if epoch is not None:
        cuts = epoch.public().get("control_local_regions", {}).get("retirement_subtrees", [])
        assert cuts
        assert all(owner.graph.nodes[cut["node_id"]].kind == "branch" for cut in cuts)


def test_every_if_elseif_else_arm_keeps_its_original_retirement(tmp_path):
    source = CONDITIONAL_SOURCE.replace("call relocate(left,right)\nendif",
        "call relocate(left,right)\nelse if(stage>1)then\ncall relocate(right,left)\n"
        "else\ncall relocate(left,right)\nendif")
    _builder, owner, epoch = issued(tmp_path, source)
    cuts = epoch.public()["control_local_regions"]["retirement_subtrees"]
    assert len(cuts) == 3
    assert len({tuple(owner.graph.nodes[cut["node_id"]].guard) for cut in cuts}) == 3
    assert [row["procedure"] for row in epoch.public()["calls"]] == ["epochs::producer", "epochs::consumer"]


def test_one_unchecked_elseif_predicate_moves_the_cut_before_the_whole_if(tmp_path):
    source = CONDITIONAL_SOURCE.replace("call relocate(left,right)\nendif",
        "call relocate(left,right)\nelse if(allocated(right))then\ncall relocate(right,left)\nendif")
    _builder, owner, epoch = issued(tmp_path, source)
    cut, = epoch.public()["control_local_regions"]["retirement_subtrees"]
    assert owner.graph.nodes[cut["node_id"]].kind == "branch"
    assert cut["node_id"] == epoch.selected[1]


def test_same_callee_at_a_retirement_site_cannot_borrow_a_retained_sites_authority(tmp_path):
    from compiler.scopes.graph_epochs import _mapped_child_bindings
    from compiler.scopes.module_owner import ModuleOwner

    source = CONDITIONAL_SOURCE.replace("if(change)then", "if(allocated(left))then")
    source = source.replace("call relocate(left,right)\nendif", "call producer(right,n)\nendif")
    builder, owner, epoch = issued(tmp_path, source)
    builder._graph_storage_epoch = epoch
    retained_id = next(identity for identity, procedure, _mappings in epoch._calls if procedure == "epochs::producer")
    retained_call, = owner.graph.source_nodes(retained_id)
    resolved = builder.execution_source_call(owner.routine, retained_call)
    routine = builder.analysis.routines[resolved.procedure]
    mappings = {mapping.formal_binding.root: owner.canonical(mapping.binding)
                for mapping in resolved.mappings if mapping.formal_binding.rank}
    owner.active.append(routine.qualified)
    try:
        member = ModuleOwner(owner, routine, mappings, original_call=retained_call)
    finally:
        owner.active.pop()
    owner.members[routine.qualified] = member
    closure = epoch._descriptor.public()["source_call_closure"]
    result, _actuals = _mapped_child_bindings(epoch, member.builder, builder.analysis, routine.qualified, closure)
    assert result is routine
    cut_id = next(identity for identity, node in owner.graph.nodes.items()
                  if node.kind == "call" and node.details.get("procedure") == "epochs::producer"
                  and identity != retained_id)
    assert not epoch.covers(builder.analysis, owner.routine.qualified, (cut_id,))
    member.original_source_call, = owner.graph.source_nodes(cut_id)
    with pytest.raises(CompilationError, match="escapes its proved original closure"):
        _mapped_child_bindings(epoch, member.builder, builder.analysis, routine.qualified, closure)
