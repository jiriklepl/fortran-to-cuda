"""Native-only definition state retains reads without device layout authority."""

from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.native_definitions import (
    intersect_rectangles,
    prove_native_definition_state,
    prove_native_definitions,
)
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.scopes.immutable_inputs import immutable_call_inputs

SOURCE = """module native_stage
implicit none
real(8),parameter::coefficients(-2:0)=[1.d0,2.d0,3.d0]
contains
subroutine invoke(previous,b,c,n,stage,scale)
real(8),intent(in)::previous(-1:,-1:)
real(8),intent(out)::b(-1:,-1:),c(-1:,-1:)
integer,intent(in)::n,stage
real(8),intent(in)::scale
call initialize(previous,b,c,n,stage,scale,coefficients)
end subroutine
subroutine initialize(previous,b,c,n,stage,scale,weights)
real(8),intent(in)::previous(-1:,-1:),weights(1:3)
real(8),intent(out)::b(-1:,-1:),c(-1:,-1:)
integer,intent(in)::n,stage
real(8),intent(in)::scale
integer::i,j
if(stage>1) then
!$omp parallel private(i,j)
!$omp do
do j=1,n
do i=1,n
b(i,j)=previous(i,j)*weights(stage)*scale
enddo
enddo
!$omp end do nowait
!$omp do
do j=1,n
do i=1,n
c(i,j)=previous(i,j)+weights(stage)
enddo
enddo
!$omp end do
!$omp end parallel
else
!$omp parallel private(i,j)
!$omp do
do j=1,n
do i=1,n
b(i,j)=0.d0
enddo
enddo
!$omp end do nowait
!$omp do
do j=1,n
do i=1,n
c(i,j)=0.d0
enddo
enddo
!$omp end do
!$omp end parallel
endif
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "native.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    caller, routine = (analysis.routines["native_stage::" + name] for name in ("invoke", "initialize"))
    call, = caller.execution.content
    resolved = resolve_source_call(analysis, caller.scope, call)
    associations = immutable_call_inputs(analysis, caller, call, resolved)
    groups, body = [], []
    branch, = routine.execution.content
    for node in branch.content:
        if type(node).__name__ in {"If_Then_Stmt", "Else_Stmt", "End_If_Stmt"}:
            if body:
                groups.extend(item.original_nodes for item in grouped_original_sequence(body, {})
                              if isinstance(item, NativeGroupCandidate))
            body = []
        else:
            body.append(node)
    outputs = tuple(routine.scope.bindings[name] for name in ("b", "c"))
    return path, analysis, routine, tuple(groups), outputs, associations


def issue(analysis, routine, selection, outputs, associations):
    completion = analysis.joined_completion(routine.qualified, selection)
    return prove_native_definition_state(analysis, routine.qualified, selection, outputs=outputs,
                                          completion=completion, immutable_inputs=associations)


def test_joined_arms_retain_reads_and_only_discharge_requested_output_interiors(tmp_path):
    path, analysis, routine, groups, outputs, associations = fixture(tmp_path)
    original = path.read_bytes()
    active = issue(analysis, routine, groups[0], outputs, associations)
    pending = active.transfer()
    assert not pending.reads_covered
    assert {read.resource for read in pending.uncovered_reads} == {"argument::previous"}
    assert {read.resource for read in pending.immutable_reads} == {"argument::weights"}
    assert all(not boxes for boxes in pending.definitions.values())
    assert all(pending.candidate_definitions.values())
    entry = {"argument::previous": (pending.uncovered_reads[0].rectangle,)}
    defined = active.transfer(entry)
    zero = issue(analysis, routine, groups[1], outputs, associations).transfer()
    assert defined.reads_covered
    assert zero.reads_covered
    assert set(defined.definitions) == {"argument::b", "argument::c"}
    for root in defined.definitions:
        box, = intersect_rectangles(defined.definitions[root], zero.definitions[root])
        assert all(axis.lower.value == 1 and axis.upper.resource == "argument::n" for axis in box.axes)
        assert len(box.iterations) == 2
        assert all(dependency.binding is routine.scope.bindings["n"] for resource in active.steps[0].sections.resources
                   for dependency in resource.dependencies if dependency.resource == "argument::n")
    reads = defined.original_reads
    assert {read.source.binding.root for read in reads if read.source.kind == "scalar_read"} >= {
        "argument::stage", "argument::scale", "argument::n"}
    assert any(read.source.binding is routine.scope.bindings["weights"]
               and str(read.source.node).lower() == "weights(stage)" for read in reads)
    public = active.public()
    assert public["physical_mapping_authority"] is False
    assert public["descriptor_lifetime_authority"] is False
    assert defined.public()["requires_original_scalar_and_index_read_validity"]
    assert "argument::weights" not in defined.definitions
    assert path.read_bytes() == original


def test_fixed_array_input_still_requires_physical_mapping_for_coherence(tmp_path):
    _path, analysis, routine, groups, outputs, associations = fixture(tmp_path)
    completion = analysis.joined_completion(routine.qualified, groups[0])
    physical = prove_native_definitions(analysis, routine.qualified, groups[0], completion=completion)
    physical.validate(analysis, routine.qualified, groups[0])
    coefficient, = [item for item in physical.public()["resources"] if item["resource"] == "argument::weights"]
    assert coefficient["guarded_explicit_dummy_extents"] == [3]
    assert coefficient["logical_lower_bounds"] == [1]
    for step in physical.steps:
        coefficient, = [item for item in step.sections.resources if item.resource == "argument::weights"]
        assert coefficient.reads
        assert not coefficient.writes
        assert not coefficient.overwrites
    logical = issue(analysis, routine, groups[0], outputs, associations)
    assert logical.public()["physical_mapping_authority"] is False
    assert logical.public()["descriptor_lifetime_authority"] is False
    assert logical.transfer().candidate_definitions


def test_missing_immutable_authority_keeps_fixed_array_read_uncovered(tmp_path):
    _path, analysis, routine, groups, outputs, _associations = fixture(tmp_path)
    proof = issue(analysis, routine, groups[0], outputs, {})
    pending = proof.transfer()
    assert {read.resource for read in pending.uncovered_reads} == {"argument::previous", "argument::weights"}
    assert not pending.immutable_reads
    assert all(not boxes for boxes in pending.definitions.values())


@pytest.mark.parametrize("mutation", ["copy", "output", "association", "selection"])
def test_native_logical_state_rejects_foreign_authorities(tmp_path, mutation):
    _path, analysis, routine, groups, outputs, associations = fixture(tmp_path)
    proof = issue(analysis, routine, groups[0], outputs, associations)
    if mutation == "copy":
        with pytest.raises(CompilationError, match="registered exact"):
            copy(proof).validate(analysis, routine.qualified, groups[0])
    elif mutation == "output":
        with pytest.raises(CompilationError, match="original array binding"):
            issue(analysis, routine, groups[0], (copy(outputs[0]),), associations)
    elif mutation == "association":
        original, = associations.values()
        changed = replace(original, actual=copy(original.actual))
        with pytest.raises(CompilationError, match="PARAMETER actual|source authority"):
            issue(analysis, routine, groups[0], outputs, {changed.resource: changed})
    else:
        with pytest.raises(CompilationError, match="registered exact"):
            proof.validate(analysis, routine.qualified, groups[1])


def test_native_logical_state_observes_changed_original_source(tmp_path):
    path, analysis, routine, groups, outputs, associations = fixture(tmp_path)
    proof = issue(analysis, routine, groups[0], outputs, associations)
    path.write_text(path.read_text().replace("weights(stage)*scale", "weights(stage+1)*scale"))
    with pytest.raises(CompilationError, match="changed"):
        proof.validate(analysis, routine.qualified, groups[0])


def test_logical_read_cannot_substitute_equal_but_foreign_binding(tmp_path):
    _path, analysis, routine, groups, outputs, associations = fixture(tmp_path)
    proof = issue(analysis, routine, groups[0], outputs, associations)
    step, *remaining = proof.steps
    read, *reads = step.sections.original_reads
    changed = replace(read, binding=copy(read.binding))
    sections = replace(step.sections, original_reads=(changed, *reads))
    object.__setattr__(proof, "steps", (replace(step, sections=sections), *remaining))
    with pytest.raises(CompilationError, match="read authority changed"):
        proof.transfer()
    with pytest.raises(CompilationError, match="cached authority changed"):
        issue(analysis, routine, groups[0], outputs, associations)


def test_uninitialized_private_input_cannot_grant_output_credit(tmp_path):
    source = SOURCE.replace("integer::i,j", "integer::i,j\nreal(8)::scratch(1)")
    source = source.replace("private(i,j)", "private(i,j,scratch)").replace("previous(i,j)", "scratch(1)")
    _path, analysis, routine, groups, outputs, associations = fixture(tmp_path, source)
    proof = issue(analysis, routine, groups[0], outputs, associations)
    result = proof.transfer()
    assert not result.reads_covered
    assert {read.resource for read in result.uncovered_reads} == {"native_stage::initialize::scratch"}
    assert all(not boxes for boxes in result.definitions.values())


def test_serial_state_uses_same_logical_walker_and_preserves_branch_meet(tmp_path):
    source = SOURCE
    for line in SOURCE.splitlines():
        if line.startswith("!$omp"):
            source = source.replace(line + "\n", "")
    _path, analysis, routine, _groups, outputs, associations = fixture(tmp_path, source)
    selection = tuple(routine.execution.content)
    proof = prove_native_definition_state(analysis, routine.qualified, selection,
                                           outputs=outputs, immutable_inputs=associations)
    pending = proof.transfer()
    defined = proof.transfer({"argument::previous": (pending.uncovered_reads[0].rectangle,)})
    assert defined.reads_covered
    assert all(len(boxes) == 1 for boxes in defined.definitions.values())
    assert any(read.source.binding is routine.scope.bindings["stage"] for read in defined.original_reads)


def test_joined_state_cannot_use_other_nowait_units_as_entry_coverage(tmp_path):
    source = SOURCE.replace("c(i,j)=previous(i,j)+weights(stage)", "c(i,j)=b(i,j)+weights(stage)")
    _path, analysis, routine, groups, outputs, associations = fixture(tmp_path, source)
    proof = issue(analysis, routine, groups[0], outputs, associations)
    pending = proof.transfer()
    previous = next(read.rectangle for read in pending.uncovered_reads if read.resource == "argument::previous")
    result = proof.transfer({"argument::previous": (previous,)})
    assert {read.resource for read in result.uncovered_reads} == {"argument::b"}
    assert all(not boxes for boxes in result.definitions.values())


@pytest.mark.parametrize("subset", [False, True])
def test_shallow_projection_issuance_cannot_mutate_parent_registry(tmp_path, subset):
    source = "\n".join(line for line in SOURCE.splitlines() if not line.startswith("!$omp")) + "\n"
    _path, analysis, routine, _groups, outputs, associations = fixture(tmp_path, source)
    selection = tuple(routine.execution.content)
    parent = prove_native_definition_state(analysis, routine.qualified, selection,
        outputs=outputs, immutable_inputs=associations)
    registry = analysis._native_definition_state_proofs
    retained = dict(registry)
    projected = analysis.private_projection()
    child = prove_native_definition_state(projected, routine.qualified, selection,
        outputs=outputs[:1] if subset else outputs, immutable_inputs=associations)
    assert projected._native_definition_state_proofs is not registry
    assert analysis._native_definition_state_proofs is registry
    assert registry == retained
    assert child is not parent
    assert child.validate(projected, routine.qualified, selection) is child
    assert parent.validate(analysis, routine.qualified, selection) is parent
    with pytest.raises(CompilationError, match="registered exact"):
        child.validate(analysis, routine.qualified, selection)
    with pytest.raises(CompilationError, match="registered exact"):
        parent.validate(projected, routine.qualified, selection)
