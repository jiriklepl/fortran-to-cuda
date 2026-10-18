"""Reached source reads grant exact conditional requirements, never prefix facts."""

import json
from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.reached_read_requirements import (
    ReachedReadRequirementsProof,
    ReachedReadRequirementsUnsupported,
    prove_reached_read_requirements,
)
from compiler.ir import CompilationError
from compiler.tests.test_numerical_effects import ENTRY, SOURCE, fixture, resource


def issue(directory, source=SOURCE):
    path, analysis, selection, prepared = fixture(directory, source)
    proof = prove_reached_read_requirements(analysis, ENTRY, selection, prepared)
    return path, analysis, selection, prepared, proof


def test_exact_unconditional_self_update_and_stencil_union(tmp_path):
    source = SOURCE.replace("b(i)=a(i)+real(i,8)", "b(i)=b(i)+a(i-1)+a(i+1)\nb(i)=b(i)*2.0_8")
    _path, analysis, selection, prepared, proof = issue(tmp_path, source)
    assert isinstance(proof, ReachedReadRequirementsProof)
    assert proof.prepared is prepared
    assert proof.validate(analysis, ENTRY, selection, prepared) is proof
    assert prove_reached_read_requirements(analysis, ENTRY, selection, prepared) is proof
    assert len(resource(proof, "a").reads) == 2
    assert {box.axes[0].lower.value for box in resource(proof, "a").reads} == {-4, -2}
    assert len(resource(proof, "b").reads) == 1
    assert resource(proof, "b").reads[0].axes[0].lower.value == -3
    assert all(not item.writes and not item.overwrites for item in proof.sections.resources)
    public = proof.public()
    assert public == json.loads(json.dumps(public))
    for authority in ("prefix_definitions", "capture", "descriptor_lifetime", "alias", "execution"):
        assert public[authority + "_authorized"] is False


def test_read_after_local_write_does_not_invent_entry_definition(tmp_path):
    source = SOURCE.replace("b(i)=a(i)+real(i,8)", "b(i)=1.0_8\nc(i)=b(i)")
    with pytest.raises(ReachedReadRequirementsUnsupported, match="before writes"):
        issue(tmp_path, source)


def test_shifted_self_read_cannot_borrow_previous_iteration_definition(tmp_path):
    source = SOURCE.replace("b(i)=a(i)+real(i,8)", "b(i)=b(i-1)+a(i)")
    with pytest.raises(CompilationError, match="(self-update|loop-carried|independent|offload)"):
        issue(tmp_path, source)


def test_conditional_preservation_upload_is_not_an_original_must_read(tmp_path):
    source = SOURCE.replace("b(i)=a(i)+real(i,8)",
                            "if(i>=lbound(b,1).and.a(i)>0.0_8)then\nb(i)=a(i)\nendif")
    with pytest.raises(ReachedReadRequirementsUnsupported, match="unconditional"):
        issue(tmp_path, source)


def test_conditional_merge_operands_cannot_supply_unconditional_requirements(tmp_path):
    source = SOURCE.replace("b(i)=a(i)+real(i,8)", "b(i)=merge(a(i),c(i),i>0)")
    with pytest.raises(ReachedReadRequirementsUnsupported, match="conditional expression"):
        issue(tmp_path, source)


def test_source_empty_loop_and_tiled_domain_remain_guarded(tmp_path):
    source = SOURCE.replace("sweep(a,b,c,n)", "sweep(a,b,c,n,tile)")
    source = source.replace("integer,intent(in)::n", "integer,intent(in)::n,tile")
    source = source.replace("integer::i", "integer::i,block")
    source = source.replace("private(i)", "private(i,block)")
    source = source.replace("do i=-3,n\nb(i)=a(i)+real(i,8)\nenddo", """do block=-3,n,tile
do i=block,min(block+tile-1,n)
b(i)=b(i)+a(i)
enddo
enddo""")
    _path, _analysis, _selection, _prepared, proof = issue(tmp_path, source)
    assert {item.role for item in proof.preconditions} == {"positive_tile_step", "tile_integer_range"}
    for item in proof.sections.resources:
        for box in item.reads:
            iteration, = box.iterations
            assert iteration.lower.value == -3
            assert iteration.upper.resource == "argument::n"
            assert box.axes[0].lower.value == -3


def test_multidimensional_faces_are_not_replaced_by_their_bounding_hull(tmp_path):
    source = SOURCE.replace("a(-4:)", "a(-4:,-4:)").replace("b(-4:),c(-4:)", "b(-4:,-4:),c(-4:,-4:)")
    source = source.replace("integer::i", "integer::i,j").replace("private(i)", "private(i,j)")
    source = source.replace("do i=-3,n\nb(i)=a(i)+real(i,8)\nenddo", """do j=-3,n
do i=-3,n
b(i,j)=b(i,j)+a(i-1,j)+a(i+1,j)+a(i,j-1)+a(i,j+1)
enddo
enddo""")
    _path, _analysis, _selection, _prepared, proof = issue(tmp_path, source)
    incoming = resource(proof, "a")
    assert len(incoming.reads) == 4
    assert {tuple(axis.lower.value for axis in box.axes) for box in incoming.reads} == {
        (-4, -3), (-2, -3), (-3, -4), (-3, -2)}
    assert all(len(box.iterations) == 2 for box in incoming.reads)
    assert len(resource(proof, "b").reads) == 1


def joined_source(second):
    opening = SOURCE.index("!$omp parallel do")
    closing = SOURCE.index("end subroutine")
    return SOURCE[:opening] + """!$omp parallel private(i)
!$omp do
do i=-3,n
b(i)=b(i)+a(i)
enddo
!$omp end do nowait
!$omp do
do i=-3,n
""" + second + """
enddo
!$omp end do
!$omp end parallel
""" + SOURCE[closing:]


def test_independent_joined_nowait_units_use_same_entry_snapshot(tmp_path):
    _path, _analysis, _selection, _prepared, proof = issue(tmp_path, joined_source("c(i)=c(i)+a(i)"))
    assert len(proof.units) == 2
    assert [{item.resource for item in unit.sections.resources} for unit in proof.units] == [
        {"argument::a", "argument::b"}, {"argument::a", "argument::c"}]
    assert len(resource(proof, "a").reads) == 1


def test_cross_unit_read_cannot_borrow_other_units_write(tmp_path):
    with pytest.raises(ReachedReadRequirementsUnsupported, match="group entry"):
        issue(tmp_path, joined_source("c(i)=b(i)"))


@pytest.mark.parametrize("mutation", ["units", "sections", "record", "selection", "prepared", "copy", "registry"])
def test_registered_payload_and_source_authority_cannot_be_replaced(tmp_path, mutation):
    _path, analysis, selection, prepared, proof = issue(tmp_path)
    if mutation == "units":
        object.__setattr__(proof, "units", ())
    elif mutation == "sections":
        object.__setattr__(proof, "sections", replace(proof.sections, resources=()))
    elif mutation == "record":
        object.__setattr__(proof, "_record", {})
    elif mutation == "selection":
        selection = tuple(reversed(selection))
    elif mutation == "prepared":
        prepared = copy(prepared)
    elif mutation == "copy":
        proof = copy(proof)
    else:
        analysis = copy(analysis)
    with pytest.raises(CompilationError, match="authority|payload"):
        proof.validate(analysis, ENTRY, selection, prepared)


def test_dependency_binding_identity_cannot_be_replaced_by_equal_public_metadata(tmp_path):
    _path, analysis, selection, prepared, proof = issue(tmp_path)
    incoming = resource(proof, "a")
    box, = incoming.reads
    axis, = box.axes
    dependency, = axis.upper.dependencies
    replacement = replace(dependency, binding=copy(dependency.binding))
    altered_axis = replace(axis, upper=replace(axis.upper, dependencies=(replacement,)))
    altered = replace(incoming, reads=(replace(box, axes=(altered_axis,)),))
    object.__setattr__(proof, "sections", replace(proof.sections, resources=(altered,)))
    assert proof.sections.public() == proof.public()["sections"]
    with pytest.raises(CompilationError, match="binding authority"):
        proof.validate(analysis, ENTRY, selection, prepared)


def test_source_disk_mutation_is_rejected_by_live_source_analysis(tmp_path):
    path, analysis, selection, prepared, proof = issue(tmp_path)
    path.write_text(SOURCE.replace("b(i)=a(i)+real(i,8)", "b(i)=1.0_8"))
    with pytest.raises(CompilationError, match="changed"):
        proof.validate(analysis, ENTRY, selection, prepared)
    with pytest.raises(CompilationError, match="changed"):
        analysis.verify_publication()


def test_original_preparation_configuration_remains_authenticated(tmp_path):
    _path, analysis, selection, prepared, proof = issue(tmp_path)
    object.__setattr__(prepared._provider, "config", replace(prepared._provider.config, host_threads=3))
    with pytest.raises(CompilationError, match="configuration|authority"):
        proof.validate(analysis, ENTRY, selection, prepared)
