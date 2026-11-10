"""Boundary planes can alias only within their original loop iteration."""

import pytest

from compiler.ir import CompilationError
from compiler.tests.test_numerical_effects import SOURCE, resource
from compiler.tests.test_reached_read_requirements import issue


def face_source(axis, target="0", incoming="n", *, body=None):
    source = SOURCE.replace("a(-4:)", "a(-4:,-4:,-4:)")
    source = source.replace("b(-4:),c(-4:)", "b(-4:,-4:,-4:),c(-4:,-4:,-4:)")
    source = source.replace("integer::i", "integer::i,j").replace("private(i)", "private(i,j)")
    write, read = ["i", "j"], ["i", "j"]
    write.insert(axis, target)
    read.insert(axis, incoming)
    statement = body or f"b({','.join(write)})=b({','.join(read)})+1.0_8"
    return source.replace("do i=-3,n\nb(i)=a(i)+real(i,8)\nenddo",
                          f"do j=1,n\ndo i=1,n\n{statement}\nenddo\nenddo")


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize(("target", "incoming"), [("0", "1"), ("n+1", "n"), ("0", "n"), ("n", "n")])
def test_face_reads_keep_exact_original_planes_and_iteration_guards(tmp_path, axis, target, incoming):
    _path, analysis, selection, prepared, proof = issue(tmp_path, face_source(axis, target, incoming))
    assert proof.validate(analysis, prepared.procedure, selection, prepared) is proof
    incoming_resource = resource(proof, "b")
    read, = incoming_resource.reads
    assert len(read.axes) == 3
    assert len(read.iterations) == 2
    assert read.axes[axis].lower == read.axes[axis].upper
    assert all(iteration.lower.value == 1 and iteration.upper.resource == "argument::n"
               for iteration in read.iterations)
    assert not incoming_resource.writes
    assert not incoming_resource.overwrites
    assert not proof.public()["prefix_definitions_authorized"]


@pytest.mark.parametrize("body", [
    "b(0,i,j)=b(n,i+1,j)",
    "b(0,i,j)=b(n,j,i)",
    "b(0,i,j)=b(n,1,j)",
    "b(0,i,j)=b(0,i-1,j)",
])
def test_unproved_cross_iteration_reads_still_decline(tmp_path, body):
    with pytest.raises(CompilationError, match="self-update|loop-carried|independent|offload"):
        issue(tmp_path, face_source(0, body=body))


def test_different_planes_do_not_credit_reads_after_a_local_write(tmp_path):
    _path, analysis, selection, prepared, proof = issue(
        tmp_path, face_source(0, body="b(0,i,j)=1.0_8\nb(n,i,j)=b(0,i,j)"))
    assert proof.validate(analysis, prepared.procedure, selection, prepared) is proof
    assert not proof.sections.resources
    assert not proof.public()["prefix_definitions_authorized"]


def test_multiple_incoming_planes_remain_separate(tmp_path):
    _path, _analysis, _selection, _prepared, proof = issue(
        tmp_path, face_source(0, body="b(0,i,j)=b(1,i,j)+b(2,i,j)"))
    reads = resource(proof, "b").reads
    assert len(reads) == 2
    assert {box.axes[0].lower.value for box in reads} == {1, 2}
