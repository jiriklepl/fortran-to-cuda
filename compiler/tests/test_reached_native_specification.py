"""Reached native accesses do not re-evaluate unrelated entry declarations."""
from hashlib import sha256

import pytest

from compiler.frontend.reached_read_requirements import native_read_sections, prove_native_read_requirements
from compiler.frontend.source_effects import SourceEffects

SOURCE = """module reached
implicit none
integer,protected::tile(3)=[-7,4,-9]
contains
subroutine work()
integer,parameter::index=2
integer::width
real(8)::scratch(-1:tile(index)+2)
width=tile(index)
end subroutine
end module
"""


def setup(tmp_path, *, sealed=False, body="width=tile(index)"):
    path = tmp_path / "original.f90"
    path.write_text(SOURCE.replace("width=tile(index)", body))
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    procedure = "reached::work"
    routine = analysis.routines[procedure]
    selection = tuple(routine.execution.content)
    return path, analysis, procedure, selection


@pytest.mark.parametrize("sealed", [False, True])
def test_reached_point_reads_ignore_unreferenced_automatic_shape(tmp_path, sealed):
    path, analysis, procedure, selection = setup(tmp_path, sealed=sealed)
    before = sha256(path.read_bytes()).hexdigest()
    sections = analysis.native_sections_for_nodes(procedure, selection, capture_locals=True)
    assert sections.available, sections.reason
    resource, = sections.resources
    assert resource.resource == "reached::tile"
    box, = resource.reads
    assert len(box.axes) == 1
    assert box.axes[0].point
    assert box.axes[0].lower.value == box.axes[0].upper.value == 2
    assert not resource.writes
    assert not resource.overwrites
    proof = prove_native_read_requirements(analysis, procedure, selection)
    assert proof is not None
    assert native_read_sections(analysis, procedure, selection, proof).public() == sections.public()
    assert sha256(path.read_bytes()).hexdigest() == before
    analysis.verify_publication()


def test_whole_entry_still_cannot_omit_payload_shape_evaluation(tmp_path):
    _path, analysis, procedure, selection = setup(tmp_path)
    for sections in (analysis.native_sections(procedure),
                     analysis.native_sections_for_nodes(procedure, selection, include_entry=True, capture_locals=True)):
        assert not sections.available
        assert "affine scalar arithmetic: tile(index)" in sections.reason
        assert not sections.resources


@pytest.mark.parametrize("body", ["scratch(1)=1.0_8", "width=int(scratch(1))"])
def test_reached_use_of_automatic_array_keeps_its_descriptor_obligations(tmp_path, body):
    _path, analysis, procedure, selection = setup(tmp_path, body=body)
    sections = analysis.native_sections_for_nodes(procedure, selection, capture_locals=True)
    assert not sections.available
    assert "affine scalar arithmetic: tile(index)" in sections.reason
    assert not sections.resources
