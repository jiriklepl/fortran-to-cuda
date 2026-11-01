"""Complete original line producer/consumer IR and source-bound emission."""

from dataclasses import replace

import pytest

from compiler.emission.cuda.tile_private import generate_tile_function
from compiler.frontend.line_snapshot import prove_line_snapshot
from compiler.frontend.lowering import lower_line_snapshot, lower_tile_private
from compiler.ir import ArrayAccess, Assignment, Binary, Block, CompilationError, Loop, Size
from compiler.ir.tile_private import TilePrivateRegion
from compiler.tests.test_line_snapshot import fixture, source


@pytest.mark.parametrize("axis", [1, 2, 3])
@pytest.mark.parametrize("kind", [4, 8])
def test_original_snapshot_uses_descriptor_extent_and_separate_transverse_axes(tmp_path, axis, kind):
    _, analysis, routine, nodes = fixture(tmp_path, source(axis, kind))
    proof = prove_line_snapshot(analysis, routine, nodes)
    lowered = lower_line_snapshot(proof)
    assert lower_tile_private(proof) == lowered
    assert isinstance(lowered.region, TilePrivateRegion)
    assert len(lowered.region.outer_loops) == 2
    (scratch,) = lowered.region.scratch_symbols
    (producer,) = lowered.region.producer.statements
    assert isinstance(producer, Loop)
    assert isinstance(producer.upper, Binary)
    assert producer.upper.left == Size(scratch, 1)
    (assignment,) = producer.body.statements
    assert isinstance(assignment, Assignment)
    assert isinstance(assignment.value, ArrayAccess)
    assert assignment.target.symbol is scratch
    assert lowered.line_snapshot_contract.scratch_lower == -2
    assert lowered.line_snapshot_contract.line_axis == axis
    assert lowered.line_snapshot_contract.public()["execution_authorized"] is False
    assert len(lowered.lower_bounds) == 6
    emitted = "\n".join(generate_tile_function(lowered.region, addressing=lowered.addressing, device=False))
    assert "Complete original producer phase" in emitted
    assert "Same-thread phase barrier" in emitted
    assert "ubound" not in emitted.lower()
    assert "tile_remaining" in emitted


def test_altered_emission_ir_and_original_proof_do_not_authorize_source_substitution(tmp_path):
    _, analysis, routine, nodes = fixture(tmp_path)
    lowered = lower_line_snapshot(prove_line_snapshot(analysis, routine, nodes))
    for candidate in [
        replace(lowered.region, producer=Block(())),
        replace(lowered.region, consumer=Block(())),
        replace(lowered.region, outer_loops=lowered.region.outer_loops[::-1]),
    ]:
        with pytest.raises(ValueError, match="substitute"):
            generate_tile_function(candidate)


def test_first_slice_declines_dynamic_private_lower_bound(tmp_path):
    text = (
        source()
        .replace("advance(a,mask,nx,ny,nz)", "advance(a,mask,nx,ny,nz,lb)")
        .replace("::nx,ny,nz", "::nx,ny,nz,lb")
        .replace("tmp(-2:", "tmp(lb:")
    )
    _, analysis, routine, nodes = fixture(tmp_path, text)
    proof = prove_line_snapshot(analysis, routine, nodes)
    with pytest.raises(CompilationError, match="constant private lower bound"):
        lower_line_snapshot(proof)
