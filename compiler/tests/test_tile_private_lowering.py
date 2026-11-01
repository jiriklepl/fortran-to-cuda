"""Original typed lowering preserves indexed scratch and source authority."""

from copy import copy

import pytest

from compiler.frontend.lowering import lower_file, lower_tile_private
from compiler.frontend.tile_private import prove_tile_private
from compiler.ir import ArrayAccess, Assignment, CompilationError, Loop, ScalarType, walk_expr
from compiler.tests.test_tile_private import SOURCE, fixture


def assignments(block):
    for item in block.statements:
        if isinstance(item, Assignment):
            yield item
        elif isinstance(item, Loop):
            yield from assignments(item.body)
        else:
            yield from assignments(item.then_body)
            yield from assignments(item.else_body)


@pytest.mark.parametrize("kind", [4, 8])
def test_original_indexed_array_is_lowered_without_scalarization_or_recomputation(tmp_path, kind):
    source = SOURCE.replace("real(8)", f"real({kind})").replace("3.0_8", f"3.0_{kind}")
    path, analysis, routine, node = fixture(tmp_path, source)
    with pytest.raises(CompilationError):
        lower_file(path, routine.qualified)
    proof = prove_tile_private(analysis, routine, node)
    before = path.read_bytes()
    lowered = lower_tile_private(proof, region_id=7)
    region = lowered.region
    assert region.id == 7
    assert region.source_proof is proof
    assert [x.name for x in region.scratch_symbols] == ["tmp"]
    assert region.scratch_symbols[0].dtype is (ScalarType.REAL32 if kind == 4 else ScalarType.REAL)
    assert not region.scratch_symbols[0].parameter
    assert region.scratch_symbols[0] not in region.parameters
    assert [(binding.name, axis) for binding, axis, _ in lowered.lower_bounds] == [("a", 1), ("b", 1)]
    producer, consumer = list(assignments(region.producer)), list(assignments(region.consumer))
    assert len(producer) == 2
    assert len(consumer) == 2
    assert sum(isinstance(x.target, ArrayAccess) and x.target.symbol in region.scratch_symbols for x in producer) == 1
    scratch_reads = [x for statement in consumer for x in walk_expr(statement.value)
                     if isinstance(x, ArrayAccess) and x.symbol in region.scratch_symbols]
    assert len(scratch_reads) == 2
    assert all(item.mode == "wide" for item in lowered.addressing.decisions)
    assert path.read_bytes() == before


def test_tile_lowering_revalidates_exact_original_source_and_issuer(tmp_path):
    path, analysis, routine, node = fixture(tmp_path)
    proof = prove_tile_private(analysis, routine, node)
    with pytest.raises(CompilationError, match="exact compiler-issued"):
        lower_tile_private(copy(proof))
    path.write_text(SOURCE + "! changed\n")
    with pytest.raises(CompilationError):
        lower_tile_private(proof)


def test_private_value_from_another_work_unit_does_not_become_a_capture(tmp_path):
    source = SOURCE.replace("integer::bi,i,li", "integer::bi,i,li\nreal(8)::factor")
    source = source.replace("private(bi,i,li,tmp)", "private(bi,i,li,tmp,factor)")
    source = source.replace("do bi=1,n,t", "factor=3.0_8\ndo bi=1,n,t")
    source = source.replace("3.0_8*a(i)", "factor*a(i)")
    _, analysis, routine, node = fixture(tmp_path, source)
    # The original thread-private scalar exists before this selected unit.
    # It cannot be read from the serial caller's different storage instance.
    with pytest.raises(CompilationError, match="private|PRIVATE"):
        lower_tile_private(prove_tile_private(analysis, routine, node))


def test_original_readonly_output_is_rejected(tmp_path):
    source = SOURCE.replace("real(8),intent(inout)::b", "real(8),intent(in)::b")
    _, analysis, routine, node = fixture(tmp_path, source)
    proof = prove_tile_private(analysis, routine, node)
    with pytest.raises(CompilationError, match=r"INTENT\(IN\)"):
        lower_tile_private(proof)


def test_original_constant_vector_and_short_reduction_use_the_ordinary_typed_frontend(tmp_path):
    source = SOURCE.replace("integer::bi,i,li", "integer::bi,i,li\ninteger,parameter::mask(3)=[-1,9,-1]")
    source = source.replace("3.0_8*a(i)", "sum(mask*a(i-1:i+1))")
    path, analysis, routine, node = fixture(tmp_path, source)
    before = path.read_bytes()
    proof = prove_tile_private(analysis, routine, node)
    lowered = lower_tile_private(proof)
    assert lowered.immutable_constants == ("stencil::advance::mask",)
    assert lowered.requires_numerical_environment is True
    assert "mask" not in {symbol.name for symbol in lowered.region.parameters}
    producer = list(assignments(lowered.region.producer))
    reads = [x for statement in producer for x in walk_expr(statement.value)
             if isinstance(x, ArrayAccess) and x.symbol.name == "a"]
    assert len(reads) == 3
    assert path.read_bytes() == before


@pytest.mark.parametrize("region_id", [-1, True, "0"])
def test_tile_identity_requires_a_nonnegative_integer(tmp_path, region_id):
    _, analysis, routine, node = fixture(tmp_path)
    proof = prove_tile_private(analysis, routine, node)
    with pytest.raises(CompilationError, match="region ID"):
        lower_tile_private(proof, region_id=region_id)
