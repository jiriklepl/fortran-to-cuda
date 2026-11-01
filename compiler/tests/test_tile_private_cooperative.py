"""A separate source-issued scalar-independence gate for cooperative tiles."""

from copy import copy
from dataclasses import replace

import pytest

from compiler.emission.cuda.tile_private import generate_cooperative_kernel, generate_kernel
from compiler.frontend.tile_private import prove_cooperative_tile_private, prove_tile_private
from compiler.ir import CompilationError
from compiler.tests.test_tile_private import SOURCE, SOURCE_3D, fixture


@pytest.mark.parametrize("source", [SOURCE, SOURCE_3D])
def test_cooperative_mapping_requires_and_retains_its_distinct_original_source_proof(tmp_path, source):
    _, analysis, routine, node = fixture(tmp_path, source)
    original = prove_tile_private(analysis, routine, node)
    proof = prove_cooperative_tile_private(original)
    assert proof.validate() is proof
    assert prove_cooperative_tile_private(original) is proof
    record = proof.public()
    assert record["source_proof_identity"] == original.identity
    assert record["definitions_authorized"] is False
    assert record["placement_authorized"] is False
    assert record["scalar_liveout_authorized"] is False
    if source == SOURCE_3D:
        assert record["version"] == 2
        assert record["point_distribution"]["advance"].startswith("mixed-radix carry")
    else:
        assert record["version"] == 1
        assert record["mapping"] == "one CTA per tile; thread-strided innermost points"
    code = "\n".join(generate_cooperative_kernel(proof.lowered.region, proof))
    assert code.count("__syncthreads();") == 2
    assert "fort_tile_slot = blockIdx.x" in code
    if source == SOURCE_3D:
        assert "fort_point_carry=threadIdx.x" in code
        assert "fort_point_carry=blockDim.x" in code
        assert code.count("while (fort_point_carry==0)") == 2
    else:
        assert "= threadIdx.x" in code
        assert "+=blockDim.x" in code
    # The original default remains one thread/tile, with no cooperative barrier.
    assert "__syncthreads" not in "\n".join(generate_kernel(proof.lowered.region))
    record["placement_authorized"] = True
    assert proof.public()["placement_authorized"] is False
    for forged in (copy(proof), replace(proof)):
        with pytest.raises(CompilationError, match="exact compiler-issued"):
            forged.validate()


@pytest.mark.parametrize("consumer", [
    "  b(i)=b(i)+carry+(tmp(li+1)-tmp(li-1))",
    "  carry=carry+(tmp(li+1)-tmp(li-1))\n  b(i)=b(i)+carry",
])
def test_cross_phase_or_cross_point_private_scalar_state_is_not_parallelized(tmp_path, consumer):
    source = SOURCE.replace("integer::bi,i,li", "integer::bi,i,li\nreal(8)::carry")
    source = source.replace("private(bi,i,li,tmp)", "private(bi,i,li,tmp,carry)")
    source = source.replace("  tmp(li)=3.0_8*a(i)", "  carry=3.0_8*a(i)\n  tmp(li)=carry")
    source = source.replace("  b(i)=b(i)+(tmp(li+1)-tmp(li-1))", consumer)
    _, analysis, routine, node = fixture(tmp_path, source)
    original = prove_tile_private(analysis, routine, node)
    with pytest.raises(CompilationError, match="private scalar"):
        prove_cooperative_tile_private(original)


def test_original_private_terminal_value_observer_outside_tiles_is_not_erased(tmp_path):
    source = SOURCE.replace("!$omp end parallel", "if(i>0)call observe(i)\n!$omp end parallel")
    _, analysis, routine, node = fixture(tmp_path, source)
    with pytest.raises(CompilationError, match="private scalar has an observer"):
        prove_tile_private(analysis, routine, node)


def test_substitute_ir_or_changed_source_cannot_borrow_cooperative_permission(tmp_path):
    path, analysis, routine, node = fixture(tmp_path)
    proof = prove_cooperative_tile_private(prove_tile_private(analysis, routine, node))
    forged = replace(proof.lowered.region, consumer=proof.lowered.region.producer)
    with pytest.raises(CompilationError, match="substitute lowered work"):
        generate_cooperative_kernel(forged, proof)
    path.write_text(SOURCE + "! changed\n")
    with pytest.raises(CompilationError):
        proof.validate()
