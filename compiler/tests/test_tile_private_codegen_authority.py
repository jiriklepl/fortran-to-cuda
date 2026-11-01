"""A real source proof cannot authenticate substitute kernel IR or addressing."""

from dataclasses import replace

import pytest

from compiler.emission.cuda.tile_private import generate_tile_function
from compiler.frontend.lowering import lower_tile_private
from compiler.frontend.tile_private import prove_tile_private
from compiler.tests.test_tile_private import fixture


@pytest.mark.parametrize("substitute", ["consumer", "private", "parameters", "scratch", "addressing"])
def test_default_tile_codegen_requires_the_exact_original_lowering(tmp_path, substitute):
    path, analysis, routine, node = fixture(tmp_path)
    before = path.read_bytes()
    lowered = lower_tile_private(prove_tile_private(analysis, routine, node))
    region, addressing = lowered.region, lowered.addressing
    # Authentic implicit and explicit addressing produce the identical kernel.
    assert generate_tile_function(region) == generate_tile_function(region, addressing=addressing)
    if substitute == "consumer":
        region = replace(region, consumer=region.producer)
    elif substitute == "private":
        region = replace(region, private_symbols=())
    elif substitute == "parameters":
        region = replace(region, parameters=tuple(reversed(region.parameters)))
    elif substitute == "scratch":
        region = replace(region, scratch_symbols=(replace(region.scratch_symbols[0], id=999),))
    else:
        addressing = replace(addressing, decisions=())
    with pytest.raises(ValueError, match="substitute source-lowered work|addressing differs"):
        generate_tile_function(region, addressing=addressing)
    assert path.read_bytes() == before
