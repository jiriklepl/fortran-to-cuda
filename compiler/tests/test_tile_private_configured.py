"""Configured parser offsets never index raw source or borrow inactive PRIVATE."""

from copy import copy
from hashlib import sha256

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects, _children
from compiler.frontend.tile_private import prove_tile_private, prove_tile_private_team
from compiler.ir import CompilationError
from compiler.scopes.segments import directive
from compiler.tests.test_tile_private import SOURCE


def configured(tmp_path, *, unmapped=None, source=SOURCE):
    source = source.replace("!$omp parallel private(", "#if 0\n!$omp parallel private(wrong)\n#endif\n!$omp parallel private(")
    original, prepared = tmp_path / "original.f90", tmp_path / "prepared.f90"
    original.write_text(source)
    selected, mapping = ["! configured padding"] * 100, [None] * 100
    active = True
    for number, line in enumerate(source.splitlines(), 1):
        if line == "#if 0":
            active = False
        elif line == "#endif":
            active = True
        elif active:
            selected.append(line)
            mapping.append(None if unmapped is not None and unmapped(line) else number)
    prepared.write_text("\n".join(selected) + "\n")
    document = {"schema_version": 1, "source_inputs": {str(original): sha256(original.read_bytes()).hexdigest()},
                "preserves_source_order": True, "configuration": {"defines": []}, "dependencies": {},
                "entries": [{"source": str(original), "path": str(prepared), "sha256": sha256(prepared.read_bytes()).hexdigest(),
                             "line_map": mapping}]}
    analysis = SourceEffects([original], analysis_sources=document)
    routine = analysis.routines["stencil::advance"]
    node = next(part for part in walk(routine.execution) if type(part).__name__ == "Block_Nonlabel_Do_Construct")
    children = tuple(_children(routine.execution))
    first = next(i for i, child in enumerate(children) if (directive(child) or "").startswith("parallel"))
    last = next(i for i, child in enumerate(children) if directive(child) == "end parallel")
    return original, analysis, routine, node, children[first:last + 1]


def test_configured_private_scan_uses_prepared_offsets_and_publishes_original_span(tmp_path):
    original, analysis, routine, node, team_nodes = configured(tmp_path)
    before = original.read_bytes()
    proof = prove_tile_private(analysis, routine, node)
    lines = original.read_text().splitlines()
    start, stop = proof.public()["owner_span"]
    assert lines[start - 1] == "!$omp parallel private(bi,i,li,tmp)"
    assert lines[stop - 1] == "!$omp end parallel"
    assert [item["name"] for item in proof.public()["scratch"]] == ["tmp"]
    team = prove_tile_private_team(analysis, routine, team_nodes)
    assert team.public()["span"] == [start, stop]
    assert len(team.units) == 1
    assert original.read_bytes() == before


@pytest.mark.parametrize("unmapped", [
    lambda line: line.startswith("!$omp parallel"),
    lambda line: line.startswith("!$omp end parallel"),
    lambda line: line.startswith("do bi="),
])
def test_unmapped_owner_or_original_loop_never_borrows_private_authority(tmp_path, unmapped):
    _path, analysis, routine, node, _team = configured(tmp_path, unmapped=unmapped)
    with pytest.raises(CompilationError):
        prove_tile_private(analysis, routine, node)


def test_copied_routine_or_changed_prepared_span_fails_closed(tmp_path):
    _path, analysis, routine, node, _team = configured(tmp_path)
    with pytest.raises(CompilationError):
        prove_tile_private(analysis, copy(routine), node)
    opening = node.content[0].item
    opening.span = (1000000, 1000000)
    with pytest.raises(CompilationError):
        prove_tile_private(analysis, routine, node)


def test_configured_private_scalar_observer_outside_tile_fails_closed(tmp_path):
    source = SOURCE.replace("!$omp end parallel", "b(1)=real(li,8)\n!$omp end parallel")
    _path, analysis, routine, node, _team = configured(tmp_path, source=source)
    with pytest.raises(CompilationError, match="scalar has an observer"):
        prove_tile_private(analysis, routine, node)
