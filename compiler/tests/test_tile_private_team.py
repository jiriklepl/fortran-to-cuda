"""Whole source-team authority, with no dropped directive or native action."""

from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.source_effects import SourceEffects, _children
from compiler.frontend.tile_private import prove_tile_private_team
from compiler.ir import CompilationError
from compiler.scopes.segments import directive
from compiler.tests.test_tile_private import SOURCE, fixture


def team_fixture(tmp_path, source=SOURCE):
    path, analysis, routine, _ = fixture(tmp_path, source)
    nodes = tuple(_children(routine.execution))
    first = next(index for index, node in enumerate(nodes) if (directive(node) or "").startswith("parallel"))
    last = next(index for index, node in enumerate(nodes) if directive(node) == "end parallel")
    return path, analysis, routine, nodes[first:last + 1]


def test_complete_team_issues_original_fields_scalar_and_descriptor_inventory(tmp_path):
    path, analysis, routine, nodes = team_fixture(tmp_path)
    before = path.read_bytes()
    proof = prove_tile_private_team(analysis, routine, nodes)
    assert proof.validate(analysis, routine, nodes) is proof
    assert prove_tile_private_team(analysis, routine, nodes) is proof
    assert [(binding.name, written) for binding, written in proof.fields] == [("a", False), ("b", True)]
    assert {binding.name for binding in proof.scalars} == {"n", "t"}
    assert [binding.name for binding in proof.scratch_bindings] == ["tmp"]
    assert proof.public()["domain"] == [{"lower": "1", "upper": "n", "width": "t"}]
    assert proof.public()["definitions_authorized"] is False
    assert proof.public()["placement_authorized"] is False
    assert path.read_bytes() == before
    for forged in (copy(proof), replace(proof)):
        with pytest.raises(CompilationError, match="exact compiler-issued whole"):
            forged.validate()
    with pytest.raises(CompilationError):
        proof.validate(nodes=nodes[1:])


@pytest.mark.parametrize("replacement", [
    ("!$omp end do", "!$omp end do nowait"),
    ("!$omp end parallel", "call observer(a)\n!$omp end parallel"),
    ("!$omp end parallel", "b(1)=1\n!$omp end parallel"),
    ("parallel private(", "parallel firstprivate(n) private("),
    ("!$omp do schedule(runtime)", "!$omp do ordered schedule(runtime)"),
])
def test_whole_team_never_erases_extra_native_work_or_unqualified_directives(tmp_path, replacement):
    _, analysis, routine, nodes = team_fixture(tmp_path, SOURCE.replace(*replacement))
    with pytest.raises(CompilationError, match="tile-private scratch"):
        prove_tile_private_team(analysis, routine, nodes)


def test_partial_or_copied_team_cannot_authorize_original_source_replacement(tmp_path):
    path, analysis, routine, nodes = team_fixture(tmp_path)
    other = SourceEffects([path])
    other_nodes = tuple(_children(other.routines[routine.qualified].execution))
    copied = tuple(node for node in other_nodes if directive(node) is not None or type(node).__name__ == "Block_Nonlabel_Do_Construct")
    for selection in (nodes[1:], nodes[:-1], copied):
        with pytest.raises(CompilationError):
            prove_tile_private_team(analysis, routine, selection)
