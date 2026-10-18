"""Bounded discovery distributes attempts across original source positions."""

import pytest

from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.ir import CompilationError
from compiler.scopes.closed_selection import interval_candidates
from compiler.scopes.window_discovery import MAX_ADJACENT_UNITS, MAX_ATTEMPTS
from compiler.tests.test_original_interval_owner import SOURCE, builder


def test_nested_early_branches_do_not_hide_later_original_call_pair(tmp_path):
    early = ""
    for index in range(5):
        early += f"if(n>{index}) then\n"
        early += "call producer(old,viscosity,n)\n" * 3
        early += "else\ncall producer(old,viscosity,n)\nendif\n"
    source = SOURCE.replace("call producer(old,viscosity,n)", early + "call producer(old,viscosity,n)", 1)
    selected = builder(tmp_path / "discovery.f90", source=source)
    originals = tuple(selected.entry.execution.content)
    final_pair = originals[-2:]
    choices = interval_candidates(selected)
    assert len(choices) == MAX_ATTEMPTS
    assert any(len(site.originals) == 2 and all(a is b for a, b in zip(site.originals, final_pair, strict=True))
               for site in choices)
    assert all(len(site.originals) <= MAX_ADJACENT_UNITS for site in choices)
    assert not selected.outputs
    assert not selected.edits
    assert not selected.inline.generated


def test_maximal_positions_precede_shorter_overlapping_cuts(tmp_path):
    calls = "\n".join(f"call producer(old,viscosity,{index})" for index in range(7))
    source = SOURCE.replace("call producer(old,viscosity,n)", calls, 1)
    selected = builder(tmp_path / "discovery.f90", source=source)
    nodes = tuple(selected.entry.execution.content)
    choices = interval_candidates(selected)
    assert [site.originals[0] for site in choices[:len(nodes)]] == list(nodes)
    assert [len(site.originals) for site in choices[:len(nodes)]] == [4, 4, 4, 4, 4, 3, 2, 1]
    assert len(choices) == MAX_ATTEMPTS


def test_counted_body_precedes_any_selection_crossing_its_backedge(tmp_path):
    selected = builder(tmp_path / "repeated.f90", repeated=True)
    choices = interval_candidates(selected)
    assert choices[0].loops
    assert [type(node).__name__ for node in choices[0].originals] == ["Call_Stmt", "Call_Stmt"]
    assert choices[0].controls == (choices[0].loops[0].content[0],)


def test_complete_original_team_remains_one_indivisible_unit(tmp_path):
    team = """!$omp parallel do private(i) schedule(runtime)
do i=0,n-1
viscosity(i)=2.d0*old(i)
enddo
!$omp end parallel do
"""
    source = SOURCE.replace("call producer(old,viscosity,n)", "integer::i\n" + team, 1)
    selected = builder(tmp_path / "team.f90", source=source)
    group, = [item.original_nodes for item in grouped_original_sequence(selected.entry.execution.content, {})
              if isinstance(item, NativeGroupCandidate)]
    choices = interval_candidates(selected)
    for site in choices:
        overlap = [node for node in group if any(node is candidate for candidate in site.originals)]
        assert not overlap or overlap == list(group)
    assert any(all(any(node is candidate for candidate in site.originals) for node in group) for site in choices)


def test_full_position_inventory_keeps_the_existing_operation_budget(tmp_path):
    selected = builder(tmp_path / "budget.f90")
    selected.analysis.operation_limit = 1
    with pytest.raises(CompilationError, match="source discovery budget exhausted"):
        interval_candidates(selected)
