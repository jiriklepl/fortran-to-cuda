"""Original-team emitters inherit placement without inheriting new legality.

These emitter tests stage source-proved units and install an internal placement
marker after scanning. They do not publish a runnable owner, admit a measured
window, or manufacture calibration evidence. Public admission is tested by the
exact-window suite and remains restricted to its existing serial source class.
"""

from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.combined_team import CombinedNumericalTeam
from compiler.scopes.joined_team import JoinedTeam
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.ordinary_consumption import _OrdinaryPairConsumption
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_joined_team_source import CONSUMER, CORRECTION, OPENING, PRODUCER, SOURCE, TEAM
from compiler.tests.test_source_scopes import FACT


def emit_child_team(directory, *, mixed, complete_window):
    """Retain registered proofs while exercising only the emission boundary."""
    team = OPENING + PRODUCER + (CORRECTION if mixed else "") + CONSUMER + "!$omp end parallel\n"
    source = SOURCE.replace(
        "module joined_owner\n",
        "module joined_owner\nuse,intrinsic::ieee_exceptions,only:ieee_set_halting_mode,ieee_invalid\n",
    ).replace(TEAM, "call advance()\n")
    source = source.replace(
        "visits=visits+11\nend subroutine",
        "visits=visits+11\ncontains\nsubroutine advance\ninteger::i\n"
        "!$omp parallel\ncall ieee_set_halting_mode(ieee_invalid,.false.)\n!$omp end parallel\n"
        + team
        + "end subroutine\nend subroutine",
    )
    path = directory / "owner.f90"
    path.write_text(source)
    facts = {
        "schema_version": 1,
        "participation": "serial",
        "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
        "captures": {"argument::" + name: FACT for name in ("a", "b", "out")},
    }
    builder = ScopeBuilder(
        [path],
        "joined_owner::step",
        facts=facts,
        options=CompilerOptions(),
        config=OffloadConfig(policy="auto", scope_execution="reached", host_threads=4),
    )
    nodes = builder.inline.prepare(builder.entry.execution.content)
    builder.generated.update(builder.inline.generated)
    builder.numerical_ir.update(builder.inline.ir)
    builder.numerical_reasons.update({name: None for name in builder.inline.regions})
    with _OrdinaryPairConsumption(builder):
        owner = LexicalOwner(builder)
        owner.scan(nodes)
        owner.leaves = {leaf for member in owner.members.values() for unit in member.units for leaf in unit.leaves}
        assert all(not member.boundaries for member in owner.members.values())
        # The admitted whole-window selector is deliberately outside this
        # emitter test. No generated source is published or executed here.
        owner.exact_window = {} if complete_window else None
        child = owner.members["joined_owner::step::advance"]
        assert child.exact_window is None
        assert child.owner is owner
        assert child.price_units is (not complete_window)
        (unit,) = (unit for unit in child.units if isinstance(unit.scope, (JoinedTeam, CombinedNumericalTeam)))
        assert isinstance(unit.scope, JoinedTeam if mixed else CombinedNumericalTeam)
        patches = child.emit_unit(unit, child.units.index(unit))
        lines = source.splitlines(keepends=True)
        for first, last, text in sorted(patches, reverse=True):
            lines[first - 1 : last] = [text]
        return "".join(lines)


@pytest.mark.parametrize("mixed", [False, True], ids=["combined", "joined"])
@pytest.mark.parametrize("complete_window", [False, True], ids=["per-unit", "whole-window"])
def test_original_child_team_obeys_owning_placement_and_keeps_execution_proofs(tmp_path, mixed, complete_window):
    text = emit_child_team(tmp_path, mixed=mixed, complete_window=complete_window)
    assert ("fort_team_choose" in text) is (not complete_window)
    assert f"fort_mode = {1 if complete_window else 2}" in text
    assert "fort_scope_plan_validate(fort_context)" in text
    assert "fort_scope_numerical_environment_supported() == 0" in text
    assert "fort_team_omp_level() /= 1" in text
    assert "fort_team_omp_threads() /= 4" in text
    assert "!$omp atomic write" in text
    assert "fort_scope_wait(fort_context)" in text
    assert "fort_scope_close(" in text
    assert text.count("b(i)=2*a(i)+real(i,8)") == 1
    assert text.count("out(i)=a(i)+b(i)") == 1
    if mixed:
        assert text.count("b(i)=b(i)+sum(a)") == 1
        assert "fort_scope_host_begin(" in text
        assert "fort_scope_host_end(" in text
        assert "error stop 'original team numerical execution failed'" in text
        if complete_window:
            assert "fort_team_run = .true." in text
    else:
        assert "error stop 'combined original team numerical execution failed'" in text
        assert "error stop 'combined original team completion failed'" in text
        assert text.index("fort_scope_plan_validate(") < text.index("fort_team_run = .true.")
