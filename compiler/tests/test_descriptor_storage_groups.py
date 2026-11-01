"""Deferred native source groups use existing whole-team storage authorities."""
import sys
from copy import copy

import pytest

from compiler.frontend.descriptor_storage_groups import original_storage_group
from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_mutation_effects import _Scan, prove_storage_mutation_effects
from compiler.ir import CompilationError


def source(*, tail="", opening="parallel private(i)", declarations="", count=76):
    return ("module storage_group\ncontains\nsubroutine run(a,n)\n"
            "real(8),intent(inout)::a(:)\ninteger,intent(in)::n\ninteger::i\n"+declarations+"\n!$omp "+opening+"\n"
            + "".join("!$omp do\ndo i=1,n\na(i)=a(i)+1.0_8\nenddo\n!$omp end do nowait\n" for _ in range(count))
            + tail+"\n!$omp end parallel\nend subroutine\nend module\n")


def fixture(tmp_path, text=None, *, seal=True):
    path = tmp_path / "source.f90"
    path.write_text(source() if text is None else text)
    analysis = SourceEffects([path])
    if seal:
        analysis.seal_session()
    routine = analysis.routines["storage_group::run"]
    graph = analysis.structure(routine.qualified)
    identity, = graph.native_groups
    return path, analysis, routine, graph, identity


def test_all_76_original_units_are_scanned_without_numerical_or_payload_authority(tmp_path):
    _path, analysis, routine, graph, identity = fixture(tmp_path)
    assert graph.raw_structure_rejection == ("bounded structured source operation budget exhausted",)
    assert len(graph.native_groups[identity].units) == 76
    proof = original_storage_group(analysis, routine, graph, identity)
    public = proof.public()
    assert public["storage_mutation"]["complete"] is True
    assert public["storage_mutation"]["version"] == 4
    assert analysis._summary_authority()["descriptor_window_storage_group_version"] == 13
    assert public["storage_mutation"]["original_selection"]["selected_node_ids"] == [identity]
    assert public["team_storage"]["original_directives"].count("end do nowait") == 76
    for key in ("entry_definition_authority", "payload_access_authorized", "alias_authority",
                "execution_authorized", "numerical_completion_authorized"):
        assert public[key] is False
    assert proof.validate() is proof


@pytest.mark.parametrize("case", ["allocation", "unknown", "alias", "pointer"])
def test_unknown_or_mutating_tail_and_storage_association_reject_whole_group(tmp_path, case):
    changes = {"allocation": dict(tail="deallocate(hidden)", declarations="real(8),allocatable,save::hidden(:)"),
               "unknown": dict(tail="call export_unknown(a)"),
               "alias": dict(declarations="real(8)::x,y\nequivalence(x,y)"),
               "pointer": dict(declarations="real(8),pointer::p(:)")}
    _path, analysis, routine, graph, identity = fixture(tmp_path, source(**changes[case]))
    with pytest.raises(CompilationError):
        original_storage_group(analysis, routine, graph, identity)


@pytest.mark.parametrize("case", ["foreign_graph", "wrong_analysis", "truncated_units", "storage_token", "team_token"])
def test_consumed_group_analysis_unit_inventory_and_registered_tokens_are_exact(tmp_path, case):
    _path, analysis, routine, graph, identity = fixture(tmp_path, seal=False)
    proof = original_storage_group(analysis, routine, graph, identity)
    if case == "foreign_graph":
        with pytest.raises(CompilationError):
            original_storage_group(analysis, routine, copy(graph), identity)
        return
    if case == "wrong_analysis":
        object.__setattr__(proof, "_analysis", analysis.private_projection())
        with pytest.raises(CompilationError):
            proof.validate()
        return
    if case == "truncated_units":
        group = graph.native_groups[identity]
        object.__setattr__(group, "units", group.units[:-1])
    elif case == "storage_token":
        analysis._storage_mutation_effects[proof._storage.identity] = copy(proof._storage)
    else:
        analysis._descriptor_team_proofs[proof._team.identity] = copy(proof._team)
    with pytest.raises(CompilationError):
        proof.validate()


def test_no_joined_token_retains_original_concurrency_frontier_and_selected_schema3(tmp_path):
    _path, analysis, routine, graph, identity = fixture(tmp_path)
    plain = analysis.storage_mutation_effects(routine.qualified, selection=(identity,)).public()
    assert plain["version"] == 3
    assert plain["complete"] is False
    assert "joined_storage_authority" not in plain
    assert {row["kind"] for row in plain["frontiers"]} == {"concurrency requires separate joined storage authority"}
    team = prove_descriptor_team(analysis, routine.qualified, graph.source_nodes(identity))
    qualified = analysis.storage_mutation_effects(routine.qualified, selection=(identity,), joined_storage=team)
    assert qualified.public()["complete"] is True
    assert qualified._joined_storage is team


@pytest.mark.parametrize("case", ["copy", "public", "foreign_analysis", "wrong_selection", "no_selection", "cached_replacement", "retag"])
def test_joined_scanner_context_requires_exact_registered_token_and_whole_selection(tmp_path, case):
    _path, analysis, routine, graph, identity = fixture(tmp_path, seal=False)
    team = prove_descriptor_team(analysis, routine.qualified, graph.source_nodes(identity))
    target, selected, supplied = analysis, (identity,), team
    if case == "copy":
        supplied = copy(team)
    elif case == "public":
        supplied = team.public()
    elif case == "foreign_analysis":
        target = analysis.private_projection()
    elif case == "wrong_selection":
        selected = (graph.root_id,)
    elif case == "no_selection":
        selected = None
    elif case == "cached_replacement":
        prove_storage_mutation_effects(analysis, routine.qualified, selection=(identity,), joined_storage=team)
        analysis._descriptor_team_proofs[team.identity] = copy(team)
    else:
        object.__setattr__(team, "identity", "b"*64)
        analysis._descriptor_team_proofs[team.identity] = team
    with pytest.raises(CompilationError):
        prove_storage_mutation_effects(target, routine.qualified, selection=selected, joined_storage=supplied)


@pytest.mark.parametrize("change", ["token", "unit_inventory"])
def test_fresh_scanner_publication_callbacks_cannot_replace_consumed_authority(tmp_path, monkeypatch, change):
    _path, analysis, routine, graph, identity = fixture(tmp_path, seal=False)
    team = prove_descriptor_team(analysis, routine.qualified, graph.source_nodes(identity))
    original = _Scan.finish

    def finish(scanner):
        original(scanner)
        if change == "token":
            analysis._descriptor_team_proofs[team.identity] = copy(team)
        else:
            group = graph.native_groups[identity]
            object.__setattr__(group, "units", group.units[:-1])

    monkeypatch.setattr(_Scan, "finish", finish)
    with pytest.raises(CompilationError):
        original_storage_group(analysis, routine, graph, identity)


def test_joined_root_token_does_not_cover_concurrency_in_an_original_callee(tmp_path):
    text = source(tail="call observer(a,n)").replace("end module", """subroutine observer(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
!$omp parallel do private(i)
do i=1,n
a(i)=a(i)+1
enddo
!$omp end parallel do
end subroutine
end module""")
    _path, analysis, routine, graph, identity = fixture(tmp_path, text)
    team = prove_descriptor_team(analysis, routine.qualified, graph.source_nodes(identity))
    storage = prove_storage_mutation_effects(analysis, routine.qualified, selection=(identity,), joined_storage=team)
    assert storage.public()["complete"] is False
    assert any(row["kind"] == "concurrency requires separate joined storage authority"
               for row in storage.public()["frontiers"])
    with pytest.raises(CompilationError):
        original_storage_group(analysis, routine, graph, identity)


def test_window_uses_conditional13_and_preserves_private_issuer_registries(tmp_path):
    _path, analysis, routine, graph, identity = fixture(tmp_path)
    proof = prove_descriptor_window(analysis, routine.qualified, (identity,), ("argument::a",))
    public = proof.public()
    assert public["schema_version"] == 13
    requirement, = public["native_storage_group_requirements"]
    group = requirement["storage_group"]
    assert group["storage_mutation"]["version"] == 4
    assert group["storage_mutation"]["complete"] is True
    assert group["execution_authorized"] is False
    assert group["numerical_completion_authorized"] is False
    assert not getattr(analysis, "_descriptor_team_proofs", {})
    assert not getattr(analysis, "_storage_mutation_effects", {})
    assert proof.validate(analysis, routine.qualified, (identity,)) is proof


@pytest.mark.parametrize("change", ["storage_copy", "storage_foreign", "team_copy", "team_retag"])
def test_final_composition_publication_cannot_detach_same_public_issuer(tmp_path, monkeypatch, change):
    _path, analysis, routine, graph, identity = fixture(tmp_path, seal=False)
    proof = original_storage_group(analysis, routine, graph, identity)
    original = analysis.verify_publication

    def publication():
        original()
        caller = sys._getframe(1)
        if caller.f_code.co_name != "validate" or not caller.f_code.co_filename.endswith("descriptor_storage_groups.py"):
            return
        if change.startswith("storage"):
            detached = copy(proof._storage)
            if change == "storage_foreign":
                foreign = analysis.private_projection()
                foreign._storage_mutation_effects = {detached.identity: detached}
                object.__setattr__(detached, "_analysis", foreign)
            object.__setattr__(proof, "_storage", detached)
        elif change == "team_copy":
            object.__setattr__(proof, "_team", copy(proof._team))
        else:
            object.__setattr__(proof._team, "identity", "b"*64)
            analysis._descriptor_team_proofs[proof._team.identity] = proof._team

    monkeypatch.setattr(analysis, "verify_publication", publication)
    with pytest.raises(CompilationError, match="final publication"):
        proof.validate()
