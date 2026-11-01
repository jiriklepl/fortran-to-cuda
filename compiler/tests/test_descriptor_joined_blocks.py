"""Scalar reduction/SINGLE/BLOCK storage is independent of numerical units."""
from copy import copy

import pytest

from compiler.frontend.descriptor_storage_groups import original_storage_group
from compiler.frontend.descriptor_teams import prove_descriptor_team
from compiler.frontend.descriptor_windows import prove_descriptor_window
from compiler.frontend.storage_mutation_effects import prove_storage_mutation_effects
from compiler.ir import CompilationError
from compiler.tests.test_descriptor_storage_groups import fixture


def source(*, reduction="+:total", single="single", ending="end single", declarations="real(8)::total", body="", tail=""):
    loop = """!$omp do reduction(""" + reduction + """)
do i=1,n
total=total+a(i)
enddo
!$omp end do
"""
    if "logical" in declarations.lower():
        loop = loop.replace("total=total+a(i)", "total=total.or.(a(i)>0)")
    text = """module storage_group
contains
subroutine run(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
""" + declarations + """
total=0
!$omp parallel private(i)
""" + loop * 76 + "!$omp " + single + """
block
use,intrinsic::iso_fortran_env,only:real64
real(real64)::value
value=total
a(1)=value
""" + body + """
end block
!$omp """ + ending + "\n" + tail + """
!$omp end parallel
end subroutine
end module
"""
    if "logical" in declarations.lower():
        text = text.replace("total=0", "total=.false.").replace("value=total", "value=merge(1.0_8,0.0_8,total)")
    return text


@pytest.mark.parametrize("sealed", [False, True])
@pytest.mark.parametrize("ending", ["end single", "end single nowait"])
def test_whole_source_joined_team_scans_real_block_without_numerical_inventory_or_permissions(tmp_path, sealed, ending):
    _path, analysis, routine, graph, identity = fixture(tmp_path, source(ending=ending), seal=sealed)
    assert graph.native_groups[identity].available is False
    proof = original_storage_group(analysis, routine, graph, identity)
    record = proof.public()
    team = record["team_storage"]
    assert team["version"] == 2
    assert team["retained_single_regions"] == 1
    assert team["parent_parallel_join_required"] is True
    assert team["intrinsic_scalar_reductions"][0]["operator"] == "+"
    storage = record["storage_mutation"]
    assert storage["version"] == 7
    assert storage["complete"] is True, storage["frontiers"]
    assert len(storage["fixed_scalar_blocks"]) == 1
    assert storage["fixed_scalar_blocks"][0]["requires_complete_original_body"] is True
    for flags in (record, storage):
        for name in ("entry_definition_authority", "payload_access_authorized", "execution_authorized", "alias_authority"):
            assert flags[name] is False
    window = prove_descriptor_window(analysis, routine.qualified, (identity,), ("argument::a",)).public()
    assert window["schema_version"] == 16
    assert analysis._summary_authority()["descriptor_team_version"] == 2
    assert window["native_storage_group_requirements"][0]["storage_group"]["storage_mutation"]["version"] == 7
    assert analysis.summarize(routine.qualified)["complete"] is False
    with pytest.raises(CompilationError):
        analysis.joined_completion(routine.qualified, graph.native_groups[identity].original_nodes)
    with pytest.raises(CompilationError):
        analysis.numerical_joined_completion(routine.qualified, graph.native_groups[identity].original_nodes)


@pytest.mark.parametrize(("identifier", "declaration"), [
    ("+:total", "real(4)::total"), ("*:total", "real(8)::total"), ("-:total", "integer(8)::total"),
    ("max:total", "real(8)::total"), ("min:total", "integer(4)::total"),
    ("iand:total", "integer(8)::total"), ("ior:total", "integer(4)::total"), ("ieor:total", "integer(4)::total"),
    (".and.:total", "logical(4)::total"), (".or.:total", "logical(8)::total"),
    (".eqv.:total", "logical(4)::total"), (".neqv.:total", "logical(4)::total")])
def test_supported_intrinsic_reductions_classify_scalar_storage_without_numerical_rewrite(tmp_path, identifier, declaration):
    _path, analysis, routine, graph, identity = fixture(tmp_path, source(reduction=identifier, declarations=declaration))
    group = graph.native_groups[identity]
    team = prove_descriptor_team(analysis, routine.qualified, group.original_nodes).public()
    assert team["version"] == 2
    assert team["payload_access_authorized"] is False
    assert team["execution_authorized"] is False


@pytest.mark.parametrize(("identifier", "declaration"), [
    ("custom:total", "real(8)::total"), ("+:total(1)", "real(8)::total(2)"),
    ("+:total", "real(8),pointer::total"), ("+:total", "real(8),target::total"),
    ("+:total", "real(8),save::total"), ("+:total", "real(8),allocatable::total"),
    ("+:total", "real(16)::total"), ("+:total", "logical::total"), ("iand:total", "real(8)::total"),
    ("+:total,total", "real(8)::total"), ("+:total", "real(8),parameter::total=0"),
    ("+:total", "type(unknown_type)::total")])
def test_nonintrinsic_array_unsafe_unknown_and_duplicate_reduction_items_reject_whole_team(tmp_path, identifier, declaration):
    _path, analysis, routine, graph, identity = fixture(tmp_path, source(reduction=identifier, declarations=declaration))
    with pytest.raises(CompilationError):
        original_storage_group(analysis, routine, graph, identity)


@pytest.mark.parametrize("change", ["missing_join", "single_clause", "copyprivate", "single_nested", "task", "unsafe_tail"])
def test_incomplete_or_unsupported_synchronization_and_late_unknown_effects_are_not_hidden(tmp_path, change):
    text = source()
    if change == "missing_join":
        text = text.replace("!$omp end parallel", "")
    elif change == "single_clause":
        text = text.replace("!$omp single", "!$omp single private(total)")
    elif change == "copyprivate":
        text = text.replace("!$omp end single", "!$omp end single copyprivate(total)")
    elif change == "single_nested":
        text = text.replace("!$omp single", "!$omp single\n!$omp single").replace("!$omp end single", "!$omp end single\n!$omp end single")
    elif change == "task":
        text = text.replace("value=total", "!$omp task\nvalue=total\n!$omp end task")
    else:
        text = text.replace("a(1)=value", "a(1)=value\ncall unknown_export(a)")
    if change == "missing_join":
        # Even the source grouping has no whole-team interval to offer.
        from compiler.frontend.source_effects import SourceEffects

        path = tmp_path / "incomplete.f90"
        path.write_text(text)
        analysis = SourceEffects([path])
        routine = analysis.routines["storage_group::run"]
        with pytest.raises(CompilationError):
            prove_descriptor_team(analysis, routine.qualified, tuple(routine.execution.content))
        return
    _path, analysis, routine, graph, identity = fixture(tmp_path, text)
    with pytest.raises(CompilationError):
        original_storage_group(analysis, routine, graph, identity)


@pytest.mark.parametrize("tail", ["deallocate(hidden)", "call unknown_export(a)"])
def test_unavailable_numerical_group_does_not_hide_later_allocation_or_escape(tmp_path, tail):
    text = source(tail=tail, declarations="real(8)::total\nreal(8),allocatable,save::hidden(:)")
    _path, analysis, routine, graph, identity = fixture(tmp_path, text)
    assert graph.native_groups[identity].available is False
    with pytest.raises(CompilationError):
        original_storage_group(analysis, routine, graph, identity)


def test_registered_team_copy_and_identity_retag_do_not_supply_new_joined_authority(tmp_path):
    _path, analysis, routine, graph, identity = fixture(tmp_path, source())
    team = prove_descriptor_team(analysis, routine.qualified, graph.native_groups[identity].original_nodes)
    with pytest.raises(CompilationError):
        copy(team).validate(analysis, routine.qualified, (identity,))
    object.__setattr__(team, "identity", "0" * 64)
    analysis._descriptor_team_proofs[team.identity] = team
    with pytest.raises(CompilationError):
        team.validate(analysis, routine.qualified, (identity,))


def test_new_unavailable_inventory_route_without_block_still_uses_conditional16(tmp_path):
    from compiler.tests.test_descriptor_storage_groups import source as old_source

    _path, analysis, routine, graph, identity = fixture(tmp_path, old_source(tail="a(1)=a(1)+1.0_8"))
    assert graph.native_groups[identity].available is False
    proof = original_storage_group(analysis, routine, graph, identity)
    assert proof.public()["team_storage"]["version"] == 1
    assert proof.public()["storage_mutation"]["version"] == 4
    assert prove_descriptor_window(analysis, routine.qualified, (identity,), ("argument::a",)).public()["schema_version"] == 16


def test_plain_scanner_retains_concurrency_and_block_frontiers_and_known_tail_mutation(tmp_path):
    text = source(body="deallocate(hidden)", declarations="real(8)::total\nreal(8),allocatable,save::hidden(:)")
    _path, analysis, routine, graph, identity = fixture(tmp_path, text)
    record = prove_storage_mutation_effects(analysis, routine.qualified, selection=(identity,)).public()
    assert record["version"] == 3
    assert any(item["kind"] == "BLOCK requires complete original joined storage authority" for item in record["frontiers"])
    assert any(item["kind"] == "deallocate" for item in record["mutations"])


@pytest.mark.parametrize("where", ["local", "host", "use"])
def test_user_reduction_declarations_are_not_primitive_scalar_combiner_evidence(tmp_path, where):
    text = source()
    declaration = "!$omp declare reduction(+:real(8):omp_out=omp_out+omp_in)\n"
    if where == "local":
        text = text.replace("real(8)::total", declaration + "real(8)::total")
    elif where == "host":
        text = text.replace("module storage_group\n", "module storage_group\n" + declaration)
    else:
        text = "module custom_reduction\n" + declaration + "end module\n" + text
        text = text.replace("subroutine run(a,n)\n", "subroutine run(a,n)\nuse custom_reduction\n")
    _path, analysis, routine, graph, identity = fixture(tmp_path, text)
    with pytest.raises(CompilationError, match="user reduction"):
        original_storage_group(analysis, routine, graph, identity)


def test_window_reissues_exact_private_tokens_and_leaves_parent_registry_unchanged(tmp_path):
    _path, analysis, routine, graph, identity = fixture(tmp_path, source())
    proof = original_storage_group(analysis, routine, graph, identity)
    teams, blocks = dict(analysis._descriptor_team_proofs), dict(analysis._descriptor_blocks)
    registry, block_registry = analysis._descriptor_team_proofs, analysis._descriptor_blocks
    window = prove_descriptor_window(analysis, routine.qualified, (identity,), ("argument::a",))
    assert window.public()["schema_version"] == 16
    assert analysis._descriptor_team_proofs is registry
    assert registry == teams
    assert analysis._descriptor_blocks is block_registry
    assert block_registry == blocks
    assert proof.validate() is proof


@pytest.mark.parametrize("mutation", ["binding", "registry", "record"])
def test_last_block_validation_cannot_detach_an_earlier_consumed_declaration(tmp_path, monkeypatch, mutation):
    from compiler.frontend.descriptor_blocks import FixedScalarBlock

    text = source().replace("!$omp end single", "block\nreal(8)::later\nlater=a(1)\nend block\n!$omp end single")
    _path, analysis, routine, graph, identity = fixture(tmp_path, text)
    team = prove_descriptor_team(analysis, routine.qualified, graph.native_groups[identity].original_nodes)
    validate, calls = FixedScalarBlock.validate, {}

    def inspect(self, *args, **kwargs):
        result = validate(self, *args, **kwargs)
        calls[self.identity] = calls.get(self.identity, 0) + 1
        if "later" in self._scope.bindings and calls[self.identity] == 2:
            previous = next(proof for proof in analysis._descriptor_blocks.values() if "value" in proof._scope.bindings)
            if mutation == "binding":
                object.__setattr__(previous._scope.bindings["value"], "kind", 4)
            elif mutation == "registry":
                analysis._descriptor_blocks[previous.identity] = copy(previous)
            else:
                object.__setattr__(previous, "_record", "{}")
        return result

    monkeypatch.setattr(FixedScalarBlock, "validate", inspect)
    with pytest.raises(CompilationError):
        prove_storage_mutation_effects(analysis, routine.qualified, selection=(identity,), joined_storage=team)
