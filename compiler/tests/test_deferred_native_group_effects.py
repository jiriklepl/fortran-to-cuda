"""Whole native groups retain bounded source proofs without granting cuts."""

from dataclasses import replace
from hashlib import sha256

import pytest

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.native_atomic import summarize
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "renamed_boundary_owner::step"


def source(*, count=48, nowait=False, body=None, scalar_bounds=False):
    def units(face):
        operation = body or f"b({face},j)=b({face},j)+sum(a({face}:{face},j))"
        bound = "m-1" if scalar_bounds else "2"
        return ("!$omp do\ndo j=0," + bound + "\n" + operation + "\nenddo\n!$omp end do"
                + (" nowait" if nowait else "") + "\n") * count

    return """module renamed_boundary_owner
implicit none
integer::visits=0
contains
subroutine step(a,b,out,n,m,flag,escape)
real(8),intent(in)::a(-2:,-1:)
real(8),intent(inout)::b(-2:,-1:),out(-2:,-1:)
integer,intent(in)::n,m
logical,intent(in)::flag,escape
integer::i,j
visits=visits+1
do j=0,m-1
do i=0,n-1
b(i,j)=2*a(i,j)+real(i+3*j,8)
enddo
enddo
!$omp parallel private(j)
if(flag) then
""" + units("-2") + "else\n" + units("n+1" if scalar_bounds else "-1") + """endif
!$omp end parallel
if(escape) call opaque(b)
do j=0,m-1
do i=0,n-1
out(i,j)=b(i,j)+b(-2,j)+b(-1,j)
enddo
enddo
end subroutine
end module
"""


def analyzed(tmp_path, **kwargs):
    path = tmp_path / "independent.f90"
    path.write_text(source(**kwargs))
    analysis = SourceEffects([path])
    graph = analysis.structure(ENTRY)
    assert graph.available, graph.reasons
    group, = graph.native_groups.values()
    return path, analysis, graph, group


@pytest.mark.parametrize("nowait", [False, True])
def test_complete_native_group_demands_units_and_deduplicates_exact_faces(tmp_path, nowait):
    path, analysis, graph, group = analyzed(tmp_path, nowait=nowait)
    original_bytes = path.read_bytes()
    assert group.available, group.reason
    assert len(group.units) == 97
    proof = analysis.joined_completion(ENTRY, group.original_nodes)
    assert proof.public()["internal_cuts_authorized"] is False
    with pytest.raises(CompilationError, match="whole original native completion"):
        analysis.segment_summary(ENTRY, group.original_nodes, capture_locals=True)
    summary = summarize(analysis, ENTRY, group.original_nodes, proof)
    assert summary["complete"], summary["reasons"]
    record = summary["native_atomic"]
    assert max(record["unit_operation_counts"]) <= 256
    assert sum(record["unit_operation_counts"]) > 256
    assert record["effect_count"] <= 256
    assert summary["guaranteed_whole_overwrites"] == []
    effects = {(item["resource"], item["kind"]) for item in summary["operations"] if item["rank"]}
    assert effects == {("argument::a", "read"), ("argument::b", "read"), ("argument::b", "write")}
    sections = analysis.native_sections_for_nodes(ENTRY, group.original_nodes, completion=proof, capture_locals=True)
    assert sections.available, sections.reason
    resources = {item.resource: item for item in sections.resources}
    assert len(resources["argument::a"].reads) == 2
    assert len(resources["argument::b"].reads) == len(resources["argument::b"].writes) == 2
    assert resources["argument::b"].overwrites == ()
    assert path.read_bytes() == original_bytes
    assert graph.nodes[group.node_id].kind == "boundary"
    assert not analysis.descriptor_stability(ENTRY)["complete"]


def test_guarded_scalar_bounds_keep_whole_effects_without_early_evaluation(tmp_path):
    _path, analysis, _graph, group = analyzed(tmp_path, scalar_bounds=True)
    proof = analysis.joined_completion(ENTRY, group.original_nodes)
    assert summarize(analysis, ENTRY, group.original_nodes, proof)["complete"]
    sections = analysis.native_sections_for_nodes(ENTRY, group.original_nodes, completion=proof, capture_locals=True)
    assert not sections.available
    assert "unproved early scalar bound" in sections.reason


def test_read_and_write_rectangles_share_one_bounded_union(tmp_path):
    repeated = "b(100,j)=sum(b(200:200,j))"
    text = source(body=repeated)
    for index in range(96):
        read, write = index % 17, 100 + index % 17
        text = text.replace(repeated, f"b({write},j)=sum(b({read}:{read},j))", 1)
    path = tmp_path / "many_faces.f90"
    path.write_text(text)
    analysis = SourceEffects([path])
    group, = analysis.structure(ENTRY).native_groups.values()
    proof = analysis.joined_completion(ENTRY, group.original_nodes)
    assert summarize(analysis, ENTRY, group.original_nodes, proof)["complete"]
    sections = analysis.native_sections_for_nodes(ENTRY, group.original_nodes, completion=proof, capture_locals=True)
    assert not sections.available
    assert "union exceeds bounded refinement" in sections.reason


def test_group_members_or_copied_tokens_never_acquire_whole_native_authority(tmp_path):
    _path, analysis, _graph, group = analyzed(tmp_path)
    proof = analysis.joined_completion(ENTRY, group.original_nodes)
    selections = [(group.original_nodes[0],), (group.original_nodes[-1],),
                  group.units[1].original_nodes, tuple(reversed(group.original_nodes))]
    for selected in selections:
        with pytest.raises(CompilationError):
            summarize(analysis, ENTRY, selected, proof)
    with pytest.raises(CompilationError, match="whole original joined-group authority"):
        summarize(analysis, ENTRY, group.original_nodes, replace(proof))
    with pytest.raises(CompilationError, match="exact whole original source selection"):
        analysis.joined_completion(ENTRY, group.node_id)
    with pytest.raises(CompilationError, match="exact whole original source selection"):
        analysis.joined_completion(ENTRY, (group.node_id,))
    with pytest.raises(CompilationError, match="exact whole original source selection"):
        analysis.native_sections_for_nodes(ENTRY, (group.node_id,), completion=proof)


def test_ancestor_selections_cannot_hide_the_native_only_boundary(tmp_path):
    text = source().replace("!$omp parallel private(j)", "if(.not.escape) then\n!$omp parallel private(j)")
    text = text.replace("!$omp end parallel\nif(escape)", "!$omp end parallel\nendif\nif(escape)")
    path = tmp_path / "guarded_owner.f90"
    path.write_text(text)
    analysis = SourceEffects([path])
    graph = analysis.structure(ENTRY)
    group, = graph.native_groups.values()
    branch, = [node for node in graph.nodes.values()
               if node.kind == "branch" and group.node_id in graph.deferred_native_descendants((node.id,))]
    for identity in (graph.root_id, branch.id):
        assert graph.deferred_native_descendants((identity,)) == (group.node_id,)
        with pytest.raises(CompilationError, match="whole original native completion"):
            analysis.segment_summary(ENTRY, (identity,), capture_locals=True)
        with pytest.raises(CompilationError, match="exact whole original source selection"):
            analysis.native_sections_for_nodes(ENTRY, (identity,), capture_locals=True)
        with pytest.raises(CompilationError, match="exact whole original source selection"):
            analysis.joined_completion(ENTRY, (identity,))
    assert all(type(node).__name__ != "NativeGroupCandidate" for node in graph.source_nodes(graph.root_id))
    proof = analysis.joined_completion(ENTRY, group.original_nodes)
    assert summarize(analysis, ENTRY, group.original_nodes, proof)["complete"]


@pytest.mark.parametrize("body", ["call opaque(b)", "return", "allocate(scratch(3))"])
def test_calls_or_lifetime_and_exit_effects_remain_boundaries(tmp_path, body):
    text = source(body=body).replace("integer::i,j", "integer::i,j\nreal(8),allocatable::scratch(:)")
    path = tmp_path / "unsupported.f90"
    path.write_text(text)
    analysis = SourceEffects([path])
    graph = analysis.structure(ENTRY)
    group, = graph.native_groups.values()

    def prove_effects():
        proof = analysis.joined_completion(ENTRY, group.original_nodes)
        return summarize(analysis, ENTRY, group.original_nodes, proof)

    if group.available:
        with pytest.raises(CompilationError):
            prove_effects()
    else:
        with pytest.raises(CompilationError, match="completion unavailable"):
            analysis.joined_completion(ENTRY, group.original_nodes)


def test_deferred_team_retains_one_owner_between_numerical_producer_and_consumer(tmp_path):
    path = tmp_path / "owner.f90"
    path.write_text(source())
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}}
    outputs, report = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="reached")).run()
    owner, = report["scopes"]
    assert len(owner["gpu_leaves"]) == 2
    assert len(owner["boundaries"]) == 1
    assert "opaque" in owner["boundaries"][0]["reason"]
    native, = [operation for segment in owner["planning_segments"]
               for operation in segment["operations"]["native_operations"]
               if operation["kind"] == "joined native OpenMP"]
    assert native["sections"]["available"], native["sections"]
    assert native["atomic_effects"]["deferred_native_group"]["unit_count"] == 97
    text = outputs[report["sources"][str(path)]["replacement"]]
    assert text.count("!$omp parallel private(j)") == 2
    assert text.count("!$omp end parallel") == 2
    assert path.read_text() == source()
