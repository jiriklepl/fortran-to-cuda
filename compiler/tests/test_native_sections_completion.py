"""Original SECTIONS acquire whole native effects, never worksharing cuts."""

from dataclasses import replace
from hashlib import sha256

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.driver.options import CompilerOptions
from compiler.frontend.native_completion import _joined_completion_facts
from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.native_atomic import summarize
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT

ENTRY = "renamed_sections::step"


def loop(face, *, statement=None, iterator="j", bound="2"):
    statement = statement or f"b({face},{iterator})=b({face},{iterator})+sum(a({face}:{face},{iterator}))"
    return f"do {iterator}=0,{bound}\n{statement}\nenddo\n"


def sections(*, nowait=False, first=None, second=None):
    return ("!$omp sections\n!$omp section\n" + (first or loop("-2"))
            + "!$omp section\n" + (second or loop("18"))
            + "!$omp end sections" + (" nowait" if nowait else "") + "\n")


def fixture(tmp_path, body=None, *, operations=256, private="j", declaration=""):
    path = tmp_path / "original_sections.f90"
    text = ("module renamed_sections\nimplicit none\ncontains\n"
            "subroutine step(a,b,flag,n)\nreal(8),intent(in)::a(-2:,-1:)\n"
            "real(8),intent(inout)::b(-2:,-1:)\nlogical,intent(in)::flag\n"
            "integer,intent(in)::n\ninteger::i,j\n" + declaration + "\ncontinue\n"
            + "!$omp parallel" + (" private(" + private + ")" if private else "") + "\n"
            + (sections() if body is None else body) + "!$omp end parallel\n"
            + "end subroutine\nend module\n")
    path.write_text(text)
    analysis = SourceEffects([path], operations=operations)
    graph = analysis.structure(ENTRY)
    assert graph.available, graph.reasons
    nodes = tuple(node for node in analysis.routines[ENTRY].execution.content
                  if type(node).__name__ != "Continue_Stmt")
    group = next(iter(graph.native_groups.values()), None)
    selected = group.original_nodes if group is not None else nodes
    return path, analysis, graph, selected, group


@pytest.mark.parametrize("nowait", [False, True])
def test_whole_original_sections_retains_join_and_exact_opposite_faces(tmp_path, nowait):
    path, analysis, _graph, selected, _group = fixture(tmp_path, sections(nowait=nowait))
    before = path.read_bytes()
    proof = analysis.joined_completion(ENTRY, selected)
    public = proof.public()
    assert public["schema_version"] == 6
    assert public["original_section_count"] == 2
    assert public["retains_original_team_and_directives"]
    assert public["native_only"]
    assert public["internal_cuts_authorized"] is False
    assert public["section_independence_established"] is public["gpu_legality_established"] is False
    physical = analysis.native_sections_for_nodes(ENTRY, selected, completion=proof, capture_locals=True)
    assert physical.available, physical.reason
    resources = {item.resource: item for item in physical.resources}
    assert {box.axes[0].lower.value for box in resources["argument::b"].writes} == {-2, 18}
    assert len(resources["argument::a"].reads) == 2
    assert path.read_bytes() == before


def test_large_guarded_groups_demand_bounded_section_units_without_flat_expansion(tmp_path):
    body = "if(flag) then\n" + sections() * 24 + "else\n" + sections() * 24 + "endif\n"
    _, analysis, graph, selected, group = fixture(tmp_path, body)
    assert group is not None
    assert group.available, group.reason
    assert group.public()["schema_version"] == 2
    assert graph.version == 5
    assert len(group.units) == 97
    assert [unit.kind for unit in group.units] == ["condition"] + ["section"] * 96
    assert group.units[1].guard == ("flag",)
    assert group.units[-1].guard == (".not.(flag)",)
    proof = analysis.joined_completion(ENTRY, selected)
    summary = summarize(analysis, ENTRY, selected, proof)
    assert summary["complete"], summary["reasons"]
    assert sum(summary["native_atomic"]["unit_operation_counts"]) > 256
    assert max(summary["native_atomic"]["unit_operation_counts"]) <= 256
    assert summary["guaranteed_whole_overwrites"] == []
    physical = analysis.native_sections_for_nodes(ENTRY, selected, completion=proof, capture_locals=True)
    assert physical.available, physical.reason
    output = next(item for item in physical.resources if item.resource == "argument::b")
    assert len(output.reads) == len(output.writes) == 2
    assert output.overwrites == ()


@pytest.mark.parametrize("body", [
    sections().replace("!$omp end sections\n", ""),
    "!$omp section\n" + loop("-2"),
    sections().replace("!$omp section\n", "", 1),
    sections().replace("!$omp sections\n", "!$omp sections private(j)\n"),
    sections().replace("!$omp end sections\n", "!$omp end sections nowait extra\n"),
    sections().replace("!$omp section\n", "!$omp section private(j)\n", 1),
    sections(first="!$omp sections\n!$omp section\n" + loop("-2") + "!$omp end sections\n"),
    sections(first=loop("-2") + loop("-1")),
    sections(first="!$omp do\n" + loop("-2") + "!$omp end do\n"),
])
def test_malformed_or_unsupported_sections_remain_whole_boundaries(tmp_path, body):
    _, analysis, _graph, selected, _group = fixture(tmp_path, body)
    with pytest.raises(CompilationError):
        analysis.joined_completion(ENTRY, selected)


@pytest.mark.parametrize(("body", "private"), [
    (sections(), ""),
    (sections(first="do i=0,1\n" + loop("-2") + "enddo\n"), "j"),
    (sections(first="do while(flag)\nb(-2,0)=1\nenddo\n"), "j"),
])
def test_section_counted_indices_need_explicit_original_private_storage(tmp_path, body, private):
    _, analysis, _graph, selected, _group = fixture(tmp_path, body, private=private)
    with pytest.raises(CompilationError, match="(PRIVATE|counted DO)"):
        analysis.joined_completion(ENTRY, selected)


def test_nested_private_counted_loops_are_retained_as_one_section_body(tmp_path):
    body = sections(first="do i=0,1\n" + loop("-2") + "enddo\n")
    _, analysis, _graph, selected, _group = fixture(tmp_path, body, private="i,j")
    assert analysis.joined_completion(ENTRY, selected).public()["original_section_count"] == 2


@pytest.mark.parametrize("nowait", [False, True])
def test_any_sections_rejects_worksharing_extraction_without_partial_collection(tmp_path, nowait):
    ordinary = "!$omp do\n" + loop("-2") + "!$omp end do\n"
    _, analysis, _graph, selected, _group = fixture(tmp_path, ordinary + sections(nowait=nowait))
    proof = analysis.joined_completion(ENTRY, selected)
    assert proof.public()["original_section_count"] == 2
    collected = []
    with pytest.raises(CompilationError, match="worksharing cuts"):
        _joined_completion_facts(analysis, ENTRY, selected, worksharing=collected)
    assert collected == []
    original_loop = walk(analysis.routines[ENTRY].execution, F.Block_Nonlabel_Do_Construct)[0]
    with pytest.raises(CompilationError, match="worksharing cuts"):
        analysis.worksharing_completion(ENTRY, selected, (original_loop,))
    with pytest.raises(CompilationError, match="worksharing cuts"):
        analysis.worksharing_native_completion(ENTRY, selected, (original_loop,))


def test_copied_partial_and_stale_section_authority_never_grants_effects(tmp_path):
    path, analysis, _graph, selected, _group = fixture(tmp_path)
    proof = analysis.joined_completion(ENTRY, selected)
    with pytest.raises(CompilationError, match="whole original joined-group authority"):
        analysis.native_sections_for_nodes(ENTRY, selected, completion=replace(proof))
    loop_node = walk(analysis.routines[ENTRY].execution, F.Block_Nonlabel_Do_Construct)[0]
    with pytest.raises(CompilationError, match="whole original joined-group authority"):
        analysis.native_sections_for_nodes(ENTRY, (loop_node,), completion=proof)
    path.write_text(path.read_text().replace("sum(a", "2*sum(a"))
    with pytest.raises(CompilationError, match="changed"):
        analysis.native_sections_for_nodes(ENTRY, selected, completion=proof)


@pytest.mark.parametrize("statement", ["call opaque(b)", "allocate(scratch(3))", "return"])
def test_section_unknown_call_lifetime_or_exit_effects_cannot_disappear(tmp_path, statement):
    body = sections(first=loop("-2", statement=statement)) * 48
    _, analysis, _graph, selected, group = fixture(tmp_path, body,
        declaration="real(8),allocatable::scratch(:)")
    assert group is not None
    with pytest.raises(CompilationError):
        analysis.joined_completion(ENTRY, selected)


def test_section_source_unit_budget_rejects_without_truncated_authority(tmp_path):
    _, analysis, _graph, selected, group = fixture(tmp_path, sections() * 7, operations=12)
    assert group is not None
    assert not group.available
    assert group.units == ()
    assert "source-unit budget" in group.reason
    with pytest.raises(CompilationError, match="completion unavailable"):
        analysis.joined_completion(ENTRY, selected)


def test_preceding_do_private_clause_cannot_grant_section_iterator_ownership(tmp_path):
    body = "!$omp do private(j)\n" + loop("-2") + "!$omp end do\n" + sections()
    _, analysis, _graph, selected, _group = fixture(tmp_path, body, private="")
    with pytest.raises(CompilationError, match="explicit original PRIVATE"):
        analysis.joined_completion(ENTRY, selected)


def test_overlarge_section_body_cannot_use_other_units_to_hide_its_budget(tmp_path):
    repeated = "\n".join("b(-2,j)=a(-2,j)" for _ in range(13))
    _, analysis, _graph, selected, group = fixture(tmp_path,
        sections(first=loop("-2", statement=repeated)), operations=12)
    assert group is not None
    assert not group.available
    assert group.units == ()
    assert "unit skeleton unavailable" in group.reason
    with pytest.raises(CompilationError, match="completion unavailable"):
        analysis.joined_completion(ENTRY, selected)


def test_end_sections_never_replaces_the_original_parallel_join(tmp_path):
    path, analysis, _graph, _selected, _group = fixture(tmp_path)
    path.write_text(path.read_text().replace("!$omp end parallel\n", ""))
    analysis = SourceEffects([path])
    selected = tuple(node for node in analysis.routines[ENTRY].execution.content
                     if type(node).__name__ != "Continue_Stmt")
    with pytest.raises(CompilationError, match="joined END PARALLEL"):
        analysis.joined_completion(ENTRY, selected)


@pytest.mark.parametrize("count", [1, 48])
def test_reached_owner_retains_original_sections_between_two_numerical_regions(tmp_path, count):
    path, _analysis, _graph, _selected, _group = fixture(tmp_path, sections() * count)
    text = path.read_text().replace("implicit none\ncontains", "implicit none\ninteger::visits=0\ncontains")
    text = text.replace("step(a,b,flag,n)", "step(a,b,out,flag,n)")
    text = text.replace("b(-2:,-1:)\n", "b(-2:,-1:),out(-2:,-1:)\n", 1)
    text = text.replace("continue\n", "visits=visits+1\n"
        "do j=0,2\ndo i=0,n-1\nb(i,j)=2*a(i,j)+real(i+3*j,8)\nenddo\nenddo\ni=0\n", 1)
    text = text.replace("!$omp end parallel\n", "!$omp end parallel\n"
        "do j=0,2\ndo i=0,n-1\nout(i,j)=b(i,j)+b(-2,j)+b(18,j)\nenddo\nenddo\n", 1)
    path.write_text(text)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: FACT for name in ("a", "b", "out")}}
    outputs, report = ScopeBuilder([path], ENTRY, facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="reached")).run()
    owner, = report["scopes"]
    assert len(owner["gpu_leaves"]) == 2
    assert owner["boundaries"] == []
    native, = [operation for segment in owner["planning_segments"]
               for operation in segment["operations"]["native_operations"]
               if operation["kind"] == "joined native OpenMP"]
    assert native["sections"]["available"], native["sections"]
    assert native["completion"]["original_section_count"] == count * 2
    generated = outputs[report["sources"][str(path)]["replacement"]]
    assert generated.count("!$omp sections\n") == count * 2
    assert generated.count("!$omp end sections\n") == count * 2
    assert generated.count("!$omp end parallel\n") == 2
    assert path.read_text() == text
