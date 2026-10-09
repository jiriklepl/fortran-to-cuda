"""Exact native effects require the complete original joined-group token."""

from __future__ import annotations

from dataclasses import replace

import pytest

from compiler.frontend.source_effects import SourceEffects
from compiler.ir import CompilationError

PROCEDURE = "grouped::step"


def fixture(tmp_path, body=None, *, declaration="", opening="parallel private(i)", ending="end parallel"):
    path = tmp_path / "joined.f90"
    body = body or "!$omp do\ndo i=1,n\na(-2,i)=b(1,i)\nenddo\n!$omp end do nowait"
    path.write_text("module grouped\nimplicit none\ncontains\nsubroutine step(a,b,n,flag)\n"
        "real(8),intent(inout)::a(-2:,:)\nreal(8),intent(in)::b(:,:)\ninteger,intent(in)::n\n"
        "logical,intent(in)::flag\ninteger::i,j\n" + declaration + "\ncontinue\n!$omp " + opening
        + "\n" + body + ("\n!$omp " + ending if ending else "") + "\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    nodes = tuple(analysis.routines[PROCEDURE].execution.content)
    # CONTINUE separates specification comments from the executable group;
    # opening directives may remain attached to their original DO construct.
    nodes = tuple(node for node in nodes if type(node).__name__ != "Continue_Stmt")
    return path, analysis, nodes


def test_registered_whole_group_token_refines_faces_and_retains_join(tmp_path):
    _path, analysis, nodes = fixture(tmp_path)
    proof = analysis.joined_completion(PROCEDURE, nodes)
    public = proof.public()
    assert public["available"] and public["retains_original_team_and_directives"]
    assert public["caller_contract"] == "serial_source_scope"
    assert "grouped::step::i" in public["private_resources"]
    sections = analysis.native_sections_for_nodes(PROCEDURE, nodes, completion=proof, capture_locals=True)
    assert sections.available, sections.reason
    output = next(item for item in sections.resources if item.resource == "argument::a")
    assert len(output.writes) == len(output.overwrites) == 1
    assert output.writes[0].axes[0].point
    assert output.writes[0].axes[0].lower.value == -2
    assert not analysis.native_sections(PROCEDURE).available
    public["private_resources"].clear()
    assert proof.private_roots


def test_optional_combined_end_uses_original_loop_completion(tmp_path):
    from compiler.scopes.segments import grouped_nodes
    _, analysis, nodes = fixture(tmp_path, 'do i=1,n\na(-2,i)=b(1,i)\nenddo',
                                 opening='parallel do private(i)', ending='')
    group, = grouped_nodes(nodes)
    assert isinstance(group, tuple)
    proof = analysis.joined_completion(PROCEDURE, nodes)
    assert proof.public()['join'] == 'implicit combined-loop completion'
    assert analysis.native_sections_for_nodes(PROCEDURE, nodes, completion=proof).available


def test_optional_worksharing_end_keeps_the_enclosing_join(tmp_path):
    _, analysis, nodes = fixture(tmp_path, '!$omp do\ndo i=1,n\na(-2,i)=b(1,i)\nenddo')
    proof = analysis.joined_completion(PROCEDURE, nodes)
    assert proof.public()['join'] == 'explicit original parallel end'
    assert analysis.native_sections_for_nodes(PROCEDURE, nodes, completion=proof).available


def test_public_or_copied_completion_facts_cannot_authorize_sections(tmp_path):
    _path, analysis, nodes = fixture(tmp_path)
    proof = analysis.joined_completion(PROCEDURE, nodes)
    for foreign in (proof.public(), replace(proof)):
        with pytest.raises(CompilationError, match="(proof token|whole original joined-group authority)"):
            analysis.native_sections_for_nodes(PROCEDURE, nodes, completion=foreign)
    independent = SourceEffects(analysis.inputs.paths)
    foreign_nodes = tuple(node for node in independent.routines[PROCEDURE].execution.content
                          if type(node).__name__ != "Continue_Stmt")
    with pytest.raises(CompilationError, match="whole original joined-group authority"):
        independent.native_sections_for_nodes(PROCEDURE, foreign_nodes, completion=proof)


def test_token_does_not_authorize_a_child_loop_of_a_larger_region(tmp_path):
    body = "!$omp do\ndo i=1,n\na(-2,i)=1\nenddo\n!$omp end do nowait\n"
    body += "!$omp do\ndo i=1,n\na(2,i)=2\nenddo\n!$omp end do nowait"
    _path, analysis, nodes = fixture(tmp_path, body)
    proof = analysis.joined_completion(PROCEDURE, nodes)
    first_loop = next(node for node in nodes if type(node).__name__ == "Block_Nonlabel_Do_Construct")
    with pytest.raises(CompilationError, match="whole original joined-group authority"):
        analysis.native_sections_for_nodes(PROCEDURE, (first_loop,), completion=proof)
    executable = tuple(node for node in nodes if type(node).__name__ != "Comment")
    sections = analysis.native_sections_for_nodes(PROCEDURE, executable, completion=proof)
    assert sections.available, sections.reason


def test_uniform_alternate_faces_are_possible_writes_without_overwrite_claim(tmp_path):
    body = "if(flag) then\n!$omp do\ndo i=1,n\na(-2,i)=b(1,i)\nenddo\n!$omp end do nowait\n"
    body += "else\n!$omp do\ndo i=1,n\na(2,i)=b(2,i)\nenddo\n!$omp end do nowait\nendif"
    _path, analysis, nodes = fixture(tmp_path, body)
    proof = analysis.joined_completion(PROCEDURE, nodes)
    sections = analysis.native_sections_for_nodes(PROCEDURE, nodes, completion=proof)
    assert sections.available, sections.reason
    output = next(item for item in sections.resources if item.resource == "argument::a")
    assert len(output.writes) == 2
    assert output.overwrites == ()
    assert [box.axes[0].lower.value for box in output.writes] == [-2, 2]


def test_private_fixed_array_is_original_native_storage_not_a_shared_capture(tmp_path):
    body = "!$omp do\ndo i=1,n\nlocal(0)=b(1,i)\na(-2,i)=local(0)\nenddo\n!$omp end do"
    _path, analysis, nodes = fixture(tmp_path, body, declaration="real(8)::local(-1:1)",
                                     opening="parallel private(i,local)")
    proof = analysis.joined_completion(PROCEDURE, nodes)
    assert "grouped::step::local" in proof.private_roots
    sections = analysis.native_sections_for_nodes(PROCEDURE, nodes, completion=proof, capture_locals=True)
    assert sections.available, sections.reason
    assert {item.resource for item in sections.resources} == {"argument::a", "argument::b"}


@pytest.mark.parametrize("body,opening,ending,reason", [
    ("do i=1,n\na(-2,i)=1\nenddo", "parallel do private(i)", "end parallel", "combined PARALLEL DO"),
    ("!$omp task\na(-2,1)=1\n!$omp end task", "parallel", "end parallel", "unsupported joined"),
    ("!$omp do\ndo i=1,n\ncall unknown(a)\nenddo\n!$omp end do", "parallel private(i)", "end parallel", "source-call proof"),
    ("if(i>0) then\n!$omp do\ndo j=1,n\na(-2,j)=1\nenddo\n!$omp end do\nendif",
     "parallel private(i,j)", "end parallel", "condition is private"),
])
def test_unsafe_completion_or_nonuniform_worksharing_remains_a_boundary(tmp_path, body, opening, ending, reason):
    _path, analysis, nodes = fixture(tmp_path, body, opening=opening, ending=ending)
    with pytest.raises(CompilationError, match=reason):
        analysis.joined_completion(PROCEDURE, nodes)


def test_changed_original_source_invalidates_completion_token(tmp_path):
    path, analysis, nodes = fixture(tmp_path)
    proof = analysis.joined_completion(PROCEDURE, nodes)
    path.write_text(path.read_text().replace("end parallel", "end parallel nowait"))
    with pytest.raises(CompilationError, match="changed"):
        analysis.native_sections_for_nodes(PROCEDURE, nodes, completion=proof)
