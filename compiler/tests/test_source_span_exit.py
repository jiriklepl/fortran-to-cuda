"""A branch-ending original CALL group has a real, non-sibling cleanup site."""
from copy import copy
from dataclasses import replace

import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_span_exit import prove_original_sequence_exit
from compiler.ir import CompilationError

SOURCE = """module groups
contains
subroutine run(flag)
logical::flag
call unknown_prefix()
if(flag)then
call first()
call second()
call third()
else
call native_other()
endif
call unknown_suffix()
end subroutine
subroutine first()
end subroutine
subroutine second()
end subroutine
subroutine third()
end subroutine
end module
"""


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "groups.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["groups::run"]
    calls = tuple(node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"
                  and str(node.items[0]).lower() in {"first", "second", "third"})
    return path, analysis, calls


def test_real_branch_exit_binds_parent_last_call_and_next_original_arm(tmp_path):
    _path, analysis, calls = fixture(tmp_path)
    proof = prove_original_sequence_exit(analysis, "groups::run", calls)
    public = proof.public()
    assert proof.validate() is proof
    assert public["exit_anchor"]["parser_kind"] == "Else_Stmt"
    assert public["exit_anchor"]["source_span"][0] > public["source_span"][1]
    assert public["last_call_node_id"] == public["selected_node_ids"][-1]
    assert not public["cross_exit_retention_authorized"]
    assert not public["entry_definition_authority"]
    assert not public["execution_authorized"]
    public["exit_anchor"].clear()
    assert proof.public()["exit_anchor"]


def test_endif_is_the_exit_of_last_original_branch(tmp_path):
    _path, analysis, calls = fixture(tmp_path, SOURCE.replace("else\ncall native_other()\n", ""))
    assert prove_original_sequence_exit(analysis, "groups::run", calls).public()["exit_anchor"]["parser_kind"] == "End_If_Stmt"


def test_private_projection_cannot_reuse_a_foreign_exit_registry(tmp_path):
    _path, analysis, calls = fixture(tmp_path)
    proof = prove_original_sequence_exit(analysis, "groups::run", calls)
    projection = analysis.private_projection()
    assert not projection._original_sequence_exits
    with pytest.raises(CompilationError, match="registered original"):
        proof.validate(projection)


@pytest.mark.parametrize("selection", ["prefix", "gap", "foreign", "reversed", "single"])
def test_call_group_cannot_move_or_forge_a_sequence_end(tmp_path, selection):
    _path, analysis, calls = fixture(tmp_path)
    chosen = calls[:2] if selection == "prefix" else (calls[0], calls[2]) if selection == "gap" else (
        copy(calls[0]), *calls[1:]) if selection == "foreign" else tuple(reversed(calls)) if selection == "reversed" else calls[-1:]
    with pytest.raises(CompilationError):
        prove_original_sequence_exit(analysis, "groups::run", chosen)


def test_same_line_branch_end_has_no_safe_insertion_span(tmp_path):
    _path, analysis, calls = fixture(tmp_path, SOURCE.replace("call third()\nelse", "call third();else"))
    with pytest.raises(CompilationError, match="insertion span"):
        prove_original_sequence_exit(analysis, "groups::run", calls)


@pytest.mark.parametrize("mutation", ["copy", "registry", "anchor", "source", "selection"])
def test_issued_exit_revalidates_exact_source_objects_and_live_inputs(tmp_path, mutation):
    path, analysis, calls = fixture(tmp_path)
    proof = prove_original_sequence_exit(analysis, "groups::run", calls)
    if mutation == "copy":
        proof = copy(proof)
    elif mutation == "registry":
        analysis._original_sequence_exits[proof.identity] = replace(proof)
    elif mutation == "anchor":
        object.__setattr__(proof, "_anchor", copy(proof._anchor))
    elif mutation == "source":
        path.write_text(path.read_text()+"!source changed\n")
    else:
        object.__setattr__(proof, "_originals", (copy(calls[0]), *calls[1:]))
    with pytest.raises(CompilationError):
        proof.validate()
