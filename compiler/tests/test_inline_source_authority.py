"""Structured demands consume original nodes behind approved inline workers."""

import copy
from hashlib import sha256

import pytest
from fparser.two import Fortran2003 as F

from compiler.driver.options import CompilerOptions
from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_source_scopes import FACT


SOURCE = """module renamed_owner
contains
subroutine advance(a,n,flag)
real(8),intent(inout)::a(:)
integer,intent(in)::n
logical,intent(in)::flag
integer::i
if(flag)then
do i=1,n
a(i)=a(i)+1
enddo
else
call unknown_native(a)
endif
end subroutine
end module
"""


def prepared(tmp_path, source=SOURCE):
    path = tmp_path / "original.f90"
    path.write_text(source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::a": FACT}}
    builder = ScopeBuilder([path], "renamed_owner::advance", facts=facts,
                           options=CompilerOptions(), config=OffloadConfig(policy="sections"))
    nodes = tuple(_children(builder.entry.execution))
    result = builder.inline.prepare(nodes)
    assert len(builder.inline.nodes) == 1, builder.boundaries
    facade, = builder.inline.nodes.values()
    return builder, nodes, result, facade, path


def test_worker_and_branch_keep_distinct_original_node_authority(tmp_path):
    builder, original, prepared_nodes, facade, _ = prepared(tmp_path)
    branch, = (node for node in original if _kind(node) == "If_Construct")
    cloned, = (node for node in prepared_nodes if _kind(node) == "If_Construct")
    assert cloned is not branch
    assert builder.inline.original_selection(cloned) == (branch,)
    selection = builder.inline.original_selection(facade)
    assert len(selection) == 1 and _kind(selection[0]) == "Block_Nonlabel_Do_Construct"
    assert any(selection[0] is node for node in branch.content)
    assert "unknown_native" in str(builder.inline.original_selection(cloned)[0])


@pytest.mark.parametrize("kind", ["copied", "same_text", "mutated_text", "mutated_span", "changed_proof"])
def test_facade_attributes_text_and_spans_cannot_grant_authority(tmp_path, kind):
    builder, _, _, facade, _ = prepared(tmp_path)
    if kind == "copied":
        facade = copy.copy(facade)
    elif kind == "same_text":
        forged = F.Call_Stmt(str(facade))
        forged.item = facade.item
        forged.fort_inline_region = facade.fort_inline_region
        facade = forged
    elif kind == "mutated_text":
        facade.items = (F.Name("different_call"), None)
    elif kind == "mutated_span":
        facade.item.fort_original_span = (1, 2)
    else:
        from dataclasses import replace
        procedure = facade.fort_inline_region
        builder.inline.regions[procedure] = replace(builder.inline.regions[procedure], source_identity="forged")
    with pytest.raises(CompilationError, match="source|facade|projection"):
        builder.inline.original_selection(facade)
    with pytest.raises(CompilationError):
        builder.inline.call(facade)


def test_source_change_invalidates_prepared_authority(tmp_path):
    builder, _, _, facade, path = prepared(tmp_path)
    path.write_text(path.read_text().replace("a(i)+1", "a(i)+2"))
    with pytest.raises(CompilationError):
        builder.inline.original_selection(facade)


def test_joined_directive_projection_recovers_original_loop_once(tmp_path):
    source = SOURCE.replace("if(flag)then\n", "i=0\n!$omp parallel do private(i)\n").replace(
        "else\ncall unknown_native(a)\nendif", "!$omp end parallel do")
    builder, original, _, facade, _ = prepared(tmp_path, source)
    selection = builder.inline.original_selection(facade)
    assert len({id(node) for node in selection}) == len(selection)
    assert any(_kind(node) == "Block_Nonlabel_Do_Construct" for node in selection)
    assert all(any(node is candidate for candidate in original) for node in selection)


def test_single_line_if_projection_retains_original_action(tmp_path):
    builder, original, _, _, _ = prepared(tmp_path, SOURCE.replace(
        "if(flag)then", "if(flag) call untouched(a)\nif(flag)then"))
    header = next(node for node in original if _kind(node) == "If_Stmt")
    action = header.items[1]
    projected = builder.inline.statement_projection(action, header)
    assert projected is not action
    assert builder.inline.original_selection(projected) == (action,)
    assert not hasattr(action, "item") or action.item is None
    with pytest.raises(CompilationError, match="original IF action"):
        builder.inline.statement_projection(copy.copy(action), header)
    with pytest.raises(CompilationError, match="source selection"):
        builder.inline.statement_projection(action, copy.copy(header))


def test_regrouping_does_not_grant_a_copied_loop_authority(tmp_path):
    source = SOURCE.replace("if(flag)then\n", "i=0\n!$omp parallel do private(i)\n").replace(
        "else\ncall unknown_native(a)\nendif", "!$omp end parallel do")
    builder, original, _, _, _ = prepared(tmp_path, source)
    first = builder.inline.grouped_nodes(original)
    for group in first:
        for node in group if isinstance(group, tuple) else (group,):
            assert builder.inline.original_selection(node)
    loop = next(node for group in first for node in (group if isinstance(group, tuple) else (group,))
                if _kind(node) == "Block_Nonlabel_Do_Construct")
    with pytest.raises(CompilationError, match="source selection"):
        builder.inline.grouped_nodes((copy.copy(loop),))
