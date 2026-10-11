"""Fresh syntax checks retain original authority without rendering whole ASTs."""

from copy import copy

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import Base, walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.source_syntax import SourceSyntaxSnapshot
from compiler.ir import CompilationError

ENTRY = "syntax_origin::advance"
SOURCE = """module syntax_origin
implicit none
integer,parameter::marker=3
! module declaration authority
contains
subroutine advance(a,b)
real(8),intent(inout)::a(:),b(:)
a(1)=b(1)+1
a(2)=b(2)+2
end subroutine
end module
"""


def fixture(tmp_path):
    path = tmp_path / "syntax.f90"
    path.write_text(SOURCE)
    analysis = SourceEffects([path])
    return path, analysis, analysis.routines[ENTRY]


def mutate(analysis, routine, change):
    first, second = walk(routine.execution, F.Assignment_Stmt)
    if change == "leaf":
        name, = walk(first.items[2], F.Name)
        name.string = "a"
    elif change == "operator":
        expression = first.items[2]
        expression.items = (expression.items[0], "-", expression.items[2])
    elif change == "separator":
        arguments, = walk(routine.scope.node, F.Dummy_Arg_List)
        arguments.separator = ";"
    elif change == "same_text_clone":
        original = first.items[2]
        cloned = copy(original)
        assert str(cloned) == str(original)
        first.items = (*first.items[:2], cloned)
    elif change == "order":
        body = routine.execution.content
        left = next(index for index, node in enumerate(body) if node is first)
        right = next(index for index, node in enumerate(body) if node is second)
        body[left], body[right] = body[right], body[left]
    elif change == "module_specification":
        declaration = next(node for node in walk(analysis.modules["syntax_origin"].node, F.Type_Declaration_Stmt)
                           if "marker" in str(node).lower())
        literal, = walk(declaration, F.Int_Literal_Constant)
        literal.items = ("4", literal.items[1])
    elif change == "comment":
        comment = next(node for node in walk(analysis.modules["syntax_origin"].node, F.Comment)
                       if "module declaration authority" in str(node))
        comment.items[0] += " changed"
    elif change == "item":
        first.item = copy(first.item)
    elif change == "span":
        first.item.span = tuple(line + 1 for line in first.item.span)
    elif change == "label":
        first.item.label = "170"
    elif change == "construct_name":
        first.item.name = "different_origin"
    else:
        assert change == "provenance"
        first.item.fort_original_span = (1, 1)


@pytest.mark.parametrize("change", [
    "leaf", "operator", "separator", "same_text_clone", "order", "module_specification", "comment",
    "item", "span", "label", "construct_name", "provenance",
])
def test_new_checkpoint_detects_original_node_and_render_metadata_mutation(tmp_path, change):
    _path, analysis, routine = fixture(tmp_path)
    module = analysis.modules["syntax_origin"].node
    snapshot = SourceSyntaxSnapshot((module, routine.execution))
    before = snapshot.checkpoint()
    assert before.unchanged(module)
    assert before.unchanged(routine.execution)
    mutate(analysis, routine, change)
    assert not snapshot.checkpoint().unchanged(module)


def test_unchanged_checkpoint_does_not_render_ast_or_accept_equal_looking_foreign_root(tmp_path, monkeypatch):
    _path, analysis, routine = fixture(tmp_path)
    module = analysis.modules["syntax_origin"].node
    snapshot = SourceSyntaxSnapshot((module, routine.execution))
    foreign = copy(module)

    def unexpected_render(_node):
        pytest.fail("syntax authentication recursively rendered an AST")

    monkeypatch.setattr(Base, "__str__", unexpected_render)
    checkpoint = snapshot.checkpoint()
    assert checkpoint.unchanged(module)
    assert checkpoint.unchanged(routine.execution)
    assert not checkpoint.unchanged(foreign)


def test_syntax_authority_ignores_parser_parent_backreferences(tmp_path):
    _path, _analysis, routine = fixture(tmp_path)
    snapshot = SourceSyntaxSnapshot((routine.execution,))
    first, _second = walk(routine.execution, F.Assignment_Stmt)
    first.parent = object()
    assert snapshot.checkpoint().unchanged(routine.execution)


@pytest.mark.parametrize("change", ["cycle", "opaque", "budget"])
def test_snapshot_rejects_unbounded_or_opaque_parser_state(tmp_path, change):
    _path, _analysis, routine = fixture(tmp_path)
    if change == "cycle":
        routine.execution.content.append(routine.execution)
    elif change == "opaque":
        routine.execution.content.append(object())
    with pytest.raises(CompilationError, match="original source syntax"):
        SourceSyntaxSnapshot((routine.execution,), limit=1 if change == "budget" else 1_048_576)


def test_checkpoint_rejects_added_syntax_state_and_nested_mutable_sequence(tmp_path):
    _path, _analysis, routine = fixture(tmp_path)
    first, _second = walk(routine.execution, F.Assignment_Stmt)
    first.items = [first.items[0], first.items[1], [first.items[2]]]
    snapshot = SourceSyntaxSnapshot((routine.execution,))
    assert snapshot.checkpoint().unchanged(routine.execution)
    first.items[2].append("changed")
    assert not snapshot.checkpoint().unchanged(routine.execution)
    first.items[2].pop()
    assert snapshot.checkpoint().unchanged(routine.execution)
    first.separator = ";"
    assert not snapshot.checkpoint().unchanged(routine.execution)


def test_overlapping_layout_eviction_cannot_recapture_mutated_syntax(tmp_path):
    _path, analysis, routine = fixture(tmp_path)
    module = analysis.modules["syntax_origin"].node
    procedures, = walk(module, F.Module_Subprogram_Part)
    snapshot = SourceSyntaxSnapshot((module,))
    assert snapshot.checkpoint().unchanged(module)
    assert id(module) in snapshot._layouts
    # These naturally overlapping subtrees exceed the finite retained-layout
    # weight. Eviction must discard only an index, never original evidence.
    for node in (procedures, routine.scope.node, routine.execution):
        assert snapshot.checkpoint().unchanged(node)
        assert snapshot._layout_weight <= snapshot._layout_limit
    assert id(module) not in snapshot._layouts
    first, _second = walk(routine.execution, F.Assignment_Stmt)
    original_items = first.items[2].items
    mutate(analysis, routine, "operator")
    assert not snapshot.checkpoint().unchanged(module)
    assert not snapshot.checkpoint().unchanged(routine.execution)
    first.items[2].items = original_items
    assert snapshot.checkpoint().unchanged(module)


def test_warmed_layout_rejects_source_item_class_mutation_with_identical_fields(tmp_path):
    _path, _analysis, routine = fixture(tmp_path)
    snapshot = SourceSyntaxSnapshot((routine.execution,))
    assert snapshot.checkpoint().unchanged(routine.execution)
    first, _second = walk(routine.execution, F.Assignment_Stmt)
    item = first.item
    original_kind, original_fields = type(item), dict(vars(item))

    class ChangedSourceItem(original_kind):
        pass

    item.__class__ = ChangedSourceItem
    assert first.item is item
    assert vars(item) == original_fields
    assert not snapshot.checkpoint().unchanged(routine.execution)
    item.__class__ = original_kind
    assert snapshot.checkpoint().unchanged(routine.execution)


@pytest.mark.parametrize("change", [
    "module_specification", "leaf", "operator", "same_text_clone", "comment", "label", "span", "provenance",
])
def test_cached_public_structure_still_rejects_changed_original_syntax(tmp_path, change):
    _path, analysis, routine = fixture(tmp_path)
    graph = analysis.structure(ENTRY)
    assert graph.available
    assert analysis.structure(ENTRY) is graph
    mutate(analysis, routine, change)
    with pytest.raises(CompilationError, match="source.*authority"):
        analysis.structure(ENTRY)


def test_cached_public_structure_still_verifies_source_files(tmp_path):
    path, analysis, _routine = fixture(tmp_path)
    analysis.structure(ENTRY)
    path.write_text(SOURCE + "! changed after cached structure\n")
    with pytest.raises(CompilationError, match="source changed"):
        analysis.structure(ENTRY)


@pytest.mark.parametrize("change", ["binding", "imports"])
def test_unchanged_syntax_does_not_replace_original_namespace_validation(tmp_path, change):
    _path, analysis, routine = fixture(tmp_path)
    analysis.structure(ENTRY)
    if change == "binding":
        routine.scope.bindings["a"].kind = 4
    else:
        routine.scope.imports["unrelated_alias"] = ("foreign", "storage")
    with pytest.raises(CompilationError, match="source-backed procedure authority"):
        analysis.structure(ENTRY)


def test_changed_contracts_still_change_structured_cache_authority(tmp_path):
    _path, analysis, _routine = fixture(tmp_path)
    original = analysis.structure(ENTRY)
    analysis.contracts["unrelated"] = {"identity": "new-contract"}
    changed = analysis.structure(ENTRY)
    assert changed.identity != original.identity
    assert analysis.structure(ENTRY) is changed


def test_original_empty_procedure_does_not_become_a_private_projection(tmp_path):
    path = tmp_path / "empty.f90"
    path.write_text("module empty_origin\ncontains\nsubroutine untouched\nend subroutine\nend module\n")
    analysis = SourceEffects([path])
    assert analysis.routines["empty_origin::untouched"].execution is None
    assert analysis._summary_authority()["role"] == "source"
    assert analysis._summary_authority()["projections"] == []
    analysis.structure("empty_origin::untouched")


def test_private_execution_projection_recomputes_effects_without_source_authority(tmp_path):
    _path, analysis, routine = fixture(tmp_path)
    original = analysis.summarize(ENTRY)
    original_sections = analysis.native_sections(ENTRY)
    analysis.structure(ENTRY)
    assert original["complete"], original["reasons"]
    assert original["summary_role"] == "source"
    routine.execution.content[:] = [F.Assignment_Stmt("a(3)=9")]
    projected = analysis.summarize(ENTRY)
    assert projected["complete"], projected["reasons"]
    assert projected["summary_role"] == "private_projection"
    assert projected["summary_identity"] != original["summary_identity"]
    sections = analysis.native_sections(ENTRY)
    assert sections is not original_sections
    assert sections.resources[0].resource == "argument::a"
    assert sections.resources[0].writes[0].axes[0].lower.value == 3
    assert analysis.native_sections(ENTRY) is sections
    with pytest.raises(CompilationError, match="source.*authority"):
        analysis.structure(ENTRY)
