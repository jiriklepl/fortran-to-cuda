"""Session syntax is immutable; private projections remain mutable and distinct."""

from copy import copy, deepcopy

import pytest
from fparser.two import Fortran2003 as F
from fparser.two.utils import Base, walk

from compiler.frontend.sealed_syntax import mutable_copy, sealed
from compiler.ir import CompilationError
from compiler.tests.test_source_syntax_authority import fixture, mutate


@pytest.mark.parametrize("change", [
    "leaf", "operator", "separator", "same_text_clone", "order", "module_specification", "comment",
    "item", "span", "label", "construct_name", "provenance",
])
def test_original_syntax_rejects_mutation_at_the_write(tmp_path, change):
    _path, analysis, routine = fixture(tmp_path)
    analysis._source_syntax = analysis._source_syntax.seal()
    assert sealed(routine.execution)
    with pytest.raises(CompilationError, match="immutable"):
        mutate(analysis, routine, change)
    assert analysis.structure(routine.qualified).available


def test_original_dictionary_class_and_nested_list_are_read_only(tmp_path):
    _path, analysis, routine = fixture(tmp_path)
    snapshot = analysis._source_syntax.seal()
    first, _ = walk(routine.execution, F.Assignment_Stmt)
    with pytest.raises(TypeError):
        vars(first)["items"] = ()
    with pytest.raises(CompilationError, match="immutable"):
        first.__class__ = F.Assignment_Stmt
    with pytest.raises(CompilationError, match="immutable"):
        routine.execution.content.append(first)
    with pytest.raises(TypeError):
        vars(first.item)["span"] = (1, 1)
    assert snapshot.checkpoint().unchanged(routine.execution)


def test_sealing_detaches_preexisting_mutable_dictionary_and_list_aliases(tmp_path):
    _path, analysis, routine = fixture(tmp_path)
    first, _ = walk(routine.execution, F.Assignment_Stmt)
    old_node, old_item, old_body = vars(first), vars(first.item), routine.execution.content
    original_source, original_label = str(routine.execution), first.item.label
    snapshot = analysis._source_syntax.seal()
    old_node["items"] = ()
    old_item["label"] = "919"
    old_body.clear()
    assert str(routine.execution) == original_source
    assert first.item.label == original_label
    assert snapshot.checkpoint().unchanged(routine.execution)


@pytest.mark.parametrize("state", [lambda node: node.__getstate__(),
                                   lambda node: node.__reduce_ex__(4)[2]])
def test_serialization_state_cannot_mutate_original_syntax_or_provenance(tmp_path, state):
    _path, analysis, routine = fixture(tmp_path)
    snapshot = analysis._source_syntax.seal()
    first, _ = walk(routine.execution, F.Assignment_Stmt)
    original_source, original_label = str(first), first.item.label
    state(first)["items"] = ()
    state(first.item)["label"] = "919"
    assert str(first) == original_source
    assert first.item.label == original_label
    assert snapshot.checkpoint().unchanged(routine.execution)


@pytest.mark.parametrize("copier", [copy, deepcopy])
def test_copy_creates_mutable_projection_without_original_authority(tmp_path, copier):
    _path, analysis, routine = fixture(tmp_path)
    snapshot = analysis._source_syntax.seal()
    original = routine.execution
    projection = copier(original)
    assert not sealed(projection)
    assert str(projection) == str(original)
    assert type(projection.content) is list
    projection.content.pop()
    assert not snapshot.checkpoint().unchanged(projection)
    assert snapshot.checkpoint().unchanged(original)
    assert len(original.content) == len(projection.content) + 1


def test_sealed_checkpoint_uses_identity_without_rewalking_original_tree(tmp_path, monkeypatch):
    _path, analysis, routine = fixture(tmp_path)
    snapshot = analysis._source_syntax.seal()

    def unexpected_walk(_node):
        pytest.fail("sealed original syntax was traversed again")

    monkeypatch.setattr(snapshot, "_layout", unexpected_walk)
    for node in walk(routine.scope.node, Base):
        assert snapshot.checkpoint().unchanged(node)


def test_sealed_source_spans_do_not_need_repeated_full_provenance_scan(tmp_path):
    path, analysis, routine = fixture(tmp_path)
    analysis._source_syntax = analysis._source_syntax.seal()

    class NoRepeatedScan(dict):
        def values(self):
            pytest.fail("sealed original source provenance was traversed again")

    analysis._source_provenance = NoRepeatedScan(analysis._source_provenance)
    assert analysis._require_original(routine.qualified) is routine
    path.write_text(path.read_text() + "! changed after immutable provenance\n")
    with pytest.raises(CompilationError, match="source changed"):
        analysis._require_original(routine.qualified)


@pytest.mark.parametrize("freeze", [False, True])
def test_comment_copy_preserves_provenance_and_is_a_mutable_projection(tmp_path, freeze):
    _path, analysis, routine = fixture(tmp_path)
    if freeze:
        analysis._source_syntax = analysis._source_syntax.seal()
    comment = walk(routine.scope.parent.node, F.Comment)[0]
    original = str(comment)
    projection = mutable_copy(comment)
    assert not sealed(projection)
    assert str(projection) == original
    assert projection.item is comment.item
    projection.items = ["! private generated comment"]
    assert str(comment) == original


def test_file_namespace_and_private_projection_authentication_remain_active(tmp_path):
    path, analysis, routine = fixture(tmp_path)
    analysis._source_syntax = analysis._source_syntax.seal()
    graph = analysis.structure(routine.qualified)
    assert analysis.structure(routine.qualified) is graph
    routine.scope.bindings["a"].kind = 4
    with pytest.raises(CompilationError, match="source-backed procedure authority"):
        analysis.structure(routine.qualified)
    routine.scope.bindings["a"].kind = 8
    path.write_text(path.read_text() + "! changed after session creation\n")
    with pytest.raises(CompilationError, match="source changed"):
        analysis.structure(routine.qualified)


def test_sealing_refuses_a_changed_snapshot(tmp_path):
    _path, analysis, routine = fixture(tmp_path)
    mutate(analysis, routine, "operator")
    with pytest.raises(CompilationError, match="cannot seal changed"):
        analysis._source_syntax.seal()
