"""Immutable source-parent indexes preserve exact original read positions."""
import pytest
from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects
from compiler.frontend.storage_component_reads import _read_position
from compiler.ir import CompilationError


def fixture(tmp_path,sealed=True,lhs_index=False):
    tmp_path.mkdir(parents=True,exist_ok=True)
    path=tmp_path/'read.f90'
    statement='out(points(1)%index)=1' if lhs_index else 'out(1)=points(1)%value'
    path.write_text('module reads\nimplicit none\ntype record\ninteger::index\nreal(8)::value\nend type\n'
        'type(record),allocatable::points(:)\ncontains\nsubroutine step(out)\nreal(8),intent(out)::out(:)\n'
        'real(8)::temporary\n'+'temporary=0\n'*500+statement+'\nend subroutine\nend module\n')
    analysis=SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine=analysis.routines['reads::step']
    node,=[node for node in walk(routine.execution) if type(node).__name__=='Data_Ref']
    return analysis,routine,node


@pytest.mark.parametrize('lhs_index',[False,True])
def test_indexed_position_matches_fresh_tree_walk_without_rewalking_siblings(tmp_path,lhs_index):
    visits=[]
    analysis,routine,node=fixture(tmp_path/'sealed',lhs_index=lhs_index)
    result=_read_position(analysis,routine,node,lambda:visits.append(True))
    assert len(visits)<100
    fresh,owner,original=fixture(tmp_path/'fresh',sealed=False,lhs_index=lhs_index)
    census=[]
    expected=_read_position(fresh,owner,original,lambda:census.append(True))
    assert len(census)>1000
    assert [(type(parent).__name__,branch) for parent,branch in result]==[
        (type(parent).__name__,branch) for parent,branch in expected]


def test_missing_parent_index_keeps_fresh_occurrence_verdict(tmp_path,monkeypatch):
    analysis,routine,node=fixture(tmp_path)
    monkeypatch.setattr(type(analysis._source_session),'execution_overlaps',lambda *args:None)
    visits=[]
    position=_read_position(analysis,routine,node,lambda:visits.append(True))
    assert position
    assert len(visits)>1000


def test_cached_parent_index_rejects_removed_current_child(tmp_path):
    analysis,routine,node=fixture(tmp_path)
    _read_position(analysis,routine,node,lambda:None)
    parent=analysis._source_session.lexical_parent(node)
    children=tuple(getattr(parent,'items',()))
    assert any(item is node for item in children)
    object.__setattr__(parent,'items',tuple(item for item in children if item is not node))
    with pytest.raises(CompilationError):
        _read_position(analysis,routine,node,lambda:None)


@pytest.mark.parametrize('sealed',[False,True])
def test_original_function_helper_keeps_existing_occurrence_walk(tmp_path,sealed):
    path=tmp_path/'function.f90'
    path.write_text('module helpers\nimplicit none\ntype record\nreal(8)::value\nend type\n'
        'type(record),allocatable::points(:)\ncontains\nfunction value() result(out)\n'
        'real(8)::out\nout=points(1)%value\nend function\nend module\n')
    analysis=SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine=analysis.numerical_helpers['helpers::value']
    assert 'helpers::value' not in analysis.routines
    node,=[node for node in walk(routine.execution) if type(node).__name__=='Data_Ref']
    position=_read_position(analysis,routine,node,lambda:None)
    assert any(type(parent).__name__=='Assignment_Stmt' and branch==2 for parent,branch in position)


def leaf_fixture(directory, *, sealed):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'leaf.f90'
    path.write_text('module leaf_reads\ncontains\nsubroutine step(q,out)\n'
        'real(8),allocatable,intent(in)::q(:)\nreal(8),intent(out)::out(:)\n'
        'out(1)=q(1)\nend subroutine\nend module\n')
    analysis = SourceEffects([path])
    if sealed:
        analysis.seal_session()
    routine = analysis.routines['leaf_reads::step']
    node, = [item for item in walk(routine.execution) if type(item).__name__ == 'Name' and str(item).lower() == 'q']
    return analysis, routine, node


def test_unindexed_original_leaf_retains_exact_fresh_read_position(tmp_path):
    analysis, routine, node = leaf_fixture(tmp_path / 'sealed', sealed=True)
    assert analysis._source_session.execution_nodes(analysis, routine).get(id(node)) is node
    with pytest.raises(CompilationError, match='lexical parent requires an original source node'):
        analysis._source_session.lexical_parent(node)
    position = _read_position(analysis, routine, node, lambda: None)
    fresh, owner, original = leaf_fixture(tmp_path / 'fresh', sealed=False)
    expected = _read_position(fresh, owner, original, lambda: None)
    assert [(type(parent).__name__, branch) for parent, branch in position] == [
        (type(parent).__name__, branch) for parent, branch in expected]


def test_unindexed_leaf_fallback_rejects_removed_current_occurrence(tmp_path):
    from fparser.two import Fortran2003 as F

    analysis, routine, node = leaf_fixture(tmp_path, sealed=True)
    assert analysis._source_session.execution_nodes(analysis, routine).get(id(node)) is node
    parent, = [item for item in walk(routine.execution)
               if type(item).__name__ == 'Part_Ref' and item.items[0] is node]
    object.__setattr__(parent, 'items', (F.Name('other'), parent.items[1]))
    with pytest.raises(CompilationError, match='exact original executable occurrence'):
        _read_position(analysis, routine, node, lambda: None)
