"""Only issued original line consumers admit relational constant-vector reads."""
from copy import copy
from dataclasses import replace

import pytest

from compiler.frontend.line_snapshot import prove_line_snapshot
from compiler.frontend.lowering import lower_file, lower_line_snapshot_consumer
from compiler.ir import Assignment, Binary, CompilationError, If, IntrinsicCall, Literal, Loop, Reference, ScalarType, walk_expr
from compiler.tests.test_line_snapshot import fixture, source


def blocks(block):
    for item in block.statements:
        yield item
        if isinstance(item, Loop):
            yield from blocks(item.body)
        elif isinstance(item, If):
            yield from blocks(item.then_body)
            yield from blocks(item.else_body)


def expressions(block):
    for item in blocks(block):
        for value in ([item.value] if isinstance(item, Assignment) else [item.condition] if isinstance(item, If) else [item.lower,item.upper]):
            yield from walk_expr(value)


def lowered(tmp_path, text=None):
    path,analysis,routine,nodes=fixture(tmp_path,text)
    proof=prove_line_snapshot(analysis,routine,nodes)
    before=path.read_bytes()
    result=lower_line_snapshot_consumer(proof)
    assert path.read_bytes()==before and proof.validate() is proof
    return result


@pytest.mark.parametrize('axis',[1,2,3])
@pytest.mark.parametrize('kind',[4,8])
def test_original_inner_loop_and_typed_selected_values_remain_ordered(tmp_path,axis,kind):
    result=lowered(tmp_path,source(axis,kind))
    assert len(result.outer_loops)==2
    consumer,=result.consumer.statements
    assert isinstance(consumer,Loop)
    loops=[item for item in blocks(result.consumer) if isinstance(item,Loop)]
    assert [item.iterator.name for item in loops]=={1:['i','ii'],2:['j','jj'],3:['k','kk']}[axis]
    assert len(result.lookup_obligations)==2
    for obligation in result.lookup_obligations:
        assert obligation.index_bounds==(-3,3)
        assert len(obligation.original_loops)==4
        assert obligation.public()['execution_authorized'] is False
        assert 'terminal value' in obligation.public()['required_prework']
    all_expressions=list(expressions(result.consumer))
    assert not any(isinstance(x,IntrinsicCall) and x.name=='merge' for x in all_expressions)
    threshold,=[x for x in all_expressions if isinstance(x,Literal) and x.value=='0.74']
    assert threshold.dtype is ScalarType.REAL32
    values=[s for s in result.symbols if s.name=='fort_line_constant_value']
    assert len(values)==2 and all(s.dtype is (ScalarType.REAL if kind==8 else ScalarType.REAL32) for s in values)
    updates=[s for s in blocks(result.consumer) if isinstance(s,Assignment) and s.target.symbol.name in {'s','q'}]
    assert [s.target.symbol.name for s in updates]==['s','q','s','q']
    assert isinstance(updates[2].value,Binary) and updates[2].value.operator=='+'
    assert isinstance(updates[2].value.right,Binary) and updates[2].value.right.operator=='*'
    assert isinstance(updates[3].value,Binary) and updates[3].value.operator=='+'
    assert result.scratch.rank==1 and result.scratch not in result.parameters
    assert len({s.id for s in result.symbols})==len(result.symbols)


def test_renamed_shifted_constant_bounds_are_source_derived(tmp_path):
    text=source().replace('w(-3:3)','coeff(5:11)').replace('w(ii-i)','coeff(ii-i+8)')
    result=lowered(tmp_path,text)
    assert [x.index_bounds for x in result.lookup_obligations]==[(5,11),(5,11)]
    assert all(x.array_binding.name=='coeff' for x in result.lookup_obligations)
    assert all(str(x.original_access).lower()=='coeff(ii - i + 8)' for x in result.lookup_obligations)


@pytest.mark.parametrize('change',[
    ('w(ii-i)','w(ii-i+1)'),
    ('w(ii-i)','w(ii-i-1)'),
    ('w(ii-i)','w(ii)'),
    ('w(ii-i)','w(ii-j)'),
    ('do ii=i-3,i+3','do ii=i-3,j+3'),
    ('do ii=i-3,i+3','do ii=i+3,i-3'),
    ('do ii=i-3,i+3','do ii=i-128,i+128'),
])
def test_unproved_or_out_of_vector_relations_reject(tmp_path,change):
    path,analysis,routine,nodes=fixture(tmp_path,source().replace(*change))
    proof=prove_line_snapshot(analysis,routine,nodes)
    with pytest.raises(CompilationError,match='line constant lookup'):
        lower_line_snapshot_consumer(proof)


def test_original_intermediate_and_terminal_checks_remain_exposed_not_folded(tmp_path):
    text=source().replace('do ii=i-3,i+3','do ii=(i+2147483647)-2147483647-3,i+3')
    result=lowered(tmp_path,text)
    inner=[x for x in blocks(result.consumer) if isinstance(x,Loop)][1]
    assert isinstance(inner.lower,Binary)
    constants=[x.value for x in walk_expr(inner.lower) if isinstance(x,Literal)]
    assert constants==['2147483647','2147483647','3']
    for obligation in result.lookup_obligations:
        assert '2147483647' in obligation.public()['original_loop_domains'][-1]['lower']
        assert 'terminal' in obligation.public()['required_prework']


def test_plain_frontend_still_rejects_dynamic_constant_index_loop(tmp_path):
    path=tmp_path/'ordinary.f90'
    path.write_text('''module ordinary
contains
subroutine f(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i,ii
real(8),parameter::w(-1:1)=[1.d0,2.d0,3.d0]
do i=2,n-1
 do ii=i-1,i+1
  a(i)=a(i)+w(ii-i)
 enddo
enddo
end subroutine
end module
''')
    with pytest.raises(CompilationError,match='bounded constant ranges'):
        lower_file(path,'f')


def test_lookup_needs_registered_exact_original_line_proof(tmp_path):
    path,analysis,routine,nodes=fixture(tmp_path)
    proof=prove_line_snapshot(analysis,routine,nodes)
    for forged in [object(),copy(proof),replace(proof)]:
        with pytest.raises(CompilationError):lower_line_snapshot_consumer(forged)
    path.write_text(path.read_text()+'! changed\n')
    with pytest.raises(CompilationError):lower_line_snapshot_consumer(proof)
