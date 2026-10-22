"""Graph compute estimates exclude address normalization and memory stores."""
from dataclasses import replace

from compiler.driver.options import CompilerOptions
from compiler.driver.pipeline import prepare_function
from compiler.frontend import lower_source
from compiler.ir import Binary, Block, Literal, Loop, Reference, ScalarType, Symbol
from compiler.offload.analysis import _compute_arithmetic_operations
from compiler.offload.graph_costs import graph_work_proxy

SOURCE = '''module work_proxy
implicit none
contains
subroutine evaluate(n,lb,a,b,scale)
integer,intent(in)::n,lb
real(8),intent(in)::a(:),scale
real(8),intent(out)::b(:)
integer::i
do i=1,n
  b(i-lb+1)=a(i-lb+1)*scale+1.0_8
end do
end subroutine
end module'''


def region(source=SOURCE):
    _, plan = prepare_function(lower_source(source, 'evaluate', source_name='real_proxy.f90'),
                              options=CompilerOptions(gpu_policy='auto'))
    return plan.regions[0]


def test_array_rebasing_does_not_inflate_real_arithmetic():
    rebased = region()
    ordinary, reason = _compute_arithmetic_operations(rebased.body)
    assert reason is None
    assert ordinary == 2
    assert graph_work_proxy(rebased) == (ordinary, ())
    direct = region(SOURCE.replace('i-lb+1', 'i'))
    assert graph_work_proxy(direct) == graph_work_proxy(rebased)


def test_assignment_only_has_memory_work_and_zero_compute_flops():
    copied = region(SOURCE.replace('a(i-lb+1)*scale+1.0_8', 'a(i-lb+1)'))
    assert graph_work_proxy(copied) == (0, ())
    filled = region(SOURCE.replace('a(i-lb+1)*scale+1.0_8', '0.0_8'))
    assert graph_work_proxy(filled) == (0, ())


def test_integer_arithmetic_and_comparisons_have_no_float_work():
    integer = region(SOURCE.replace('real(8)', 'integer').replace('1.0_8', '1'))
    assert graph_work_proxy(integer) == (0, ())
    conditional = region(SOURCE.replace('  b(i-lb+1)=', '  if(i-lb>0) b(i-lb+1)='))
    assert graph_work_proxy(conditional) == (2, ())


def test_real_intrinsic_and_division_weights_remain_explicit_heuristics():
    expensive = region(SOURCE.replace('a(i-lb+1)*scale+1.0_8',
                                     'sqrt(a(i-lb+1))+cos(a(i-lb+1))/scale'))
    assert graph_work_proxy(expensive) == (8 + 64 + 4 + 1, ())
    constant = region(SOURCE.replace('a(i-lb+1)*scale+1.0_8', 'sqrt(4.0_8)+2.0_8'))
    assert graph_work_proxy(constant) == (0, ())


def test_dynamic_retained_loop_stays_explicitly_unpriced():
    base = region()
    iterator = Symbol(100000, 'retained_j', ScalarType.INTEGER)
    bound = Symbol(100001, 'retained_n', ScalarType.INTEGER)
    loop = Loop(iterator, Literal('1', ScalarType.INTEGER), Reference(bound),
                base.body, base.body.statements[0].location)
    nested = replace(base, body=Block((loop,)))
    assert graph_work_proxy(nested) == (2, ('retained_j',))
    constant_loop = replace(loop, upper=Binary('+', Literal('1', ScalarType.INTEGER),
                                               Literal('2', ScalarType.INTEGER)))
    assert graph_work_proxy(replace(base, body=Block((constant_loop,)))) == (6, ())
