"""Structured dependence queries using only the baseline loop language."""

from dataclasses import replace

import pytest

from compiler.analysis import (
    Affine,
    ParallelizationError,
    build_execution_plan,
    dependence,
    format_plan,
    prove_region,
)
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    CompilationError,
    FunctionIR,
    If,
    Literal,
    Loop,
    Reference,
    ScalarType,
    SourceLocation,
    Symbol,
)

LOCATION = SourceLocation("proof_fixture.f90", 12)
A = Symbol(0, "a", ScalarType.INTEGER, rank=1, intent="inout", parameter=True)
SOURCE = Symbol(1, "source", ScalarType.INTEGER, rank=1, intent="in", parameter=True)
INDICES = Symbol(2, "indices", ScalarType.INTEGER, rank=1, intent="in", parameter=True)
N = Symbol(3, "n", ScalarType.INTEGER, intent="in", parameter=True)
ITERATOR = Symbol(4, "i", ScalarType.INTEGER)
TEMP = Symbol(5, "temp", ScalarType.INTEGER)
ONE = Literal("1", ScalarType.INTEGER)
POINT = ArrayAccess(A, (Reference(ITERATOR),))


def function_with_assignment(target, value):
    assignment = Assignment(target, value, LOCATION)
    loop = Loop(ITERATOR, ONE, Reference(N), Block((assignment,)), LOCATION)
    return FunctionIR(
        "entry",
        "proof_case",
        (A, SOURCE, INDICES, N),
        (A, SOURCE, INDICES, N, ITERATOR, TEMP),
        Block((loop,)),
        LOCATION.path,
    )


def query(function):
    return prove_region(0, function.body.statements[0], {N: Affine(terms=(("p3", 1),))}, {N}, Block(()))


@pytest.mark.parametrize("guarded", [False, True])
def test_integer_guard_proves_periodic_halo_copy_only_on_disjoint_domain(guarded):
    three = Literal("3", ScalarType.INTEGER)
    target = ArrayAccess(A, (Binary("+", Reference(N), Reference(ITERATOR)),))
    assignment = Assignment(target, POINT, LOCATION)
    body = (If(Binary(">=", Reference(N), three), Block((assignment,)), Block(()), LOCATION)
            if guarded else assignment)
    loop = Loop(ITERATOR, ONE, three, Block((body,)), LOCATION)
    function = replace(function_with_assignment(POINT, ONE), body=Block((loop,)))
    proof = query(function)
    assert proof.proven is guarded
    if guarded:
        assert not proof.region.report.conservative


def test_integer_guard_does_not_hide_conflicts_in_else_arm():
    three = Literal("3", ScalarType.INTEGER)
    safe = Assignment(ArrayAccess(A, (Binary("+", Reference(N), Reference(ITERATOR)),)), POINT, LOCATION)
    unsafe = Assignment(POINT, ArrayAccess(A, (Binary("-", Reference(ITERATOR), ONE),)), LOCATION)
    branch = If(Binary(">=", Reference(N), three), Block((safe,)), Block((unsafe,)), LOCATION)
    loop = Loop(ITERATOR, ONE, three, Block((branch,)), LOCATION)
    assert not query(replace(function_with_assignment(POINT, ONE), body=Block((loop,)))).proven


def test_comment_before_if_keeps_original_branch_header():
    from compiler.frontend import lower_source

    source = """module comment_case
contains
subroutine step(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
! Original branch explanation.
if (a(i)>0) then
  a(i)=a(i)*2
else
  a(i)=0
endif
enddo
end subroutine
end module
"""
    function = lower_source(source, "comment_case::step", source_name="comment_case.f90")
    assert isinstance(function.body.statements[0].body.statements[0], If)


@pytest.mark.parametrize("indirect", [False, True], ids=["exact", "conservative"])
def test_success_preserves_model_precision_and_existing_entry_points(indirect):
    value = ArrayAccess(SOURCE, (ArrayAccess(INDICES, (Reference(ITERATOR),)),)) if indirect else POINT
    function = function_with_assignment(POINT, value)
    result = query(function)
    assert result.proven
    assert result.failure is None
    plan = build_execution_plan(function)
    assert result.region == plan.regions[0]
    assert result.region.report.conservative is indirect
    assert f"dependence model: {'conservative' if indirect else 'exact'}" in format_plan(plan)
    assert dependence.build_execution_plan(function) == plan
    assert dependence.format_plan(plan) == format_plan(plan)


@pytest.mark.parametrize(
    ("target", "value", "kind", "conservative"),
    [
        (POINT, ArrayAccess(A, (Binary("-", Reference(ITERATOR), ONE),)), "RAW", False),
        (POINT, ArrayAccess(A, (Binary("+", Reference(ITERATOR), ONE),)), "WAR", False),
        (ArrayAccess(A, (ONE,)), ONE, "WAW", False),
        (ArrayAccess(A, (ArrayAccess(INDICES, (Reference(ITERATOR),)),)), ONE, "WAW", True),
    ],
    ids=["exact-raw", "exact-war", "exact-waw", "conservative-waw"],
)
def test_conflicts_return_evidence_and_strict_compilation_still_rejects(target, value, kind, conservative):
    function = function_with_assignment(target, value)
    result = query(function)
    assert not result.proven
    assert result.region is None
    failure = result.failure
    assert kind in failure.reason
    assert failure.location == LOCATION
    assert failure.witness
    assert failure.witness in failure.reason
    assert failure.conservative is conservative
    with pytest.raises(ParallelizationError) as error:
        build_execution_plan(function)
    assert isinstance(error.value, CompilationError)
    assert error.value.failure == failure
    assert str(LOCATION) in str(error.value)


@pytest.mark.parametrize("invalid", ["undefined-read", "zero-stride", "iterator-write"])
def test_compilation_errors_propagate_instead_of_becoming_proof_results(invalid):
    target = Reference(ITERATOR) if invalid == "iterator-write" else POINT
    value = Reference(TEMP) if invalid == "undefined-read" else ONE
    function = function_with_assignment(target, value)
    if invalid == "zero-stride":
        function = replace(function, body=Block((replace(function.body.statements[0], step=0),)))
    with pytest.raises(CompilationError) as error:
        query(function)
    assert not isinstance(error.value, ParallelizationError)


def test_internal_errors_propagate(monkeypatch):
    def broken_region(*args, **kwargs):
        raise RuntimeError("proof engine failed")

    monkeypatch.setattr(dependence, "_region", broken_region)
    with pytest.raises(RuntimeError, match="proof engine failed"):
        query(function_with_assignment(POINT, ONE))


def conditional_function(target, *, predicate=None):
    function = function_with_assignment(target, ONE)
    loop = function.body.statements[0]
    condition = predicate or Binary("==", Reference(ITERATOR), ONE)
    branch = If(condition, loop.body, Block(()), LOCATION)
    return replace(function, body=Block((replace(loop, body=Block((branch,))),)))


def test_affine_branch_success_reports_exact_dependence_model():
    function = conditional_function(POINT)
    proof = query(function)
    assert proof.proven
    assert not proof.region.report.conservative
    assert "dependence model: exact" in format_plan(build_execution_plan(function))


def test_affine_branch_excludes_a_conflict_that_cannot_execute():
    # Only i=1 writes. The exact predicate eliminates the spurious WAW witness.
    function = conditional_function(ArrayAccess(A, (ONE,)))
    proof = query(function)
    assert proof.proven
    assert not proof.region.report.conservative


def test_branch_predicate_reads_remain_in_conservative_dependence_model():
    predicate = Binary(">", ArrayAccess(A, (Binary("-", Reference(ITERATOR), ONE),)), ONE)
    proof = query(conditional_function(POINT, predicate=predicate))
    assert not proof.proven
    assert proof.failure.conservative
    assert "RAW possible loop-carried conflict" in proof.failure.reason
    assert proof.failure.witness
