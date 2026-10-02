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
    assert f"access model: {'conservative' if indirect else 'exact'}" in format_plan(plan)
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
