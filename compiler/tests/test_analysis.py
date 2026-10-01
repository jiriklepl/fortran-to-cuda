"""Dependence mathematics and scalar lifetime checks, independent of emission."""

from itertools import product

import islpy as isl
import pytest

from compiler.analysis import build_execution_plan, ordered_conflicts
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

LOCATION = SourceLocation("analysis_fixture.f90", 12)


def integer(value):
    return Literal(str(value), ScalarType.INTEGER)


def function_with_body(body, symbols, parameters):
    return FunctionIR("kernel", "kernel_module", tuple(parameters), tuple(symbols), Block(tuple(body)), LOCATION.path)


def one_loop(assignments, *, after=(), before=(), upper=None):
    arr = Symbol(0, "arr", ScalarType.REAL, 1, "inout", True)
    n = Symbol(1, "n", ScalarType.INTEGER, intent="in", parameter=True)
    i = Symbol(2, "i", ScalarType.INTEGER)
    temp = Symbol(3, "temp", ScalarType.REAL)
    body = assignments(arr, i, temp)
    loop = Loop(i, integer(1), upper or Reference(n), Block(tuple(body)), LOCATION)
    return function_with_body((*before, loop, *after), (arr, n, i, temp), (arr, n))


@pytest.mark.parametrize(("offset", "kind"), [(-1, "RAW"), (1, "WAR")])
def test_neighbor_recurrences_report_relation_and_witness(offset, kind):
    def body(arr, i, temp):
        return [
            Assignment(
                ArrayAccess(arr, (Reference(i),)),
                ArrayAccess(arr, (Binary("+", Reference(i), integer(offset)),)),
                LOCATION,
            )
        ]

    with pytest.raises(CompilationError, match=kind) as error:
        build_execution_plan(one_loop(body))
    assert "witness" in str(error.value)
    assert "analysis_fixture.f90:12" in str(error.value)
    assert "arr" in str(error.value)


def test_constant_element_writes_are_not_parallel():
    with pytest.raises(CompilationError, match="WAW"):
        build_execution_plan(
            one_loop(lambda arr, i, temp: [Assignment(ArrayAccess(arr, (integer(1),)), integer(0), LOCATION)])
        )


def test_same_iteration_order_and_private_temporaries_are_allowed():
    def body(arr, i, temp):
        return [
            Assignment(Reference(temp), ArrayAccess(arr, (Reference(i),)), LOCATION),
            Assignment(ArrayAccess(arr, (Reference(i),)), Binary("*", Reference(temp), integer(2)), LOCATION),
            Assignment(ArrayAccess(arr, (Reference(i),)), ArrayAccess(arr, (Reference(i),)), LOCATION),
        ]

    region = build_execution_plan(one_loop(body)).regions[0]
    assert [symbol.name for symbol in region.private_symbols] == ["temp"]
    assert "S1" in region.report.raw
    assert "S2" in region.report.raw


def test_scalar_reduction_is_not_privatized():
    def body(arr, i, temp):
        return [Assignment(Reference(temp), Binary("+", Reference(temp), ArrayAccess(arr, (Reference(i),))), LOCATION)]

    with pytest.raises(CompilationError, match="read before its per-iteration definition"):
        build_execution_plan(one_loop(body))


def test_scalar_liveout_and_final_induction_value_are_rejected():
    for symbol in (Symbol(3, "temp", ScalarType.REAL), Symbol(2, "i", ScalarType.INTEGER)):
        after = Assignment(Reference(Symbol(4, "result", ScalarType.REAL)), Reference(symbol), LOCATION)
        with pytest.raises(CompilationError, match="live after"):
            build_execution_plan(
                one_loop(lambda arr, i, temp: [Assignment(Reference(temp), integer(2), LOCATION)], after=(after,))
            )


def test_redefinition_after_region_kills_previous_private_value():
    temp = Symbol(3, "temp", ScalarType.REAL)
    result = Symbol(4, "result", ScalarType.REAL)
    after = (
        Assignment(Reference(temp), integer(3), LOCATION),
        Assignment(Reference(result), Reference(temp), LOCATION),
    )
    plan = build_execution_plan(
        one_loop(lambda arr, i, temp: [Assignment(Reference(temp), integer(2), LOCATION)], after=after)
    )
    assert len(plan.regions) == 1


def test_inner_dimension_dependence_is_rejected():
    arr = Symbol(0, "arr", ScalarType.REAL, 2, "inout", True)
    n = Symbol(1, "n", ScalarType.INTEGER, intent="in", parameter=True)
    j = Symbol(2, "j", ScalarType.INTEGER)
    i = Symbol(3, "i", ScalarType.INTEGER)
    assignment = Assignment(
        ArrayAccess(arr, (Reference(i), Reference(j))),
        ArrayAccess(arr, (Binary("-", Reference(i), integer(1)), Reference(j))),
        LOCATION,
    )
    inner = Loop(i, integer(2), Reference(n), Block((assignment,)), LOCATION)
    outer = Loop(j, integer(1), Reference(n), Block((inner,)), LOCATION)
    with pytest.raises(CompilationError, match="RAW"):
        build_execution_plan(function_with_body((outer,), (arr, n, j, i), (arr, n)))


def test_indirect_access_and_nonrectangular_bound_are_rejected():
    def body(arr, i, temp):
        return [Assignment(ArrayAccess(arr, (ArrayAccess(arr, (Reference(i),)),)), integer(2), LOCATION)]

    with pytest.raises(CompilationError, match="non-affine"):
        build_execution_plan(one_loop(body))
    i = Symbol(2, "i", ScalarType.INTEGER)
    with pytest.raises(CompilationError, match="invariant affine integer"):
        build_execution_plan(one_loop(lambda arr, i, temp: [], upper=Reference(i)))


def test_affine_host_integer_and_size_bounds():
    arr = Symbol(0, "arr", ScalarType.REAL, 1, "inout", True)
    n = Symbol(1, "n", ScalarType.INTEGER, intent="in", parameter=True)
    i = Symbol(2, "i", ScalarType.INTEGER)
    limit = Symbol(3, "limit", ScalarType.INTEGER)
    host = Assignment(Reference(limit), Binary("+", Reference(n), integer(1)), LOCATION)
    stmt = Assignment(ArrayAccess(arr, (Reference(i),)), integer(1), LOCATION)
    loop = Loop(i, integer(1), Reference(limit), Block((stmt,)), LOCATION)
    report = build_execution_plan(function_with_body((host, loop), (arr, n, i, limit), (arr, n))).regions[0].report
    assert "p1" in report.domain


# Each specification describes (storage name, subscript offset); None is constant 0.
ACCESS_CASES = [
    ([[("A", 0)]], [[("A", 0)]]),
    ([[("A", -1)]], [[("A", 0)]]),
    ([[("A", 1)]], [[("A", 0)]]),
    ([[]], [[("A", None)]]),
    ([[("B", 1)]], [[("A", 0)]]),
    ([[("A", 0)], [("T", 0)]], [[("T", 0)], [("A", 0)]]),
    ([[], [("A", -1)]], [[("A", 0)], [("B", 0)]]),
]


@pytest.mark.parametrize(("reads_spec", "writes_spec"), ACCESS_CASES)
@pytest.mark.parametrize("n", [0, 1, 2, 4])
def test_isl_conflict_relations_match_brute_force(reads_spec, writes_spec, n):
    def memory_accesses(spec):
        clauses = []
        for statement, accesses in enumerate(spec):
            for storage, offset in accesses:
                index = "0" if offset is None else f"i + {offset}"
                clauses.append(f"S{statement}[i] -> {storage}[{index}] : 0 <= i < n")
        return isl.UnionMap.read_from_str(isl.DEFAULT_CONTEXT, "[n] -> { " + "; ".join(clauses) + " }")

    schedule = isl.UnionMap.read_from_str(
        isl.DEFAULT_CONTEXT,
        "[n] -> { " + "; ".join(f"S{s}[i] -> T[i,{s}] : 0 <= i < n" for s in range(len(reads_spec))) + " }",
    )
    actual = ordered_conflicts(memory_accesses(reads_spec), memory_accesses(writes_spec), schedule)
    instances = [(s, i) for i in range(n) for s in range(len(reads_spec))]

    def accesses(spec, instance):
        s, i = instance
        return {(storage, 0 if offset is None else i + offset) for storage, offset in spec[s]}

    parameter = isl.Set.read_from_str(isl.DEFAULT_CONTEXT, f"[n] -> {{ : n = {n} }}")
    for kind, source, sink in (
        ("RAW", writes_spec, reads_spec),
        ("WAR", reads_spec, writes_spec),
        ("WAW", writes_spec, writes_spec),
    ):
        expected_pairs = [
            (a, b)
            for a, b in product(instances, repeat=2)
            if (a[1], a[0]) < (b[1], b[0]) and accesses(source, a) & accesses(sink, b)
        ]
        expected = isl.UnionMap.read_from_str(
            isl.DEFAULT_CONTEXT,
            "[n] -> { " + "; ".join(f"S{a[0]}[{a[1]}] -> S{b[0]}[{b[1]}] : n = {n}" for a, b in expected_pairs) + " }",
        )
        assert actual[kind].intersect_params(parameter).is_equal(expected), kind


@pytest.mark.parametrize("bound_source", ["private", "outer_iterator", "inner_iterator"])
def test_inner_bounds_cannot_change_even_with_a_prior_host_definition(bound_source):
    arr = Symbol(0, "arr", ScalarType.REAL, 2, "inout", True)
    i = Symbol(1, "i", ScalarType.INTEGER)
    j = Symbol(2, "j", ScalarType.INTEGER)
    limit = Symbol(3, "limit", ScalarType.INTEGER)
    symbol = {"private": limit, "outer_iterator": j, "inner_iterator": i}[bound_source]
    host = Assignment(Reference(symbol), integer(3), LOCATION)
    body = [Assignment(ArrayAccess(arr, (Reference(i), Reference(j))), integer(1), LOCATION)]
    if bound_source == "private":
        body.insert(0, Assignment(Reference(limit), integer(1), LOCATION))
    inner = Loop(i, integer(1), Reference(symbol), Block(tuple(body)), LOCATION)
    outer = Loop(j, integer(1), integer(3), Block((inner,)), LOCATION)
    with pytest.raises(CompilationError, match="invariant rectangular bounds"):
        build_execution_plan(function_with_body((host, outer), (arr, i, j, limit), (arr,)))


def test_nonaffine_host_integer_is_a_frozen_invariant_parameter():
    arr = Symbol(0, "arr", ScalarType.REAL, 1, "inout", True)
    n = Symbol(1, "n", ScalarType.INTEGER, intent="in", parameter=True)
    i = Symbol(2, "i", ScalarType.INTEGER)
    limit = Symbol(3, "limit", ScalarType.INTEGER)
    host = Assignment(Reference(limit), Binary("/", Reference(n), integer(2)), LOCATION)
    stmt = Assignment(ArrayAccess(arr, (Reference(i),)), integer(1), LOCATION)
    loop = Loop(i, integer(1), Reference(limit), Block((stmt,)), LOCATION)
    function = function_with_body((host, loop), (arr, n, i, limit), (arr, n))
    region = build_execution_plan(function).regions[0]
    assert "h0_3" in region.report.domain
    assert limit in region.captured_symbols
