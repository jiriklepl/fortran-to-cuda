"""Known work/span systems and rejection of unidentifiable native costs."""

from dataclasses import replace

import pytest

from compiler.ir import ScalarType
from compiler.offload.compute_dependencies import ComputeDependencies, ComputeOperation
from compiler.offload.cpu_dependency_model import (
    Coefficient,
    DependencyModelError,
    evaluate_work_span,
    fit_basis_pair,
)


def chains(width, steps, *, family="multiply", join=True):
    operations, ends = [], []
    for _ in range(width):
        parent = ()
        for _ in range(steps):
            operations.append(ComputeOperation(family, ScalarType.REAL, (parent, ()), False, False))
            parent = (len(operations) - 1,)
        ends.append(parent)
    if join:
        parent = ends[0]
        for other in ends[1:]:
            operations.append(ComputeOperation("add", ScalarType.REAL, (parent, other), False, False))
            parent = (len(operations) - 1,)
        ends = [parent]
    return ComputeDependencies(True, operations=tuple(operations), outputs=tuple(ends),
                               floating_dtypes=(ScalarType.REAL,))


def test_recovers_distinct_work_and_span_from_predeclared_basis():
    single, wide = chains(1, 32), chains(4, 32)
    actual = Coefficient(2e-9, 5e-9)
    single_seconds = evaluate_work_span(single, {"ordinary": actual}, precision_bits=64).seconds_per_item
    wide_seconds = evaluate_work_span(wide, {"ordinary": actual}, precision_bits=64).seconds_per_item
    fitted = fit_basis_pair(single, wide, single_seconds, wide_seconds,
                            family="ordinary", known={}, precision_bits=64)
    assert fitted.work_seconds == pytest.approx(actual.work_seconds)
    assert fitted.span_seconds == pytest.approx(actual.span_seconds)
    # An independent diamond with a different width/depth uses both estimates.
    diamond = chains(3, 7)
    assert evaluate_work_span(diamond, {"ordinary": fitted}, precision_bits=64).seconds_per_item == pytest.approx(46e-9)


def test_identifies_primitive_after_removing_known_ordinary_work():
    known = {"ordinary": Coefficient(1e-9, 2e-9)}
    single, wide = chains(1, 32, family="sqrt"), chains(4, 32, family="sqrt")
    expected = Coefficient(3e-9, 9e-9)
    model = {**known, "sqrt": expected}
    a = evaluate_work_span(single, model, precision_bits=64).seconds_per_item
    b = evaluate_work_span(wide, model, precision_bits=64).seconds_per_item
    fitted = fit_basis_pair(single, wide, a, b, family="sqrt", known=known, precision_bits=64)
    assert fitted.work_seconds == pytest.approx(3e-9)
    assert fitted.span_seconds == pytest.approx(9e-9)


def test_division_is_not_double_charged_and_uniform_flags_do_not_hoist():
    graph = chains(1, 3, family="divide_dynamic")
    graph = replace(graph, operations=tuple(replace(op, constant=True, invariant=True) for op in graph.operations))
    result = evaluate_work_span(graph, {"divide_dynamic": Coefficient(2e-9, 4e-9)}, precision_bits=64)
    assert result.work_seconds == pytest.approx(6e-9)
    assert result.span_seconds == pytest.approx(12e-9)


@pytest.mark.parametrize(("single_seconds", "wide_seconds"), [(32e-9, 1e-9), (32e-9, 1000e-9)])
def test_rejects_incompatible_basis_without_switching_fitting_roles(single_seconds, wide_seconds):
    with pytest.raises(DependencyModelError, match="cone|both work and span"):
        fit_basis_pair(chains(1, 32), chains(4, 32), single_seconds, wide_seconds,
                       family="ordinary", known={}, precision_bits=64)


def test_negative_primitive_residual_is_rejected_instead_of_clamped():
    with pytest.raises(DependencyModelError, match="positive"):
        fit_basis_pair(chains(1, 32, family="cos"), chains(4, 32, family="cos"),
                       1e-6, 1e-9, family="cos", known={"ordinary": Coefficient(1e-9, 2e-9)}, precision_bits=64)


@pytest.mark.parametrize("value", [True, 0, -1, float("inf"), float("nan"), 10**1000])
def test_invalid_coefficients_never_become_costs(value):
    with pytest.raises(DependencyModelError):
        Coefficient(value, 1.0)


def test_missing_primitive_mixed_precision_and_unavailable_graph_remain_unpriced():
    graph = chains(1, 2, family="acos")
    with pytest.raises(DependencyModelError, match="missing"):
        evaluate_work_span(graph, {}, precision_bits=64)
    with pytest.raises(DependencyModelError, match="precision"):
        evaluate_work_span(graph, {"acos": Coefficient(1.0, 2.0)}, precision_bits=32)
    with pytest.raises(DependencyModelError, match="unavailable"):
        evaluate_work_span(ComputeDependencies(False, "conditional"), {}, precision_bits=64)


def test_cyclic_or_forward_dependencies_cannot_be_priced():
    graph = chains(1, 2)
    graph = replace(graph, operations=(replace(graph.operations[0], operands=((1,),)), *graph.operations[1:]))
    with pytest.raises(DependencyModelError, match="ordering"):
        evaluate_work_span(graph, {"ordinary": Coefficient(1.0, 2.0)}, precision_bits=64)


@pytest.mark.parametrize("changes", [{"schema_version": True}, {"available": "yes"}, {"operations": (None,)}])
def test_malformed_graph_cannot_become_an_estimate(changes):
    with pytest.raises(DependencyModelError):
        evaluate_work_span(replace(chains(1, 2), **changes), {"ordinary": Coefficient(1.0, 2.0)}, precision_bits=64)


def test_invalid_known_coefficient_is_a_checked_unavailable_result():
    with pytest.raises(DependencyModelError, match="known"):
        fit_basis_pair(chains(1, 32, family="cos"), chains(4, 32, family="cos"),
                       1e-6, 2e-6, family="cos", known={"ordinary": None}, precision_bits=64)


@pytest.mark.parametrize("types", [(), (ScalarType.REAL32, ScalarType.REAL),
    (ScalarType.REAL, ScalarType.REAL), (ScalarType.INTEGER,), [ScalarType.REAL], ([],)])
def test_precision_metadata_cannot_hide_casts_or_missing_provenance(types):
    graph = replace(chains(1, 2), floating_dtypes=types)
    with pytest.raises(DependencyModelError, match="precision"):
        evaluate_work_span(graph, {"ordinary": Coefficient(1.0, 2.0)}, precision_bits=64)


def test_empty_integer_graph_has_zero_work_without_floating_metadata():
    result = evaluate_work_span(ComputeDependencies(True), {}, precision_bits=64)
    assert result.seconds_per_item == 0


@pytest.mark.parametrize("unpriced", [(('integer_divide', 32),), (('real_conversion', 1),), []])
def test_unpriced_numerical_work_cannot_become_a_zero_cost_estimate(unpriced):
    with pytest.raises(DependencyModelError, match="unpriced"):
        evaluate_work_span(replace(chains(1, 2), unpriced_numerical_operations=unpriced),
            {"ordinary": Coefficient(1.0, 2.0)}, precision_bits=64)
