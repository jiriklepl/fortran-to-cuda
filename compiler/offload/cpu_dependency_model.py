"""A bounded source work/span hypothesis, independent of application timings.

The coefficients include the calibrated host-thread participation. They are
not instruction latencies, a SIMD proof, or permission to offload. Callers need
independent size, expression, memory and input-domain validation before using
the result for placement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from compiler.ir import ScalarType
from compiler.offload.compute_dependencies import MAX_EDGES, MAX_OPERATIONS, ComputeDependencies, ComputeOperation

ORDINARY_FAMILIES = frozenset({"add", "subtract", "multiply", "negate", "abs", "min", "max"})
PRIMITIVE_FAMILIES = frozenset({"divide_dynamic", "divide_constant", "sqrt", "acos", "cos"})
COEFFICIENT_FAMILIES = ("ordinary", *sorted(PRIMITIVE_FAMILIES))


class DependencyModelError(ValueError):
    """The cost hypothesis cannot describe the supplied evidence."""


def _positive(value, name):
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float)) and
                 math.isfinite(value) and value > 0)
    except OverflowError:
        valid = False
    if not valid:
        raise DependencyModelError(name + " must be finite and positive")
    return float(value)


@dataclass(frozen=True)
class Coefficient:
    work_seconds: float
    span_seconds: float

    def __post_init__(self):
        _positive(self.work_seconds, "work coefficient")
        _positive(self.span_seconds, "span coefficient")
        if self.span_seconds < self.work_seconds:
            raise DependencyModelError("span coefficient is below aggregate work coefficient")

    def to_dict(self):
        return {"work_seconds": self.work_seconds, "span_seconds": self.span_seconds}


@dataclass(frozen=True)
class WorkSpan:
    work_seconds: float
    span_seconds: float

    @property
    def seconds_per_item(self):
        return max(self.work_seconds, self.span_seconds)

    def to_dict(self):
        return {"work_seconds": self.work_seconds, "span_seconds": self.span_seconds,
                "seconds_per_item": self.seconds_per_item}


def coefficient_family(family):
    if family in ORDINARY_FAMILIES:
        return "ordinary"
    if family in PRIMITIVE_FAMILIES:
        return family
    raise DependencyModelError("unsupported dependency family " + str(family))


def _checked_graph(graph, precision_bits):
    if (not isinstance(graph, ComputeDependencies) or type(graph.schema_version) is not int or
            graph.schema_version != 1 or graph.available is not True or not isinstance(graph.operations, tuple) or
            len(graph.operations) > MAX_OPERATIONS):
        raise DependencyModelError("bounded dependency graph unavailable")
    if type(precision_bits) is not int or precision_bits not in {32, 64}:
        raise DependencyModelError("unsupported dependency precision")
    if not isinstance(graph.unpriced_numerical_operations, tuple) or graph.unpriced_numerical_operations:
        raise DependencyModelError("unpriced integer/logical numerical operations")
    dtype = ScalarType.REAL if precision_bits == 64 else ScalarType.REAL32
    if (not isinstance(graph.floating_dtypes, tuple) or
            any(not isinstance(value, ScalarType) or value not in {ScalarType.REAL, ScalarType.REAL32}
                for value in graph.floating_dtypes) or
            len(set(graph.floating_dtypes)) != len(graph.floating_dtypes) or
            any(value is not dtype for value in graph.floating_dtypes) or
            (graph.operations and dtype not in graph.floating_dtypes)):
        raise DependencyModelError("mixed, missing or incompatible dependency precision metadata")
    edges = 0
    for index, operation in enumerate(graph.operations):
        if (not isinstance(operation, ComputeOperation) or not isinstance(operation.operands, tuple) or
                any(not isinstance(operand, tuple) for operand in operation.operands)):
            raise DependencyModelError("invalid dependency operation or operands")
        edges += sum(len(operand) for operand in operation.operands)
        if edges > MAX_EDGES:
            raise DependencyModelError("dependency edge budget exceeded")
        if operation.dtype is not dtype:
            raise DependencyModelError("mixed or incompatible dependency precision")
        coefficient_family(operation.family)
        if (type(operation.work_units) is not int or not 1 <= operation.work_units <= MAX_OPERATIONS or
                any(type(parent) is not int or not 0 <= parent < index for parent in operation.predecessors)):
            raise DependencyModelError("invalid dependency graph ordering or work")


def evaluate_work_span(graph, coefficients: Mapping[str, Coefficient], *, precision_bits):
    """Price every source occurrence, including constants and invariants.

    Those flags alone do not establish folding or hoisting by the original
    native compiler. Division is one complete operation, not ordinary work
    plus an additional division charge. No host-thread division occurs here.
    """
    _checked_graph(graph, precision_bits)
    if not isinstance(coefficients, Mapping):
        raise DependencyModelError("dependency coefficients missing")
    work, weights = [], {}
    for operation in graph.operations:
        family = coefficient_family(operation.family)
        coefficient = coefficients.get(family)
        if not isinstance(coefficient, Coefficient):
            raise DependencyModelError("missing dependency coefficient " + family)
        work.append(coefficient.work_seconds * operation.work_units)
        weights[operation.family] = coefficient.span_seconds
    try:
        total = math.fsum(work)
        span = graph.weighted_span(weights)
    except (OverflowError, ValueError) as error:
        raise DependencyModelError("dependency work/span is unrepresentable") from error
    if not math.isfinite(total) or not math.isfinite(span):
        raise DependencyModelError("dependency work/span is unrepresentable")
    return WorkSpan(total, span)


def fit_basis_pair(single, wide, single_seconds, wide_seconds, *, family,
                   known: Mapping[str, Coefficient], precision_bits, maximum_error=0.25):
    """Identify one coefficient from a predeclared one-/four-chain pair.

    The single chain must be span-limited and the wide graph work-limited.
    These roles are fixed before measurement. An incompatible cone is a
    rejection, rather than a reason to select another equation or clip a
    negative residual. Fit points never replace independent holdouts.
    """
    if (not isinstance(known, Mapping) or any(name not in COEFFICIENT_FAMILIES or
            not isinstance(value, Coefficient) for name, value in known.items())):
        raise DependencyModelError("invalid known dependency coefficients")
    if family not in COEFFICIENT_FAMILIES or family in known:
        raise DependencyModelError("unknown or already identified coefficient family")
    if (isinstance(maximum_error, bool) or not isinstance(maximum_error, (int, float)) or
            not math.isfinite(maximum_error) or not 0 < maximum_error <= 0.25):
        raise DependencyModelError("invalid dependency fit tolerance")
    for graph in (single, wide):
        _checked_graph(graph, precision_bits)
        if not any(coefficient_family(op.family) == family for op in graph.operations):
            raise DependencyModelError("basis does not contain its coefficient family")
        if any(coefficient_family(op.family) not in {*known, family} for op in graph.operations):
            raise DependencyModelError("basis depends on an unidentified coefficient")
    single_seconds = _positive(single_seconds, "single-chain variable slope")
    wide_seconds = _positive(wide_seconds, "wide-chain variable slope")
    try:
        known_work = math.fsum(known[coefficient_family(op.family)].work_seconds * op.work_units
                              for op in wide.operations if coefficient_family(op.family) != family)
    except OverflowError as error:
        raise DependencyModelError("basis work is unrepresentable") from error
    if not math.isfinite(known_work):
        raise DependencyModelError("basis work is unrepresentable")
    count = sum(op.work_units for op in wide.operations if coefficient_family(op.family) == family)
    work = (wide_seconds - known_work) / count
    _positive(work, "basis residual work coefficient")

    def span_at(value):
        weights = {op.family: (value if coefficient_family(op.family) == family else
                              known[coefficient_family(op.family)].span_seconds)
                   for op in single.operations}
        try:
            return single.weighted_span(weights)
        except (ValueError, OverflowError) as error:
            raise DependencyModelError("basis span is unrepresentable") from error

    # Positive work is a lower bound on latency. If this already exceeds the
    # observed single-chain slope, the prescribed model is unidentifiable.
    if span_at(work) > single_seconds:
        raise DependencyModelError("basis is incompatible with the work/span cone")
    if span_at(0.0) >= single_seconds:
        raise DependencyModelError("basis has no positive identifiable span residual")
    lower, upper = work, single_seconds
    if span_at(upper) < single_seconds:
        raise DependencyModelError("basis does not identify a span coefficient")
    for _ in range(64):
        middle = lower + (upper - lower) / 2
        if span_at(middle) < single_seconds:
            lower = middle
        else:
            upper = middle
    coefficient = Coefficient(work, upper)
    fitted = {**known, family: coefficient}
    single_model = evaluate_work_span(single, fitted, precision_bits=precision_bits)
    wide_model = evaluate_work_span(wide, fitted, precision_bits=precision_bits)
    # Each graph must expose the promised active component; numerical error
    # is allowed only in prediction, never to swap the identification roles.
    epsilon = 1e-12
    if (single_model.span_seconds + epsilon * single_seconds < single_model.work_seconds or
            wide_model.work_seconds + epsilon * wide_seconds < wide_model.span_seconds):
        raise DependencyModelError("basis does not expose both work and span")
    for predicted, actual in ((single_model.seconds_per_item, single_seconds),
                              (wide_model.seconds_per_item, wide_seconds)):
        if abs(predicted - actual) / actual > maximum_error:
            raise DependencyModelError("basis fit error exceeds the validation ceiling")
    return coefficient
