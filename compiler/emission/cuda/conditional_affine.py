"""Emit one source-bound conditional domain before any source-read admission."""
from __future__ import annotations

from compiler.analysis.semantics import expression_type
from compiler.emission.cuda.tile_task_provider import _Ranges, _SUPPORT
from compiler.frontend.conditional_numerical import ConditionalNumericalContext
from compiler.ir import Binary, CompilationError, IntrinsicCall, ScalarType, SourceLocation, Unary


class _OriginalRanges(_Ranges):
    def addresses(self, value, environment, access="reads"):
        if (isinstance(value, (Binary, Unary, IntrinsicCall))
                and expression_type(value, SourceLocation("conditional source prework")) is ScalarType.INTEGER):
            self.expression(value, environment)
        super().addresses(value, environment, access)


def conditional_prework(alternative, prepared):
    """Return reused range support, original prework and checked domain guard.

    A zero-trip unit declines the complete optional team before inspecting any
    later axis/unit bound. Thus all scalar inputs of a successful guard were
    reached by original unconditional headers/assignments, without inventing
    source prefix definitions. Existing physical layout/alias admission remains
    separate. Neither this function nor its result grants execution authority.
    """
    context = prepared.conditional_context
    if type(context) is not ConditionalNumericalContext:
        raise CompilationError("conditional prework requires its exact source context")
    context.validate(prepared._provider)
    domain = context.domain(alternative)
    if domain is None:
        raise CompilationError("conditional prework cannot replace an empty original path")
    function = prepared.function
    parameters = {item.name: item for item in prepared.extraction.parameters}
    arrays = [symbol for symbol in function.parameters if symbol.rank]
    scalar = {symbol: "*input_"+symbol.cpp_name for symbol in function.parameters if not symbol.rank}
    field_index = {symbol: index for index, symbol in enumerate(arrays)}
    by_root = {parameters[symbol.name].resource: symbol for symbol in arrays}
    lowers = {(by_root[parameters[symbol.name].resource], parameters[symbol.name].lower_bound_dimension): symbol
              for symbol in scalar if parameters[symbol.name].lower_bound_dimension is not None}
    ranges = _OriginalRanges(None, scalar, field_index, lowers, "conditional_", descriptors="layouts")
    lines, emitted = [], 0
    for unit in prepared.offload.units:
        environment = {}
        for loop in unit.region.loops:
            if loop.step != 1:
                raise CompilationError("conditional original prework first slice requires positive unit strides")
            lo = ranges.expression(loop.lower, environment)
            hi = ranges.expression(loop.upper, environment)
            lines += ranges.lines[emitted:]
            emitted = len(ranges.lines)
            lines += [f"if ({lo}.lo>{hi}.hi) return fail(FORT_SCOPE_BOUNDARY);",
                      f"if ({hi}.hi==INT_MAX) return fail(FORT_SCOPE_ARGUMENT);"]
            span = ranges.new(f"range_span({lo},{hi})")
            lines += ranges.lines[emitted:]
            emitted = len(ranges.lines)
            environment[loop.iterator] = span
        ranges.block(unit.region.body, environment)
        lines += ranges.lines[emitted:]
        emitted = len(ranges.lines)
    # Reused ranges return the status directly; this provider also preserves
    # the status in its checked before/after submission report.
    lines = [line.replace("return FORT_SCOPE_ARGUMENT;", "return fail(FORT_SCOPE_ARGUMENT);") for line in lines]
    names = {name: "*input_" + symbol.cpp_name for name, symbol in domain.parameters}
    guard = ["bool conditional_legal=true;"]
    for clause_index, clause in enumerate(domain.clauses):
        guard += ["{", "bool conditional_clause=false;"]
        for number, arm in enumerate(clause.alternatives):
            guard.append("{")
            checks = []
            for index, condition in enumerate(arm):
                value = f"number({condition.constant}LL)"
                for name, coefficient in condition.coefficients:
                    value = f"plus({value},times(number({coefficient}LL),number({names[name]})))"
                label = f"conditional_guard_{clause_index}_{number}_{index}"
                guard.append(f"const auto {label}={value};")
                checks.append(f"({label}.ok && {label}.value{'==' if condition.equality else '>='}0)")
            guard += ["conditional_clause=conditional_clause || (" + (" && ".join(checks) or "true") + ");", "}"]
        guard += ["conditional_legal=conditional_legal && conditional_clause;", "}"]
    guard.append("if (!conditional_legal) return fail(FORT_SCOPE_BOUNDARY);")
    return _SUPPORT, lines, guard
