"""Metadata-only prework for the exact source-issued private line lowering.

This reuses tile source-address checks. A successful result checks shape and
integer/address coverage only; runtime alias, initialized-region, numerical
state, allocation budget and stream/lifetime admission are still required.
"""

from __future__ import annotations

import re

from compiler.analysis.semantics import expression_type
from compiler.emission.c.declarations import cpp_declaration
from compiler.emission.common.abi import abi_arguments, dimension_name
from compiler.emission.cuda.tile_task_provider import _SUPPORT, _Ranges
from compiler.frontend.lowering import lower_line_snapshot
from compiler.ir import ArrayAccess, Binary, CompilationError, IntrinsicCall, ScalarType, Size, SourceLocation, Unary


class _LineRanges(_Ranges):
    def __init__(self, *args, scratch_lower="fort_line_scratch_lower",
                 scratch_extent="fort_line_scratch_extent", **kwargs):
        super().__init__(*args, **kwargs)
        self.scratch_lower_expression = scratch_lower
        self.scratch_extent_expression = scratch_extent

    def expression(self, value, environment):
        if isinstance(value, Size) and value.symbol is self.lowered.region.scratch_symbols[0]:
            return self.new(f"range_extent({self.scratch_extent_expression})")
        return super().expression(value, environment)

    def addresses(self, value, environment, access="reads"):
        # Integer operations in predicates and values also retain the original
        # default-INTEGER behavior. Bounds checks alone would miss e.g. i*i in
        # a condition. Payload-dependent arithmetic conservatively declines;
        # immutable coordinate operations use the same checked range algebra.
        if (
            isinstance(value, (Binary, Unary, IntrinsicCall))
            and expression_type(
                value, SourceLocation(str(self.lowered.line_snapshot_contract.proof._routine.scope.path))
            )
            is ScalarType.INTEGER
        ):
            self.expression(value, environment)
        if isinstance(value, ArrayAccess) and value.symbol is self.lowered.region.scratch_symbols[0]:
            if len(value.indices) != 1:
                raise CompilationError("line metadata requires its original rank-one private snapshot")
            bounds = self.expression(value.indices[0], environment)
            self.lines.append(
                f"if (!inside({bounds},{self.scratch_lower_expression},{self.scratch_extent_expression})) return FORT_SCOPE_ARGUMENT;"
            )
            return
        super().addresses(value, environment, access)

    def loop_terminal(self, item, lower, upper):
        # The original unit-stride loop must retain its default-INTEGER terminal
        # value, including translated inner loops such as ii=i-3:i+3. The range
        # checks are deliberately conservative; no arithmetic is reassociated.
        self.new(f"range_add({upper},range_value(1))")


def generate_line_metadata_check(lowered, *, name="fort_line_metadata_check"):
    """Return a bounds-only C++ checker: 0 checked, 1 no work, 2 declined.

    Arguments use the existing ordinary field ABI, but no payload pointer is
    dereferenced. Descriptor/scratch checks follow all original outer-domain
    zero-trip checks. No caller-supplied IR, range table or geometry is trusted.
    """
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError("line metadata function needs an ordinary identifier")
    contract = lowered.line_snapshot_contract
    if contract is None or lower_line_snapshot(contract.proof, region_id=lowered.region.id) != lowered:
        raise CompilationError("line metadata requires its exact source-lowered producer and consumer")
    region = lowered.region
    arrays = [symbol for symbol in region.parameters if symbol.rank]
    field_index = {symbol: i for i, symbol in enumerate(arrays)}
    scalar = {symbol: symbol.cpp_name for symbol in region.parameters if not symbol.rank}
    binding_symbols = {binding.root: symbol for binding, symbol in lowered.bindings}
    lower = {(binding_symbols[binding.root], axis): symbol for binding, axis, symbol in lowered.lower_bounds}
    ranges = _LineRanges(lowered, scalar, field_index, lower, "line_", descriptors="views")
    parameters = [cpp_declaration(arg) for arg in abi_arguments(region.parameters)]
    parameters += [
        "int fort_line_scratch_lower",
        "std::size_t fort_line_scratch_extent",
        "std::size_t* fort_line_total",
    ]
    lines = [f"static int {name}(" + ",".join(parameters) + ") {", "*fort_line_total=0;", "std::size_t total=1;"]
    environment = {}
    emitted = 0
    for axis, loop in enumerate(region.outer_loops):
        if loop.step != 1:
            raise CompilationError("line metadata requires its original unit-stride transverse domains")
        lo, hi = ranges.expression(loop.lower, {}), ranges.expression(loop.upper, {})
        lines += ranges.lines[emitted:]
        emitted = len(ranges.lines)
        lines += [
            f"if ({lo}.lo>{hi}.hi) return 1;",
            f"if ({hi}.hi==INT_MAX) return 2;",  # Original terminal, before any payload work.
            f"const auto axis_count{axis}=static_cast<unsigned long long>({hi}.hi-{lo}.lo)+1;",
            f"if (axis_count{axis}>INT_MAX || !multiply(total,static_cast<std::size_t>(axis_count{axis}),total)) return 2;",
        ]
        span = ranges.new(f"range_span({lo},{hi})")
        lines += ranges.lines[emitted:]
        emitted = len(ranges.lines)
        environment[loop.iterator] = span
    lines += [
        f"if (fort_line_scratch_lower!={contract.scratch_lower} || !fort_line_scratch_extent || fort_line_scratch_extent>INT_MAX) return 2;",
        "if (static_cast<long long>(fort_line_scratch_lower)+static_cast<long long>(fort_line_scratch_extent)-1>INT_MAX) return 2;",
        f"struct line_view {{ std::size_t extents[3]; }} views[{len(arrays)}] = {{",
    ]
    for symbol in arrays:
        lines += [
            "{{"
            + ",".join([dimension_name(symbol, i) for i in range(1, symbol.rank + 1)] + ["0"] * (3 - symbol.rank))
            + "}},"
        ]
    lines += ["};"]
    source_symbol = binding_symbols[contract.proof.source_binding.root]
    lines += [
        f"if (views[{field_index[source_symbol]}].extents[{contract.line_axis - 1}]!=fort_line_scratch_extent) return 2;"
    ]
    for symbol in arrays:
        for axis in range(1, symbol.rank + 1):
            extent = f"views[{field_index[symbol]}].extents[{axis - 1}]"
            lb = scalar[lower[symbol, axis]]
            lines += [
                f"if (!{extent} || {extent}>INT_MAX || static_cast<long long>({lb})+static_cast<long long>({extent})-1>INT_MAX) return 2;"
            ]
    ranges.block(region.producer, environment)
    ranges.block(region.consumer, environment)
    lines += ranges.lines[emitted:]
    lines += ["*fort_line_total=total;", "return 0;", "}"]
    return [
        "#include <climits>",
        "#include <algorithm>",
        _SUPPORT,
        *(line.replace("return FORT_SCOPE_ARGUMENT;", "return 2;") for line in lines),
    ]
