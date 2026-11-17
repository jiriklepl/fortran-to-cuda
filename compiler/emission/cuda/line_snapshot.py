"""Metadata-only prework for the exact source-issued private line lowering.

This reuses tile source-address checks. A successful result checks shape and
integer/address coverage only; runtime alias, initialized-region, numerical
state, allocation budget and stream/lifetime admission are still required.
"""

from __future__ import annotations

from compiler.analysis.semantics import expression_type
from compiler.emission.cuda.tile_task_provider import _Ranges
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
