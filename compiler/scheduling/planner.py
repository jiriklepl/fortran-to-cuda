"""Deterministic locality schedules for already-proved parallel regions."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from compiler.analysis.semantics import constant_integer
from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    CompilationError,
    ConditionalRegion,
    If,
    Loop,
    ParallelRegion,
    Reference,
    RegionSchedule,
    Unary,
    referenced_symbols,
    walk_expr,
)

if TYPE_CHECKING:
    from compiler.driver.options import CompilerOptions
    from compiler.ir import Block, ExecutionPlan, Expr, FunctionIR, Symbol


def _constant_integer(expression: Expr | int) -> int | None:
    return expression if isinstance(expression, int) else constant_integer(expression)


def _coefficient(expression: Expr, iterator: Symbol) -> int | None:
    """Return a known affine coefficient, withholding credit for unknown accesses."""
    if iterator not in referenced_symbols(expression):
        return 0
    if isinstance(expression, Reference):
        return 1
    if isinstance(expression, Unary) and expression.operator in {"+", "-"}:
        operand = _coefficient(expression.operand, iterator)
        return None if operand is None else operand * (1 if expression.operator == "+" else -1)
    if isinstance(expression, Binary):
        if expression.operator in {"+", "-"}:
            left, right = _coefficient(expression.left, iterator), _coefficient(expression.right, iterator)
            if left is not None and right is not None:
                return left + right if expression.operator == "+" else left - right
        if expression.operator == "*":
            for constant, value in ((expression.left, expression.right), (expression.right, expression.left)):
                factor = _constant_integer(constant)
                coefficient = _coefficient(value, iterator)
                if factor is not None and coefficient is not None:
                    return factor * coefficient
        if expression.operator == "/":
            denominator = _constant_integer(expression.right)
            numerator = _coefficient(expression.left, iterator)
            if denominator in {-1, 1} and numerator is not None:
                return numerator * denominator
    return None


def _accesses(block: Block):
    for statement in block.statements:
        if isinstance(statement, Assignment):
            if isinstance(statement.target, ArrayAccess):
                yield statement.target, 2
                for subscript in statement.target.indices:
                    yield from ((node, 1) for node in walk_expr(subscript) if isinstance(node, ArrayAccess))
            yield from ((node, 1) for node in walk_expr(statement.value) if isinstance(node, ArrayAccess))
        elif isinstance(statement, If):
            yield from ((node, 1) for node in walk_expr(statement.condition) if isinstance(node, ArrayAccess))
            yield from _accesses(statement.then_body)
            yield from _accesses(statement.else_body)
        elif isinstance(statement, Loop):
            expressions = (statement.lower, statement.upper)
            if not isinstance(statement.step, int):
                expressions += (statement.step,)
            for expression in expressions:
                yield from ((node, 1) for node in walk_expr(expression) if isinstance(node, ArrayAccess))
            yield from _accesses(statement.body)


def _axis_scores(region: ParallelRegion) -> tuple[tuple[int, ...], ...]:
    accesses = tuple(_accesses(region.body))
    rank = max((access.symbol.rank for access, _ in accesses), default=0)
    scores = []
    for loop in region.loops:
        score = [0] * rank
        stride = _constant_integer(loop.step)
        if stride in {-1, 1}:
            for access, weight in accesses:
                coefficients = tuple(_coefficient(index, loop.iterator) for index in access.indices)
                varying = [index for index, coefficient in enumerate(coefficients) if coefficient != 0]
                if len(varying) == 1 and coefficients[varying[0]] in {-1, 1}:
                    score[varying[0]] += weight
        scores.append(tuple(score))
    return tuple(scores)


def schedule_plan(function: FunctionIR, plan: ExecutionPlan, *, options: CompilerOptions) -> ExecutionPlan:
    """Attach schedules without changing source loops or their bound evaluation order."""
    maximum_rank = max((len(region.loops) for region in plan.regions), default=0)
    if len(options.tile_sizes) > maximum_rank:
        raise CompilationError(
            f"Tile specification has {len(options.tile_sizes)} axes but maximum mapped rank is {maximum_rank}"
        )
    if any(size <= 0 or size > 2**64 - 1 for size in options.tile_sizes):
        raise CompilationError("Tile sizes must be positive and fit in an unsigned 64-bit integer")

    def schedule(current: ExecutionPlan) -> ExecutionPlan:
        steps = []
        reports = list(current.reports)
        for step in current.steps:
            if isinstance(step, ConditionalRegion):
                steps.append(replace(step, then_plan=schedule(step.then_plan), else_plan=schedule(step.else_plan)))
                continue
            if not isinstance(step, ParallelRegion):
                steps.append(step)
                continue
            scores = _axis_scores(step)
            axes = tuple(reversed(range(len(step.loops))))
            if options.resolved_schedule == "auto":
                axes = tuple(sorted(axes, key=lambda axis: (scores[axis], axis), reverse=True))
            tiles: tuple[int, ...] = ()
            if options.tile_sizes:
                by_axis = [1] * len(axes)
                for axis, size in zip(axes, options.tile_sizes, strict=False):
                    by_axis[axis] = size
                tiles = tuple(by_axis)
            steps.append(replace(step, schedule=RegionSchedule(axes, tiles, 256)))
            order = ",".join(step.loops[axis].iterator.name for axis in axes)
            reports.append(
                f"region {step.id}: {options.resolved_schedule} schedule fastest-first [{order}], "
                f"locality scores={scores}, tiles={tuple(tiles[axis] for axis in axes) if tiles else 'none'}"
            )
        return replace(current, steps=tuple(steps), reports=tuple(reports))

    return schedule(plan)
