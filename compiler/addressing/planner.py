"""Attach checked addressing decisions without modifying source expressions."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING

from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    ConditionalRegion,
    Expr,
    HostBlock,
    If,
    IntrinsicCall,
    Literal,
    Loop,
    ParallelRegion,
    Reference,
    RegionAddressing,
    SequentialRegion,
    Size,
    SubscriptDecision,
    Unary,
    walk_expr,
)

from .ranges import iterator_range, prove_range

if TYPE_CHECKING:
    from compiler.driver.options import CompilerOptions
    from compiler.ir import ExecutionPlan, Symbol


def _body_subscripts(block: Block):
    for statement in block.statements:
        if isinstance(statement, Assignment):
            expressions = (statement.target, statement.value)
        elif isinstance(statement, If):
            expressions = (statement.condition,)
        elif isinstance(statement, Loop):
            expressions = ()
        else:
            raise TypeError(f"Unknown statement: {type(statement).__name__}")
        for expression in expressions:
            for node in walk_expr(expression):
                if isinstance(node, ArrayAccess):
                    yield from node.indices
        if isinstance(statement, If):
            yield from _body_subscripts(statement.then_body)
            yield from _body_subscripts(statement.else_body)
        elif isinstance(statement, Loop):
            # Sequential headers retain source lowering, including any accesses.
            yield from _body_subscripts(statement.body)


def _mapped_references(expression: Expr, mapped: set[Symbol]) -> set[Symbol]:
    if isinstance(expression, Reference):
        return {expression.symbol} & mapped
    if isinstance(expression, Unary):
        return _mapped_references(expression.operand, mapped)
    if isinstance(expression, Binary):
        return _mapped_references(expression.left, mapped) | _mapped_references(expression.right, mapped)
    if isinstance(expression, IntrinsicCall):
        return set().union(*(_mapped_references(argument, mapped) for argument in expression.arguments))
    # An array load is one source-width value. Its subscripts are separate sites.
    return set()


def _expression_text(expression: Expr) -> str:
    if isinstance(expression, Literal):
        return expression.value
    if isinstance(expression, Reference):
        return expression.symbol.name
    if isinstance(expression, ArrayAccess):
        return f"{expression.symbol.name}({', '.join(_expression_text(index) for index in expression.indices)})"
    if isinstance(expression, Size):
        return f"size({expression.symbol.name}, {expression.dimension})"
    if isinstance(expression, Unary):
        return f"({expression.operator}{_expression_text(expression.operand)})"
    if isinstance(expression, Binary):
        return f"({_expression_text(expression.left)} {expression.operator} {_expression_text(expression.right)})"
    if isinstance(expression, IntrinsicCall):
        return f"{expression.name}({', '.join(_expression_text(argument) for argument in expression.arguments)})"
    raise TypeError(f"Unknown expression: {type(expression).__name__}")


def _region_addressing(region: ParallelRegion, *, enabled: bool) -> RegionAddressing:
    ranges = tuple((loop.iterator, iterator_range(loop)) for loop in region.loops)
    environment = dict(ranges)
    mapped = set(environment)
    body = region.body if region.body is not None else Block(region.assignments)
    # Facts are uniform across a region: only mapped iterators are refined. Equal
    # expressions can share decisions without conflating scalar lifetimes.
    occurrences = Counter(_body_subscripts(body))
    decisions = []
    wide_iterators = set()
    for expression, count in occurrences.items():
        proof = prove_range(expression, environment)
        references = _mapped_references(expression, mapped)
        mode = "source"
        reason = proof.reason
        if not enabled:
            reason = "source indexing requested"
        elif not references:
            reason = "no mapped iterator in subscript arithmetic"
        elif proof.interval is not None:
            mode = "wide"
            wide_iterators.update(references)
        decisions.append(SubscriptDecision(expression, mode, proof.interval, reason, count))
    return RegionAddressing(
        tuple(decisions), ranges, tuple(loop.iterator for loop in region.loops if loop.iterator in wide_iterators)
    )


def plan_addressing(plan: ExecutionPlan, *, options: CompilerOptions) -> ExecutionPlan:
    """Select wide-safe subscripts after scheduling, retaining all source IR."""
    steps = []
    reports = list(plan.reports)
    for step in plan.steps:
        if isinstance(step, ConditionalRegion):
            steps.append(
                replace(
                    step,
                    then_plan=plan_addressing(step.then_plan, options=options),
                    else_plan=plan_addressing(step.else_plan, options=options),
                )
            )
        elif isinstance(step, ParallelRegion):
            addressing = _region_addressing(step, enabled=options.resolved_indexing == "auto")
            steps.append(replace(step, addressing=addressing))
            total = sum(decision.occurrences for decision in addressing.decisions)
            wide = sum(decision.occurrences for decision in addressing.decisions if decision.mode == "wide")
            reports.append(f"region {step.id}: {options.resolved_indexing} indexing, {wide}/{total} subscripts widened")
            for decision in addressing.decisions:
                interval = decision.interval
                bounds = f" [{interval.lower}, {interval.upper}]" if interval is not None else ""
                reports.append(
                    f"  {_expression_text(decision.expression)}: {decision.mode}{bounds}; {decision.reason} "
                    f"({decision.occurrences} occurrence(s))"
                )
        elif isinstance(step, (HostBlock, SequentialRegion)):
            steps.append(step)
        else:
            raise TypeError(f"Unknown execution step: {type(step).__name__}")
    return replace(plan, steps=tuple(steps), reports=tuple(reports))
