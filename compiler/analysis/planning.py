"""Execution policy: validated source, proof queries, and explicit host fallback."""

from __future__ import annotations

from typing import TYPE_CHECKING

from compiler.ir import (
    Assignment,
    Block,
    CompilationError,
    ConditionalRegion,
    ExecutionPlan,
    FunctionIR,
    HostBlock,
    If,
    ParallelRegion,
    Reference,
    ScalarType,
    SequentialRegion,
    block_writes,
)

from .dependence import Affine, affine_expression, prove_region
from .effects import all_loops, array_effects
from .proof import ParallelizationError
from .semantics import validate_block, validate_function

if TYPE_CHECKING:
    from compiler.driver.options import CompilerOptions


def build_execution_plan(function: FunctionIR, *, options: CompilerOptions | None = None) -> ExecutionPlan:
    from compiler.driver.options import CompilerOptions

    options = options or CompilerOptions()
    validate_function(function)
    environment = {
        symbol: Affine(terms=((f"p{symbol.id}", 1),))
        for symbol in function.parameters
        if not symbol.rank and symbol.dtype is ScalarType.INTEGER
    }
    defined = {symbol for symbol in function.parameters if not symbol.rank}
    next_region = [0]
    next_snapshot = [0]

    def snapshot(symbol):
        value = Affine(terms=((f"h{next_snapshot[0]}_{symbol.id}", 1),))
        next_snapshot[0] += 1
        return value

    def build(block, environment, defined, continuation):
        environment, defined = dict(environment), set(defined)
        steps, host = [], []

        def flush_host():
            if host:
                reads, writes = array_effects(Block(tuple(host)))
                steps.append(HostBlock(tuple(host), reads, writes))
                host.clear()

        for index, statement in enumerate(block.statements):
            later = Block((*block.statements[index + 1 :], *continuation.statements))
            if isinstance(statement, Assignment):
                if isinstance(statement.target, Reference):
                    symbol = statement.target.symbol
                    if symbol.dtype is ScalarType.INTEGER:
                        try:
                            environment[symbol] = affine_expression(
                                statement.value, environment, {}, statement.location
                            )
                        except CompilationError:
                            environment[symbol] = snapshot(symbol)
                    defined.add(symbol)
                host.append(statement)
                continue
            flush_host()
            if isinstance(statement, If):
                then_plan = build(statement.then_body, environment, defined, later)
                else_plan = build(statement.else_body, environment, defined, later)
                reads, writes = array_effects(Block((statement,)))
                steps.append(
                    ConditionalRegion(statement.condition, then_plan, else_plan, statement.location, reads, writes)
                )
                defined = validate_block(Block((statement,)), defined)
                written = block_writes(statement.then_body) | block_writes(statement.else_body)
            else:
                region_id = next_region[0]
                next_region[0] += 1
                proof = prove_region(region_id, statement, environment, defined, later)
                if proof.proven:
                    steps.append(proof.region)
                elif options.fallback == "host":
                    body = Block((statement,))
                    reads, writes = array_effects(body)
                    steps.append(SequentialRegion(region_id, body, proof.failure.reason, reads, writes))
                else:
                    failure = proof.failure
                    raise ParallelizationError(
                        failure.reason, failure.location, witness=failure.witness, conservative=failure.conservative
                    )
                defined = validate_block(Block((statement,)), defined)
                written = block_writes(Block((statement,)))
            for symbol in written:
                environment.pop(symbol, None)
                if symbol in defined and symbol.dtype is ScalarType.INTEGER and not symbol.rank:
                    environment[symbol] = snapshot(symbol)
        flush_host()
        return ExecutionPlan(tuple(steps))

    return build(function.body, environment, defined, Block(()))


def format_plan(plan):
    lines = list(plan.reports)
    for step in plan.steps:
        if isinstance(step, HostBlock):
            lines.append(f"host block: {len(step.assignments)} ordered assignment(s)")
            lines.append(f"  array reads: {', '.join(s.name for s in step.read_symbols) or '(none)'}")
            lines.append(f"  array writes: {', '.join(s.name for s in step.write_symbols) or '(none)'}")
        elif isinstance(step, SequentialRegion):
            lines.append(f"region {step.id}: sequential host fallback: {step.reason}")
        elif isinstance(step, ConditionalRegion):
            lines.append(f"host conditional at {step.location}")
            lines.extend("  then: " + line for line in format_plan(step.then_plan).splitlines())
            lines.extend("  else: " + line for line in format_plan(step.else_plan).splitlines())
        elif isinstance(step, ParallelRegion):
            lines.append(f"region {step.id}: {len(step.loops)} mapped dimensions, parallel legality PROVEN")
            lines.append(f"  retained sequential loops: {len(tuple(all_loops(step.body)))}")
            lines.append(f"  access model: {'conservative' if step.report.conservative else 'exact'}")
            lines.append(f"  private: {', '.join(s.cpp_name for s in step.private_symbols) or '(none)'}")
            for name in ("domain", "schedule", "raw", "war", "waw"):
                lines.append(
                    f"  {name.upper() if name in {'raw', 'war', 'waw'} else name}: {getattr(step.report, name)}"
                )
    return "\n".join(lines)
