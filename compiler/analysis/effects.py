"""Scalar liveness and structural effects, independent of dependence mathematics."""

from compiler.ir import (
    ArrayAccess,
    Assignment,
    If,
    Literal,
    Loop,
    Reference,
    ScalarType,
    block_writes,
    referenced_symbols,
    statement_reads,
    walk_expr,
)


def loop_step(loop):
    return Literal(str(loop.step), ScalarType.INTEGER) if isinstance(loop.step, int) else loop.step


def header_reads(loop):
    return referenced_symbols(loop.lower) | referenced_symbols(loop.upper) | referenced_symbols(loop_step(loop))


def all_loops(block):
    for statement in block.statements:
        if isinstance(statement, Loop):
            yield statement
            yield from all_loops(statement.body)
        elif isinstance(statement, If):
            yield from all_loops(statement.then_body)
            yield from all_loops(statement.else_body)


def assignments(block):
    for statement in block.statements:
        if isinstance(statement, Assignment):
            yield statement
        elif isinstance(statement, If):
            yield from assignments(statement.then_body)
            yield from assignments(statement.else_body)
        else:
            yield from assignments(statement.body)


def mapped_prefix(loop):
    """Map a rectangular perfect prefix; retain the remaining ordered body."""
    loops, body = [loop], loop.body
    written = block_writes(body)
    while len(body.statements) == 1 and isinstance(body.statements[0], Loop):
        inner = body.statements[0]
        header_values = frozenset(
            node.symbol
            for expression in (inner.lower, inner.upper, loop_step(inner))
            for node in walk_expr(expression)
            if isinstance(node, (Reference, ArrayAccess))
        )
        if header_values & (written | {item.iterator for item in loops}):
            break
        loops.append(inner)
        body = inner.body
    return tuple(loops), body


def definite_writes(block):
    """Scalars defined on every control-flow path, excluding possibly empty loops."""
    defined = set()
    for statement in block.statements:
        if isinstance(statement, Assignment) and isinstance(statement.target, Reference):
            defined.add(statement.target.symbol)
        elif isinstance(statement, If):
            defined.update(definite_writes(statement.then_body) & definite_writes(statement.else_body))
        elif isinstance(statement, Loop):
            defined.add(statement.iterator)
    return frozenset(defined)


def upward_reads(block):
    reads, defined = set(), set()
    for statement in block.statements:
        if isinstance(statement, Assignment):
            reads.update(statement_reads(statement) - defined)
            if isinstance(statement.target, Reference):
                defined.add(statement.target.symbol)
        elif isinstance(statement, If):
            incoming = (
                referenced_symbols(statement.condition)
                | upward_reads(statement.then_body)
                | upward_reads(statement.else_body)
            )
            reads.update(incoming - defined)
            defined.update(definite_writes(statement.then_body) & definite_writes(statement.else_body))
        else:
            incoming = header_reads(statement) | (upward_reads(statement.body) - {statement.iterator})
            reads.update(incoming - defined)
            defined.add(statement.iterator)
    return frozenset(reads)


def array_effects(block):
    """Read data, excluding SIZE metadata; writes conservatively union branches."""
    reads, writes = set(), set()

    def add_reads(expression):
        reads.update(node.symbol for node in walk_expr(expression) if isinstance(node, ArrayAccess))

    for statement in block.statements:
        if isinstance(statement, Assignment):
            add_reads(statement.value)
            if isinstance(statement.target, ArrayAccess):
                writes.add(statement.target.symbol)
                for index in statement.target.indices:
                    add_reads(index)
        elif isinstance(statement, If):
            add_reads(statement.condition)
            for body in (statement.then_body, statement.else_body):
                branch_reads, branch_writes = array_effects(body)
                reads.update(branch_reads)
                writes.update(branch_writes)
        else:
            for expression in (statement.lower, statement.upper, loop_step(statement)):
                add_reads(expression)
            body_reads, body_writes = array_effects(statement.body)
            reads.update(body_reads)
            writes.update(body_writes)
    return tuple(sorted(reads, key=lambda s: s.id)), tuple(sorted(writes, key=lambda s: s.id))
