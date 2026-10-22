"""Construct and independently check dependencies for hierarchical task graphs.

All effect proofs originate in lowering/source analysis. This module only takes
their conservative union and checks conflicts; executable-on-GPU is never used as
an effect proof. Original task order defines conflicting/observable order, while
unrelated native work is free to move within its structured control region.
"""

from __future__ import annotations

from collections.abc import Iterable

from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    ConditionalRegion,
    ExecutionPlan,
    FunctionIR,
    HostBlock,
    If,
    Literal,
    Loop,
    ParallelRegion,
    Reference,
    ScalarType,
    SequentialRegion,
    Size,
    Unary,
    block_reads,
    block_writes,
    referenced_symbols,
    walk_expr,
)
from compiler.ir.task_graph import (
    BranchRegion,
    DependencyKind,
    ExecutionTarget,
    LoopRegion,
    ResourceAccess,
    Task,
    TaskDependency,
    TaskEffects,
    TaskGraph,
)

from .effects import array_effects

_NO_EFFECTS = TaskEffects()


class TaskGraphError(ValueError):
    """Invalid effect summaries, dependencies, or task coverage."""


def union_effects(effects: Iterable[TaskEffects]) -> TaskEffects:
    """A safe may-effect summary, without promoting conditional overwrites."""
    values = tuple(effects)
    return TaskEffects(
        reads=tuple(access for value in values for access in value.reads),
        may_writes=tuple(access for value in values for access in value.may_writes),
        scalar_reads=frozenset(name for value in values for name in value.scalar_reads),
        scalar_writes=frozenset(name for value in values for name in value.scalar_writes),
        lifetime_resources=frozenset(name for value in values for name in value.lifetime_resources),
        observable_tokens=frozenset(name for value in values for name in value.observable_tokens),
        opaque=any(value.opaque for value in values),
        metadata_reads=frozenset(name for value in values for name in value.metadata_reads),
        ordered=any(value.ordered for value in values),
    )


def summarize_graph(graph: TaskGraph) -> TaskEffects:
    return union_effects(task.effects for task in graph.tasks)


def _children(structure: BranchRegion | LoopRegion) -> tuple[TaskGraph, ...]:
    if isinstance(structure, BranchRegion):
        return structure.then_graph, structure.else_graph
    return (structure.body,)


def structured_task(
    task_id: str,
    structure: BranchRegion | LoopRegion,
    *,
    header_effects: TaskEffects = _NO_EFFECTS,
    payload: object = None,
    predecessors: frozenset[str] = frozenset(),
) -> Task:
    """Keep branches/loops atomic to their parent scheduler, with checked children.

    Header effects describe predicate/bounds evaluation. The caller must include
    loop iterator writes if the iterator is visible after the loop.
    """
    effects = union_effects((header_effects, *(summarize_graph(child) for child in _children(structure))))
    return Task(
        task_id, effects, predecessors=predecessors, structure=structure, header_effects=header_effects, payload=payload
    )


def required_dependencies(tasks: Iterable[Task]) -> tuple[TaskDependency, ...]:
    """Derive every source-ordered conflict, without source-window cutoffs."""
    tasks = tuple(tasks)
    positions = {task.id: index for index, task in enumerate(tasks)}
    if len(positions) != len(tasks):
        raise TaskGraphError("Task identities must be unique within a graph")
    edges: set[TaskDependency] = set()
    reads: dict[str, list[tuple[str, ResourceAccess]]] = {}
    writes: dict[str, list[tuple[str, ResourceAccess]]] = {}
    scalar_reads: dict[str, list[str]] = {}
    scalar_writes: dict[str, list[str]] = {}
    resource_users: dict[str, list[str]] = {}
    lifetimes: dict[str, list[str]] = {}
    observations: dict[str, str] = {}
    last_opaque: tuple[str, bool] | None = None
    since_opaque: list[str] = []
    for index, task in enumerate(tasks):
        for predecessor in task.predecessors:
            if predecessor not in positions or positions[predecessor] >= index:
                raise TaskGraphError(f"Task {task.id} has an absent or non-preceding control predecessor")
            edges.add(TaskDependency(predecessor, task.id, DependencyKind.CONTROL))
        effects = task.effects
        if last_opaque is not None:
            before, opaque = last_opaque
            edges.add(
                TaskDependency(
                    before,
                    task.id,
                    DependencyKind.OPAQUE if opaque or effects.opaque else DependencyKind.OBSERVABLE,
                    "" if opaque or effects.opaque else "execution_order",
                )
            )
        if effects.opaque or effects.ordered:
            for before in since_opaque:
                edges.add(
                    TaskDependency(
                        before,
                        task.id,
                        DependencyKind.OPAQUE if effects.opaque else DependencyKind.OBSERVABLE,
                        "" if effects.opaque else "execution_order",
                    )
                )
            last_opaque, since_opaque = (task.id, effects.opaque), []
        else:
            since_opaque.append(task.id)

        # Sparse indexes avoid comparing the thousands of unrelated scalar and
        # descriptor tasks that can separate numerical regions. Dense conflicts
        # still need dense evidence; no arbitrary task-count window is imposed.
        for current, history, kind in (
            (effects.reads, writes, DependencyKind.RAW),
            (effects.may_writes, reads, DependencyKind.WAR),
            (effects.may_writes, writes, DependencyKind.WAW),
        ):
            for access in current:
                for before, previous in history.get(access.resource, ()):
                    if previous.overlaps(access):
                        edges.add(TaskDependency(before, task.id, kind, access.resource))
        for names, history, kind in (
            (effects.scalar_reads, scalar_writes, DependencyKind.RAW),
            (effects.scalar_writes, scalar_reads, DependencyKind.WAR),
            (effects.scalar_writes, scalar_writes, DependencyKind.WAW),
        ):
            for name in names:
                for before in history.get(name, ()):
                    edges.add(TaskDependency(before, task.id, kind, f"scalar:{name}"))
        for resource in effects.resources:
            for before in lifetimes.get(resource, ()):
                edges.add(TaskDependency(before, task.id, DependencyKind.LIFETIME, resource))
        for resource in effects.lifetime_resources:
            for before in resource_users.get(resource, ()):
                edges.add(TaskDependency(before, task.id, DependencyKind.LIFETIME, resource))
        for token in effects.observable_tokens:
            if token in observations:
                edges.add(TaskDependency(observations[token], task.id, DependencyKind.OBSERVABLE, token))
            observations[token] = task.id
        for access in effects.reads:
            reads.setdefault(access.resource, []).append((task.id, access))
        for access in effects.may_writes:
            writes.setdefault(access.resource, []).append((task.id, access))
        for name in effects.scalar_reads:
            scalar_reads.setdefault(name, []).append(task.id)
        for name in effects.scalar_writes:
            scalar_writes.setdefault(name, []).append(task.id)
        for resource in effects.resources:
            resource_users.setdefault(resource, []).append(task.id)
        for resource in effects.lifetime_resources:
            lifetimes.setdefault(resource, []).append(task.id)
    return tuple(
        sorted(edges, key=lambda edge: (positions[edge.before], positions[edge.after], edge.kind.value, edge.resource))
    )


def build_task_graph(tasks: Iterable[Task], *, name: str = "") -> TaskGraph:
    tasks = tuple(tasks)
    graph = TaskGraph(tasks, required_dependencies(tasks), name)
    verify_task_graph(graph)
    return graph


def _covers(summary: tuple[ResourceAccess, ...], needed: tuple[ResourceAccess, ...]) -> bool:
    return all(any(a.resource == b.resource and a.region.contains(b.region) for a in summary) for b in needed)


def verify_task_graph(graph: TaskGraph) -> None:
    """Recompute required edges and authenticate structured summaries recursively.

    Extra source-ordered edges are safe restrictions. Required edges cannot be
    silently omitted by a scheduler or by hand-constructed graph input.
    """
    if not isinstance(graph, TaskGraph) or any(not isinstance(task, Task) for task in graph.tasks):
        raise TaskGraphError("Expected a TaskGraph containing Task instances")

    def edge_key(edge):
        return edge.before, edge.after, edge.kind, edge.resource

    # The owning execution-plan freezer seals dataclasses with immutable
    # subclasses. Compare semantic fields rather than dataclass exact-class
    # equality, so a sealed graph is checked as strictly as a fresh graph.
    required = {edge_key(edge) for edge in required_dependencies(graph.tasks)}
    actual = {edge_key(edge) for edge in graph.dependencies}
    if len(actual) != len(graph.dependencies):
        raise TaskGraphError("Duplicate graph dependencies")
    positions = {task.id: index for index, task in enumerate(graph.tasks)}
    for edge in graph.dependencies:
        if not isinstance(edge, TaskDependency) or not isinstance(edge.kind, DependencyKind):
            raise TaskGraphError("Invalid dependency record")
        if (
            edge.before not in positions
            or edge.after not in positions
            or positions[edge.before] >= positions[edge.after]
        ):
            raise TaskGraphError("Dependencies must connect existing tasks in original execution order")
    missing = required - actual
    if missing:
        before, after, kind, _resource = min(missing, key=lambda item: (item[0], item[1], item[2].value, item[3]))
        raise TaskGraphError(f"Missing required {kind.value} dependency {before} -> {after}")
    for task in graph.tasks:
        if task.structure is None:
            continue
        if not isinstance(task.structure, (BranchRegion, LoopRegion)):
            raise TaskGraphError("Unknown structured task payload")
        children = _children(task.structure)
        for child in children:
            verify_task_graph(child)
        needed = union_effects((task.header_effects, *(summarize_graph(child) for child in children)))
        effects = task.effects
        if (
            not _covers(effects.reads, needed.reads)
            or not _covers(effects.may_writes, needed.may_writes)
            or not needed.scalar_reads <= effects.scalar_reads
            or not needed.scalar_writes <= effects.scalar_writes
            or not needed.lifetime_resources <= effects.lifetime_resources
            or not needed.metadata_reads <= effects.metadata_reads
            or not needed.observable_tokens <= effects.observable_tokens
            or (needed.opaque and not effects.opaque)
            or (needed.ordered and not effects.ordered)
        ):
            raise TaskGraphError(f"Structured task {task.id} omits child effects")
        if effects.must_overwrites:
            raise TaskGraphError("Structured tasks require a control-flow overwrite proof; none is issued here")


def _symbol_key(symbol) -> str:
    return f"symbol:{symbol.id}:{symbol.name}"


def _body_effects(body: Block, *, private=frozenset()) -> TaskEffects:
    # Whole-resource summaries are deliberate: exact affine/indirect footprints
    # require the existing dependence/access proofs and may be supplied directly.
    reads = block_reads(body) - private
    writes = block_writes(body) - private
    array_reads, array_writes = array_effects(body)
    return TaskEffects(
        reads=tuple(ResourceAccess(_symbol_key(symbol)) for symbol in array_reads),
        may_writes=tuple(ResourceAccess(_symbol_key(symbol)) for symbol in array_writes),
        scalar_reads=frozenset(_symbol_key(symbol) for symbol in reads if not symbol.rank),
        scalar_writes=frozenset(_symbol_key(symbol) for symbol in writes if not symbol.rank),
        metadata_reads=frozenset(_symbol_key(symbol) for symbol in reads if symbol.rank and symbol not in array_reads),
    )


def _total_metadata_expression(expression) -> bool:
    """Small total scalar subset eligible for movement without trap proofs."""
    if isinstance(expression, (Literal, Reference)):
        dtype = expression.dtype if isinstance(expression, Literal) else expression.symbol.dtype
        return dtype in {ScalarType.INTEGER, ScalarType.LOGICAL}
    if isinstance(expression, Size):
        return True
    if isinstance(expression, Unary):
        return expression.operator.lower() in {"+", "-", ".not."} and _total_metadata_expression(expression.operand)
    if isinstance(expression, Binary):
        return (
            expression.operator.lower()
            in {
                "+",
                "-",
                "*",
                ".and.",
                ".or.",
                ".eqv.",
                ".neqv.",
                "==",
                "/=",
                "<",
                ">",
                "<=",
                ">=",
                ".eq.",
                ".ne.",
                ".lt.",
                ".gt.",
                ".le.",
                ".ge.",
            }
            and _total_metadata_expression(expression.left)
            and _total_metadata_expression(expression.right)
        )
    return False


def _total_integer_expression(expression) -> bool:
    """Integer native work on the same executed path may move as one task.

    All referenced storage and index operands remain graph dependencies. No
    access is speculated onto a new path, and the complete native loop keeps its
    original order. Division, real arithmetic, and calls need stronger proofs.
    """
    if isinstance(expression, ArrayAccess):
        return expression.symbol.dtype in {ScalarType.INTEGER, ScalarType.LOGICAL} and all(
            _total_integer_expression(index) for index in expression.indices
        )
    if isinstance(expression, Unary):
        return expression.operator.lower() in {"+", "-", ".not."} and _total_integer_expression(expression.operand)
    if isinstance(expression, Binary):
        # Reuse the operator admission independently of operand storage.
        placeholder = Literal("1", ScalarType.INTEGER)
        return (
            _total_metadata_expression(Binary(expression.operator, placeholder, placeholder))
            and _total_integer_expression(expression.left)
            and _total_integer_expression(expression.right)
        )
    return _total_metadata_expression(expression)


def _total_integer_block(block) -> bool:
    from compiler.analysis.semantics import constant_integer

    for statement in block.statements:
        if isinstance(statement, Assignment):
            if not _total_integer_expression(statement.target) or not _total_integer_expression(statement.value):
                return False
        elif isinstance(statement, Loop):
            stride = statement.step if isinstance(statement.step, int) else constant_integer(statement.step)
            if (
                stride is None
                or stride == 0
                or not _total_integer_expression(statement.lower)
                or not _total_integer_expression(statement.upper)
                or not _total_integer_block(statement.body)
            ):
                return False
        elif isinstance(statement, If):
            if (
                not _total_integer_expression(statement.condition)
                or not _total_integer_block(statement.then_body)
                or not _total_integer_block(statement.else_body)
            ):
                return False
        else:
            return False
    return True


def execution_plan_graph(function: FunctionIR, plan: ExecutionPlan, *, prefix: str = "task") -> TaskGraph:
    """Adapt existing checked numerical plans without changing their semantics.

    Parallel regions retain their already-proven CUDA implementation. Native
    regions remain ordinary tasks rather than terminating graph discovery.
    Branches remain hierarchical so untaken and zero-trip work is not hoisted.
    This adapter deliberately claims no overwrite proofs from parameter INTENT.
    """
    tasks = []
    for index, step in enumerate(plan.steps):
        task_id = f"{prefix}:{index}"
        if isinstance(step, ConditionalRegion):
            then_graph = execution_plan_graph(function, step.then_plan, prefix=f"{task_id}:then")
            else_graph = execution_plan_graph(function, step.else_plan, prefix=f"{task_id}:else")
            predicate_symbols = referenced_symbols(step.condition)
            data_symbols = {node.symbol for node in walk_expr(step.condition) if isinstance(node, ArrayAccess)}
            header = TaskEffects(
                reads=tuple(
                    ResourceAccess(_symbol_key(symbol)) for symbol in sorted(data_symbols, key=lambda item: item.id)
                ),
                scalar_reads=frozenset(_symbol_key(symbol) for symbol in predicate_symbols if not symbol.rank),
                metadata_reads=frozenset(
                    _symbol_key(symbol) for symbol in predicate_symbols if symbol.rank and symbol not in data_symbols
                ),
                ordered=not _total_metadata_expression(step.condition),
            )
            tasks.append(
                structured_task(
                    task_id, BranchRegion(step.condition, then_graph, else_graph), header_effects=header, payload=step
                )
            )
        elif isinstance(step, HostBlock):
            effects = _body_effects(Block(step.assignments))
            safe_motion = all(
                isinstance(assignment.target, Reference)
                and assignment.target.symbol.dtype in {ScalarType.INTEGER, ScalarType.LOGICAL}
                and _total_metadata_expression(assignment.value)
                for assignment in step.assignments
            )
            if not safe_motion:
                effects = union_effects((effects, TaskEffects(ordered=True)))
            tasks.append(Task(task_id, effects, payload=step))
        elif isinstance(step, SequentialRegion):
            effects = _body_effects(step.body)
            if not _total_integer_block(step.body):
                effects = union_effects((effects, TaskEffects(ordered=True)))
            tasks.append(Task(task_id, effects, payload=step))
        elif isinstance(step, ParallelRegion):
            body = Block(step.loops) if step.loops else step.body
            private = frozenset((*step.private_symbols, *(loop.iterator for loop in step.loops)))
            effects = _body_effects(body, private=private)
            tasks.append(Task(task_id, effects, frozenset({ExecutionTarget.CPU, ExecutionTarget.GPU}), payload=step))
        else:
            raise TaskGraphError(f"Unsupported execution-plan step {type(step).__name__}")
    return build_task_graph(tasks, name=function.name)
