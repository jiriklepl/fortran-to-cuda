"""Checked ready-task placement with persistent-resource cost accounting.

The scheduler may hoist/sink independent tasks inside a control region. It never
speculates a branch, unrolls a loop, changes numerical kernels, or treats a native
task as a reason to publish unrelated arrays. Costs are estimates, not permission
to omit runtime coherence checks. Runtime section versions remain authoritative.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from compiler.analysis.task_dependencies import TaskGraphError, verify_task_graph
from compiler.ir.task_graph import BranchRegion, ExecutionTarget, LoopRegion, Task, TaskGraph


@dataclass(frozen=True)
class TaskCost:
    cpu: float = 1.0
    gpu: float = 1.0

    def __post_init__(self):
        if any(not math.isfinite(value) or value < 0 for value in (self.cpu, self.gpu)):
            raise ValueError("Task costs must be finite and nonnegative")


@dataclass(frozen=True)
class TransferEstimate:
    resource: str
    target: ExecutionTarget
    nbytes: int | None
    reason: str = "access"


@dataclass(frozen=True)
class ScheduledTask:
    task_id: str
    target: ExecutionTarget
    estimated_cost: float = 0.0
    transfers: tuple[TransferEstimate, ...] = ()
    evictions: tuple[str, ...] = ()
    children: tuple[TaskSchedule, ...] = ()


@dataclass(frozen=True)
class ExecutionGroup:
    """Consecutive same-target work; grouping does not itself prove fusion."""

    target: ExecutionTarget
    task_ids: tuple[str, ...]


@dataclass(frozen=True)
class TaskSchedule:
    placements: tuple[ScheduledTask, ...]
    groups: tuple[ExecutionGroup, ...] = ()

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(placement.task_id for placement in self.placements)

    @property
    def estimated_cost(self) -> float:
        return sum(placement.estimated_cost for placement in self.placements)

    def public(self) -> dict:
        return {
            "estimated_cost": self.estimated_cost,
            "placements": [
                {
                    "task": placement.task_id,
                    "target": placement.target.value,
                    "estimated_cost": placement.estimated_cost,
                    "transfers": [
                        {
                            "resource": transfer.resource,
                            "target": transfer.target.value,
                            "nbytes": transfer.nbytes,
                            "reason": transfer.reason,
                        }
                        for transfer in placement.transfers
                    ],
                    "evictions": list(placement.evictions),
                    "children": [child.public() for child in placement.children],
                }
                for placement in self.placements
            ],
            "groups": [{"target": group.target.value, "tasks": list(group.task_ids)} for group in self.groups],
        }


def _groups(placements: Iterable[ScheduledTask]) -> tuple[ExecutionGroup, ...]:
    groups: list[ExecutionGroup] = []
    for placement in placements:
        if groups and groups[-1].target == placement.target and not placement.children:
            previous = groups.pop()
            groups.append(ExecutionGroup(placement.target, (*previous.task_ids, placement.task_id)))
        else:
            groups.append(ExecutionGroup(placement.target, (placement.task_id,)))
    return tuple(groups)


def verify_schedule(graph: TaskGraph, schedule: TaskSchedule) -> None:
    """Verify coverage, feasibility, dependencies, and unchanged control regions."""
    verify_task_graph(graph)
    if not isinstance(schedule, TaskSchedule):
        raise TaskGraphError("Expected a TaskSchedule")
    if len(schedule.task_ids) != len(set(schedule.task_ids)) or set(schedule.task_ids) != set(graph.task_ids):
        raise TaskGraphError("Schedule must execute every task exactly once in its original control region")
    positions = {task_id: index for index, task_id in enumerate(schedule.task_ids)}
    for edge in graph.dependencies:
        if positions[edge.before] >= positions[edge.after]:
            raise TaskGraphError(f"Schedule violates {edge.kind.value} dependency {edge.before} -> {edge.after}")
    tasks = {task.id: task for task in graph.tasks}
    for placement in schedule.placements:
        task = tasks[placement.task_id]
        if not isinstance(placement.target, ExecutionTarget) or placement.target not in task.implementations:
            raise TaskGraphError(f"Task {task.id} has no {placement.target} implementation")
        if not math.isfinite(placement.estimated_cost) or placement.estimated_cost < 0:
            raise TaskGraphError("Schedule contains invalid estimated cost")
        if isinstance(task.structure, BranchRegion):
            children = task.structure.then_graph, task.structure.else_graph
        elif isinstance(task.structure, LoopRegion):
            children = (task.structure.body,)
        else:
            children = ()
        if len(children) != len(placement.children):
            raise TaskGraphError(f"Schedule changes the control structure of task {task.id}")
        for child_graph, child_schedule in zip(children, placement.children, strict=True):
            verify_schedule(child_graph, child_schedule)
    if schedule.groups and tuple((group.target, group.task_ids) for group in schedule.groups) != tuple(
        (group.target, group.task_ids) for group in _groups(schedule.placements)
    ):
        raise TaskGraphError("Execution groups do not match the actual schedule")


@dataclass
class _State:
    # Cost credit is deliberately at canonical resource granularity. Any partial
    # write invalidates opposite-side credit, which is pessimistic but safe. It
    # does not widen the exact write/defined regions consumed by the runtime.
    valid: dict[str, set[ExecutionTarget]]
    allocated: set[str]
    previous: ExecutionTarget | None = None

    def copy(self) -> _State:
        return _State({name: set(targets) for name, targets in self.valid.items()}, set(self.allocated), self.previous)


def _merge_states(states: tuple[_State, ...]) -> _State:
    names = set().union(*(state.valid for state in states))
    valid = {
        name: set.intersection(*(state.valid.get(name, {ExecutionTarget.CPU}) for state in states)) for name in names
    }
    allocated = set().union(*(state.allocated for state in states))
    # After conditional execution a submission adjacency credit is not proved.
    return _State(valid, allocated)


def _all_tasks(graph: TaskGraph):
    for task in graph.tasks:
        yield task
        if isinstance(task.structure, BranchRegion):
            yield from _all_tasks(task.structure.then_graph)
            yield from _all_tasks(task.structure.else_graph)
        elif isinstance(task.structure, LoopRegion):
            yield from _all_tasks(task.structure.body)


def schedule_tasks(
    graph: TaskGraph,
    costs: Mapping[str, TaskCost] | None = None,
    *,
    required_targets: Mapping[str, ExecutionTarget | str] | None = None,
    initial_residency: Iterable[str] = (),
    resource_bytes: Mapping[str, int] | None = None,
    memory_budget_bytes: int | None = None,
    transfer_cost: float = 1.0,
    transfer_byte_cost: float = 0.0,
    launch_cost: float = 0.0,
    target_switch_cost: float = 0.0,
    preserve_source_order: bool = False,
) -> TaskSchedule:
    """Greedily place ready tasks using execution, transfers, and live memory.

    Repeated consumers of the same version share transfer cost. Native work only
    invalidates GPU credit for the resources it may write. Unknown calls publish
    and invalidate conservatively. Proven whole-resource overwrites need no
    incoming value; partial/conditional writes always preserve the old version.

    A memory budget requires sizes for GPU-touched resources. Evictions account
    for publishing a sole device copy. Dynamic branches use the maximum arm cost
    and intersect guaranteed validity at the join; counted loops are costed as
    one body traversal while retaining their actual bounds and zero-trip path.
    These approximations affect profitability, never execution semantics.
    """
    verify_task_graph(graph)
    costs = dict(costs or {})
    required = {name: ExecutionTarget(target) for name, target in (required_targets or {}).items()}
    sizes = dict(resource_bytes or {})
    tasks = tuple(_all_tasks(graph))
    all_ids = {task.id for task in tasks}
    if len(tasks) != len(all_ids):
        raise TaskGraphError("Hierarchical task identities must be globally unique for scheduling")
    if (set(costs) | set(required)) - all_ids:
        raise TaskGraphError("Costs or target constraints reference an absent task")
    if any(not isinstance(cost, TaskCost) for cost in costs.values()):
        raise TypeError("Task cost mappings require TaskCost values")
    if any(not isinstance(size, int) or size < 0 for size in sizes.values()):
        raise ValueError("Resource sizes must be nonnegative integers")
    if memory_budget_bytes is not None and (not isinstance(memory_budget_bytes, int) or memory_budget_bytes < 0):
        raise ValueError("Device memory budget must be a nonnegative integer")
    if any(
        not math.isfinite(value) or value < 0
        for value in (transfer_cost, transfer_byte_cost, launch_cost, target_switch_cost)
    ):
        raise ValueError("Scheduling overhead costs must be finite and nonnegative")
    for task in tasks:
        if task.id in required and required[task.id] not in task.implementations:
            raise TaskGraphError(f"Required target cannot execute task {task.id}")

    resources = set().union(*(task.effects.resources for task in tasks)) if tasks else set()
    resident = set(initial_residency)
    resources.update(resident)
    initial = _State({resource: {ExecutionTarget.CPU} for resource in resources}, resident)
    for resource in resident:
        initial.valid[resource].add(ExecutionTarget.GPU)
    if memory_budget_bytes is not None and (
        any(name not in sizes for name in resident) or sum(sizes[name] for name in resident) > memory_budget_bytes
    ):
        raise TaskGraphError("Initial residency does not fit the known device memory budget")

    def execute(current: TaskGraph, incoming: _State) -> tuple[TaskSchedule, _State]:
        state = incoming.copy()
        remaining = {task.id: task for task in current.tasks}
        positions = {task.id: index for index, task in enumerate(current.tasks)}
        predecessors = {task.id: set() for task in current.tasks}
        for edge in current.dependencies:
            predecessors[edge.after].add(edge.before)
        done: set[str] = set()
        placements = []

        def candidate(task: Task, target: ExecutionTarget) -> tuple[ScheduledTask, _State] | None:
            after = state.copy()
            model = costs.get(task.id, TaskCost())
            estimate = model.cpu if target == ExecutionTarget.CPU else model.gpu + launch_cost
            if after.previous is not None and after.previous != target:
                estimate += target_switch_cost
            transfers = []
            evictions = []
            effects = task.header_effects if task.structure is not None else task.effects

            def ensure(resource: str, destination: ExecutionTarget, reason="access"):
                nonlocal estimate
                valid = after.valid.setdefault(resource, {ExecutionTarget.CPU})
                if destination not in valid:
                    nbytes = sizes.get(resource)
                    estimate += transfer_cost + (nbytes or 0) * transfer_byte_cost
                    transfers.append(TransferEstimate(resource, destination, nbytes, reason))
                    valid.add(destination)

            touched = effects.data_resources
            if effects.opaque:
                touched = frozenset(after.valid)
            if target == ExecutionTarget.GPU:
                needed = set(touched) - after.allocated
                if memory_budget_bytes is not None:
                    if any(resource not in sizes for resource in touched | after.allocated):
                        return None
                    if sum(sizes[resource] for resource in touched) > memory_budget_bytes:
                        return None
                    occupied = sum(sizes[resource] for resource in after.allocated | needed)
                    victims = after.allocated - set(touched)

                    # Prefer values with distant/no source-order future use;
                    # legality does not depend on this eviction cost heuristic.
                    def next_use(resource):
                        return min(
                            (
                                positions[value.id]
                                for value in remaining.values()
                                if resource in value.effects.resources
                            ),
                            default=math.inf,
                        )

                    for resource in sorted(victims, key=lambda name: (-next_use(name), name)):
                        if occupied <= memory_budget_bytes:
                            break
                        ensure(resource, ExecutionTarget.CPU, "eviction")
                        after.valid[resource].discard(ExecutionTarget.GPU)
                        after.allocated.remove(resource)
                        evictions.append(resource)
                        occupied -= sizes[resource]
                    if occupied > memory_budget_bytes:
                        return None
                after.allocated.update(needed)
            overwritten = {access.resource for access in effects.must_overwrites if access.region.intervals is None}
            read_resources = {access.resource for access in effects.reads}
            for resource in sorted(touched - (overwritten - read_resources)):
                ensure(resource, target)
            for resource in {access.resource for access in effects.may_writes}:
                after.valid[resource] = {target}
            if effects.opaque:
                after.valid = {resource: {target} for resource in after.valid}
            for resource in effects.lifetime_resources:
                after.valid[resource] = {ExecutionTarget.CPU}
                after.allocated.discard(resource)
            children = ()
            if isinstance(task.structure, BranchRegion):
                then_schedule, then_state = execute(task.structure.then_graph, after)
                else_schedule, else_state = execute(task.structure.else_graph, after)
                children = then_schedule, else_schedule
                estimate += max(child.estimated_cost for child in children)
                after = _merge_states((then_state, else_state))
            elif isinstance(task.structure, LoopRegion):
                body_schedule, body_state = execute(task.structure.body, after)
                children = (body_schedule,)
                estimate += body_schedule.estimated_cost
                after = _merge_states((after, body_state))
            after.previous = None if children else target
            return ScheduledTask(task.id, target, estimate, tuple(transfers), tuple(evictions), children), after

        while remaining:
            ready = [task for task in remaining.values() if predecessors[task.id] <= done]
            if preserve_source_order:
                ready = [min(ready, key=lambda task: positions[task.id])]
            choices = []
            for task in ready:
                targets = (
                    (required[task.id],)
                    if task.id in required
                    else sorted(task.implementations, key=lambda value: value.value)
                )
                for target in targets:
                    proposal = candidate(task, target)
                    if proposal is not None:
                        placement, next_state = proposal
                        # Equal costs prefer the current submission target,
                        # then original order, then CPU for reproducibility.
                        key = (placement.estimated_cost, target != state.previous, positions[task.id], target.value)
                        choices.append((key, placement, next_state))
            if not choices:
                raise TaskGraphError("No ready implementation fits target constraints and device memory budget")
            _key, selected, state = min(choices, key=lambda item: item[0])
            placements.append(selected)
            done.add(selected.task_id)
            del remaining[selected.task_id]
        result = TaskSchedule(tuple(placements), _groups(placements))
        verify_schedule(current, result)
        return result, state

    result, _state = execute(graph, initial)
    verify_schedule(graph, result)
    return result
