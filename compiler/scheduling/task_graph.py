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
from compiler.ir.task_graph import BranchRegion, DeferredPlacement, ExecutionTarget, LoopRegion, Task, TaskGraph


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
    target: ExecutionTarget | None
    estimated_cost: float | None = 0.0
    transfers: tuple[TransferEstimate, ...] = ()
    evictions: tuple[str, ...] = ()
    children: tuple[TaskSchedule, ...] = ()
    deferred_placement: DeferredPlacement | None = None


@dataclass(frozen=True)
class ExecutionGroup:
    """Consecutive same-target work; grouping does not itself prove fusion."""

    target: ExecutionTarget | None
    task_ids: tuple[str, ...]


@dataclass(frozen=True)
class TaskSchedule:
    placements: tuple[ScheduledTask, ...]
    groups: tuple[ExecutionGroup, ...] = ()

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(placement.task_id for placement in self.placements)

    @property
    def estimated_cost(self) -> float | None:
        if any(placement.estimated_cost is None for placement in self.placements):
            return None
        return sum(placement.estimated_cost for placement in self.placements)

    def public(self) -> dict:
        return {
            "estimated_cost": self.estimated_cost,
            "placements": [
                {
                    "task": placement.task_id,
                    "target": placement.target.value if placement.target is not None else None,
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
                    **({"deferred_placement": placement.deferred_placement.public()}
                       if placement.deferred_placement is not None else {}),
                }
                for placement in self.placements
            ],
            "groups": [{"target": group.target.value if group.target is not None else None,
                        "tasks": list(group.task_ids)} for group in self.groups],
        }


def _groups(placements: Iterable[ScheduledTask]) -> tuple[ExecutionGroup, ...]:
    groups: list[ExecutionGroup] = []
    for placement in placements:
        if (groups and placement.target is not None and groups[-1].target == placement.target
                and not placement.children):
            previous = groups.pop()
            groups.append(ExecutionGroup(placement.target, (*previous.task_ids, placement.task_id)))
        else:
            groups.append(ExecutionGroup(placement.target, (placement.task_id,)))
    return tuple(groups)


def _verify_schedule(graph: TaskGraph, schedule: TaskSchedule, *, incoming_uncertain=False) -> None:
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
    original_positions = {task.id: index for index, task in enumerate(graph.tasks)}
    remaining = set(tasks)
    uncertain = incoming_uncertain
    for placement in schedule.placements:
        task = tasks[placement.task_id]
        if uncertain and placement.task_id != min(remaining, key=original_positions.__getitem__):
            raise TaskGraphError("Unresolved runtime placement must retain remaining original source order")
        remaining.remove(placement.task_id)
        deferred = task.deferred_placement
        if deferred is not None:
            if (placement.target is not None or placement.deferred_placement is None
                    or placement.deferred_placement.public() != deferred.public()
                    or placement.estimated_cost is not None or placement.transfers or placement.evictions):
                raise TaskGraphError("Runtime placement must retain its exact policy and unavailable estimate")
            uncertain = True
        elif (placement.deferred_placement is not None or not isinstance(placement.target, ExecutionTarget)
              or placement.target not in task.implementations):
            raise TaskGraphError(f"Task {task.id} has no {placement.target} implementation")
        if (placement.estimated_cost is not None
                and (not math.isfinite(placement.estimated_cost) or placement.estimated_cost < 0)):
            raise TaskGraphError("Schedule contains invalid estimated cost")
        if uncertain and (placement.estimated_cost is not None or placement.transfers or placement.evictions):
            raise TaskGraphError("Unresolved runtime placement cannot grant later residency cost credit")
        if isinstance(task.structure, BranchRegion):
            children = task.structure.then_graph, task.structure.else_graph
        elif isinstance(task.structure, LoopRegion):
            children = (task.structure.body,)
        else:
            children = ()
        if len(children) != len(placement.children):
            raise TaskGraphError(f"Schedule changes the control structure of task {task.id}")
        for child_graph, child_schedule in zip(children, placement.children, strict=True):
            _verify_schedule(child_graph, child_schedule, incoming_uncertain=uncertain)
        if any(child.estimated_cost is None for child in placement.children):
            if placement.estimated_cost is not None:
                raise TaskGraphError("Structured task cannot hide an unavailable child estimate")
            uncertain = True
        if placement.estimated_cost is None and not uncertain:
            raise TaskGraphError("Unavailable estimate requires an unresolved runtime placement")
    if schedule.groups and tuple((group.target, group.task_ids) for group in schedule.groups) != tuple(
        (group.target, group.task_ids) for group in _groups(schedule.placements)
    ):
        raise TaskGraphError("Execution groups do not match the actual schedule")


def verify_schedule(graph: TaskGraph, schedule: TaskSchedule) -> None:
    """Verify complete placement and runtime bindings from a known graph entry."""
    _verify_schedule(graph, schedule)


@dataclass
class _State:
    # Cost credit is deliberately at canonical resource granularity. Any partial
    # write invalidates opposite-side credit, which is pessimistic but safe. It
    # does not widen the exact write/defined regions consumed by the runtime.
    valid: dict[str, set[ExecutionTarget]]
    allocated: set[str]
    previous: ExecutionTarget | None = None
    uncertain: bool = False

    def copy(self) -> _State:
        return _State({name: set(targets) for name, targets in self.valid.items()}, set(self.allocated),
                      self.previous, self.uncertain)


def _merge_states(states: tuple[_State, ...]) -> _State:
    names = set().union(*(state.valid for state in states))
    valid = {
        name: set.intersection(*(state.valid.get(name, {ExecutionTarget.CPU}) for state in states)) for name in names
    }
    allocated = set().union(*(state.allocated for state in states))
    # After conditional execution a submission adjacency credit is not proved.
    return _State(valid, allocated, uncertain=any(state.uncertain for state in states))


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
    group_search_budget: int = 64,
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

    Connected GPU proposals also compare a complete ready prefix with its
    independently priced alternative. This can amortize the first upload over
    several consumers. The budget bounds proposal evaluations per ready-list
    decision; it never limits graph coverage, groups or storage lifetime.
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
    if type(group_search_budget) is not int or group_search_budget < 0:
        raise ValueError("Group search budget must be a nonnegative integer")
    if any(
        not math.isfinite(value) or value < 0
        for value in (transfer_cost, transfer_byte_cost, launch_cost, target_switch_cost)
    ):
        raise ValueError("Scheduling overhead costs must be finite and nonnegative")
    for task in tasks:
        if task.deferred_placement is not None and task.id in required:
            raise TaskGraphError("Runtime-deferred policy cannot also have a static required target")
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
        search_remaining = 0

        def candidate(task: Task, target: ExecutionTarget, before: _State | None = None
                      ) -> tuple[ScheduledTask, _State] | None:
            if task.deferred_placement is not None:
                raise TaskGraphError("Runtime-deferred task cannot be priced as a fixed physical target")
            after = (state if before is None else before).copy()
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

                    for resource in sorted(victims, key=lambda name: (
                            ExecutionTarget.CPU not in after.valid.get(name, set()), -next_use(name), name)):
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
                if any(child.estimated_cost is None for child in children):
                    estimate = None
                else:
                    estimate += max(child.estimated_cost for child in children)
                after = _merge_states((then_state, else_state))
            elif isinstance(task.structure, LoopRegion):
                body_schedule, body_state = execute(task.structure.body, after)
                children = (body_schedule,)
                if body_schedule.estimated_cost is None:
                    estimate = None
                else:
                    estimate += body_schedule.estimated_cost
                after = _merge_states((after, body_state))
            after.previous = None if children else target
            return ScheduledTask(task.id, target, estimate, tuple(transfers), tuple(evictions), children), after

        def unresolved(task):
            """Retain source order without inventing physical replica/cost facts.

            A runtime choice may evict any replica. Its actual accesses and
            memory budget remain runtime responsibilities; clearing this cost
            state emits no copy, eviction, publication or allocation operation.
            """
            after = state.copy()
            after.uncertain = True
            after.valid = {name: set() for name in after.valid}
            after.allocated.clear()
            after.previous = None
            deferred = task.deferred_placement
            target = None if deferred is not None else required.get(task.id, (
                ExecutionTarget.CPU if ExecutionTarget.CPU in task.implementations else ExecutionTarget.GPU))
            if target == ExecutionTarget.GPU and memory_budget_bytes is not None:
                touched = task.effects.data_resources
                if (any(name not in sizes for name in touched)
                        or sum(sizes[name] for name in touched) > memory_budget_bytes):
                    raise TaskGraphError("Required GPU working set cannot fit the known device memory budget")
            children = ()
            if isinstance(task.structure, BranchRegion):
                left, left_state = execute(task.structure.then_graph, after)
                right, right_state = execute(task.structure.else_graph, after)
                children = left, right
                after = _merge_states((left_state, right_state))
            elif isinstance(task.structure, LoopRegion):
                body, body_state = execute(task.structure.body, after)
                children = (body,)
                after = _merge_states((after, body_state))
            return ScheduledTask(task.id, target, None, children=children, deferred_placement=deferred), after

        def connected_group(task, placement, next_state):
            """Price, then commit one dependency-valid connected GPU prefix.

            Alternative states are private estimates. Neither search exhaustion
            nor an unprofitable suffix removes work from the actual schedule.
            Structured controls remain single tasks and keep their own schedules.
            """
            nonlocal search_remaining
            if (not search_remaining or preserve_source_order or task.structure is not None
                    or placement.estimated_cost is None or next_state.uncertain
                    or task.effects.opaque or task.effects.lifetime_resources
                    or ExecutionTarget.CPU not in task.implementations or task.id in required):
                return (placement,), next_state, placement.estimated_cost
            baseline = candidate(task, ExecutionTarget.CPU)
            if baseline is None:
                return (placement,), next_state, placement.estimated_cost
            base_placement, base_state = baseline
            base_total, gpu_total = base_placement.estimated_cost, placement.estimated_cost
            first_difference = base_total - gpu_total
            best = (placement,), next_state, placement.estimated_cost
            selected = [placement]
            selected_ids = {task.id}
            connected = set(task.effects.data_resources)
            while search_remaining:
                frontier = [value for value in remaining.values()
                            if value.id not in selected_ids and predecessors[value.id] <= done | selected_ids
                            and value.deferred_placement is None
                            and value.structure is None and not value.effects.opaque
                            and not value.effects.lifetime_resources
                            and ExecutionTarget.GPU in value.implementations
                            and required.get(value.id, ExecutionTarget.GPU) == ExecutionTarget.GPU
                            and connected & value.effects.data_resources]
                proposals = []
                # A small deterministic frontier preserves budget for deeper
                # shared-read chains instead of spending it all on siblings.
                for value in frontier[:4]:
                    if not search_remaining:
                        break
                    search_remaining -= 1
                    gpu = candidate(value, ExecutionTarget.GPU, next_state)
                    targets = ((required[value.id],) if value.id in required else
                               sorted(value.implementations, key=lambda item: item.value))
                    alternatives = [item for target in targets
                                    if (item := candidate(value, target, base_state)) is not None]
                    if gpu is None or not alternatives:
                        continue
                    alternative = min(alternatives, key=lambda item: (
                        item[0].estimated_cost, item[0].target.value))
                    proposals.append(((gpu[0].estimated_cost - alternative[0].estimated_cost,
                                       positions[value.id]), value, gpu, alternative))
                if not proposals:
                    break
                _key, value, gpu, alternative = min(proposals, key=lambda item: item[0])
                selected.append(gpu[0])
                selected_ids.add(value.id)
                connected.update(value.effects.data_resources)
                next_state, base_state = gpu[1], alternative[1]
                gpu_total += gpu[0].estimated_cost
                base_total += alternative[0].estimated_cost
                credit = base_total - gpu_total - first_difference
                if credit > 0 and placement.estimated_cost - credit < best[2]:
                    best = tuple(selected), next_state, placement.estimated_cost - credit
            return best

        while remaining:
            ready = [task for task in remaining.values() if predecessors[task.id] <= done]
            if state.uncertain or any(task.deferred_placement is not None for task in ready):
                task = min(ready, key=lambda value: positions[value.id])
                placement, state = unresolved(task)
                placements.append(placement)
                done.add(task.id)
                del remaining[task.id]
                continue
            if preserve_source_order:
                ready = [min(ready, key=lambda task: positions[task.id])]
            choices = []
            search_remaining = group_search_budget
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
                        selected, next_state, score = (connected_group(task, placement, next_state)
                            if target == ExecutionTarget.GPU else ((placement,), next_state, placement.estimated_cost))
                        # Equal costs prefer the current submission target,
                        # then original order, then CPU for reproducibility.
                        key = (score if score is not None else math.inf,
                               target != state.previous, positions[task.id], target.value)
                        choices.append((key, selected, next_state))
            if not choices:
                raise TaskGraphError("No ready implementation fits target constraints and device memory budget")
            _key, selected, state = min(choices, key=lambda item: item[0])
            placements.extend(selected)
            for placement in selected:
                done.add(placement.task_id)
                del remaining[placement.task_id]
        result = TaskSchedule(tuple(placements), _groups(placements))
        _verify_schedule(current, result, incoming_uncertain=incoming.uncertain)
        return result, state

    result, _state = execute(graph, initial)
    verify_schedule(graph, result)
    return result
