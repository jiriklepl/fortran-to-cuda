"""Immutable scheduling IR above numerical regions and native execution.

Canonical resource names identify storage, including allocation generations when
needed. Views carry byte ranges in that storage, never dummy-relative indices.
Unknown footprints use ``MemoryRegion.whole()``. A footprint is an effect proof
supplied by analysis, not inferred from argument INTENT or target feasibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ExecutionTarget(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


@dataclass(frozen=True)
class MemoryRegion:
    """Union of normalized half-open byte intervals; None means unknown/whole.

    An empty tuple denotes an empty footprint. Coordinates must already have
    been rebased to the canonical allocation by alias/view analysis.
    """

    intervals: tuple[tuple[int, int], ...] | None = None

    def __post_init__(self):
        if self.intervals is None:
            return
        intervals = sorted(tuple(pair) for pair in self.intervals)
        merged: list[tuple[int, int]] = []
        for lower, upper in intervals:
            if not isinstance(lower, int) or not isinstance(upper, int) or lower < 0 or upper < lower:
                raise ValueError("Memory regions require nonnegative half-open byte intervals")
            if lower == upper:
                continue
            if merged and lower <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(upper, merged[-1][1]))
            else:
                merged.append((lower, upper))
        object.__setattr__(self, "intervals", tuple(merged))

    @classmethod
    def whole(cls) -> MemoryRegion:
        return cls()

    @property
    def empty(self) -> bool:
        return self.intervals == ()

    @property
    def nbytes(self) -> int | None:
        return None if self.intervals is None else sum(upper - lower for lower, upper in self.intervals)

    def overlaps(self, other: MemoryRegion) -> bool:
        if self.empty or other.empty:
            return False
        if self.intervals is None or other.intervals is None:
            return True
        left = right = 0
        while left < len(self.intervals) and right < len(other.intervals):
            a, b = self.intervals[left], other.intervals[right]
            if a[0] < b[1] and b[0] < a[1]:
                return True
            if a[1] <= b[0]:
                left += 1
            else:
                right += 1
        return False

    def contains(self, other: MemoryRegion) -> bool:
        if self.intervals is None or other.empty:
            return True
        if other.intervals is None:
            return False
        return all(any(a <= lower and upper <= b for a, b in self.intervals) for lower, upper in other.intervals)


@dataclass(frozen=True)
class ResourceAccess:
    resource: str
    region: MemoryRegion = MemoryRegion()

    def __post_init__(self):
        if not isinstance(self.resource, str) or not self.resource:
            raise ValueError("Resource accesses require a canonical storage identity")
        if not isinstance(self.region, MemoryRegion):
            raise TypeError("Resource access footprints must be MemoryRegion instances")

    def overlaps(self, other: ResourceAccess) -> bool:
        return self.resource == other.resource and self.region.overlaps(other.region)


@dataclass(frozen=True)
class TaskEffects:
    reads: tuple[ResourceAccess, ...] = ()
    may_writes: tuple[ResourceAccess, ...] = ()
    must_overwrites: tuple[ResourceAccess, ...] = ()
    scalar_reads: frozenset[str] = frozenset()
    scalar_writes: frozenset[str] = frozenset()
    lifetime_resources: frozenset[str] = frozenset()
    observable_tokens: frozenset[str] = frozenset()
    opaque: bool = False
    metadata_reads: frozenset[str] = frozenset()
    ordered: bool = False

    def __post_init__(self):
        for name in ("reads", "may_writes", "must_overwrites"):
            values = tuple(getattr(self, name))
            if any(not isinstance(value, ResourceAccess) for value in values):
                raise TypeError(f"{name} requires ResourceAccess instances")
            object.__setattr__(self, name, tuple(dict.fromkeys(values)))
        for name in ("scalar_reads", "scalar_writes", "lifetime_resources", "observable_tokens", "metadata_reads"):
            values = frozenset(getattr(self, name))
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} requires nonempty identity strings")
            object.__setattr__(self, name, values)
        for overwrite in self.must_overwrites:
            if not any(
                write.resource == overwrite.resource and write.region.contains(overwrite.region)
                for write in self.may_writes
            ):
                raise ValueError("A proven overwrite must be covered by a possible write")

    @property
    def data_resources(self) -> frozenset[str]:
        return frozenset(access.resource for access in (*self.reads, *self.may_writes))

    @property
    def resources(self) -> frozenset[str]:
        return self.data_resources | self.lifetime_resources | self.metadata_reads


class DependencyKind(str, Enum):
    RAW = "RAW"
    WAR = "WAR"
    WAW = "WAW"
    CONTROL = "control"
    LIFETIME = "lifetime"
    OBSERVABLE = "observable"
    OPAQUE = "opaque"


@dataclass(frozen=True)
class TaskDependency:
    before: str
    after: str
    kind: DependencyKind
    resource: str = ""


@dataclass(frozen=True)
class BranchRegion:
    """Predicate is evaluated once; only the selected child graph executes."""

    predicate: object
    then_graph: TaskGraph
    else_graph: TaskGraph


@dataclass(frozen=True)
class LoopRegion:
    """A sequential counted loop, retained structurally including zero trips.

    The body schedule is reused each iteration. No task can be moved across the
    loop boundary and no loop-carried dependence is removed by this IR.
    """

    iterator: str
    lower: object
    upper: object
    step: object
    body: TaskGraph


@dataclass(frozen=True)
class Task:
    id: str
    effects: TaskEffects = TaskEffects()
    implementations: frozenset[ExecutionTarget] = frozenset({ExecutionTarget.CPU})
    # Explicit same-region predecessors represent execution conditions/barriers
    # that are not expressed by memory effects. Structured children never escape.
    predecessors: frozenset[str] = frozenset()
    structure: BranchRegion | LoopRegion | None = None
    header_effects: TaskEffects = TaskEffects()
    payload: object = field(default=None, compare=False, hash=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("Tasks require a nonempty identity")
        targets = frozenset(ExecutionTarget(target) for target in self.implementations)
        if not targets:
            raise ValueError("Every task requires an executable implementation")
        object.__setattr__(self, "implementations", targets)
        object.__setattr__(self, "predecessors", frozenset(self.predecessors))
        if self.structure is not None and targets != {ExecutionTarget.CPU}:
            raise ValueError("Structured control is executed by the host; child tasks retain their own targets")


@dataclass(frozen=True)
class TaskGraph:
    tasks: tuple[Task, ...]
    dependencies: tuple[TaskDependency, ...] = ()
    name: str = ""

    def __post_init__(self):
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(task.id for task in self.tasks)

    def public(self) -> dict:
        """JSON-compatible diagnostics; payloads are deliberately not serialized."""
        result = {"name": self.name, "tasks": [], "dependencies": []}
        for task in self.tasks:
            item = {
                "id": task.id,
                "implementations": sorted(target.value for target in task.implementations),
                "reads": sorted({access.resource for access in task.effects.reads}),
                "may_writes": sorted({access.resource for access in task.effects.may_writes}),
                "scalar_reads": sorted(task.effects.scalar_reads),
                "scalar_writes": sorted(task.effects.scalar_writes),
                "opaque": task.effects.opaque,
                "ordered": task.effects.ordered,
            }
            if isinstance(task.structure, BranchRegion):
                predicate = task.structure.predicate
                item["predicate"] = (
                    [str(value) for value in predicate] if isinstance(predicate, tuple) else str(predicate)
                )
                item["branch"] = [task.structure.then_graph.public(), task.structure.else_graph.public()]
            elif isinstance(task.structure, LoopRegion):
                item["loop_control"] = {
                    "iterator": task.structure.iterator,
                    "lower": str(task.structure.lower),
                    "upper": str(task.structure.upper),
                    "step": str(task.structure.step),
                }
                item["loop"] = task.structure.body.public()
            result["tasks"].append(item)
        result["dependencies"] = [
            {"before": edge.before, "after": edge.after, "kind": edge.kind.value, "resource": edge.resource}
            for edge in self.dependencies
        ]
        return result
