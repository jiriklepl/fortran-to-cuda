"""Checked execution plan shared by analysis and target emitters."""

from __future__ import annotations

from dataclasses import dataclass

from .nodes import Assignment, Block, Expr, Loop, SourceLocation, Symbol


@dataclass(frozen=True)
class RegionReport:
    domain: str
    reads: str
    writes: str
    schedule: str
    raw: str
    war: str
    waw: str
    conservative: bool = False


@dataclass(frozen=True)
class HostBlock:
    assignments: tuple[Assignment, ...]
    read_symbols: tuple[Symbol, ...] = ()
    write_symbols: tuple[Symbol, ...] = ()


@dataclass(frozen=True)
class RegionSchedule:
    axis_order: tuple[int, ...]
    tile_sizes: tuple[int, ...] = ()
    cuda_threads: int = 256


@dataclass(frozen=True)
class ParallelRegion:
    id: int
    loops: tuple[Loop, ...]
    assignments: tuple[Assignment, ...]
    private_symbols: tuple[Symbol, ...]
    captured_symbols: tuple[Symbol, ...]
    report: RegionReport
    body: Block
    read_symbols: tuple[Symbol, ...] = ()
    write_symbols: tuple[Symbol, ...] = ()

    schedule: RegionSchedule | None = None


@dataclass(frozen=True)
class ConditionalRegion:
    condition: Expr
    then_plan: ExecutionPlan
    else_plan: ExecutionPlan
    location: SourceLocation
    read_symbols: tuple[Symbol, ...] = ()
    write_symbols: tuple[Symbol, ...] = ()


@dataclass(frozen=True)
class ExecutionPlan:
    steps: tuple[HostBlock | ParallelRegion | ConditionalRegion, ...]
    reports: tuple[str, ...] = ()

    @property
    def regions(self) -> tuple[ParallelRegion, ...]:
        result = []
        for step in self.steps:
            if isinstance(step, ParallelRegion):
                result.append(step)
            elif isinstance(step, ConditionalRegion):
                result.extend(step.then_plan.regions)
                result.extend(step.else_plan.regions)
        return tuple(result)
