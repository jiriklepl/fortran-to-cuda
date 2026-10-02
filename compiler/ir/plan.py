"""Checked execution plan shared by analysis and target emitters."""

from __future__ import annotations

from dataclasses import dataclass

from .nodes import Assignment, Block, Loop, Symbol


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


@dataclass(frozen=True)
class ExecutionPlan:
    steps: tuple[HostBlock | ParallelRegion, ...]

    @property
    def regions(self) -> tuple[ParallelRegion, ...]:
        return tuple(step for step in self.steps if isinstance(step, ParallelRegion))
