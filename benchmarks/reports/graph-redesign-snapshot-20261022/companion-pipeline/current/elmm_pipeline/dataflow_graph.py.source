"""Canonical handles, context dataflow graph, min-cut placement optimizer, and residency scheduler.

This module unifies isolated compiler scopes into a whole-program heterogeneous
dataflow graph. It computes canonical allocation identities via value-numbering,
optimizes stage placement via min-cut graph partitioning, and schedules
optimistic GPU residency, lazy fallback fences, and proactive DtoH transfers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import itertools
import re
from typing import Any


class BufferState(Enum):
    HOST_CURRENT = "HOST_CURRENT"
    DEVICE_CURRENT = "DEVICE_CURRENT"
    BOTH_VALID = "BOTH_VALID"


class ExecutionTarget(Enum):
    CPU = "CPU"
    GPU = "GPU"


class ResidencyActionType(Enum):
    RESIDENT_HANDOFF = "RESIDENT_HANDOFF"
    HtoD_TRANSFER = "HtoD_TRANSFER"
    DtoH_TRANSFER = "DtoH_TRANSFER"
    PROACTIVE_ASYNC_DtoH = "PROACTIVE_ASYNC_DtoH"
    LAZY_SYNC_FENCE = "LAZY_SYNC_FENCE"


@dataclass(frozen=True)
class CanonicalBuffer:
    handle_id: int
    canonical_name: str
    shape: tuple[int, ...]
    element_bytes: int
    bytes: int


class CanonicalBufferCatalog:
    """Assigns and unifies canonical allocation identities across routines and scopes."""

    def __init__(self) -> None:
        self._symbol_to_id: dict[str, int] = {}
        self._buffers: dict[int, CanonicalBuffer] = {}
        self._aliases: dict[str, str] = {}
        self._local_aliases: dict[tuple[str, int], int] = {}
        self._next_id: int = 1

    @staticmethod
    def normalize_name(symbol: str) -> str:
        return symbol.strip().lower()

    def _resolve_symbol(self, symbol: str) -> str:
        norm = self.normalize_name(symbol)
        visited = set()
        while norm in self._aliases:
            if norm in visited:
                break
            visited.add(norm)
            norm = self._aliases[norm]
        return norm

    def get_or_create(
        self,
        symbol: str,
        shape: tuple[int, ...],
        element_bytes: int = 8,
    ) -> CanonicalBuffer:
        resolved = self._resolve_symbol(symbol)
        if resolved in self._symbol_to_id:
            return self._buffers[self._symbol_to_id[resolved]]

        total_bytes = element_bytes
        for dim in shape:
            total_bytes *= dim

        handle_id = self._next_id
        self._next_id += 1

        buf = CanonicalBuffer(
            handle_id=handle_id,
            canonical_name=resolved,
            shape=shape,
            element_bytes=element_bytes,
            bytes=total_bytes,
        )
        self._symbol_to_id[resolved] = handle_id
        self._buffers[handle_id] = buf
        return buf

    def unify(self, alias_symbol: str, canonical_symbol: str) -> None:
        norm_alias = self.normalize_name(alias_symbol)
        norm_canon = self._resolve_symbol(canonical_symbol)
        self._aliases[norm_alias] = norm_canon

        # If alias was already assigned a separate buffer, point its symbol to canonical
        if norm_alias in self._symbol_to_id and norm_canon in self._symbol_to_id:
            old_id = self._symbol_to_id[norm_alias]
            canon_id = self._symbol_to_id[norm_canon]
            if old_id != canon_id:
                self._symbol_to_id[norm_alias] = canon_id

    def lookup(self, symbol: str) -> CanonicalBuffer:
        resolved = self._resolve_symbol(symbol)
        if resolved not in self._symbol_to_id:
            raise KeyError(f"Symbol '{symbol}' (resolved to '{resolved}') not in catalog")
        return self._buffers[self._symbol_to_id[resolved]]

    def register_local_alias(self, scope: str, local_id: int, canonical_symbol: str) -> None:
        norm_scope = scope.strip().lower()
        buf = self.lookup(canonical_symbol)
        self._local_aliases[(norm_scope, local_id)] = buf.handle_id

    def resolve_local_id(self, scope: str, local_id: int) -> CanonicalBuffer:
        norm_scope = scope.strip().lower()
        key = (norm_scope, local_id)
        if key not in self._local_aliases:
            raise KeyError(f"Scope '{scope}' local ID {local_id} not registered")
        return self._buffers[self._local_aliases[key]]


@dataclass
class HardwareCostModel:
    pcie_latency_seconds: float = 1e-5
    pcie_bandwidth_bytes_per_sec: float = 16e9  # 16 GB/s
    kernel_launch_overhead_seconds: float = 5e-6

    def transfer_time(self, bytes_count: int) -> float:
        if bytes_count <= 0:
            return 0.0
        return self.pcie_latency_seconds + (bytes_count / self.pcie_bandwidth_bytes_per_sec)


@dataclass
class StageNode:
    name: str
    cpu_cost: float
    gpu_cost: float | None = None
    has_gpu: bool = False


@dataclass
class DataflowEdge:
    producer: str
    consumer: str
    buffer: CanonicalBuffer
    transfer_bytes: int


class ContextDataflowGraph:
    """Represents computation stages and buffer dataflow dependencies."""

    def __init__(self) -> None:
        self.stages: dict[str, StageNode] = {}
        self.edges: list[DataflowEdge] = []

    def add_stage(
        self,
        name: str,
        cpu_cost: float,
        gpu_cost: float | None = None,
        has_gpu: bool = False,
    ) -> StageNode:
        stage = StageNode(
            name=name,
            cpu_cost=cpu_cost,
            gpu_cost=gpu_cost,
            has_gpu=has_gpu and (gpu_cost is not None),
        )
        self.stages[name] = stage
        return stage

    def add_dependency(
        self,
        producer: str,
        consumer: str,
        buffer: CanonicalBuffer,
        transfer_bytes: int | None = None,
    ) -> DataflowEdge:
        if producer not in self.stages:
            raise KeyError(f"Producer stage '{producer}' not in graph")
        if consumer not in self.stages:
            raise KeyError(f"Consumer stage '{consumer}' not in graph")

        bytes_count = transfer_bytes if transfer_bytes is not None else buffer.bytes
        edge = DataflowEdge(
            producer=producer,
            consumer=consumer,
            buffer=buffer,
            transfer_bytes=bytes_count,
        )
        self.edges.append(edge)
        return edge


@dataclass
class EdgePlan:
    producer: str
    consumer: str
    buffer: CanonicalBuffer
    is_cut: bool
    transfer_cost_seconds: float


@dataclass
class PlacementPlan:
    placements: dict[str, ExecutionTarget]
    edge_plans: dict[tuple[str, str], EdgePlan]
    total_cost_seconds: float

    def get_edge_plan(self, producer: str, consumer: str) -> EdgePlan:
        return self.edge_plans[(producer, consumer)]


class PlacementOptimizer:
    """Solves the heterogeneous min-cut graph partitioning problem for whole-program placement."""

    def __init__(self, hardware: HardwareCostModel | None = None) -> None:
        self.hardware = hardware or HardwareCostModel()

    def optimize(self, graph: ContextDataflowGraph) -> PlacementPlan:
        # Determine valid target candidates per stage
        candidates: dict[str, list[ExecutionTarget]] = {}
        for name, stage in graph.stages.items():
            options = [ExecutionTarget.CPU]
            if stage.has_gpu and stage.gpu_cost is not None:
                options.append(ExecutionTarget.GPU)
            candidates[name] = options

        stage_names = list(graph.stages.keys())
        all_combinations = list(itertools.product(*(candidates[name] for name in stage_names)))

        best_cost = float("inf")
        best_placement: dict[str, ExecutionTarget] = {}
        best_edge_plans: dict[tuple[str, str], EdgePlan] = {}

        for combo in all_combinations:
            placement = {name: target for name, target in zip(stage_names, combo)}
            cost = 0.0

            # Compute execution cost
            for name, target in placement.items():
                stage = graph.stages[name]
                if target == ExecutionTarget.GPU and stage.gpu_cost is not None:
                    cost += stage.gpu_cost
                else:
                    cost += stage.cpu_cost

            # Compute edge transfer costs
            current_edge_plans: dict[tuple[str, str], EdgePlan] = {}
            for edge in graph.edges:
                p_target = placement[edge.producer]
                c_target = placement[edge.consumer]
                is_cut = (p_target != c_target)
                t_cost = self.hardware.transfer_time(edge.transfer_bytes) if is_cut else 0.0
                cost += t_cost

                current_edge_plans[(edge.producer, edge.consumer)] = EdgePlan(
                    producer=edge.producer,
                    consumer=edge.consumer,
                    buffer=edge.buffer,
                    is_cut=is_cut,
                    transfer_cost_seconds=t_cost,
                )

            if cost < best_cost:
                best_cost = cost
                best_placement = placement
                best_edge_plans = current_edge_plans

        return PlacementPlan(
            placements=best_placement,
            edge_plans=best_edge_plans,
            total_cost_seconds=best_cost,
        )


@dataclass
class ResidencyAction:
    action_type: ResidencyActionType
    buffer: CanonicalBuffer
    stage: str
    details: dict[str, Any] = field(default_factory=dict)


class ResidencyCoherenceScheduler:
    """Maintains buffer state machine and generates optimistic residency / fallback actions."""

    def __init__(self, catalog: CanonicalBufferCatalog) -> None:
        self.catalog = catalog
        self._states: dict[int, BufferState] = {}
        self._active_writes: dict[str, list[CanonicalBuffer]] = {}

    def get_state(self, buffer: CanonicalBuffer) -> BufferState:
        return self._states.get(buffer.handle_id, BufferState.HOST_CURRENT)

    def set_state(self, buffer: CanonicalBuffer, state: BufferState) -> None:
        self._states[buffer.handle_id] = state

    def on_stage_begin(
        self,
        stage_name: str,
        target: ExecutionTarget,
        reads: list[CanonicalBuffer],
        writes: list[CanonicalBuffer],
    ) -> list[ResidencyAction]:
        actions: list[ResidencyAction] = []
        self._active_writes[stage_name] = list(writes)

        for buf in reads:
            state = self.get_state(buf)
            if target == ExecutionTarget.GPU:
                if state == BufferState.DEVICE_CURRENT or state == BufferState.BOTH_VALID:
                    # Memory is already on GPU: optimistic handoff
                    actions.append(
                        ResidencyAction(
                            action_type=ResidencyActionType.RESIDENT_HANDOFF,
                            buffer=buf,
                            stage=stage_name,
                            details={"reused_device_handle": buf.handle_id},
                        )
                    )
                else:
                    # HOST_CURRENT: needs upload
                    actions.append(
                        ResidencyAction(
                            action_type=ResidencyActionType.HtoD_TRANSFER,
                            buffer=buf,
                            stage=stage_name,
                        )
                    )
                    self.set_state(buf, BufferState.BOTH_VALID)
            else:  # target == ExecutionTarget.CPU
                if state == BufferState.DEVICE_CURRENT:
                    # Unlikely/un-offloaded CPU read: trigger lazy fence!
                    actions.append(
                        ResidencyAction(
                            action_type=ResidencyActionType.LAZY_SYNC_FENCE,
                            buffer=buf,
                            stage=stage_name,
                            details={"reason": "CPU read of dirty device buffer"},
                        )
                    )
                    self.set_state(buf, BufferState.BOTH_VALID)

        # Mark written buffers
        for buf in writes:
            if target == ExecutionTarget.GPU:
                self.set_state(buf, BufferState.DEVICE_CURRENT)
            else:
                self.set_state(buf, BufferState.HOST_CURRENT)

        return actions

    def on_stage_end(
        self,
        stage_name: str,
        next_predicted_stage: str | None = None,
        next_target: ExecutionTarget | None = None,
    ) -> list[ResidencyAction]:
        actions: list[ResidencyAction] = []
        writes = self._active_writes.pop(stage_name, [])

        for buf in writes:
            state = self.get_state(buf)
            if state == BufferState.DEVICE_CURRENT:
                if next_target == ExecutionTarget.CPU:
                    # Proactive asynchronous DtoH for likely CPU consumer
                    actions.append(
                        ResidencyAction(
                            action_type=ResidencyActionType.PROACTIVE_ASYNC_DtoH,
                            buffer=buf,
                            stage=stage_name,
                            details={"predicted_consumer": next_predicted_stage},
                        )
                    )
                    self.set_state(buf, BufferState.BOTH_VALID)
                elif next_target == ExecutionTarget.GPU:
                    # Remains resident on device, zero DtoH copy!
                    pass

        return actions


class UnifiedScopeBridge:
    """Generates the shared Fortran residency bridge module for multi-scope applications."""

    def __init__(self, capacity: int = 32) -> None:
        self.capacity = capacity

    def generate_fortran_bridge(self) -> str:
        return f"""! Generated by ELMM Heterogeneous Dataflow Residency Optimizer
module fort_resident_bridge
  use iso_c_binding, only: c_int, c_int32_t, c_int64_t, c_ptr, c_size_t, c_loc, c_null_ptr
  use fort_scoped_memory, only: fort_scope_create, fort_scope_close, fort_scope_set_device_budget, &
                                fort_scope_host_begin, fort_scope_host_end, fort_scope_access, &
                                fort_scope_forget_definition, &
                                FORT_SCOPE_OK, FORT_SCOPE_READ_ALL
  implicit none
  private
  public :: global_resident_context, fort_get_or_create_resident_context, &
            fort_close_resident_context, fort_ensure_host_buffer, fort_mark_device_buffer, &
            fort_invalidate_device_buffer

  integer(c_int64_t), save :: global_resident_context = 0
  logical, save :: resident_initialized = .false.

  type :: buffer_tracking
    integer(c_int64_t) :: handle = 0
    logical :: is_device_current = .false.
  end type buffer_tracking

  type(buffer_tracking), save :: tracked_buffers({self.capacity})

contains

  subroutine fort_get_or_create_resident_context(ctx)
    integer(c_int64_t), intent(out) :: ctx
    integer(c_int) :: status
    if (global_resident_context == 0) then
      status = fort_scope_create(0_c_int, global_resident_context)
      if (status == FORT_SCOPE_OK) then
        status = fort_scope_set_device_budget(global_resident_context, 268435456_c_size_t)
      end if
      resident_initialized = .true.
    end if
    ctx = global_resident_context
  end subroutine fort_get_or_create_resident_context

  subroutine fort_close_resident_context()
    integer(c_int) :: status
    if (global_resident_context /= 0) then
      status = fort_scope_close(global_resident_context)
      global_resident_context = 0
      resident_initialized = .false.
    end if
  end subroutine fort_close_resident_context

  subroutine fort_mark_device_buffer(id, buf_handle)
    integer, intent(in) :: id
    integer(c_int64_t), intent(in) :: buf_handle
    if (id >= 1 .and. id <= {self.capacity}) then
      tracked_buffers(id)%handle = buf_handle
      tracked_buffers(id)%is_device_current = .true.
    end if
  end subroutine fort_mark_device_buffer

  subroutine fort_invalidate_device_buffer(id)
    integer, intent(in) :: id
    integer(c_int) :: status
    if (id >= 1 .and. id <= {self.capacity} .and. global_resident_context /= 0) then
      if (tracked_buffers(id)%handle /= 0) then
        status = fort_scope_forget_definition(global_resident_context, tracked_buffers(id)%handle)
        tracked_buffers(id)%is_device_current = .false.
      end if
    end if
  end subroutine fort_invalidate_device_buffer

  subroutine fort_ensure_host_buffer(id)
    integer, intent(in) :: id
    integer(c_int) :: status
    type(fort_scope_access) :: access
    if (id >= 1 .and. id <= {self.capacity} .and. global_resident_context /= 0) then
      if (tracked_buffers(id)%is_device_current .and. tracked_buffers(id)%handle /= 0) then
        access%flags = FORT_SCOPE_READ_ALL
        access%read_count = 0
        access%reads = c_null_ptr
        status = fort_scope_host_begin(global_resident_context, tracked_buffers(id)%handle, access)
        status = fort_scope_host_end(global_resident_context, tracked_buffers(id)%handle)
        tracked_buffers(id)%is_device_current = .false.
      end if
    end if
  end subroutine fort_ensure_host_buffer

end module fort_resident_bridge
"""

    def remap_dynamics_identities(self, source: str, offset: int = 10) -> str:
        """Remap fort_scope_register integer IDs in dynamics to avoid collisions with SGS handles."""
        def _repl(m: re.Match) -> str:
            ident = int(m.group(2))
            return f"{m.group(1)}{ident + offset}_c_int64_t"
        return re.sub(r"(fort_scope_register\([^,]+,\s*)([0-9]+)_c_int64_t", _repl, source)

    def patch_subgrid_source(self, source: str) -> str:
        res = source
        # Inject use fort_resident_bridge
        if "use fort_resident_bridge" not in res:
            res = "use fort_resident_bridge\n" + res

        # Replace local context assignment
        res = res.replace(
            "fort_selection_context_0a0443cf2c90 = 0",
            "call fort_get_or_create_resident_context(fort_selection_context_0a0443cf2c90)",
        )
        # Replace explicit create
        res = re.sub(
            r"fort_actual_status\s*=\s*fort_scope_create\(0_c_int,\s*fort_selection_context_0a0443cf2c90\)",
            "call fort_get_or_create_resident_context(fort_selection_context_0a0443cf2c90)",
            res,
        )
        # Replace normal exit fort_scope_close with device buffer marking
        res = res.replace(
            "fort_close_status = fort_scope_close(fort_selection_context_0a0443cf2c90)",
            "call fort_mark_device_buffer(1, fort_selection_context_0a0443cf2c90)",
        )
        return res

    def patch_dynamics_source(self, source: str) -> str:
        res = source
        if "use fort_resident_bridge" not in res:
            res = "use fort_resident_bridge\n" + res

        # Replace create with get_or_create_resident_context
        res = re.sub(
            r"fort_actual_status\s*=\s*fort_scope_create\(0_c_int,\s*fort_selection_context_2d2e69bd1df9\)",
            "call fort_get_or_create_resident_context(fort_selection_context_2d2e69bd1df9)",
            res,
        )
        return res
