"""Explicit, opt-in ordinary-call GPU execution policies."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from compiler.numerical_contract import require_numerical_build_contract

if TYPE_CHECKING:
    from compiler.offload.source_applicability import SourceComputeApplicability


@dataclass(frozen=True)
class OffloadConfig:
    policy: str = "always"
    profile: dict | None = None
    host_threads: int = 4
    collective: bool = False
    profile_reason: str | None = None
    scope_transfers: str = "direct"
    scope_execution: str = "bounded"
    # None prices generated workers only. Source-backed original Fortran
    # alternatives require a separately calibrated participation contract.
    native_participation: str | None = None
    # Issued for one authenticated original region, before artifact rewriting.
    # This describes cost applicability only; it cannot authorize execution.
    source_applicability: "SourceComputeApplicability | None" = None
    # Source-bound complete-window or original-selection evidence. This is
    # independent of the per-kernel roofline profile; absent evidence never
    # authorizes AUTO.
    exact_window: dict | None = None
    # Explicit offline observation of an already-admitted complete owner.
    # This is not calibration evidence and never authorizes AUTO placement.
    scope_measurement: str | None = None

    def __post_init__(self):
        if self.policy not in {"always", "sections", "auto", "chunked", "hybrid"}:
            raise ValueError("Unknown GPU execution policy")
        if type(self.host_threads) is not int or self.host_threads < 1:
            raise ValueError("Host thread budget must be positive")
        if self.scope_transfers not in {"direct", "pinned", "pipelined", "auto", "coalesced"}:
            raise ValueError("Unknown scope transfer mode")
        if self.scope_execution not in {"bounded", "reached"}:
            raise ValueError("Unknown source scope execution mode")
        if self.native_participation not in {None, "serial", "fork_join", "fork_join_runtime", "existing_team", "unknown"}:
            raise ValueError("Unknown native Fortran participation")
        if self.source_applicability is not None:
            from compiler.offload.source_applicability import SourceComputeApplicability
            if type(self.source_applicability) is not SourceComputeApplicability:
                raise ValueError("Source compute applicability requires original-region metadata")
            if self.source_applicability.native_participation != self.native_participation:
                raise ValueError("Source compute applicability participation differs")
        if self.exact_window is not None:
            if type(self.exact_window) is not dict:
                raise ValueError("Exact window cost request must be an object")
            coverage = self.exact_window.get("coverage")
            if coverage not in {None, "original_selection", "checked_interval"}:
                raise ValueError("Unknown exact window cost coverage")
            execution = "bounded" if coverage == "original_selection" else "reached"
            if self.scope_execution != execution or self.policy not in {"auto", "sections"} or self.collective:
                raise ValueError(f"Exact window requires a serial {execution} auto/sections owner")
        if self.scope_measurement is not None and (
                type(self.scope_measurement) is not str or self.scope_measurement not in {"native", "resident"}
                or self.policy != "sections" or self.scope_execution != "reached"
                or self.collective or self.exact_window is not None):
            raise ValueError("Scope measurement requires a native/resident serial reached sections owner")
        if self.profile is not None:
            try:
                require_numerical_build_contract(self.profile.get("numerical_contract")
                                                 if isinstance(self.profile, dict) else None)
            except ValueError as error:
                # Reading legacy evidence does not authorize a changed backend.
                object.__setattr__(self, "profile", None)
                object.__setattr__(self, "profile_reason", "hardware profile " + str(error))
