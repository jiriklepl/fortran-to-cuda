"""Explicit, opt-in ordinary-call GPU execution policies."""

from dataclasses import dataclass


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

    def __post_init__(self):
        if self.policy not in {"always", "sections", "auto", "chunked", "hybrid"}:
            raise ValueError("Unknown GPU execution policy")
        if type(self.host_threads) is not int or self.host_threads < 1:
            raise ValueError("Host thread budget must be positive")
        if self.scope_transfers not in {"direct", "pinned", "pipelined", "auto"}:
            raise ValueError("Unknown scope transfer mode")
        if self.scope_execution not in {"bounded", "reached"}:
            raise ValueError("Unknown source scope execution mode")
        if self.native_participation not in {None, "serial", "fork_join", "fork_join_runtime", "existing_team", "unknown"}:
            raise ValueError("Unknown native Fortran participation")
