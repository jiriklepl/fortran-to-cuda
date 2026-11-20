"""Shared graph execution configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OffloadConfig:
    host_threads: int = 4
    collective: bool = False
    scope_transfers: str = "direct"
    # The native alternative is the original Fortran, not the generated worker.
    # Static work counts describe only the generated worker.
    original_native: bool = False
    def __post_init__(self):
        if type(self.host_threads) is not int or self.host_threads < 1:
            raise ValueError("Host thread budget must be positive")
        if self.scope_transfers == "pipelined":
            raise ValueError("Pipelined slab batching was removed; use shared graph execution")
        if self.scope_transfers not in {"direct", "pinned", "auto", "coalesced"}:
            raise ValueError("Unknown scope transfer mode")
        if type(self.original_native) is not bool:
            raise ValueError("original_native must be a boolean")

