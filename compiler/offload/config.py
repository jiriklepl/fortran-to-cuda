"""Shared graph execution configuration."""

from dataclasses import dataclass, replace

from compiler.numerical_contract import require_numerical_build_contract


@dataclass(frozen=True)
class OffloadConfig:
    policy: str = "sections"
    profile: dict | None = None
    host_threads: int = 4
    collective: bool = False
    profile_reason: str | None = None
    scope_transfers: str = "direct"
    # The native alternative is the original Fortran, not the generated worker.
    # Static work counts describe only the generated worker.
    original_native: bool = False
    def __post_init__(self):
        if self.policy not in {"sections", "auto"}:
            raise ValueError("Ordinary GPU policies were removed; use sections or auto")
        if type(self.host_threads) is not int or self.host_threads < 1:
            raise ValueError("Host thread budget must be positive")
        if self.scope_transfers == "pipelined":
            raise ValueError("Pipelined slab batching was removed; use shared graph execution")
        if self.scope_transfers not in {"direct", "pinned", "auto", "coalesced"}:
            raise ValueError("Unknown scope transfer mode")
        if type(self.original_native) is not bool:
            raise ValueError("original_native must be a boolean")
        if self.profile is not None:
            try:
                require_numerical_build_contract(self.profile.get("numerical_contract")
                                                 if isinstance(self.profile, dict) else None)
            except ValueError as error:
                # Reading legacy evidence does not authorize a changed backend.
                object.__setattr__(self, "profile", None)
                object.__setattr__(self, "profile_reason", "hardware profile " + str(error))

    def for_numerical_entry(self, **updates):
        """Keep code-generation options when entering a numerical worker."""
        return replace(self, **updates)
