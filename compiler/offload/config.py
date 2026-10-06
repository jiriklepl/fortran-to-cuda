"""Explicit, opt-in ordinary-call GPU execution policies."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OffloadConfig:
    policy: str = "always"
    profile: dict | None = None
    host_threads: int = 4
    collective: bool = False
    profile_reason: str | None = None

    def __post_init__(self):
        if self.policy not in {"always", "sections", "auto", "chunked", "hybrid"}:
            raise ValueError("Unknown GPU execution policy")
        if type(self.host_threads) is not int or self.host_threads < 1:
            raise ValueError("Host thread budget must be positive")
