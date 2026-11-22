"""Shared graph execution configuration."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OffloadConfig:
    host_threads: int = 4

    def __post_init__(self):
        if type(self.host_threads) is not int or self.host_threads < 1:
            raise ValueError("Host thread budget must be positive")

