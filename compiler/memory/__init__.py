"""Explicit host/device coherence and lifetime planning."""

from .planning import MemoryOperation, MemoryPlan, format_memory, plan_memory, validate_memory

__all__ = ["MemoryOperation", "MemoryPlan", "format_memory", "plan_memory", "validate_memory"]
