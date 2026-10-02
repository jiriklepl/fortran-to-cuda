"""Scalar lifetime and dependence analysis for checked execution plans."""

from compiler.ir import ExecutionPlan, HostBlock, ParallelRegion, RegionReport

from .dependence import Affine, affine_expression, build_execution_plan, format_plan, ordered_conflicts

__all__ = [
    "Affine",
    "ExecutionPlan",
    "HostBlock",
    "ParallelRegion",
    "RegionReport",
    "affine_expression",
    "build_execution_plan",
    "format_plan",
    "ordered_conflicts",
]
