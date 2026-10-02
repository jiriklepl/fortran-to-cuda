"""Scalar lifetime and dependence analysis for checked execution plans."""

from compiler.ir import ExecutionPlan, HostBlock, ParallelRegion, RegionReport

from .dependence import Affine, affine_expression, build_execution_plan, format_plan, ordered_conflicts, prove_region
from .proof import ParallelizationError, ParallelProof, ProofFailure

__all__ = [
    "Affine",
    "ParallelizationError",
    "ParallelProof",
    "ProofFailure",
    "prove_region",
    "ExecutionPlan",
    "HostBlock",
    "ParallelRegion",
    "RegionReport",
    "affine_expression",
    "build_execution_plan",
    "format_plan",
    "ordered_conflicts",
]
