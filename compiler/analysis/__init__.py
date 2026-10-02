"""Scalar lifetime and dependence analysis for checked execution plans."""

from compiler.ir import ExecutionPlan, HostBlock, ParallelRegion, RegionReport

from .dependence import Affine, affine_expression, ordered_conflicts, prove_region
from .planning import build_execution_plan, format_plan
from .proof import ParallelizationError, ParallelProof, ProofFailure
from .semantics import validate_function

__all__ = [
    "Affine",
    "ParallelizationError",
    "ParallelProof",
    "ProofFailure",
    "prove_region",
    "validate_function",
    "ExecutionPlan",
    "HostBlock",
    "ParallelRegion",
    "RegionReport",
    "affine_expression",
    "build_execution_plan",
    "format_plan",
    "ordered_conflicts",
]
