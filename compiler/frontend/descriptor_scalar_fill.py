"""Descriptor effects of original primitive scalar-to-array assignments.

Fortran 2023 10.2.1.3 paragraphs 3 and 5 require an already allocated array
for a scalar RHS and broadcast over its existing shape. Fixed intrinsic
numeric types have no deferred length or dynamic type that could reallocate.
This classifier grants no allocation presence, initializedness or execution.
"""
from __future__ import annotations

from compiler.frontend.source_functions import _walk, original_name_absent
from compiler.ir import CompilationError

DESCRIPTOR_WINDOW_SCALAR_FILL_VERSION = 24


class ScalarFillUnsupported(CompilationError):
    """A semantic scalar-fill capability decline, never source authentication."""


def scalar_fill_descriptor_effects(analysis, routine, assignment):
    """Classify one exact intrinsic assignment; the consumer scans both sides.

    This is not a reusable authority token or a publication boundary. The
    descriptor consumer retains original syntax/declarations before calling
    here and independently validates the complete original source closure.
    """
    if (analysis._require_original(routine.qualified) is not routine
            or type(assignment).__name__ != "Assignment_Stmt"
            or not any(node is assignment for node in _walk(routine.execution))):
        raise CompilationError("descriptor scalar fill requires exact original assignment")
    target, _equals, value = assignment.items
    if type(target).__name__ != "Name":
        raise ScalarFillUnsupported("descriptor scalar fill requires a whole original allocatable array")
    scope = analysis.source_scope_for(assignment, routine.scope)
    if not original_name_absent(analysis, scope, "assignment(=)"):
        raise ScalarFillUnsupported("descriptor scalar fill cannot consume a visible defined assignment")
    binding = analysis._binding(scope, target)
    if (binding is None or "allocatable" not in binding.attributes
            or binding.attributes & {"pointer", "optional", "target", "volatile", "asynchronous", "value"}
            or binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}
            or not 1 <= binding.rank <= 4):
        raise ScalarFillUnsupported("descriptor scalar fill requires fixed primitive numeric allocatable array")
    signature = analysis._signature(scope, value)
    if signature not in {("real", 4, 0), ("real", 8, 0), ("integer", 4, 0), ("integer", 8, 0)}:
        raise ScalarFillUnsupported("descriptor scalar fill requires a proved primitive numeric scalar RHS")
    return (target, value), {
        "schema_version": 1, "procedure": routine.qualified,
        "intrinsic": "scalar_array_assignment", "source_expression": str(assignment),
        "target": {"resource": binding.root, "type": binding.dtype,
                   "kind": binding.kind, "rank": binding.rank},
        "rhs": {"type": signature[0], "kind": signature[1], "rank": 0},
        "requires_original_allocated_array": True, "allocation_presence_proved": False,
        "unallocated_behavior": "original assignment requires same-rank RHS; scalar supplies no allocation extent",
        "shape": "existing original allocated shape, bounds and storage retained",
        "allocation_or_escape": False, "storage_writes": True,
        "payload_effect": "unchanged original intrinsic scalar broadcast; independently scanned native write",
        "all_actual_effects_require_original_source_traversal": True,
        "requires_original_execution": True, "descriptor_evidence_only": True,
        "execution_authorized": False, "capture_authorized": False,
        "entry_definition_authority": False, "gpu_legality_established": False,
        "numerical_lowering_authorized": False, "motion_authorized": False,
    }
