"""Original automatic storage retained by native OpenMP operations.

The bounds have already been evaluated on entry to the original procedure.
Their expressions must never be re-evaluated by an outlined owner. This proof
grants only native PRIVATE storage; it grants no numerical lowering or device
registration authority.
"""

from compiler.ir import CompilationError


def original_private_array(routine, binding):
    """Validate local automatic storage without evaluating specification bounds."""
    if (binding.declaring_scope is not routine.scope
            or binding.name in routine.arguments
            or not binding.root.startswith(routine.qualified + "::")
            or binding.attributes & {"save", "pointer", "allocatable", "optional", "volatile",
                                     "asynchronous", "parameter", "target", "value"}
            or binding.dtype not in {"real", "integer", "logical"}
            or binding.kind not in {4, 8}
            or not 1 <= binding.rank <= 4
            or len(binding.shape_nodes) != binding.rank
            or any(type(axis).__name__ != "Explicit_Shape_Spec" or axis.items[1] is None
                   for axis in binding.shape_nodes)):
        raise CompilationError("native PRIVATE requires original automatic numeric local array storage")
    return {"resource": binding.root, "rank": binding.rank,
            "storage": "original automatic explicit-shape local array",
            "bounds": "original allocated descriptor; specification expressions are not re-evaluated",
            "contents": "undefined original OpenMP PRIVATE contents; no capture or registration",
            "numerical_authority": False}
