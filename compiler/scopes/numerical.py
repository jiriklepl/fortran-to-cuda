"""Public source-extraction packages for independently normalized procedures.

Extraction owns normalization/capture/OpenMP facts. The compiler checks their
source identity, bindings and coordinate contract, then proves numerical legality
using its ordinary frontend. Original procedure definition events stay in clones.
"""

from __future__ import annotations

from dataclasses import dataclass

from fparser.two import Fortran2003 as F

from compiler.ir import CompilationError

# The types an entry takes as arrays and scalar values: REAL of either kind and default INTEGER.
NUMERIC_TYPES = frozenset({("real", 4), ("real", 8), ("integer", 4)})
# The LOGICAL kinds that the variables of an entry's own may have.
LOGICAL_TYPES = frozenset({("logical", 1), ("logical", 4)})


@dataclass(frozen=True)
class Parameter:
    name: str
    resource: str
    rank: int
    lower_bound_dimension: int | None = None


def resource_binding(analysis, routine, resource):
    """Resolve a source resource through actual visible bindings, including aliases."""
    if not isinstance(resource, str):
        raise CompilationError("numerical source parameter needs a canonical resource")
    if "%" in resource:
        base, fields = resource.split("%", 1)
        owner = resource_binding(analysis, routine, base)
        binding = analysis._binding(routine.scope, F.Data_Ref(owner.name + "%" + fields))
        if binding is None or binding.root != resource:
            raise CompilationError("numerical component resource differs from its original source binding")
        return binding
    if resource.startswith("argument::"):
        name = resource.split("::", 1)[1]
        binding = routine.scope.bindings.get(name)
        if name not in routine.arguments or binding is None:
            raise CompilationError("numerical source resource is not an original argument: " + resource)
        return binding
    candidates = {resource.rsplit("::", 1)[-1]}
    visited = set()

    def imported(module):
        target = analysis.modules.get(module)
        if target is None or module in visited:
            return
        visited.add(module)
        candidates.update(target.bindings)
        candidates.update(target.imports)
        for name in target.wildcards:
            imported(name)

    current = routine.scope
    while current:
        candidates.update(current.bindings)
        candidates.update(current.imports)
        for module in current.wildcards:
            imported(module)
        current = current.parent
    for name in sorted(candidates):
        binding = analysis._binding(routine.scope, name)
        if binding and binding.root == resource:
            return binding
    raise CompilationError("numerical source resource is unavailable in original procedure: " + resource)
