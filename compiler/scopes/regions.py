"""Outline source-proven numerical loops without moving their owning storage.

This module constructs a numerical candidate, not a placement decision.  The
ordinary frontend must still prove numerical legality.  Allocation guards and
saved declarations remain in the original procedure; only the reached loop is
borrowed through ordinary array arguments and explicit original lower bounds.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

from fparser.two import Fortran2003 as F
from fparser.two.utils import Base, walk

from compiler.frontend.source_effects import Binding, _children, _kind, _part
from compiler.ir import CompilationError
from compiler.scopes.numerical import Parameter
from compiler.scopes.segments import directive, fortran_lines, grouped_nodes, joined_group_completion, statement_span


@dataclass(frozen=True)
class RegionExtraction:
    nodes: tuple
    span: tuple[int, int]
    source: str
    entry: str
    parameters: tuple[Parameter, ...]
    bindings: tuple[Binding, ...]
    written_resources: frozenset[str]
    private_scalars: tuple[str, ...]
    allocation_guards: tuple[dict, ...]
    source_identity: str
    completion: dict

    def write(self, directory: Path) -> Path:
        """Publish once, rejecting an accidental identity/content collision."""
        path = Path(directory) / (self.entry.split("::")[0] + ".f90")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_text() != self.source:
            raise CompilationError("inline numerical artifact identity changed")
        path.write_text(self.source)
        return path

    def public(self):
        return {"schema_version": 1, "source_identity": self.source_identity, "entry": self.entry,
                "first_line": self.span[0], "last_line": self.span[1],
                "resources": [binding.public() for binding in self.bindings],
                "parameters": [{"name": item.name, "resource": item.resource, "rank": item.rank,
                                "lower_bound_dimension": item.lower_bound_dimension,
                                "runtime_lower_bound": item.runtime_lower_bound} for item in self.parameters],
                "written_resources": sorted(self.written_resources), "private_scalars": list(self.private_scalars),
                "allocation_guards": list(self.allocation_guards), "completion": dict(self.completion),
                "storage_owner": "original procedure; saved storage and initialization are not cloned",
                "numerical_legality": "requires ordinary frontend and dependence proofs"}


def _variable_names(node):
    """Skip syntax names such as intrinsic keywords and DO construct labels."""
    if isinstance(node, (tuple, list)):
        for child in node:
            yield from _variable_names(child)
        return
    kind = _kind(node)
    if kind == "Name":
        yield node
        return
    if kind in {"Actual_Arg_Spec", "Intrinsic_Function_Reference", "Nonlabel_Do_Stmt"}:
        children = (node.items[1],)
    elif kind in {"End_Do_Stmt", "Comment"}:
        children = ()
    else:
        children = _children(node)
    for child in children:
        yield from _variable_names(child)


def _names(analysis, routine, node):
    return {binding.root for name in _variable_names(node)
            if (binding := analysis._binding(routine.scope, name)) is not None}


def _scalar_reads(analysis, routine, node):
    return {root for root in _names(analysis, routine, node)
            if not _binding(analysis, routine, root).rank}


def _binding(analysis, routine, root):
    for name in walk(routine.scope.node, F.Name):
        candidate = analysis._binding(routine.scope, name)
        if candidate and candidate.root == root:
            return candidate
    raise CompilationError("inline resource is unavailable: " + root)


def _intrinsic_shadowed(analysis, routine, name):
    scope = routine.scope
    while scope is not None:
        if name in scope.externals or name in scope.procedure_arguments:
            return True
        scope = scope.parent
    return bool(analysis._binding(routine.scope, name) or analysis._candidates(routine.scope, F.Name(name)))


def _iterator(node):
    header = next((item for item in _children(node) if _kind(item) == "Nonlabel_Do_Stmt"), None)
    control = header.items[1] if header is not None else None
    counted = control.items[1] if control is not None and _kind(control) == "Loop_Control" else None
    if not counted or len(counted) != 2 or _kind(counted[0]) != "Name":
        raise CompilationError("inline numerical regions require counted DO loops")
    return str(counted[0]).lower(), tuple(counted[1])


def _upward_reads(analysis, routine, nodes):
    """Conservative scalar liveness, including original DO final values."""
    defined, reads = set(), set()
    for node in nodes:
        kind = _kind(node)
        if kind == "Comment":
            continue
        if kind == "Assignment_Stmt":
            target, _, value = node.items
            own = _scalar_reads(analysis, routine, value)
            if _kind(target) != "Name":
                own |= _scalar_reads(analysis, routine, target)
            reads |= own - defined
            binding = analysis._binding(routine.scope, target) if _kind(target) == "Name" else None
            if binding and not binding.rank:
                defined.add(binding.root)
        elif kind == "Block_Nonlabel_Do_Construct":
            name, bounds = _iterator(node)
            reads |= set().union(*(_scalar_reads(analysis, routine, bound) for bound in bounds)) - defined
            index = analysis._binding(routine.scope, name)
            if index:
                defined.add(index.root)
            body = tuple(item for item in _children(node) if _kind(item) not in {
                "Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"})
            reads |= _upward_reads(analysis, routine, body) - defined
        elif kind == "If_Construct":
            body = []
            for child in _children(node):
                if _kind(child) in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                    reads |= _upward_reads(analysis, routine, body) - defined
                    body = []
                    reads |= _scalar_reads(analysis, routine, child) - defined
                else:
                    body.append(child)
            reads |= _upward_reads(analysis, routine, body) - defined
            # Branch-local definitions do not establish a definition after IF.
        else:
            # Unknown calls can read any referenced scalar; their conditional
            # writes do not establish a subsequent definition.
            reads |= _scalar_reads(analysis, routine, node) - defined
    return reads


def _allocated_names(node):
    if _kind(node) != "Allocate_Stmt":
        return set()
    result = set()
    for item in walk(node):
        if _kind(item) == "Allocation":
            result.add(str(item.items[0]).lower())
    return result


def allocation_guard(analysis, routine, binding, preceding):
    """Identify original allocation sites, leaving current existence to a guard.

    A conditional allocation need not execute on this invocation.  Fresh
    ALLOCATED/bounds checks at the reached point establish the actual descriptor;
    no earlier allocation address, counter or proof is reused.  Allocation and
    unknown operations before this point remain outside this owning region.
    """
    if binding.rank == 0 or not {"save", "allocatable"}.issubset(binding.attributes):
        raise CompilationError("inline local allocations require original saved array storage")
    if binding.root != routine.qualified + "::" + binding.name:
        raise CompilationError("inline allocation proof requires original procedure-local storage")
    if binding.attributes & {"target", "pointer", "optional", "volatile", "asynchronous"}:
        raise CompilationError("inline saved allocation may escape or have uncertain association")
    if _intrinsic_shadowed(analysis, routine, "allocated"):
        raise CompilationError("inline allocation guard intrinsic is shadowed")
    sites = []
    for node in preceding:
        for item in walk(node):
            if _kind(item) == "Allocate_Stmt" and binding.name in _allocated_names(item):
                try:
                    first, last = statement_span(item)
                except CompilationError:
                    # An action statement in a single-line IF has no own
                    # reader item; its enclosing original IF owns the span.
                    first, last = statement_span(node)
                sites.append({"first_line": first, "last_line": last})
                if len(sites) > 32:
                    raise CompilationError("inline saved allocation site budget exceeded")
    if not sites:
        raise CompilationError("inline saved allocation lacks a preceding original allocation site: " + binding.root)
    return {"resource": binding.root, "kind": "original saved allocation with reached descriptor guard",
            "allocation_sites": sites, "allocation_proven": False,
            "runtime_guard": "ALLOCATED then checked original LBOUND/UBOUND/SIZE",
            "lifetime": "reached bounded region; allocation changes and unknown effects end ownership"}


def extract_region(analysis, routine, node, *, preceding=(), following=()):
    """Construct one bounded serial or complete joined OpenMP DO candidate."""
    analysis.inputs.verify()
    role = analysis._source_roles.get(routine.qualified)
    if (analysis.routines.get(routine.qualified) is not routine or role is None
            or routine.execution is not role[0] or analysis._routine_signature(routine) != role[1]
            or str(routine.execution) != role[2]):
        raise CompilationError("inline extraction requires the original source-backed routine")
    if routine.issues:
        raise CompilationError("inline original specification is unsupported: " + "; ".join(routine.issues))
    if analysis._unknown_exports(routine.scope):
        raise CompilationError("inline intrinsic authority requires complete wildcard imports")
    if any(directive(item) is not None for item in walk(_part(routine.scope.node, "Specification_Part"))):
        # fparser may attach opening executable directives to specifications.
        # A detached DO alone does not prove either serial participation or the
        # complete parallel region; retain the original operation in that case.
        raise CompilationError("inline OpenMP association requires its complete original source region")
    source_nodes = tuple(node) if isinstance(node, (tuple, list)) else (node,)
    nodes = source_nodes
    if not nodes:
        raise CompilationError("inline numerical region is empty")
    # fparser attaches an associated !$OMP DO prefix to its DO construct.
    # Peel only that original structural prefix; do not authorize a projected
    # routine or arbitrary foreign AST as source-backed numerical work.
    grouped = grouped_nodes(nodes)
    if len(grouped) == 1 and isinstance(grouped[0], tuple):
        nodes = grouped[0]
    joined = directive(nodes[0]) is not None
    if joined:
        completion = joined_group_completion(SimpleNamespace(analysis=analysis, entry=routine), nodes)
        loops = tuple(item for item in nodes if _kind(item) == "Block_Nonlabel_Do_Construct")
    else:
        if len(nodes) != 1 or _kind(nodes[0]) != "Block_Nonlabel_Do_Construct":
            raise CompilationError("inline numerical region requires one counted DO or a complete joined team")
        completion = {"available": True, "caller_contract": "serial_source_scope",
                      "reason": "original serial counted loop completes before continuation"}
        loops = nodes
    if not loops or len(loops) > 32:
        raise CompilationError("inline numerical loop budget exceeded")
    original_nodes = {id(item) for item in walk(routine.execution)}
    for item in (*nodes, *preceding, *following):
        if id(item) in original_nodes:
            continue
        related = False
        if _kind(item) == "Block_Nonlabel_Do_Construct":
            for original in walk(routine.execution, F.Block_Nonlabel_Do_Construct):
                children = list(original.content)
                while children and _kind(children[0]) == "Comment":
                    children.pop(0)
                if (len(children) == len(item.content)
                        and all(left is right for left, right in zip(children, item.content, strict=True))):
                    related = True
                    break
        if not related:
            raise CompilationError("inline extraction nodes must belong to the original execution")
    admitted = {"Block_Nonlabel_Do_Construct", "Nonlabel_Do_Stmt", "End_Do_Stmt", "Assignment_Stmt",
                "If_Stmt", "If_Construct", "If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt", "Comment"}
    for loop in loops:
        for item in walk(loop):
            if ((hasattr(item, "content") or _kind(item).endswith("_Stmt")
                 or getattr(item, "item", None) is not None) and _kind(item) not in admitted):
                raise CompilationError("unsupported inline numerical statement: " + _kind(item))
            if directive(item) is not None:
                raise CompilationError("nested inline OpenMP directives require a synchronization proof")

    used = {}
    for loop in loops:
        for name in _variable_names(loop):
            binding = analysis._binding(routine.scope, name)
            if binding is not None:
                boundary = analysis.resource_identity_boundary(binding)
                if boundary:
                    raise CompilationError(boundary)
                used[binding.root] = binding
    writes, scalar_writes, iterators = set(), set(), set()
    for loop in loops:
        for item in walk(loop):
            if _kind(item) == "Block_Nonlabel_Do_Construct":
                name, _ = _iterator(item)
                binding = analysis._binding(routine.scope, name)
                if binding is None:
                    raise CompilationError("undeclared inline loop iterator")
                scalar_writes.add(binding.root)
                iterators.add(binding.root)
            elif _kind(item) == "Assignment_Stmt":
                target = item.items[0]
                name = target.items[0] if _kind(target) == "Part_Ref" else target
                binding = analysis._binding(routine.scope, name)
                if binding is None:
                    raise CompilationError("unresolved inline assignment target")
                if binding.rank:
                    if _kind(target) != "Part_Ref":
                        raise CompilationError("inline whole-array assignments require separate allocation/effect proofs")
                    if binding.intent == "in":
                        raise CompilationError("inline region writes original INTENT(IN) storage")
                    writes.add(binding.root)
                else:
                    scalar_writes.add(binding.root)
    for root in scalar_writes:
        binding = used[root]
        if not root.startswith(routine.qualified + "::") or "save" in binding.attributes or binding.rank:
            raise CompilationError("inline scalar effects must remain in the original owner: " + root)
        if binding.attributes & {"target", "pointer", "allocatable", "optional", "volatile", "asynchronous", "value"}:
            raise CompilationError("inline private scalar association is uncertain: " + root)
        if (binding.dtype, binding.kind) not in {("real", 4), ("real", 8), ("integer", 4), ("logical", 1)}:
            raise CompilationError("unsupported inline private scalar type: " + root)
        if root in iterators and (binding.dtype, binding.kind) != ("integer", 4):
            raise CompilationError("inline loop iterators require default INTEGER: " + root)
    if scalar_writes & _upward_reads(analysis, routine, loops):
        raise CompilationError("inline private scalar reads require a definition inside the region")
    if scalar_writes & _upward_reads(analysis, routine, following):
        raise CompilationError("inline loop-written scalar is live after the region")
    if joined:
        from compiler.scopes.participation import _threadprivate
        if set(used) & _threadprivate(analysis):
            raise CompilationError("inline OpenMP capture is THREADPRIVATE")
        explicit_private = set()
        for item in nodes:
            text = directive(item) or ""
            for match in re.finditer(r"private\s*\(([^()]*)\)", text):
                for name in match.group(1).split(","):
                    binding = analysis._binding(routine.scope, name.strip())
                    if binding:
                        explicit_private.add(binding.root)
        if scalar_writes - iterators - explicit_private:
            raise CompilationError("inline OpenMP scalar writes need proven PRIVATE storage")
    private = {root: used[root] for root in scalar_writes}
    captures = {root: binding for root, binding in used.items()
                if root not in private and "parameter" not in binding.attributes}
    guards = []
    for binding in captures.values():
        if binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "value"}:
            raise CompilationError("inline capture association is uncertain: " + binding.root)
        if (binding.dtype, binding.kind) not in {("real", 4), ("real", 8), ("integer", 4), ("logical", 1)}:
            raise CompilationError("unsupported inline numerical capture type: " + binding.root)
        if "allocatable" in binding.attributes and binding.root.startswith(routine.qualified + "::"):
            guards.append(allocation_guard(analysis, routine, binding, preceding))
    if not any(binding.rank for binding in captures.values()):
        raise CompilationError("inline numerical candidate has no array resources")
    if len(captures) > 64:
        raise CompilationError("inline numerical capture budget exceeded")
    arrays = {binding.name: binding for binding in captures.values() if binding.rank}
    names = {binding.name for binding in used.values()}
    lowers = {}
    for name, binding in arrays.items():
        for axis in range(1, binding.rank + 1):
            lower = f"fort_region_lb_{name}_{axis}"
            if lower in names:
                raise CompilationError("inline lower-bound parameter namespace conflicts")
            lowers[name, axis] = lower

    def transformed(value):
        if isinstance(value, (tuple, list)):
            return type(value)(transformed(child) for child in value)
        if not isinstance(value, Base):
            return value
        kind = _kind(value)
        if kind == "Actual_Arg_Spec":
            result = copy.copy(value)
            result.items = (value.items[0], transformed(value.items[1]))
            return result
        if kind == "End_Do_Stmt":
            return copy.copy(value)
        if kind == "Nonlabel_Do_Stmt":
            result = copy.copy(value)
            result.items = (value.items[0], transformed(value.items[1]))
            return result
        if kind == "Intrinsic_Function_Reference":
            intrinsic = str(value.items[0]).lower()
            if _intrinsic_shadowed(analysis, routine, intrinsic):
                raise CompilationError("inline numerical intrinsic is shadowed: " + intrinsic)
        if kind == "Part_Ref" and str(value.items[0]).lower() not in arrays:
            name = str(value.items[0]).lower()
            if _intrinsic_shadowed(analysis, routine, name):
                raise CompilationError("inline numerical intrinsic is shadowed: " + name)
        if kind == "Part_Ref" and str(value.items[0]).lower() in arrays:
            name = str(value.items[0]).lower()
            indices = _children(value.items[1])
            if len(indices) != arrays[name].rank or any(_kind(index) == "Subscript_Triplet" for index in indices):
                raise CompilationError("inline numerical array accesses require scalar rank-preserving indices")
            rendered = [f"({transformed(index)}) - {lowers[name, axis]} + 1"
                        for axis, index in enumerate(indices, 1)]
            return F.Part_Ref(name + "(" + ",".join(rendered) + ")")
        if kind == "Intrinsic_Function_Reference" and str(value.items[0]).lower() in {"lbound", "ubound", "size"}:
            intrinsic = str(value.items[0]).lower()
            if _intrinsic_shadowed(analysis, routine, intrinsic):
                raise CompilationError("inline array inquiry intrinsic is shadowed")
            arguments = _children(value.items[1])
            if (not arguments or _kind(arguments[0]) != "Name"
                    or str(arguments[0]).lower() not in arrays):
                raise CompilationError("inline array inquiry requires an original array descriptor")
            if intrinsic == "size":
                # Full-layout shape is unchanged by coordinate rebasing.
                pass
            else:
                if len(arguments) != 2 or _kind(arguments[1]) != "Int_Literal_Constant":
                    raise CompilationError("inline bounds inquiries require a constant scalar DIM")
                name = str(arguments[0]).lower()
                axis = int(str(arguments[1]).split("_")[0])
                if (name, axis) not in lowers:
                    raise CompilationError("inline bounds inquiry DIM is outside the original rank")
                if intrinsic == "lbound":
                    return F.Name(lowers[name, axis])
                return F.Level_2_Expr(f"({lowers[name, axis]} + (SIZE({name},{axis}) - 1))")
        if kind == "Name":
            binding = analysis._binding(routine.scope, value)
            if binding and "parameter" in binding.attributes:
                # Resolve kinds/constants in the original lexical scope; a new
                # module cannot silently acquire a same-spelling imported value.
                if binding.dtype != "integer" or binding.kind != 4:
                    raise CompilationError("inline numerical constants currently require default INTEGER")
                from compiler.ir import SourceLocation
                integer = routine.scope.kinds.integer(value, SourceLocation(str(routine.scope.path)))
                if not -(2**31) <= integer < 2**31:
                    raise CompilationError("inline constant exceeds the existing INTEGER ABI")
                spelling = "(-2147483647 - 1)" if integer == -(2**31) else str(integer)
                return F.Level_2_Expr(spelling)
        result = copy.copy(value)
        for attribute in ("content", "items"):
            if hasattr(value, attribute):
                setattr(result, attribute, transformed(getattr(value, attribute)))
        return result

    span = (statement_span(nodes[0])[0], statement_span(nodes[-1])[1])
    identity = sha256((analysis.sources[str(routine.scope.path)] + "\0" + routine.qualified + "\0" +
                       str(span) + "\0" + "\n".join(map(str, nodes))).encode()).hexdigest()
    module, procedure = "fort_inline_" + identity[:12], "region"
    parameters, declarations = [], []

    def numerical_type(binding):
        # The numerical ABI uses default INTEGER and its C-bool LOGICAL model.
        # Their original kinds were checked above. Real temporaries must retain
        # their precision as well as the array/scalar captures.
        return binding.dtype if binding.dtype in {"integer", "logical"} else f"{binding.dtype}({binding.kind})"

    for binding in sorted(captures.values(), key=lambda item: item.root):
        parameters.append(Parameter(binding.name, binding.root, binding.rank))
        dtype = numerical_type(binding)
        shape = "(" + ",".join(":" for _ in range(binding.rank)) + ")" if binding.rank else ""
        intent = "inout" if binding.root in writes else "in"
        declarations.append(f"{dtype}, intent({intent}) :: {binding.name}{shape}")
        if binding.rank:
            for axis in range(1, binding.rank + 1):
                lower = lowers[binding.name, axis]
                parameters.append(Parameter(lower, binding.root, 0, axis, True))
                declarations.append(f"integer, intent(in) :: {lower}")
    declarations += [f"{numerical_type(binding)} :: {binding.name}"
                     for binding in sorted(private.values(), key=lambda item: item.name)]
    body = [str(transformed(loop)) for loop in loops]
    source = "\n".join(fortran_lines([f"module {module}", "implicit none", "contains",
                                   f"subroutine {procedure}(" + ",".join(item.name for item in parameters) + ")",
                                   *declarations, *"\n".join(body).splitlines(), "end subroutine", "end module"])) + "\n"
    analysis.inputs.verify()
    return RegionExtraction(source_nodes, span, source, module + "::" + procedure, tuple(parameters),
                            tuple(sorted(captures.values(), key=lambda item: item.root)), frozenset(writes),
                            tuple(binding.name for binding in private.values()), tuple(guards), identity, completion)
