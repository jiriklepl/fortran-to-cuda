"""Carry immutable source constants into an outlined numerical region.

Initializers retain their Fortran types and expressions. Unsupported scalar
initializers can instead supply their original Fortran value through a visible,
read-only scalar capture. Array initializers retain the complete constant proof.
Values are never folded using Python floating-point arithmetic.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from hashlib import sha256

from fparser.two import Fortran2003 as F
from fparser.two.utils import Base

from compiler.frontend.source_effects import Binding, _children, _kind, _part
from compiler.ir import CompilationError, SourceLocation


@dataclass(frozen=True)
class ConstantParameters:
    names: dict[str, str]
    declarations: tuple[str, ...]
    resources: tuple[str, ...]
    identity: str
    scalar_captures: tuple[Binding, ...] = ()


class _UnsupportedInitializer(CompilationError):
    """An original initializer is outside the outlined constant language."""


def outline_constants(analysis, routine, bindings, occupied, fixed_shape):
    """Resolve a bounded declaration closure through original lexical imports."""
    selected = []
    for binding in bindings:
        if "parameter" not in binding.attributes:
            continue
        if not binding.rank and binding.dtype == "integer" and binding.kind == 4:
            # Retain the existing exact INTEGER substitution where available.
            # Other original scalar initializers need the same capture proof
            # as REAL constants; they are not evaluated by a Python fallback.
            scope = binding.declaring_scope
            try:
                if scope is not None:
                    scope.kinds.integer(F.Name(binding.name), SourceLocation(str(scope.path)))
                    continue
            except CompilationError:
                pass
        selected.append(binding)
    if not selected:
        return ConstantParameters({}, (), (), "")
    analysis._require_original(routine.qualified)
    names, ordered, active = {}, [], set()
    captures, capture_authority = [], []
    visits = 0

    def integer(scope, value):
        result = scope.kinds.integer(value, SourceLocation(str(scope.path)))
        if not -(2**31) <= result < 2**31:
            raise CompilationError("inline constant exceeds the existing INTEGER ABI")
        return F.Level_2_Expr("(-2147483647 - 1)" if result == -(2**31) else str(result))

    def declaration(binding):
        scope = binding.declaring_scope
        owner = getattr(scope, "qualified", getattr(scope, "module", None))
        procedure = analysis.numerical_helpers.get(owner)
        module = analysis.modules.get(owner)
        role = (analysis._numerical_roles.get(owner) if procedure is not None
                else analysis._source_module_roles.get(owner))
        original_scope = procedure.scope if procedure is not None else module
        if (scope is None or scope is not original_scope or role is None
                or scope.node is not role[0] or str(scope.node) != role[-1]
                or scope.bindings.get(binding.name) is not binding or not isinstance(owner, str)
                or binding.root != owner + "::" + binding.name):
            raise CompilationError("inline constant requires its original declaring scope")
        for node in _children(_part(scope.node, "Specification_Part")):
            if _kind(node) != "Type_Declaration_Stmt":
                continue
            dtype, attributes, entities = node.items
            if not any(str(item).lower() == "parameter" for item in _children(attributes)):
                continue
            for entity in _children(entities):
                if str(entity.items[0]).lower() == binding.name and entity.items[3] is not None:
                    dimension = next((item.items[1] for item in _children(attributes)
                                      if _kind(item) == "Dimension_Attr_Spec"), None)
                    shape = entity.items[1] if entity.items[1] is not None else dimension
                    axes = tuple(_children(shape))
                    if (len(axes) != binding.rank or len(binding.shape_nodes) != len(axes)
                            or any(left is not right for left, right in zip(axes, binding.shape_nodes, strict=True))
                            or _kind(dtype) != "Intrinsic_Type_Spec"):
                        raise CompilationError("inline constant descriptor differs from its original declaration")
                    base, selector = dtype.items
                    base = str(base).lower()
                    width = 8 if base == "double precision" else 4
                    if base == "double precision":
                        base = "real"
                    if selector is not None:
                        width = scope.kinds.integer(selector.items[1], SourceLocation(str(scope.path)))
                    if (base, width) != (binding.dtype, binding.kind):
                        raise CompilationError("inline constant type differs from its original declaration")
                    return scope, entity.items[3].items[1]
        raise CompilationError("inline constant requires its original initializer")

    def require(binding, depth):
        if binding.root in active or depth >= 16:
            raise CompilationError("inline constant closure is cyclic or exceeds the depth budget")
        if binding.root in names:
            return names[binding.root]
        if ("parameter" not in binding.attributes
                or (binding.dtype, binding.kind) not in {("integer", 4), ("real", 4), ("real", 8)}
                or binding.rank > 1):
            raise CompilationError("inline constants require scalar or bounded rank-one numeric PARAMETER storage")
        if len(names) >= 256:
            raise CompilationError("inline constant closure exceeds the declaration budget")
        name = "fort_region_constant_" + str(len(names))
        if name in occupied:
            raise CompilationError("inline constant namespace conflicts with original source")
        occupied.add(name)
        names[binding.root] = name
        active.add(binding.root)
        scope, initializer = declaration(binding)
        shape = fixed_shape(routine, binding) if binding.rank else ()
        rendered = normalize(initializer, scope, depth + 1)
        active.remove(binding.root)
        dtype = "integer" if binding.dtype == "integer" else f"real({binding.kind})"
        dimensions = "(" + ",".join(shape) + ")" if shape else ""
        ordered.append((binding.root, f"{dtype}, parameter :: {name}{dimensions} = {rendered}"))
        return name

    def normalize(value, scope, depth):
        nonlocal visits
        visits += 1
        if visits > 4096:
            raise CompilationError("inline constant initializer exceeds the expression budget")
        if isinstance(value, (tuple, list)):
            return type(value)(normalize(child, scope, depth) for child in value)
        if not isinstance(value, Base):
            return value
        kind = _kind(value)
        if kind == "Name":
            binding = analysis._binding(scope, value)
            if binding is None or "parameter" not in binding.attributes:
                raise CompilationError("inline initializer requires original constant operands")
            if binding.dtype == "integer" and not binding.rank and binding.kind == 4:
                return integer(scope, value)
            return F.Name(require(binding, depth))
        if kind in {"Ac_Implied_Do", "Function_Reference", "Structure_Constructor", "Data_Ref"}:
            raise _UnsupportedInitializer("inline constant initializer requires bounded numeric expressions")
        if kind == "Intrinsic_Function_Reference":
            name = str(value.items[0]).lower()
            if (name not in {"real", "int"} or analysis._binding(scope, name)
                    or analysis._candidates(scope, F.Name(name)) or analysis._unknown_exports(scope)):
                raise _UnsupportedInitializer("inline constant initializer requires proved numeric conversions")
        if kind == "Actual_Arg_Spec":
            result = copy.copy(value)
            result.items = (value.items[0], normalize(value.items[1], scope, depth))
            return result
        if kind == "Kind_Selector":
            return F.Kind_Selector("(kind=" + str(scope.kinds.integer(
                value.items[1], SourceLocation(str(scope.path)))) + ")")
        if kind in {"Real_Literal_Constant", "Int_Literal_Constant"} and value.items[1] is not None:
            result = copy.copy(value)
            result.items = (value.items[0], str(scope.kinds.integer(value.items[1], SourceLocation(str(scope.path)))))
            return result
        result = copy.copy(value)
        if hasattr(value, "items"):
            result.items = normalize(value.items, scope, depth)
        return result

    for binding in sorted(selected, key=lambda item: item.root):
        previous_names, previous_ordered = dict(names), list(ordered)
        previous_active, previous_occupied = set(active), set(occupied)
        try:
            require(binding, 0)
        except _UnsupportedInitializer:
            if binding.rank:
                raise  # An array PARAMETER cannot depend on a runtime dummy.
            # Discard the entire attempted declaration closure. In particular,
            # no partially outlined dependency may survive an unsupported use.
            names.clear()
            names.update(previous_names)
            ordered[:] = previous_ordered
            active.clear()
            active.update(previous_active)
            occupied.clear()
            occupied.update(previous_occupied)
            from compiler.scopes.numerical import resource_binding

            try:
                visible = resource_binding(analysis, routine, binding.root)
            except CompilationError as error:
                raise CompilationError("inline scalar PARAMETER is unavailable in the original owner: "
                                       + binding.root) from error
            if visible is not binding:
                raise CompilationError("inline scalar PARAMETER requires its original visible binding") from None
            scope, initializer = declaration(binding)
            captures.append(binding)
            capture_authority.append(binding.root + "\0" + str(initializer) + "\0"
                                     + analysis.sources[str(scope.path)])
    # Include original dependency files and normalized declarations: configured
    # initializers can differ even when their original edit target is unchanged.
    authority = "\0".join([*(root + "\0" + text for root, text in ordered), *capture_authority])
    identity = sha256(authority.encode()).hexdigest()
    return ConstantParameters(names, tuple(text for _, text in ordered),
                              tuple(root for root, _ in ordered), identity, tuple(captures))
