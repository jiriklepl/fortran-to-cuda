"""Lower the supported Fortran subset into an ordered, immutable computation IR.

Parsing details stay inside this module. In particular, inlining resolves dummy
arguments to their actual storage identities before any dependence analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fparser.common.readfortran import FortranFileReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import FortranSyntaxError, walk

from compiler.ir import (
    ArrayAccess,
    Assignment,
    Binary,
    Block,
    CompilationError,
    Expr,
    FunctionIR,
    If,
    IntrinsicCall,
    Literal,
    Loop,
    PrivateArrayOrigin,
    Reference,
    ScalarType,
    Size,
    SourceLocation,
    Symbol,
    Unary,
    walk_expr,
)
from compiler.ir.integers import constant_integer, integer_literal
from compiler.ir.intrinsics import (
    ARRAY_INQUIRIES,
    INTRINSIC_ARGUMENTS,
    INTRINSICS,
    KIND_ARGUMENT,
    MODEL_INQUIRIES,
    intrinsic_kind,
    intrinsic_type,
    model_inquiry,
)


@dataclass(frozen=True)
class _Declaration:
    name: str
    dtype: ScalarType
    rank: int
    intent: str | None
    location: SourceLocation
    spelling: str
    inferred_intent: bool = False
    bounds: tuple[tuple[int, int], ...] = ()
    initializer: Any = None
    constant: bool = False


@dataclass(frozen=True)
class _Routine:
    name: str
    module: str
    arguments: tuple[str, ...]
    declarations: tuple[_Declaration, ...]
    execution: tuple[Any, ...]
    location: SourceLocation
    kinds: _KindScope
    key: tuple[str, str]
    parent: tuple[str, str] | None = None
    result: str | None = None
    pure: bool = False


@dataclass(frozen=True)
class _PrivateArray:
    """A bounded Fortran array represented by independently defined scalars."""

    name: str
    dtype: ScalarType
    bounds: tuple[tuple[int, int], ...]
    elements: tuple[Symbol, ...]

    @property
    def rank(self):
        return len(self.bounds)

    def element(self, indices: tuple[int, ...], location: SourceLocation) -> Symbol:
        offset, pitch = 0, 1
        for index, (lower, upper) in zip(indices, self.bounds, strict=True):
            if not lower <= index <= upper:
                raise CompilationError(f"private array subscript outside bounds of {self.name}", location)
            offset += (index - lower) * pitch
            pitch *= max(0, upper - lower + 1)
        return self.elements[offset]


@dataclass(frozen=True)
class _ConstantArray:
    """Immutable declared constants; these are not private runtime storage."""

    name: str
    dtype: ScalarType
    bounds: tuple[tuple[int, int], ...]
    elements: tuple[Expr, ...]

    @property
    def rank(self):
        return len(self.bounds)

    def element(self, indices, location):
        offset, pitch = 0, 1
        for index, (lower, upper) in zip(indices, self.bounds, strict=True):
            if not lower <= index <= upper:
                raise CompilationError(f"constant array subscript outside bounds of {self.name}", location)
            offset += (index - lower) * pitch
            pitch *= max(0, upper - lower + 1)
        if offset >= len(self.elements):
            raise CompilationError("PARAMETER arrays require dependency-ordered initializers", location)
        return self.elements[offset]


@dataclass(frozen=True)
class _FixedVector:
    """A transient bounded expression, expanded before the public scalar IR."""

    dtype: ScalarType
    elements: tuple[Expr, ...]


@dataclass(frozen=True)
class ProcedureCandidate:
    """A module procedure and its frontend result, before dependence checks."""

    module: str
    name: str
    location: SourceLocation
    annotated: bool
    lowerable: bool
    reason: str | None = None

    @property
    def qualified_name(self) -> str:
        return f"{self.module}::{self.name}"


class _KindScope:
    """Resolve declared INTEGER constants lazily under the supported ABI model.

    Unsupported, unrelated module declarations must not make a selected routine
    fail. In particular, an arbitrary variable named ``knd`` is never assumed to
    mean double precision.
    """

    intrinsic_kinds = {"iso_fortran_env": {"real32": 4, "real64": 8}, "iso_c_binding": {"c_float": 4, "c_double": 8}}

    def __init__(self, specification: Any | None = None, parent: _KindScope | None = None):
        self.parent = parent
        self.values: dict[str, Any] = {}
        for node in getattr(specification, "content", ()):
            if type(node).__name__ == "Use_Stmt":
                nature, _, module, only, names = node.items
                available = self.intrinsic_kinds.get(str(module).lower(), {})
                if not available or str(nature).upper() == "NON_INTRINSIC":
                    continue
                if names is None:
                    self.values.update(available)
                elif str(only).upper().replace(" ", "") == ",ONLY:":
                    for name in names.items:
                        if type(name).__name__ == "Rename":
                            local, remote = str(name.items[1]).lower(), str(name.items[2]).lower()
                        else:
                            local = remote = str(name).lower()
                        if remote in available:
                            self.values[local] = available[remote]
            elif type(node).__name__ == "Type_Declaration_Stmt":
                dtype, attributes, entities = node.items
                parameter = any(str(item).upper() == "PARAMETER" for item in getattr(attributes, "items", ()))
                default_integer = type(dtype).__name__ == "Intrinsic_Type_Spec" and dtype.items == ("INTEGER", None)
                for entity in entities.items:
                    name, dimensions, length, initializer = entity.items
                    value = None
                    if parameter and default_integer and dimensions is None and length is None and initializer:
                        value = initializer.items[1]
                    self.values[str(name).lower()] = value

    def has_constant(self, name: str) -> bool:
        if name in self.values:
            return self.values[name] is not None
        return self.parent is not None and self.parent.has_constant(name)

    def integer(self, node: Any, location: SourceLocation, active: frozenset[str] = frozenset()) -> int:
        kind = type(node).__name__
        if isinstance(node, int):
            return node
        if kind == "Name" or isinstance(node, str):
            name = str(node).lower()
            if name.isdigit():
                return int(name)
            if name in active:
                raise CompilationError(f"cyclic kind parameter {name}", location)
            if name in self.values:
                value = self.values[name]
                if value is not None:
                    return self.integer(value, location, active | {name})
            elif self.parent is not None:
                return self.parent.integer(node, location, active)
            raise CompilationError(f"unresolved INTEGER kind parameter {name}", location)
        if kind == "Int_Literal_Constant" and node.items[1] is None:
            return int(node.items[0])
        if kind == "Parenthesis":
            return self.integer(node.items[1], location, active)
        if kind == "Intrinsic_Function_Reference":
            name, arguments = node.items
            args = getattr(arguments, "items", ())
            if str(name).upper() == "KIND" and len(args) == 1:
                argument = args[0]
                if type(argument).__name__ == "Actual_Arg_Spec" and str(argument.items[0]).lower() == "x":
                    argument = argument.items[1]
                if type(argument).__name__ in {
                    "Real_Literal_Constant",
                    "Int_Literal_Constant",
                    "Logical_Literal_Constant",
                }:
                    value, selector = argument.items
                    return (
                        self.integer(selector, location, active)
                        if selector is not None
                        else (8 if type(argument).__name__ == "Real_Literal_Constant" and "d" in value.lower() else 4)
                    )
        items = getattr(node, "items", ())
        if len(items) == 2 and str(items[0]) in {"+", "-"}:
            operand = Literal(str(self.integer(items[1], location, active)), ScalarType.INTEGER)
            return constant_integer(Unary(str(items[0]), operand), location)
        if len(items) == 3 and str(items[1]) in {"+", "-", "*", "/"}:
            left = Literal(str(self.integer(items[0], location, active)), ScalarType.INTEGER)
            right = Literal(str(self.integer(items[2], location, active)), ScalarType.INTEGER)
            return constant_integer(Binary(str(items[1]), left, right), location)
        raise CompilationError(f"unsupported INTEGER kind constant {node}", location)

    def real_type(self, selector: Any, location: SourceLocation) -> ScalarType:
        value = self.integer(selector, location)
        if value not in {4, 8}:
            raise CompilationError(f"unsupported REAL kind {value}; supported kinds are 4 and 8", location)
        return ScalarType.REAL32 if value == 4 else ScalarType.REAL


class _Lowerer:
    def __init__(self, path: Path, *, require_markers: bool = False):
        self.path = str(path)
        self.symbols: list[Symbol] = []
        self.routines: dict[tuple[str, str], _Routine] = {}
        self.routine_nodes: dict[tuple[str, str], tuple[str, Any]] = {}
        self.module_kinds: dict[str, _KindScope] = {}
        self.annotated: set[tuple[str, str]] = set()
        self.require_markers = require_markers
        self.active_kinds = _KindScope()
        self.parents: dict[tuple[str, str], tuple[str, str] | None] = {}
        self.active_frames: list[tuple] = []
        self.pending: list = []
        self.inline_calls = 0
        self.private_array_groups = 0
        self.unrolled_values: dict[Symbol, int] = {}
        self.unrolled_iterations = 0
        self.vector_elements = 0
        self.constant_initialization = False
        self.requires_numerical_environment = False

    def location(self, node: Any, stack: tuple[str, ...] = ()) -> SourceLocation:
        for child in walk(node):
            span = getattr(getattr(child, "item", None), "span", None)
            if span:
                return SourceLocation(self.path, span[0], stack)
        # Embedded actions (for example single-line IF assignments/calls) inherit
        # their source span from the enclosing parser node.
        parent = getattr(node, "parent", None)
        while parent is not None:
            span = getattr(getattr(parent, "item", None), "span", None)
            if span:
                return SourceLocation(self.path, span[0], stack)
            parent = getattr(parent, "parent", None)
        return SourceLocation(self.path, 1, stack)

    def error(self, message: str, node: Any, stack: tuple[str, ...] = ()) -> CompilationError:
        return CompilationError(message, self.location(node, stack))

    def discover(self, tree: Any) -> None:
        for module in (node for node in walk(tree) if type(node).__name__ == "Module"):
            module_statement = next(child for child in module.content if type(child).__name__ == "Module_Stmt")
            module_name = str(module_statement.items[1])
            specification = next(
                (child for child in module.content if type(child).__name__ == "Specification_Part"), None
            )
            self.module_kinds[module_name.lower()] = _KindScope(specification)
            for part in module.content:
                if type(part).__name__ != "Module_Subprogram_Part":
                    continue
                for subroutine in part.content:
                    if type(subroutine).__name__ not in {"Subroutine_Subprogram", "Function_Subprogram"}:
                        continue
                    annotated = any(
                        type(child).__name__ == "Comment" and str(child).strip().lower() == "! kernel"
                        for child in subroutine.content
                    )
                    if not self.require_markers or annotated:
                        self.register_routine(subroutine, module_name, None, annotated)

    def register_routine(self, node, module, parent, annotated=False):
        statement = next(child for child in node.content
                         if type(child).__name__ in {"Subroutine_Stmt", "Function_Stmt"})
        name = str(statement.items[1]).lower()
        key = (module.lower(), (parent[1] + "::" if parent else "") + name)
        if key in self.routine_nodes:
            raise self.error(f"duplicate procedure {name}", statement)
        self.routine_nodes[key] = (module, node)
        self.parents[key] = parent
        if annotated:
            self.annotated.add(key)
        for part in node.content:
            if type(part).__name__ == "Internal_Subprogram_Part":
                for child in part.content:
                    if type(child).__name__ in {"Subroutine_Subprogram", "Function_Subprogram"}:
                        self.register_routine(child, module, key)

    def resolve_routine(self, key: tuple[str, str], provenance: tuple[str, ...] = ()) -> _Routine:
        if key not in self.routines:
            module, node = self.routine_nodes[key]
            try:
                self.routines[key] = self.routine(node, module, key)
            except CompilationError as error:
                location = error.location
                if location is not None:
                    location = SourceLocation(location.path, location.line, provenance)
                raise CompilationError(error.message, location) from error
        return self.routines[key]

    def routine(self, node: Any, module: str, key: tuple[str, str]) -> _Routine:
        statement = next(child for child in node.content
                         if type(child).__name__ in {"Subroutine_Stmt", "Function_Stmt"})
        prefix, name, argument_list, suffix = statement.items
        function = type(statement).__name__ == "Function_Stmt"
        result_name = None
        if function:
            result_name = str(suffix.items[0]).lower() if suffix is not None and suffix.items[0] else str(name).lower()
            if suffix is not None and suffix.items[1] is not None:
                raise self.error("BIND function suffixes are unsupported", statement)
        elif suffix is not None:
            raise self.error("BIND and other subroutine suffixes are unsupported", statement)
        result_type = next((value for value in getattr(prefix, "items", ())
                            if type(value).__name__ == "Intrinsic_Type_Spec"), None)
        if prefix is not None and any(str(value).upper() not in {"PURE", "RECURSIVE"}
                                      and value is not result_type for value in prefix.items):
            raise self.error(f"unsupported subroutine prefix {prefix}", statement)
        arguments = tuple(str(value).lower() for value in argument_list.items) if argument_list is not None else ()
        if len(set(arguments)) != len(arguments):
            raise self.error("duplicate dummy argument names", statement)
        declarations: list[_Declaration] = []
        execution: tuple[Any, ...] = ()
        specification = next((child for child in node.content if type(child).__name__ == "Specification_Part"), None)
        parent = self.parents[key]
        parent_kinds = self.resolve_routine(parent).kinds if parent else self.module_kinds[module.lower()]
        kinds = _KindScope(specification, parent_kinds)
        for child in node.content:
            kind = type(child).__name__
            if kind == "Specification_Part":
                declarations.extend(self.specification(child, set(arguments), kinds))
            elif kind == "Execution_Part":
                execution = tuple(child.content)
            elif kind not in {"Comment", "Subroutine_Stmt", "End_Subroutine_Stmt", "Function_Stmt",
                              "End_Function_Stmt", "Internal_Subprogram_Part"}:
                raise self.error(f"unsupported subroutine construct {kind}", child)
        names = [declaration.name for declaration in declarations]
        if len(set(names)) != len(names):
            raise self.error("duplicate declarations", statement)
        undeclared = set(arguments) - set(names)
        if undeclared:
            raise self.error(
                f"dummy arguments require explicit declarations: {', '.join(sorted(undeclared))}", statement
            )
        if result_name is not None and result_name not in names:
            if result_type is None:
                raise self.error("function result requires an explicit scalar declaration", statement)
            type_name, selector = result_type.items
            if str(type_name).upper() == "REAL":
                dtype = ScalarType.REAL32 if selector is None else kinds.real_type(selector.items[1], self.location(statement))
            elif str(type_name).upper() == "DOUBLE PRECISION" and selector is None:
                dtype = ScalarType.REAL
            elif str(type_name).upper() in {"INTEGER", "LOGICAL"} and selector is None:
                dtype = ScalarType.INTEGER if str(type_name).upper() == "INTEGER" else ScalarType.LOGICAL
            else:
                raise self.error("unsupported function result type", statement)
            declarations.append(_Declaration(result_name, dtype, 0, None, self.location(statement), result_name))
        return _Routine(str(name), module, arguments, tuple(declarations), execution,
                        self.location(statement), kinds, key, parent, result_name,
                        any(str(value).upper() == "PURE" for value in getattr(prefix, "items", ())))

    def specification(self, node: Any, arguments: set[str], kinds: _KindScope) -> list[_Declaration]:
        declarations: list[_Declaration] = []
        for child in node.content:
            kind = type(child).__name__
            if kind == "Type_Declaration_Stmt":
                type_node, attributes, entities = child.items
                if (
                    type(type_node).__name__ == "Intrinsic_Type_Spec"
                    and type_node.items == ("INTEGER", None)
                    and tuple(str(item).upper() for item in getattr(attributes, "items", ())) == ("PARAMETER",)
                    and all(entity.items[1] is None and entity.items[3] is not None for entity in entities.items)
                ):
                    if any(str(entity.items[0]).lower() in arguments for entity in entities.items):
                        raise self.error("dummy arguments cannot be PARAMETER constants", child)
                    continue
                declarations.extend(self.declaration(child, arguments, kinds))
            elif kind == "Use_Stmt" and str(child.items[2]).lower() in {
                *_KindScope.intrinsic_kinds, "ieee_arithmetic", "ieee_exceptions"
            }:
                if str(child.items[0]).upper() == "NON_INTRINSIC":
                    raise self.error("non-intrinsic module imports are unsupported", child)
            elif kind == "Implicit_Part":
                for item in child.content:
                    if type(item).__name__ == "Comment":
                        continue
                    if type(item).__name__ != "Implicit_Stmt" or str(item).strip().upper() != "IMPLICIT NONE":
                        raise self.error("only IMPLICIT NONE is supported", item)
            elif kind != "Comment":
                raise self.error(f"unsupported specification statement {kind}: {child}", child)
        return declarations

    def declaration(self, node: Any, arguments: set[str], kinds: _KindScope) -> list[_Declaration]:
        type_node, attributes, entities = node.items
        if type(type_node).__name__ != "Intrinsic_Type_Spec":
            raise self.error("only INTEGER, LOGICAL, and REAL declarations are supported", node)
        type_name, selector = type_node.items
        if str(type_name).upper() == "INTEGER" and selector is None:
            dtype = ScalarType.INTEGER
        elif str(type_name).upper() == "LOGICAL" and selector is None:
            dtype = ScalarType.LOGICAL
        elif str(type_name).upper() == "REAL" and selector is None:
            dtype = ScalarType.REAL32
        elif str(type_name).upper() == "REAL" and selector is not None and type(selector).__name__ == "Kind_Selector":
            dtype = kinds.real_type(selector.items[1], self.location(node))
        elif str(type_name).upper() == "DOUBLE PRECISION" and selector is None:
            dtype = ScalarType.REAL
        else:
            raise self.error("only default INTEGER, LOGICAL, and REAL kinds 4 and 8 are supported", node)
        intent: str | None = None
        dimensions = None
        contiguous = False
        parameter = False
        for attribute in attributes.items if attributes is not None else ():
            kind = type(attribute).__name__
            if kind == "Intent_Attr_Spec":
                if intent is not None:
                    raise self.error("duplicate INTENT attributes", node)
                intent = str(attribute.items[1]).lower()
            elif kind == "Dimension_Attr_Spec":
                dimensions = attribute.items[1]
            elif kind == "Attr_Spec" and str(attribute).upper() == "CONTIGUOUS":
                contiguous = True
            elif kind == "Attr_Spec" and str(attribute).upper() == "TARGET":
                # Identity remains the Symbol itself. Pointer association is not
                # supported, so TARGET does not change the computation model.
                continue
            elif kind == "Attr_Spec" and str(attribute).upper() == "PARAMETER":
                parameter = True
            else:
                raise self.error(f"unsupported declaration attribute {attribute}", node)
        result: list[_Declaration] = []
        for entity in entities.items:
            name_node, entity_dimensions, length, initializer = entity.items
            name = str(name_node).lower()
            if length is not None or (initializer is not None and not parameter):
                raise self.error(
                    "initialized variables, PARAMETER declarations, and character lengths are unsupported", node
                )
            if dimensions is not None and entity_dimensions is not None:
                raise self.error(f"duplicate array dimensions for {name}", node)
            shape = entity_dimensions if entity_dimensions is not None else dimensions
            rank = 0
            bounds: tuple[tuple[int, int], ...] = ()
            if shape is not None:
                rank = len(shape.items)
                if dtype is ScalarType.LOGICAL:
                    raise self.error("LOGICAL arrays are unsupported", node)
                if type(shape).__name__ == "Explicit_Shape_Spec_List":
                    bounds = tuple((1 if item.items[0] is None else kinds.integer(item.items[0], self.location(node)),
                                    kinds.integer(item.items[1], self.location(node))) for item in shape.items)
                    count = 1
                    for lower, upper in bounds:
                        count *= max(0, upper - lower + 1)
                    if count > 256:
                        raise self.error("private fixed array exceeds the 256-element scalarization budget", node)
                elif type(shape).__name__ != "Assumed_Shape_Spec_List" or any(
                    value.items != (None, None) for value in shape.items
                ):
                    raise self.error("arrays require assumed shape ':' or bounded constant explicit shape", node)
                elif name not in arguments:
                    raise self.error(f"local arrays are unsupported: {name}", node)
            elif contiguous:
                raise self.error("CONTIGUOUS requires an array", node)
            entity_intent = intent
            if name in arguments:
                # Unspecified array intent conservatively preserves both the
                # input and output storage. Scalar dummies remain value inputs;
                # any attempted write is diagnosed during lowering.
                if entity_intent is None:
                    entity_intent = "inout" if rank else "in"
            elif intent is not None:
                raise self.error(f"INTENT is only valid for dummy arguments: {name}", node)
            if parameter and (name in arguments or initializer is None or rank and (rank != 1 or not bounds)):
                raise self.error("PARAMETER declarations require initialized scalars or bounded rank-one vectors", node)
            result.append(
                _Declaration(
                    name,
                    dtype,
                    rank,
                    entity_intent,
                    self.location(node),
                    str(name_node),
                    name in arguments and intent is None,
                    bounds,
                    initializer.items[1] if initializer else None,
                    parameter,
                )
            )
        return result

    def new_symbol(self, declaration: _Declaration, parameter: bool = False,
                   *, private_array_origin: PrivateArrayOrigin | None = None) -> Symbol:
        symbol = Symbol(
            len(self.symbols), declaration.spelling, declaration.dtype, declaration.rank, declaration.intent, parameter,
            private_array_origin
        )
        self.symbols.append(symbol)
        return symbol

    def lower(self, routine: _Routine) -> FunctionIR:
        if routine.result is not None or routine.parent is not None:
            raise CompilationError("entries must be module subroutines", routine.location)
        declarations = {declaration.name: declaration for declaration in routine.declarations}
        if any(declarations[name].bounds for name in routine.arguments):
            raise CompilationError("entry arrays must have assumed shape ':' in every dimension", routine.location)
        if any(not declarations[name].rank and declarations[name].intent != "in" for name in routine.arguments):
            raise CompilationError("writable scalar entry dummy arguments are unsupported", routine.location)
        bindings = {name: self.new_symbol(declarations[name], parameter=True) for name in routine.arguments}
        parameters = tuple(bindings[name] for name in routine.arguments)
        body = self.inline(routine, bindings, (), (), frozenset())
        return FunctionIR(routine.name, routine.module, parameters, tuple(self.symbols), body, self.path,
                          requires_numerical_environment=self.requires_numerical_environment)

    def inline(
        self,
        routine: _Routine,
        parameters: dict,
        ancestors: tuple[str, ...],
        provenance: tuple[str, ...],
        active_iterators: frozenset[Symbol],
    ) -> Block:
        key = "::".join(routine.key)
        if key in ancestors:
            raise CompilationError("recursive kernel calls are unsupported", routine.location)
        if len(ancestors) >= 8 or self.inline_calls >= 128:
            raise CompilationError("numerical helper depth/call budget exhausted", routine.location)
        self.inline_calls += 1
        bindings = dict(parameters)
        declarations = {declaration.name: declaration for declaration in routine.declarations}
        if routine.parent is not None:
            host = next((frame for frame in reversed(self.active_frames) if frame[0].key == routine.parent), None)
            if host is None:
                raise CompilationError("internal helper requires its lexical host activation", routine.location)
            bindings = {**host[1], **bindings}
            declarations = {**{name: replace(value, intent="in") if routine.pure else value
                                for name, value in host[2].items()}, **declarations}
        for declaration in routine.declarations:
            if declaration.name not in parameters:
                if declaration.constant and declaration.rank:
                    bindings[declaration.name] = _ConstantArray(declaration.name, declaration.dtype,
                                                                 declaration.bounds, ())
                elif declaration.bounds:
                    count = 1
                    for lower, upper in declaration.bounds:
                        count *= max(0, upper - lower + 1)
                    group_id = self.private_array_groups
                    self.private_array_groups += 1
                    elements = tuple(self.new_symbol(replace(declaration, spelling=f"{declaration.spelling}_{index}",
                                                             rank=0, intent=None),
                                                     private_array_origin=PrivateArrayOrigin(
                                                         group_id, declaration.bounds, index))
                                     for index in range(count))
                    bindings[declaration.name] = _PrivateArray(declaration.name, declaration.dtype,
                                                               declaration.bounds, elements)
                else:
                    bindings[declaration.name] = self.new_symbol(declaration)
        previous_kinds = self.active_kinds
        self.active_kinds = routine.kinds
        frame = (routine, bindings, declarations, (*ancestors, key), provenance, active_iterators)
        self.active_frames.append(frame)
        try:
            initializers = []
            for declaration in routine.declarations:
                if declaration.initializer is not None:
                    previous_constant = self.constant_initialization
                    self.constant_initialization = declaration.constant
                    try:
                        value, prelude = self.evaluated(declaration.initializer, bindings, declaration.location)
                    finally:
                        self.constant_initialization = previous_constant
                    if declaration.rank:
                        constant = bindings[declaration.name]
                        count = max(0, declaration.bounds[0][1] - declaration.bounds[0][0] + 1)
                        values = value.elements if isinstance(value, _FixedVector) else (value,) * count
                        if len(values) != count:
                            raise CompilationError("PARAMETER vector initializer extent differs from its declaration",
                                                   declaration.location)
                        constant_symbols = {bindings[name] for name, item in declarations.items()
                                            if item.constant and isinstance(bindings.get(name), Symbol)}
                        if prelude or any(isinstance(item, (ArrayAccess, Size)) or
                                          isinstance(item, Reference) and item.symbol not in constant_symbols
                                          for expression in values for item in walk_expr(expression)):
                            raise CompilationError("PARAMETER vector requires immutable constant expressions",
                                                   declaration.location)
                        bindings[declaration.name] = replace(constant, elements=tuple(
                            self.convert(expression, declaration.dtype, declaration.location) for expression in values))
                        continue
                    if isinstance(value, _FixedVector):
                        raise CompilationError("scalar PARAMETER initializer requires a scalar expression", declaration.location)
                    initializers.extend(prelude)
                    initializers.append(Assignment(Reference(bindings[declaration.name]), value, declaration.location))
            body = self.block(
                routine.execution, routine, bindings, declarations, (*ancestors, key), provenance, active_iterators
            )
            return Block(tuple(initializers) + body.statements)
        finally:
            self.active_frames.pop()
            self.active_kinds = previous_kinds

    def evaluated(self, node, bindings, location):
        """Keep pure-call work at the exact evaluation point of its expression."""
        previous = self.pending
        self.pending = []
        try:
            expression = self.expression(node, bindings, location)
            return expression, tuple(self.pending)
        finally:
            self.pending = previous

    def block(
        self,
        nodes: tuple[Any, ...],
        routine: _Routine,
        bindings: dict[str, Symbol],
        declarations: dict[str, _Declaration],
        ancestors: tuple[str, ...],
        provenance: tuple[str, ...],
        active_iterators: frozenset[Symbol],
    ) -> Block:
        statements: list[Assignment | Loop | If] = []
        for node in nodes:
            kind = type(node).__name__
            if kind == "Comment":
                continue
            location = self.location(node, provenance)
            if kind == "Assignment_Stmt":
                target_node, _, value_node = node.items
                target, target_prelude = self.evaluated(target_node, bindings, location)
                targets = target.elements if isinstance(target, _FixedVector) else (target,)
                if any(not isinstance(item, (Reference, ArrayAccess)) for item in targets):
                    raise CompilationError("assignment targets must be scalar variables or array elements", location)
                name = str(target_node if type(target_node).__name__ == "Name" else target_node.items[0]).lower()
                if declarations[name].intent == "in" or declarations[name].constant or any(
                        item.symbol.intent == "in" for item in targets):
                    raise CompilationError(f"cannot write INTENT(IN) variable {name}", location)
                if any(item.symbol in active_iterators for item in targets):
                    raise CompilationError(f"cannot modify active loop iterator {name}", location)
                value, value_prelude = self.evaluated(value_node, bindings, location)
                if (self.dtype(target) is ScalarType.LOGICAL) != (self.dtype(value) is ScalarType.LOGICAL):
                    raise CompilationError("assignment requires compatible logical or numeric types", location)
                statements.extend(target_prelude)
                statements.extend(value_prelude)
                if isinstance(target, _FixedVector):
                    if ((not isinstance(value, _FixedVector) and self.has_floating_value(value))
                            or self.has_floating_work((*target_prelude, *value_prelude))):
                        self.requires_numerical_environment = True
                    if isinstance(value, _FixedVector):
                        if len(target.elements) != len(value.elements):
                            raise CompilationError("array assignment vector extents differ", location)
                        # Fortran evaluates all RHS values before defining any
                        # overlapping LHS element, including kind conversions.
                        snapshots = []
                        for expression in value.elements:
                            symbol = self.temporary(self.dtype(target), location, "fort_snapshot")
                            statements.append(Assignment(Reference(symbol), expression, location))
                            snapshots.append(Reference(symbol))
                    else:
                        symbol = self.temporary(self.dtype(target), location, "fort_broadcast")
                        statements.append(Assignment(Reference(symbol), value, location))
                        snapshots = [Reference(symbol)] * len(target.elements)
                    statements.extend(Assignment(destination, expression, location)
                                      for destination, expression in zip(target.elements, snapshots, strict=True))
                elif isinstance(value, _FixedVector):
                    raise CompilationError("scalar assignment requires a scalar expression", location)
                else:
                    statements.append(Assignment(target, value, location))
            elif kind in {"If_Stmt", "If_Construct"}:

                def lower_branch(children):
                    return self.block(
                        tuple(children), routine, bindings, declarations, ancestors, provenance, active_iterators
                    )

                if kind == "If_Stmt":
                    condition, prelude = self.evaluated(node.items[0], bindings, location)
                    then_body = lower_branch((node.items[1],))
                    else_body = Block(())
                else:
                    branches = []
                    children = []
                    header = node.content[0]
                    for child in node.content[1:]:
                        child_kind = type(child).__name__
                        if child_kind in {"Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                            branches.append((header, lower_branch(children)))
                            header, children = child, []
                        else:
                            children.append(child)
                    else_body = Block(())
                    for header, branch_body in reversed(branches):
                        if type(header).__name__ == "Else_Stmt":
                            else_body = branch_body
                            continue
                        branch_location = self.location(header, provenance)
                        condition, prelude = self.evaluated(header.items[0], bindings, branch_location)
                        if isinstance(condition, _FixedVector) or self.dtype(condition) is not ScalarType.LOGICAL:
                            raise CompilationError("IF condition must be LOGICAL", branch_location)
                        then_body = branch_body
                        else_body = Block(prelude + (If(condition, then_body, else_body, branch_location),))
                    statements.extend(else_body.statements)
                    continue
                if isinstance(condition, _FixedVector) or self.dtype(condition) is not ScalarType.LOGICAL:
                    raise CompilationError("IF condition must be LOGICAL", location)
                statements.extend(prelude)
                statements.append(If(condition, then_body, else_body, location))
            elif kind == "Block_Nonlabel_Do_Construct":
                loop_nodes = [child for child in node.content if type(child).__name__ != "Comment"]
                header = loop_nodes[0]
                location = self.location(header, provenance)
                if type(header).__name__ != "Nonlabel_Do_Stmt":
                    raise CompilationError("only nonlabel counted DO loops are supported", location)
                control = header.items[1]
                if (
                    control is None
                    or control.items[0] is not None
                    or control.items[1] is None
                    or any(value is not None for value in control.items[2:])
                ):
                    raise CompilationError("only counted DO loops are supported", location)
                iterator_node, ranges = control.items[1]
                iterator = self.lookup(iterator_node, bindings, location)
                if iterator.rank or iterator.dtype != ScalarType.INTEGER:
                    raise CompilationError("loop iterators must be INTEGER scalars", location)
                if iterator.intent == "in" or declarations[str(iterator_node).lower()].intent == "in":
                    raise CompilationError("loop iterators must be writable local INTEGER scalars", location)
                if iterator in active_iterators:
                    raise CompilationError("nested loops cannot reuse an active iterator", location)
                step: Expr | int = 1
                if len(ranges) == 3:
                    step, prelude = self.evaluated(ranges[2], bindings, location)
                    statements.extend(prelude)
                    if isinstance(step, _FixedVector) or self.dtype(step) != ScalarType.INTEGER:
                        raise CompilationError("loop strides must be INTEGER expressions", location)
                    constant = step
                    while isinstance(constant, Unary):
                        constant = constant.operand
                    if isinstance(constant, Literal) and int(constant.value) == 0:
                        raise CompilationError("loop stride must not be zero", location)
                lower, prelude = self.evaluated(ranges[0], bindings, location)
                statements.extend(prelude)
                upper, prelude = self.evaluated(ranges[1], bindings, location)
                statements.extend(prelude)
                if any(isinstance(value, _FixedVector) or self.dtype(value) != ScalarType.INTEGER for value in (lower, upper)):
                    raise CompilationError("loop bounds must be INTEGER expressions", location)
                private_index = any(
                    type(access).__name__ == "Part_Ref"
                    and isinstance(bindings.get(str(access.items[0]).lower()), (_PrivateArray, _ConstantArray))
                    and any(type(name).__name__ == "Name" and str(name).lower() == str(iterator_node).lower()
                            for name in walk(access.items[1]))
                    for child in loop_nodes[1:-1] for access in walk(child)
                )
                if private_index:
                    start, stop = constant_integer(lower, location), constant_integer(upper, location)
                    stride = step if isinstance(step, int) else constant_integer(step, location)
                    if None in (start, stop, stride) or stride == 0:
                        raise CompilationError("private array indexing loops require bounded constant ranges", location)
                    trip_count = max(0, (stop-start)//stride+1)
                    if self.unrolled_iterations + trip_count > 256:
                        raise CompilationError("private array indexing loop exceeds the 256-iteration unrolling budget", location)
                    self.unrolled_iterations += trip_count
                    old_values = dict(self.unrolled_values)
                    try:
                        for ordinal in range(trip_count):
                            value = start + ordinal*stride
                            statements.append(Assignment(Reference(iterator), Literal(str(value), ScalarType.INTEGER), location))
                            self.unrolled_values[iterator] = value
                            statements.extend(self.block(tuple(loop_nodes[1:-1]), routine, bindings, declarations,
                                                         ancestors, provenance, active_iterators | {iterator}).statements)
                    finally:
                        self.unrolled_values = old_values
                    final = start + trip_count*stride
                    integer_literal(str(final), location)
                    statements.append(Assignment(Reference(iterator), Literal(str(final), ScalarType.INTEGER), location))
                    continue
                body = self.block(
                    tuple(loop_nodes[1:-1]),
                    routine,
                    bindings,
                    declarations,
                    ancestors,
                    provenance,
                    active_iterators | {iterator},
                )
                statements.append(Loop(iterator, lower, upper, body, location, step))
            elif kind == "Call_Stmt":
                name_node, argument_list = node.items
                if type(name_node).__name__ != "Name":
                    raise CompilationError("only direct calls to module subroutines are supported", location)
                callee = self.call_target(str(name_node), routine, location)
                if callee.result is not None:
                    raise CompilationError("function used as a subroutine", location)
                block, _ = self.call(callee, argument_list, bindings, declarations,
                                     ancestors, provenance, active_iterators, location)
                statements.extend(block.statements)
            else:
                raise CompilationError(f"unsupported execution construct {kind}: {node}", location)
        return Block(tuple(statements))

    def call_key(self, name, routine):
        scope = routine.key
        while scope is not None:
            key = (routine.module.lower(), scope[1] + "::" + name.lower())
            if key in self.routine_nodes:
                return key
            scope = self.parents[scope]
        key = (routine.module.lower(), name.lower())
        return key if key in self.routine_nodes else None

    def call_target(self, name, routine, location):
        provenance = (*location.call_stack, f"{routine.name} at {self.path}:{location.line} -> {name}")
        key = self.call_key(name, routine)
        if key is None:
            reason = "unannotated or unknown" if self.require_markers else "unknown or external"
            raise CompilationError(f"call to {reason} kernel {name}", location)
        return self.resolve_routine(key, provenance)

    def call(self, callee, argument_list, bindings, declarations, ancestors,
             provenance, active_iterators, location):
        """Inline a source-backed call after binding its actual storage."""
        key = "::".join(callee.key)
        if key in ancestors:
            raise CompilationError(f"recursive kernel call to {callee.name} is unsupported", location)
        actual_nodes = getattr(argument_list, "items", ())
        if len(actual_nodes) != len(callee.arguments):
            raise CompilationError(f"call to {callee.name} has the wrong number of arguments", location)
        formal_declarations = {declaration.name: declaration for declaration in callee.declarations}
        actuals, prelude, storage, writable = {}, [], [], set()
        for formal, actual_node in zip(callee.arguments, actual_nodes, strict=True):
            expected = formal_declarations[formal]
            if not expected.rank and expected.intent in {"out", "inout"} and not callee.pure:
                origin = expected.location
                origin = SourceLocation(origin.path, origin.line, (*provenance,
                    f"{self.active_frames[-1][0].name} at {self.path}:{location.line} -> {callee.name}"))
                raise CompilationError("writable scalar helper arguments require a PURE numerical closure", origin)
            name = str(actual_node).lower() if type(actual_node).__name__ == "Name" else None
            actual = bindings.get(name) if name is not None else None
            if type(actual_node).__name__ == "Actual_Arg_Spec" or (name is None and not callee.pure):
                raise CompilationError("call arguments must be positional whole variables for non-pure helpers", location)
            if expected.rank:
                if actual is None or not actual.rank:
                    if actual is not None:
                        raise CompilationError(f"type or rank mismatch for argument {formal} of {callee.name}", location)
                    raise CompilationError("array call arguments must be positional whole variables", location)
                if expected.bounds:
                    if not isinstance(actual, (_PrivateArray, _ConstantArray)):
                        raise CompilationError("explicit-shape numerical helpers require private fixed arrays", location)
                    expected_extents = tuple(max(0, upper-lower+1) for lower, upper in expected.bounds)
                    actual_extents = tuple(max(0, upper-lower+1) for lower, upper in actual.bounds)
                    if expected_extents != actual_extents:
                        raise CompilationError("private array helper shape mismatch", location)
                    actual = replace(actual, bounds=expected.bounds)
                elif isinstance(actual, (_PrivateArray, _ConstantArray)):
                    actual = replace(actual, bounds=tuple((1, max(0, hi-lo+1)) for lo, hi in actual.bounds))
            elif actual is None or actual.rank:
                if expected.intent != "in":
                    raise CompilationError("writable scalar helper arguments require private whole variables", location)
                value, prefix = self.evaluated(actual_node, bindings, location)
                if isinstance(value, _FixedVector):
                    raise CompilationError("scalar helper argument requires a scalar expression", location)
                prelude.extend(prefix)
                actual = self.new_symbol(replace(expected, intent=None, rank=0))
                prelude.append(Assignment(Reference(actual), value, location))
                if self.dtype(value) is not expected.dtype:
                    raise CompilationError(f"type or rank mismatch for argument {formal} of {callee.name}", location)
            if actual.rank != expected.rank or actual.dtype != expected.dtype:
                raise CompilationError(f"type or rank mismatch for argument {formal} of {callee.name}", location)
            symbols = (set(actual.elements) if isinstance(actual, _PrivateArray) else
                       {value.symbol for expression in actual.elements for value in walk_expr(expression)
                        if isinstance(value, Reference)} if isinstance(actual, _ConstantArray) else {actual})
            modifies = expected.intent in {"out", "inout"} and not expected.inferred_intent
            if modifies:
                actual_declaration = declarations.get(name)
                if (isinstance(actual, _ConstantArray) or
                    (actual_declaration is not None and (actual_declaration.intent == "in" or actual_declaration.constant))
                    or any(symbol.intent == "in" or symbol in active_iterators for symbol in symbols)):
                    raise CompilationError(f"writable argument {formal} of {callee.name} aliases a read-only variable", location)
                if not expected.rank and (not callee.pure or any(symbol.parameter for symbol in symbols)):
                    raise CompilationError("writable scalar helper arguments require pure calls and private storage", location)
                writable.update(symbols)
            storage.append(symbols)
            actuals[formal] = actual
        if callee.pure:
            for index, symbols in enumerate(storage):
                if symbols & writable and any(symbols & other for other in storage[index+1:]):
                    raise CompilationError("pure helper arguments have overlapping writable storage", location)
        if callee.result is not None:
            if not callee.pure:
                raise CompilationError("numerical helper functions must be PURE", location)
            if any(formal_declarations[name].intent != "in" for name in callee.arguments):
                raise CompilationError("pure numerical function arguments must have INTENT(IN)", location)
            declaration = formal_declarations[callee.result]
            if declaration.rank:
                raise CompilationError("numerical helper function results must be scalar", location)
            actuals[callee.result] = self.new_symbol(replace(declaration, intent=None))
        call_provenance = (*provenance, f"{self.active_frames[-1][0].name} at {self.path}:{location.line} -> {callee.name}")
        body = self.inline(callee, actuals, ancestors, call_provenance, active_iterators)
        # INTENT(OUT) kills prior definitions, including values from a previous
        # invocation/iteration. Require complete private outputs so scalarized
        # storage cannot accidentally keep an old element alive.
        output_symbols = set()
        for formal, actual in actuals.items():
            if formal_declarations[formal].intent == "out" or formal == callee.result:
                output_symbols.update(actual.elements if isinstance(actual, _PrivateArray) else
                                      (actual,) if not actual.rank else ())
        if output_symbols:
            from compiler.analysis.semantics import validate_block
            defined = validate_block(body, set(self.symbols) - output_symbols, active_iterators)
            if not output_symbols <= defined:
                raise CompilationError("private INTENT(OUT) helper results require complete ordered definitions", location)
        return Block(tuple(prelude) + body.statements), actuals.get(callee.result)

    @staticmethod
    def lookup(node: Any, bindings: dict[str, Symbol], location: SourceLocation) -> Symbol:
        name = str(node).lower()
        if name not in bindings:
            raise CompilationError(f"unknown or undeclared variable {node}", location)
        return bindings[name]

    def expression(self, node: Any, bindings: dict[str, Symbol], location: SourceLocation) -> Expr:
        expression = self._expression(node, bindings, location)
        for value in expression.elements if isinstance(expression, _FixedVector) else (expression,):
            constant_integer(value, location)
        return expression

    def temporary(self, dtype, location, prefix):
        name = prefix + "_" + str(len(self.symbols))
        return self.new_symbol(_Declaration(name, dtype, 0, None, location, name))

    def fixed_vector(self, dtype, elements, location):
        elements = tuple(elements)
        self.vector_elements += len(elements)
        if len(elements) > 256 or self.vector_elements > 4096:
            raise CompilationError("fixed vector expansion exceeds its bounded scalarization budget", location)
        # Scalarization must not erase the floating environment contract of
        # new vector arithmetic, conversions or assignments. Address metadata
        # alone (for example SIZE of a real array) is not a floating value.
        if (dtype in {ScalarType.REAL32, ScalarType.REAL}
                or any(self.has_floating_value(value) for value in elements)
                or self.has_floating_work(self.pending)):
            self.requires_numerical_environment = True
        return _FixedVector(dtype, elements)

    @staticmethod
    def has_floating_value(expression):
        real = {ScalarType.REAL32, ScalarType.REAL}
        return any((isinstance(item, (Literal, IntrinsicCall)) and item.dtype in real)
                   or (isinstance(item, (Reference, ArrayAccess)) and item.symbol.dtype in real)
                   for item in walk_expr(expression))

    def has_floating_work(self, statements):
        for statement in statements:
            if isinstance(statement, Assignment):
                if self.has_floating_value(statement.value) or self.has_floating_value(statement.target):
                    return True
            elif isinstance(statement, Loop):
                bounds = (statement.lower, statement.upper) + (() if isinstance(statement.step, int) else (statement.step,))
                if any(self.has_floating_value(bound) for bound in bounds) or self.has_floating_work(statement.body.statements):
                    return True
            elif isinstance(statement, If) and (self.has_floating_value(statement.condition)
                    or self.has_floating_work(statement.then_body.statements)
                    or self.has_floating_work(statement.else_body.statements)):
                return True
        return False

    def convert(self, expression, dtype, location):
        source = self.dtype(expression)
        if source is dtype:
            return expression
        if ScalarType.LOGICAL in {source, dtype}:
            raise CompilationError("constant initialization requires compatible logical or numeric types", location)
        if dtype is ScalarType.INTEGER:
            return IntrinsicCall("int", (expression,), dtype)
        return IntrinsicCall("real", (expression, Literal("4" if dtype is ScalarType.REAL32 else "8",
                                                        ScalarType.INTEGER)), dtype)

    def elementwise(self, arguments, operation, dtype, location, *, scalar_arguments=frozenset()):
        vectors = [value for value in arguments if isinstance(value, _FixedVector)]
        if not vectors:
            return operation(*arguments)
        extent = len(vectors[0].elements)
        if any(len(value.elements) != extent for value in vectors):
            raise CompilationError("array expression vector extents differ", location)
        values = []
        for index, value in enumerate(arguments):
            if isinstance(value, _FixedVector):
                if index in scalar_arguments:
                    raise CompilationError("array expression requires a scalar intrinsic argument", location)
                values.append(value.elements)
            elif index in scalar_arguments or isinstance(value, Literal) or self.constant_initialization:
                values.append((value,) * extent)
            else:
                constant_integer(value, location)
                symbol = self.temporary(self.dtype(value), location, "fort_broadcast")
                self.pending.append(Assignment(Reference(symbol), value, location))
                values.append((Reference(symbol),) * extent)
        return self.fixed_vector(dtype, (operation(*items) for items in zip(*values, strict=True)), location)

    def integer_affine(self, expression, location, depth=0):
        """Cancel equal runtime origins without evaluating any runtime bound."""
        if depth > 64:
            raise CompilationError("fixed vector bound expression exceeds its depth budget", location)
        value = constant_integer(expression, location)
        if value is not None:
            return {}, value
        if isinstance(expression, Unary) and expression.operator in {"+", "-"}:
            terms, offset = self.integer_affine(expression.operand, location, depth + 1)
            factor = -1 if expression.operator == "-" else 1
            return {key: factor * value for key, value in terms.items()}, factor * offset
        if isinstance(expression, Binary) and expression.operator in {"+", "-"}:
            left, a = self.integer_affine(expression.left, location, depth + 1)
            right, b = self.integer_affine(expression.right, location, depth + 1)
            factor = -1 if expression.operator == "-" else 1
            for key, value in right.items():
                left[key] = left.get(key, 0) + factor * value
                if not left[key]:
                    del left[key]
            if len(left) > 32:
                raise CompilationError("fixed vector bound expression exceeds its term budget", location)
            return left, a + factor * b
        return {expression: 1}, 0

    def vector_section(self, array, subscripts, bindings, location):
        if len(subscripts) != array.rank:
            raise CompilationError("array section rank mismatch", location)
        axes, vector_axis = [], None
        bounded = isinstance(array, (_PrivateArray, _ConstantArray))
        for axis, subscript in enumerate(subscripts):
            if subscript is not None and type(subscript).__name__ != "Subscript_Triplet":
                index = self._expression(subscript, bindings, location)
                if isinstance(index, _FixedVector) or self.dtype(index) is not ScalarType.INTEGER:
                    raise CompilationError("array subscripts require scalar INTEGER expressions", location)
                axes.append(index)
                continue
            if vector_axis is not None:
                raise CompilationError("fixed vector sections require exactly one varying axis", location)
            vector_axis = axis
            start_node, stop_node, stride_node = (None, None, None) if subscript is None else subscript.items
            stride = 1 if stride_node is None else constant_integer(self._expression(stride_node, bindings, location), location)
            if stride is None or stride == 0:
                raise CompilationError("fixed vector sections require a constant nonzero stride", location)
            lower = Literal(str(array.bounds[axis][0]) if bounded else "1", ScalarType.INTEGER)
            upper = Literal(str(array.bounds[axis][1]), ScalarType.INTEGER) if bounded else Size(array, axis + 1)
            # Fortran's omitted first/last bounds are LBOUND/UBOUND even when
            # the stride is negative. A reverse traversal needs explicit ends.
            start = lower if start_node is None else self._expression(start_node, bindings, location)
            stop = upper if stop_node is None else self._expression(stop_node, bindings, location)
            if any(isinstance(item, _FixedVector) or self.dtype(item) is not ScalarType.INTEGER for item in (start, stop)):
                raise CompilationError("fixed vector bounds require scalar INTEGER expressions", location)
            a, c = self.integer_affine(start, location)
            b, d = self.integer_affine(stop, location)
            if a != b:
                raise CompilationError("array section requires bounded constant cardinality", location)
            extent = max(0, (d - c) // stride + 1) if (d - c) * stride >= 0 else 0
            if extent > 256:
                raise CompilationError("fixed vector section exceeds the 256-element scalarization budget", location)
            coordinates = tuple(start if index == 0 else Binary("+", start, Literal(str(index * stride), ScalarType.INTEGER))
                                for index in range(extent))
            axes.append(coordinates)
        if vector_axis is None:
            raise CompilationError("fixed vector section requires one varying axis", location)
        elements = []
        for coordinate in axes[vector_axis]:
            indices = tuple(coordinate if axis == vector_axis else value for axis, value in enumerate(axes))
            if bounded:
                constants = tuple(constant_integer(index, location) for index in indices)
                if any(index is None for index in constants):
                    raise CompilationError("private sections require constant subscripts and bounds", location)
                value = array.element(constants, location)
                elements.append(Reference(value) if isinstance(array, _PrivateArray) else value)
            else:
                elements.append(ArrayAccess(array, indices))
        return self.fixed_vector(array.dtype, elements, location)

    def array_constructor(self, node, bindings, location):
        constructor = node.items[1]
        dtype = None
        if type(constructor).__name__ == "Ac_Spec":
            type_node, constructor = constructor.items
            if type(type_node).__name__ != "Intrinsic_Type_Spec":
                raise CompilationError("array constructors require a supported numeric type", location)
            name, selector = type_node.items
            if str(name).upper() == "REAL":
                dtype = ScalarType.REAL32 if selector is None else self.active_kinds.real_type(selector.items[1], location)
            elif str(name).upper() == "DOUBLE PRECISION" and selector is None:
                dtype = ScalarType.REAL
            elif str(name).upper() == "INTEGER" and selector is None:
                dtype = ScalarType.INTEGER
            else:
                raise CompilationError("array constructors require a supported numeric type", location)
        elements = []
        for item in getattr(constructor, "items", ()):
            value = self._expression(item, bindings, location)
            if dtype is None:
                dtype = self.dtype(value)
            if type(node.items[1]).__name__ != "Ac_Spec" and self.dtype(value) is not dtype:
                raise CompilationError("untyped array constructor elements require matching kinds", location)
            elements.extend(self.convert(element, dtype, location) for element in
                            (value.elements if isinstance(value, _FixedVector) else (value,)))
            if len(elements) > 256:
                raise CompilationError("array constructor exceeds the 256-element scalarization budget", location)
        if dtype is None:
            raise CompilationError("empty array constructors require an explicit supported type", location)
        return self.fixed_vector(dtype, elements, location)

    def intrinsic_arguments(self, name: str, argument_list: Any, location: SourceLocation) -> tuple[Any, ...]:
        """Resolve positional/keyword arguments before lowering their values."""
        if name not in INTRINSIC_ARGUMENTS:
            raise CompilationError(f"unsupported intrinsic {name}", location)
        source = getattr(argument_list, "items", ())
        parameters = INTRINSIC_ARGUMENTS[name]
        if name in {"min", "max"}:
            parameters = tuple(f"a{index + 1}" for index in range(len(source)))
        arguments = [None] * len(parameters)
        keyword_seen = False
        for index, argument in enumerate(source):
            if type(argument).__name__ == "Actual_Arg_Spec":
                keyword, argument = argument.items
                keyword = str(keyword).lower()
                if keyword not in parameters:
                    raise CompilationError(f"unknown keyword {keyword} for intrinsic {name.upper()}", location)
                index = parameters.index(keyword)
                keyword_seen = True
            elif keyword_seen:
                raise CompilationError(f"positional argument after keyword for intrinsic {name.upper()}", location)
            if index >= len(arguments):
                raise CompilationError(f"wrong number of arguments for intrinsic {name.upper()}", location)
            if arguments[index] is not None:
                raise CompilationError(f"duplicate argument {parameters[index]} for intrinsic {name.upper()}", location)
            arguments[index] = argument
        required = INTRINSICS[name].minimum_arguments if name in INTRINSICS else 1
        if len(arguments) < required or any(argument is None for argument in arguments[:required]):
            raise CompilationError(f"missing required argument for intrinsic {name.upper()}", location)
        while arguments and arguments[-1] is None:
            arguments.pop()
        if name in {"min", "max"} and any(argument is None for argument in arguments):
            raise CompilationError(f"missing required argument for intrinsic {name.upper()}", location)
        return tuple(arguments)

    def kind_argument(self, node: Any, bindings: dict[str, Symbol], location: SourceLocation) -> Literal:
        if type(node).__name__ == "Name" and str(node).lower() not in bindings:
            value = self.active_kinds.integer(node, location)
        else:
            value = constant_integer(self._expression(node, bindings, location), location)
        if value is None:
            raise CompilationError("KIND must be a constant INTEGER expression", location)
        return Literal(str(value), ScalarType.INTEGER)

    def array_inquiry(
        self, name: str, args: tuple[Any, ...], bindings: dict[str, Symbol], location: SourceLocation
    ) -> Expr:
        if type(args[0]).__name__ != "Name":
            raise CompilationError(f"{name.upper()} requires a whole array", location)
        symbol = self.lookup(args[0], bindings, location)
        if not symbol.rank:
            raise CompilationError(f"{name.upper()} requires an array", location)
        if len(args) == 3 and self.kind_argument(args[2], bindings, location).value != "4":
            raise CompilationError(f"{name.upper()} supports only default INTEGER result kind 4", location)
        if len(args) == 1 or args[1] is None:
            if name != "size":
                raise CompilationError(f"{name.upper()} requires DIM; array-valued results are unsupported", location)
            if isinstance(symbol, (_PrivateArray, _ConstantArray)):
                total = 1
                for lower, upper in symbol.bounds:
                    total *= max(0, upper-lower+1)
                return Literal(str(total), ScalarType.INTEGER)
            total: Expr = Size(symbol, 1)
            for dimension in range(2, symbol.rank + 1):
                total = Binary("*", total, Size(symbol, dimension))
            return total
        dim = self._expression(args[1], bindings, location)
        if self.dtype(dim) is not ScalarType.INTEGER:
            raise CompilationError(f"{name.upper()} dimension must be INTEGER", location)
        dimension = constant_integer(dim, location)
        one = Literal("1", ScalarType.INTEGER)
        if dimension is not None:
            if not 1 <= dimension <= symbol.rank:
                raise CompilationError(
                    f"{name.upper()} dimension {dimension} is outside rank {symbol.rank} of {symbol.name}", location
                )
            if isinstance(symbol, (_PrivateArray, _ConstantArray)):
                lower, upper = symbol.bounds[dimension-1]
                value = lower if name == "lbound" else upper if name == "ubound" else max(0, upper-lower+1)
                # Fortran bounds inquiries on an empty dimension return 1/0.
                if upper < lower and name in {"lbound", "ubound"}:
                    value = 1 if name == "lbound" else 0
                return Literal(str(value), ScalarType.INTEGER)
            return one if name == "lbound" else Size(symbol, dimension)
        if isinstance(symbol, (_PrivateArray, _ConstantArray)):
            raise CompilationError("private array inquiries require constant DIM", location)
        # Assumed-shape dummy bounds start at 1. Select runtime dimensions from
        # existing extent nodes so dependence and memory analyses see metadata.
        result: Expr = one if name == "lbound" else Size(symbol, symbol.rank)
        for dimension in range(max(1, symbol.rank - 1), 0, -1):
            result = IntrinsicCall(
                "merge",
                (
                    one if name == "lbound" else Size(symbol, dimension),
                    result,
                    Binary("==", dim, Literal(str(dimension), ScalarType.INTEGER)),
                ),
                ScalarType.INTEGER,
            )
        return result

    def intrinsic(self, node: Any, bindings: dict[str, Symbol], location: SourceLocation) -> Expr:
        name_node, argument_list = node.items
        name = str(name_node).lower()
        if name == "sum":
            args = getattr(argument_list, "items", ())
            if len(args) == 1 and type(args[0]).__name__ == "Actual_Arg_Spec" and str(args[0].items[0]).lower() == "array":
                args = (args[0].items[1],)
            if len(args) != 1 or type(args[0]).__name__ == "Actual_Arg_Spec":
                raise CompilationError("short SUM supports only one fixed numeric vector without DIM or MASK", location)
            vector = self._expression(args[0], bindings, location)
            if not isinstance(vector, _FixedVector) or vector.dtype is ScalarType.LOGICAL:
                raise CompilationError("short SUM requires a fixed numeric vector", location)
            if vector.dtype in {ScalarType.REAL32, ScalarType.REAL}:
                self.requires_numerical_environment = True
            # The supported native backend's bounded serial SUM uses this
            # zero-seeded order. This is not an OpenMP reduction contract.
            result = Literal("0" if vector.dtype is ScalarType.INTEGER else "0.0", vector.dtype)
            for element in vector.elements:
                result = Binary("+", result, element)
            return result
        if name == "dot_product":
            args = getattr(argument_list, "items", ())
            if len(args) != 2:
                raise CompilationError("DOT_PRODUCT requires two private constant vector sections", location)
            left, right = (self._expression(arg, bindings, location) for arg in args)
            if not isinstance(left, _FixedVector) or not isinstance(right, _FixedVector):
                raise CompilationError("DOT_PRODUCT requires two fixed numeric vectors", location)
            if len(left.elements) != len(right.elements):
                raise CompilationError("DOT_PRODUCT vector extents differ", location)
            if left.dtype is not right.dtype or left.dtype is ScalarType.LOGICAL:
                raise CompilationError("DOT_PRODUCT requires matching numeric kinds", location)
            if left.dtype in {ScalarType.REAL32, ScalarType.REAL}:
                self.requires_numerical_environment = True
            result: Expr = Literal("0" if left.dtype is ScalarType.INTEGER else "0.0", left.dtype)
            for a, b in zip(left.elements, right.elements, strict=True):
                result = Binary("+", result, Binary("*", a, b))
            return result
        args = self.intrinsic_arguments(name, argument_list, location)
        if name in ARRAY_INQUIRIES:
            return self.array_inquiry(name, args, bindings, location)
        if name in MODEL_INQUIRIES:
            # Whole arrays and undefined scalars are valid inquiry operands.
            if type(args[0]).__name__ == "Name" and str(args[0]).lower() in bindings:
                dtype = self.lookup(args[0], bindings, location).dtype
            else:
                dtype = self.dtype(self._expression(args[0], bindings, location))
            return model_inquiry(name, dtype, location)
        arguments = tuple(
            self.kind_argument(arg, bindings, location)
            if name in KIND_ARGUMENT and index == 1
            else self._expression(arg, bindings, location)
            for index, arg in enumerate(args)
        )
        dtype = intrinsic_type(
            name, tuple(self.dtype(arg) for arg in arguments), location, kind=intrinsic_kind(name, arguments, location)
        )
        if (name in {"min", "max"} and len(arguments) > 2
                and dtype in {ScalarType.REAL32, ScalarType.REAL}
                and any(isinstance(argument, _FixedVector) for argument in arguments)):
            raise CompilationError("real vector MIN/MAX supports only two operands until native NaN ordering is proved",
                                   location)
        return self.elementwise(arguments, lambda *values: IntrinsicCall(name, values, dtype), dtype, location,
                                scalar_arguments=frozenset((1,)) if name in KIND_ARGUMENT else frozenset())

    def private_section(self, node, bindings, location):
        if type(node).__name__ == "Name":
            array = self.lookup(node, bindings, location)
            subscripts = (None,) * array.rank
        elif type(node).__name__ == "Part_Ref":
            array = self.lookup(node.items[0], bindings, location)
            subscripts = node.items[1].items
        else:
            raise CompilationError("constant sections require private fixed arrays", location)
        if not isinstance(array, _PrivateArray) or len(subscripts) != array.rank:
            raise CompilationError("constant sections require private fixed arrays", location)
        return tuple(value.symbol for value in self.vector_section(array, subscripts, bindings, location).elements)

    def _expression(self, node: Any, bindings: dict[str, Symbol], location: SourceLocation) -> Expr:
        kind = type(node).__name__
        if kind == "Name":
            if str(node).lower() not in bindings and self.active_kinds.has_constant(str(node).lower()):
                return Literal(str(self.active_kinds.integer(node, location)), ScalarType.INTEGER)
            symbol = self.lookup(node, bindings, location)
            if symbol.rank:
                return self.vector_section(symbol, (None,) * symbol.rank, bindings, location)
            if symbol in self.unrolled_values:
                return Literal(str(self.unrolled_values[symbol]), ScalarType.INTEGER)
            return Reference(symbol)
        if kind == "Logical_Literal_Constant":
            value, literal_kind = node.items
            if literal_kind is not None:
                raise CompilationError("logical literal kinds are unsupported", location)
            return Literal(str(value).lower(), ScalarType.LOGICAL)
        if kind in {"Int_Literal_Constant", "Real_Literal_Constant"}:
            value, literal_kind = node.items
            if kind == "Int_Literal_Constant" and literal_kind is not None:
                raise CompilationError(
                    "integer literal kinds are unsupported; only default INTEGER is supported", location
                )
            if kind == "Int_Literal_Constant":
                integer_literal(str(value), location, source_token=True)
                dtype = ScalarType.INTEGER
            elif literal_kind is not None:
                dtype = self.active_kinds.real_type(literal_kind, location)
            elif "d" in str(value).lower():
                dtype = ScalarType.REAL
            else:
                dtype = ScalarType.REAL32
            return Literal(str(value).replace("D", "e").replace("d", "e"), dtype)
        if kind == "Parenthesis":
            return self._expression(node.items[1], bindings, location)
        if kind == "Array_Constructor":
            return self.array_constructor(node, bindings, location)
        if kind == "Part_Ref":
            if str(node.items[0]).lower() not in bindings:
                return self.function_call(node, bindings, location)
            symbol = self.lookup(node.items[0], bindings, location)
            subscripts = node.items[1].items
            if not symbol.rank or len(subscripts) != symbol.rank:
                raise CompilationError(f"array access rank mismatch for {symbol.name}", location)
            if any(type(index).__name__ == "Subscript_Triplet" for index in subscripts):
                return self.vector_section(symbol, subscripts, bindings, location)
            indices = tuple(self._expression(index, bindings, location) for index in subscripts)
            if any(isinstance(index, _FixedVector) or self.dtype(index) != ScalarType.INTEGER for index in indices):
                raise CompilationError("array subscripts must be INTEGER expressions", location)
            if isinstance(symbol, (_PrivateArray, _ConstantArray)):
                constants = tuple(constant_integer(index, location) for index in indices)
                if any(index is None for index in constants):
                    raise CompilationError("private fixed array accesses require constant subscripts", location)
                value = symbol.element(constants, location)
                return Reference(value) if isinstance(symbol, _PrivateArray) else value
            return ArrayAccess(symbol, indices)
        if kind == "Function_Reference":
            return self.function_call(node, bindings, location)
        if kind == "Intrinsic_Function_Reference":
            name = str(node.items[0]).lower()
            if name in bindings:
                raise CompilationError("intrinsic name is shadowed by local storage: " + name, location)
            if self.call_key(name, self.active_frames[-1][0]) is not None:
                return self.function_call(node, bindings, location)
            return self.intrinsic(node, bindings, location)
        items = getattr(node, "items", ())
        if len(items) == 2 and str(items[0]).upper() in {"+", "-", ".NOT."}:
            operator = str(items[0]).lower()
            operand = self._expression(items[1], bindings, location)
            logical = self.dtype(operand) is ScalarType.LOGICAL
            if (operator == ".not.") != logical:
                raise CompilationError("invalid operand type for unary operator", location)
            return self.elementwise((operand,), lambda value: Unary(operator, value), self.dtype(operand), location)
        if len(items) == 3:
            left_node, operator, right_node = items
            operator = str(operator).lower()
            if operator == "**":
                left = self._expression(left_node, bindings, location)
                exponent = constant_integer(self._expression(right_node, bindings, location), location)
                if exponent is None or not 0 <= exponent <= 16 or self.dtype(left) is ScalarType.LOGICAL:
                    raise CompilationError("numerical powers require an INTEGER constant exponent from 0 through 16", location)
                def power(value):
                    if exponent == 0:
                        return Literal("1" if self.dtype(left) is ScalarType.INTEGER else "1.0", self.dtype(left))
                    result = value
                    for _ in range(exponent-1):
                        result = Binary("*", result, value)
                    return result
                return self.elementwise((left,), power, self.dtype(left), location)
            operators = {
                "+",
                "-",
                "*",
                "/",
                "==",
                "/=",
                "<",
                "<=",
                ">",
                ">=",
                ".eq.",
                ".ne.",
                ".lt.",
                ".le.",
                ".gt.",
                ".ge.",
                ".and.",
                ".or.",
                ".eqv.",
                ".neqv.",
            }
            if operator in operators:
                left = self._expression(left_node, bindings, location)
                right = self._expression(right_node, bindings, location)
                logical_operator = operator in {".and.", ".or.", ".eqv.", ".neqv."}
                if any((self.dtype(operand) is ScalarType.LOGICAL) != logical_operator for operand in (left, right)):
                    raise CompilationError("invalid operand types for operator " + operator, location)
                dtype = self.dtype(Binary(operator, left, right))
                return self.elementwise((left, right), lambda a, b: Binary(operator, a, b), dtype, location)
        raise CompilationError(f"unsupported expression {kind}: {node}", location)

    def function_call(self, node, bindings, location):
        routine, _, declarations, ancestors, provenance, active_iterators = self.active_frames[-1]
        callee = self.call_target(str(node.items[0]), routine, location)
        if callee.result is None:
            raise CompilationError("subroutine used as a function", location)
        body, result = self.call(callee, node.items[1], bindings, declarations,
                                 ancestors, provenance, active_iterators, location)
        self.pending.extend(body.statements)
        return Reference(result)

    def dtype(self, expression: Expr) -> ScalarType:
        if isinstance(expression, (Literal, IntrinsicCall, _FixedVector)):
            return expression.dtype
        if isinstance(expression, (Reference, ArrayAccess)):
            return expression.symbol.dtype
        if isinstance(expression, Size):
            return ScalarType.INTEGER
        if isinstance(expression, Unary):
            return self.dtype(expression.operand)
        if expression.operator not in {"+", "-", "*", "/"}:
            return ScalarType.LOGICAL
        operand_types = {self.dtype(expression.left), self.dtype(expression.right)}
        if ScalarType.REAL in operand_types:
            return ScalarType.REAL
        if ScalarType.REAL32 in operand_types:
            return ScalarType.REAL32
        return ScalarType.INTEGER


def _parse_file(path: Path, *, require_markers: bool) -> Any:
    path = Path(path)
    try:
        if require_markers:
            with path.open(encoding="utf-8") as source:
                if source.readline().strip().lower() != "! kernels":
                    raise CompilationError("kernel files must begin with '! kernels'", SourceLocation(str(path)))
        reader = FortranFileReader(str(path), ignore_comments=False)
        return ParserFactory().create(std="f2008")(reader)
    except OSError as error:
        raise CompilationError(f"cannot read source: {error}", SourceLocation(str(path))) from error
    except FortranSyntaxError as error:
        raise CompilationError(f"invalid Fortran syntax: {error}", SourceLocation(str(path))) from error


def lower_file(path: str | Path, entry_name: str, *, require_markers: bool = False) -> FunctionIR:
    """Lower an entry and its reachable same-module helpers into typed IR.

    Markers are optional unless ``require_markers`` is selected. The entry can
    be qualified as ``module::procedure`` to disambiguate a multi-module file.
    Unreachable procedures are parsed but their unsupported semantics are ignored.
    """
    path = Path(path)
    tree = _parse_file(path, require_markers=require_markers)
    return _lower_tree(tree, path, entry_name, require_markers=require_markers)


def _lower_tree(tree, path, entry_name, *, require_markers=False):
    """Share source-backed file and compiler-owned string entry resolution."""
    lowerer = _Lowerer(path, require_markers=require_markers)
    lowerer.discover(tree)
    requested = entry_name.lower().split("::")
    matches = [
        key
        for key in lowerer.routine_nodes
        if (list(key) == requested if len(requested) > 1 else key[1] == requested[0])
    ]
    if len(matches) != 1:
        reason = "not found" if not matches else "ambiguous across modules"
        qualifier = "annotated " if require_markers else ""
        raise CompilationError(f"{qualifier}entry kernel {entry_name!r} is {reason}", SourceLocation(str(path)))
    return lowerer.lower(lowerer.resolve_routine(matches[0]))


def discover_file(path: str | Path, *, require_markers: bool = False) -> tuple[ProcedureCandidate, ...]:
    """Inspect every module subroutine, reporting frontend rejections separately.

    ``lowerable`` only covers parsing, types, and supported constructs. It does
    not promise dependence legality or successful generation; callers must run
    the normal compiler pipeline before selecting a candidate for replacement.
    """
    path = Path(path)
    tree = _parse_file(path, require_markers=require_markers)
    index = _Lowerer(path, require_markers=require_markers)
    index.discover(tree)
    candidates = []
    for key, (module, node) in index.routine_nodes.items():
        if index.parents[key] is not None or type(node).__name__ != "Subroutine_Subprogram":
            continue
        statement = next(child for child in node.content if type(child).__name__ == "Subroutine_Stmt")
        reason = None
        try:
            lowerer = _Lowerer(path, require_markers=require_markers)
            lowerer.discover(tree)
            lowerer.lower(lowerer.resolve_routine(key))
        except CompilationError as error:
            reason = str(error)
        candidates.append(
            ProcedureCandidate(
                module,
                str(statement.items[1]),
                index.location(statement),
                key in index.annotated,
                reason is None,
                reason,
            )
        )
    return tuple(candidates)
