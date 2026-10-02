"""Lower the supported Fortran subset into an ordered, immutable computation IR.

Parsing details stay inside this module. In particular, inlining resolves dummy
arguments to their actual storage identities before any dependence analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    Reference,
    ScalarType,
    Size,
    SourceLocation,
    Symbol,
    Unary,
)
from compiler.ir.intrinsics import intrinsic_type


@dataclass(frozen=True)
class _Declaration:
    name: str
    dtype: ScalarType
    rank: int
    intent: str | None
    location: SourceLocation
    spelling: str
    inferred_intent: bool = False


@dataclass(frozen=True)
class _Routine:
    name: str
    module: str
    arguments: tuple[str, ...]
    declarations: tuple[_Declaration, ...]
    execution: tuple[Any, ...]
    location: SourceLocation


class _Lowerer:
    def __init__(self, path: Path):
        self.path = str(path)
        self.symbols: list[Symbol] = []
        self.routines: dict[tuple[str, str], _Routine] = {}
        self.routine_nodes: dict[tuple[str, str], tuple[str, Any]] = {}

    def location(self, node: Any, stack: tuple[str, ...] = ()) -> SourceLocation:
        item = getattr(node, "item", None)
        if item is None:
            for child in walk(node):
                item = getattr(child, "item", None)
                if item is not None:
                    break
        span = getattr(item, "span", None)
        return SourceLocation(self.path, span[0] if span else 1, stack)

    def error(self, message: str, node: Any, stack: tuple[str, ...] = ()) -> CompilationError:
        return CompilationError(message, self.location(node, stack))

    def discover(self, tree: Any) -> None:
        for module in (node for node in walk(tree) if type(node).__name__ == "Module"):
            module_statement = next(child for child in module.content if type(child).__name__ == "Module_Stmt")
            module_name = str(module_statement.items[1])
            for part in module.content:
                if type(part).__name__ != "Module_Subprogram_Part":
                    continue
                for subroutine in part.content:
                    if type(subroutine).__name__ != "Subroutine_Subprogram":
                        continue
                    if not any(
                        type(child).__name__ == "Comment" and str(child).strip().lower() == "! kernel"
                        for child in subroutine.content
                    ):
                        continue
                    statement = next(child for child in subroutine.content if type(child).__name__ == "Subroutine_Stmt")
                    name = str(statement.items[1])
                    key = (module_name.lower(), name.lower())
                    if key in self.routine_nodes:
                        raise self.error(f"duplicate annotated subroutine {name}", statement)
                    self.routine_nodes[key] = (module_name, subroutine)

    def resolve_routine(self, key: tuple[str, str], provenance: tuple[str, ...] = ()) -> _Routine:
        if key not in self.routines:
            module, node = self.routine_nodes[key]
            try:
                self.routines[key] = self.routine(node, module)
            except CompilationError as error:
                location = error.location
                if location is not None:
                    location = SourceLocation(location.path, location.line, provenance)
                raise CompilationError(error.message, location) from error
        return self.routines[key]

    def routine(self, node: Any, module: str) -> _Routine:
        statement = next(child for child in node.content if type(child).__name__ == "Subroutine_Stmt")
        prefix, name, argument_list, suffix = statement.items
        if suffix is not None:
            raise self.error("BIND and other subroutine suffixes are unsupported", statement)
        if prefix is not None and any(str(value).upper() not in {"PURE", "RECURSIVE"} for value in prefix.items):
            raise self.error(f"unsupported subroutine prefix {prefix}", statement)
        arguments = tuple(str(value).lower() for value in argument_list.items) if argument_list is not None else ()
        if len(set(arguments)) != len(arguments):
            raise self.error("duplicate dummy argument names", statement)
        declarations: list[_Declaration] = []
        execution: tuple[Any, ...] = ()
        for child in node.content:
            kind = type(child).__name__
            if kind == "Specification_Part":
                declarations.extend(self.specification(child, set(arguments)))
            elif kind == "Execution_Part":
                execution = tuple(child.content)
            elif kind not in {"Comment", "Subroutine_Stmt", "End_Subroutine_Stmt"}:
                raise self.error(f"unsupported subroutine construct {kind}", child)
        names = [declaration.name for declaration in declarations]
        if len(set(names)) != len(names):
            raise self.error("duplicate declarations", statement)
        undeclared = set(arguments) - set(names)
        if undeclared:
            raise self.error(
                f"dummy arguments require explicit declarations: {', '.join(sorted(undeclared))}", statement
            )
        return _Routine(str(name), module, arguments, tuple(declarations), execution, self.location(statement))

    def specification(self, node: Any, arguments: set[str]) -> list[_Declaration]:
        declarations: list[_Declaration] = []
        for child in node.content:
            kind = type(child).__name__
            if kind == "Type_Declaration_Stmt":
                declarations.extend(self.declaration(child, arguments))
            elif kind == "Implicit_Part":
                for item in child.content:
                    if type(item).__name__ == "Comment":
                        continue
                    if type(item).__name__ != "Implicit_Stmt" or str(item).strip().upper() != "IMPLICIT NONE":
                        raise self.error("only IMPLICIT NONE is supported", item)
            elif kind != "Comment":
                raise self.error(f"unsupported specification statement {kind}: {child}", child)
        return declarations

    def declaration(self, node: Any, arguments: set[str]) -> list[_Declaration]:
        type_node, attributes, entities = node.items
        if type(type_node).__name__ != "Intrinsic_Type_Spec":
            raise self.error("only INTEGER, LOGICAL, REAL, and REAL(knd) declarations are supported", node)
        type_name, selector = type_node.items
        if str(type_name).upper() == "INTEGER" and selector is None:
            dtype = ScalarType.INTEGER
        elif str(type_name).upper() == "LOGICAL" and selector is None:
            dtype = ScalarType.LOGICAL
        elif str(type_name).upper() == "REAL" and selector is None:
            dtype = ScalarType.REAL32
        elif (
            str(type_name).upper() == "REAL"
            and selector is not None
            and type(selector).__name__ == "Kind_Selector"
            and str(selector.items[1]).lower() == "knd"
        ):
            dtype = ScalarType.REAL
        else:
            raise self.error("only default INTEGER, LOGICAL, REAL, and REAL(knd) declarations are supported", node)
        intent: str | None = None
        dimensions = None
        contiguous = False
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
            else:
                raise self.error(f"unsupported declaration attribute {attribute}", node)
        result: list[_Declaration] = []
        for entity in entities.items:
            name_node, entity_dimensions, length, initializer = entity.items
            name = str(name_node).lower()
            if length is not None or initializer is not None:
                raise self.error(
                    "initialized variables, PARAMETER declarations, and character lengths are unsupported", node
                )
            if dimensions is not None and entity_dimensions is not None:
                raise self.error(f"duplicate array dimensions for {name}", node)
            shape = entity_dimensions if entity_dimensions is not None else dimensions
            rank = 0
            if shape is not None:
                if type(shape).__name__ != "Assumed_Shape_Spec_List" or any(
                    value.items != (None, None) for value in shape.items
                ):
                    raise self.error("arrays must have assumed shape ':' in every dimension", node)
                rank = len(shape.items)
                if dtype is ScalarType.LOGICAL:
                    raise self.error("LOGICAL arrays are unsupported", node)
                if name not in arguments:
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
                if rank == 0 and entity_intent != "in":
                    raise self.error(f"writable scalar dummy arguments are unsupported: {name}", node)
            elif intent is not None:
                raise self.error(f"INTENT is only valid for dummy arguments: {name}", node)
            result.append(
                _Declaration(
                    name,
                    dtype,
                    rank,
                    entity_intent,
                    self.location(node),
                    str(name_node),
                    name in arguments and intent is None,
                )
            )
        return result

    def new_symbol(self, declaration: _Declaration, parameter: bool = False) -> Symbol:
        symbol = Symbol(
            len(self.symbols), declaration.spelling, declaration.dtype, declaration.rank, declaration.intent, parameter
        )
        self.symbols.append(symbol)
        return symbol

    def lower(self, routine: _Routine) -> FunctionIR:
        declarations = {declaration.name: declaration for declaration in routine.declarations}
        bindings = {name: self.new_symbol(declarations[name], parameter=True) for name in routine.arguments}
        parameters = tuple(bindings[name] for name in routine.arguments)
        body = self.inline(routine, bindings, (), (), frozenset())
        return FunctionIR(routine.name, routine.module, parameters, tuple(self.symbols), body, self.path)

    def inline(
        self,
        routine: _Routine,
        parameters: dict[str, Symbol],
        ancestors: tuple[str, ...],
        provenance: tuple[str, ...],
        active_iterators: frozenset[Symbol],
    ) -> Block:
        key = routine.name.lower()
        if key in ancestors:
            raise CompilationError("recursive kernel calls are unsupported", routine.location)
        bindings = dict(parameters)
        declarations = {declaration.name: declaration for declaration in routine.declarations}
        for declaration in routine.declarations:
            if declaration.name not in bindings:
                bindings[declaration.name] = self.new_symbol(declaration)
        return self.block(
            routine.execution, routine, bindings, declarations, (*ancestors, key), provenance, active_iterators
        )

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
                target = self.expression(target_node, bindings, location)
                if not isinstance(target, (Reference, ArrayAccess)):
                    raise CompilationError("assignment targets must be scalar variables or array elements", location)
                name = str(target_node if isinstance(target, Reference) else target_node.items[0]).lower()
                if declarations[name].intent == "in" or target.symbol.intent == "in":
                    raise CompilationError(f"cannot write INTENT(IN) variable {name}", location)
                if target.symbol in active_iterators:
                    raise CompilationError(f"cannot modify active loop iterator {name}", location)
                value = self.expression(value_node, bindings, location)
                if (target.symbol.dtype is ScalarType.LOGICAL) != (self.dtype(value) is ScalarType.LOGICAL):
                    raise CompilationError("assignment requires compatible logical or numeric types", location)
                statements.append(Assignment(target, value, location))
            elif kind in {"If_Stmt", "If_Construct"}:

                def lower_branch(children):
                    return self.block(
                        tuple(children), routine, bindings, declarations, ancestors, provenance, active_iterators
                    )

                if kind == "If_Stmt":
                    condition = self.expression(node.items[0], bindings, location)
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
                        condition = self.expression(header.items[0], bindings, branch_location)
                        if self.dtype(condition) is not ScalarType.LOGICAL:
                            raise CompilationError("IF condition must be LOGICAL", branch_location)
                        then_body = branch_body
                        else_body = Block((If(condition, then_body, else_body, branch_location),))
                    statements.extend(else_body.statements)
                    continue
                if self.dtype(condition) is not ScalarType.LOGICAL:
                    raise CompilationError("IF condition must be LOGICAL", location)
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
                    step = self.expression(ranges[2], bindings, location)
                    if self.dtype(step) != ScalarType.INTEGER:
                        raise CompilationError("loop strides must be INTEGER expressions", location)
                    constant = step
                    while isinstance(constant, Unary):
                        constant = constant.operand
                    if isinstance(constant, Literal) and int(constant.value) == 0:
                        raise CompilationError("loop stride must not be zero", location)
                lower = self.expression(ranges[0], bindings, location)
                upper = self.expression(ranges[1], bindings, location)
                if self.dtype(lower) != ScalarType.INTEGER or self.dtype(upper) != ScalarType.INTEGER:
                    raise CompilationError("loop bounds must be INTEGER expressions", location)
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
                    raise CompilationError("only direct calls to annotated kernel subroutines are supported", location)
                key = (routine.module.lower(), str(name_node).lower())
                if key not in self.routine_nodes:
                    raise CompilationError(f"call to unannotated or unknown kernel {name_node}", location)
                call_provenance = (*provenance, f"{routine.name} at {self.path}:{location.line} -> {name_node}")
                callee = self.resolve_routine(key, call_provenance)
                if callee.name.lower() in ancestors:
                    raise CompilationError(f"recursive kernel call to {callee.name} is unsupported", location)
                actual_nodes = argument_list.items if argument_list is not None else ()
                if len(actual_nodes) != len(callee.arguments):
                    raise CompilationError(f"call to {callee.name} has the wrong number of arguments", location)
                formal_declarations = {declaration.name: declaration for declaration in callee.declarations}
                actuals: dict[str, Symbol] = {}
                for formal, actual_node in zip(callee.arguments, actual_nodes, strict=True):
                    if type(actual_node).__name__ != "Name":
                        raise CompilationError(
                            "call arguments must be positional whole variables; slices and expressions are unsupported",
                            location,
                        )
                    actual = self.lookup(actual_node, bindings, location)
                    expected = formal_declarations[formal]
                    if actual.rank != expected.rank or actual.dtype != expected.dtype:
                        raise CompilationError(
                            f"type or rank mismatch for argument {formal} of {callee.name}", location
                        )
                    actual_declaration = declarations[str(actual_node).lower()]
                    if (
                        expected.intent in {"out", "inout"}
                        and not expected.inferred_intent
                        and (actual.intent == "in" or actual_declaration.intent == "in" or actual in active_iterators)
                    ):
                        raise CompilationError(
                            f"writable argument {formal} of {callee.name} aliases a read-only variable", location
                        )
                    actuals[formal] = actual
                statements.extend(self.inline(callee, actuals, ancestors, call_provenance, active_iterators).statements)
            else:
                raise CompilationError(f"unsupported execution construct {kind}: {node}", location)
        return Block(tuple(statements))

    @staticmethod
    def lookup(node: Any, bindings: dict[str, Symbol], location: SourceLocation) -> Symbol:
        name = str(node).lower()
        if name not in bindings:
            raise CompilationError(f"unknown or undeclared variable {node}", location)
        return bindings[name]

    def expression(self, node: Any, bindings: dict[str, Symbol], location: SourceLocation) -> Expr:
        kind = type(node).__name__
        if kind == "Name":
            symbol = self.lookup(node, bindings, location)
            if symbol.rank:
                raise CompilationError(f"array {symbol.name} must be accessed with {symbol.rank} subscripts", location)
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
            if literal_kind is not None and str(literal_kind).lower() != "knd":
                raise CompilationError(f"unsupported literal kind {literal_kind}; only knd is supported", location)
            if kind == "Int_Literal_Constant":
                dtype = ScalarType.INTEGER
            elif literal_kind is not None or "d" in str(value).lower():
                dtype = ScalarType.REAL
            else:
                dtype = ScalarType.REAL32
            return Literal(str(value).replace("D", "e").replace("d", "e"), dtype)
        if kind == "Parenthesis":
            return self.expression(node.items[1], bindings, location)
        if kind == "Part_Ref":
            symbol = self.lookup(node.items[0], bindings, location)
            subscripts = node.items[1].items
            if not symbol.rank or len(subscripts) != symbol.rank:
                raise CompilationError(f"array access rank mismatch for {symbol.name}", location)
            indices = tuple(self.expression(index, bindings, location) for index in subscripts)
            if any(self.dtype(index) != ScalarType.INTEGER for index in indices):
                raise CompilationError("array subscripts must be INTEGER expressions", location)
            return ArrayAccess(symbol, indices)
        if kind == "Intrinsic_Function_Reference":
            name, argument_list = node.items
            if str(name).upper() != "SIZE":
                args = argument_list.items if argument_list is not None else ()
                arguments = tuple(self.expression(arg, bindings, location) for arg in args)
                dtype = intrinsic_type(str(name), tuple(self.dtype(arg) for arg in arguments), location)
                return IntrinsicCall(str(name).lower(), arguments, dtype)
            args = argument_list.items if argument_list is not None else ()
            if len(args) not in {1, 2} or type(args[0]).__name__ != "Name":
                raise CompilationError("SIZE requires a whole array and an optional literal dimension", location)
            symbol = self.lookup(args[0], bindings, location)
            if not symbol.rank:
                raise CompilationError("SIZE requires an array", location)
            if len(args) == 1:
                total: Expr = Size(symbol, 1)
                for dimension in range(2, symbol.rank + 1):
                    total = Binary("*", total, Size(symbol, dimension))
                return total
            if type(args[1]).__name__ != "Int_Literal_Constant":
                raise CompilationError("SIZE requires a whole array and an optional literal dimension", location)
            if args[1].items[1] is not None:
                raise CompilationError("SIZE dimensions require default INTEGER literals without a kind", location)
            dimension = int(args[1].items[0])
            if not 1 <= dimension <= symbol.rank:
                raise CompilationError(
                    f"SIZE dimension {dimension} is outside rank {symbol.rank} of {symbol.name}", location
                )
            return Size(symbol, dimension)
        items = getattr(node, "items", ())
        if len(items) == 2 and str(items[0]).upper() in {"+", "-", ".NOT."}:
            operator = str(items[0]).lower()
            operand = self.expression(items[1], bindings, location)
            logical = self.dtype(operand) is ScalarType.LOGICAL
            if (operator == ".not.") != logical:
                raise CompilationError("invalid operand type for unary operator", location)
            return Unary(operator, operand)
        if len(items) == 3:
            left_node, operator, right_node = items
            operator = str(operator).lower()
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
                left = self.expression(left_node, bindings, location)
                right = self.expression(right_node, bindings, location)
                logical_operator = operator in {".and.", ".or.", ".eqv.", ".neqv."}
                if any((self.dtype(operand) is ScalarType.LOGICAL) != logical_operator for operand in (left, right)):
                    raise CompilationError("invalid operand types for operator " + operator, location)
                return Binary(operator, left, right)
        raise CompilationError(f"unsupported expression {kind}: {node}", location)

    def dtype(self, expression: Expr) -> ScalarType:
        if isinstance(expression, (Literal, IntrinsicCall)):
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


def lower_file(path: str | Path, entry_name: str) -> FunctionIR:
    """Parse an annotated file and inline its selected kernel into typed IR."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as source:
            if source.readline().strip().lower() != "! kernels":
                raise CompilationError("kernel files must begin with '! kernels'", SourceLocation(str(path)))
        reader = FortranFileReader(str(path), ignore_comments=False)
        tree = ParserFactory().create(std="f2008")(reader)
    except OSError as error:
        raise CompilationError(f"cannot read source: {error}", SourceLocation(str(path))) from error
    except FortranSyntaxError as error:
        raise CompilationError(f"invalid Fortran syntax: {error}", SourceLocation(str(path))) from error
    lowerer = _Lowerer(path)
    lowerer.discover(tree)
    matches = [key for key in lowerer.routine_nodes if key[1] == entry_name.lower()]
    if len(matches) != 1:
        reason = "not found" if not matches else "ambiguous across modules"
        raise CompilationError(f"annotated entry kernel {entry_name!r} is {reason}", SourceLocation(str(path)))
    return lowerer.lower(lowerer.resolve_routine(matches[0]))
