"""Original intrinsic output may observe data without replacing storage.

This is only descriptor-lifetime evidence. Original execution, I/O ordering,
coherence, initializedness and exception/timing observations remain separate.
"""
from __future__ import annotations

from compiler.frontend.call_bindings import OriginalCallBudgetExceeded
from compiler.frontend.descriptor_termination import (
    DescriptorTerminationUnsupported,
)
from compiler.frontend.descriptor_termination import (
    _Reader as _DiagnosticReader,
)
from compiler.ir import CompilationError

_UNSAFE = {"pointer", "target", "optional", "allocatable", "volatile", "asynchronous", "value"}
_NUMERIC = {"real", "integer", "logical"}


class DescriptorOutputUnsupported(CompilationError):
    """Original intrinsic output was not established by this narrow policy."""


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _default_character(binding):
    """Read the original selector: generic binding.kind also encodes LEN."""
    scope = binding.declaring_scope
    specification = next((node for node in _children(scope.node) if _kind(node) == "Specification_Part"), None)
    for declaration in _children(specification):
        if _kind(declaration) != "Type_Declaration_Stmt":
            continue
        dtype, _attributes, entities = declaration.items
        matching = [entity for entity in _children(entities) if str(entity.items[0]).lower() == binding.name]
        if not matching:
            continue
        if (len(matching) != 1 or matching[0].items[2] is not None
                or _kind(dtype) != "Intrinsic_Type_Spec" or str(dtype.items[0]).lower() != "character"):
            return False
        selector = dtype.items[1]
        if selector is None:
            return True
        if _kind(selector) == "Length_Selector":
            length, kind = selector.items[1], None
        elif _kind(selector) == "Char_Selector":
            length, kind = selector.items
        else:
            return False
        return ((length is None or _kind(length) == "Int_Literal_Constant" and length.items[1] is None)
                and (kind is None or _kind(kind) == "Int_Literal_Constant" and kind.items == ("1", None)))
    return False


def _signature(check, routine, node, depth=0):
    # The ordinary semantic-unit resolution walk runs first. This defensive
    # recursion fence matches its existing 256-node bound; it is not a bound
    # on the selected lifetime or the number of original output statements.
    if depth >= 256:
        raise OriginalCallBudgetExceeded("descriptor output original expression unit budget exhausted")
    check.tick()
    scope = check.analysis.source_scope_for(node, routine.scope)
    kind = _kind(node)
    if kind == "Char_Literal_Constant" and node.items[1] is None:
        return "character", 1, 0
    if kind in {"Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}:
        result = check.analysis._signature(scope, node)
        if result is not None and result[0] in _NUMERIC and result[1] in {4, 8}:
            return result
    if kind == "Name":
        binding = check.analysis._binding(scope, node)
        if (binding is not None and not binding.attributes & _UNSAFE
                and (binding.dtype in _NUMERIC and binding.kind in {4, 8}
                     or binding.dtype == "character" and _default_character(binding))
                and binding.rank in {0, 1}
                and (not binding.rank or binding.dtype != "character"
                     and all(_kind(axis) == "Explicit_Shape_Spec" for axis in binding.shape_nodes))):
            return binding.dtype, 1 if binding.dtype == "character" else binding.kind, binding.rank
    if kind == "Parenthesis":
        return _signature(check, routine, node.items[1], depth + 1)
    items = _children(node)
    if len(items) == 2 and str(items[0]).lower() in {"+", "-", ".not."}:
        child = _signature(check, routine, items[1], depth + 1)
        if child[2] == 0 and ((str(items[0]).lower() == ".not." and child[0] == "logical")
                             or str(items[0]) in {"+", "-"} and child[0] in {"real", "integer"}):
            return child
    if len(items) == 3 and str(items[1]).lower() in {
            "+", "-", "*", "/", "==", "/=", "<", "<=", ">", ">=", ".eq.", ".ne.", ".lt.", ".le.",
            ".gt.", ".ge.", ".and.", ".or.", ".eqv.", ".neqv.", "//"}:
        left, right = (_signature(check, routine, items[index], depth + 1) for index in (0, 2))
        operator = str(items[1]).lower()
        if not left[2] and not right[2]:
            if operator == "//" and left[0] == right[0] == "character":
                return "character", 1, 0
            if operator in {".and.", ".or.", ".eqv.", ".neqv."} and left[0] == right[0] == "logical":
                return "logical", max(left[1], right[1]), 0
            if left[0] in {"integer", "real"} and right[0] in {"integer", "real"}:
                if operator in {"+", "-", "*", "/"}:
                    reals = [value[1] for value in (left, right) if value[0] == "real"]
                    return ("real", max(reals), 0) if reals else ("integer", max(left[1], right[1]), 0)
                if operator in {"==", "/=", "<", "<=", ">", ">=", ".eq.", ".ne.", ".lt.", ".le.", ".gt.", ".ge."}:
                    return "logical", 4, 0
    if kind == "Part_Ref":
        binding = check.analysis._binding(scope, node.items[0])
        if binding is not None and binding.rank == 1:
            result = _signature(check, routine, node.items[0], depth + 1)
            arguments = _children(node.items[1])
            if len(arguments) == 1 and _signature(check, routine, arguments[0], depth + 1) in {
                    ("integer", 4, 0), ("integer", 8, 0)}:
                return result[0], result[1], 0
    if kind == "Intrinsic_Function_Reference":
        name, arguments = node.items
        name = str(name).lower()
        if (check.analysis._binding(scope, name) is None and not check.analysis._candidates(scope, name)
                and not check.analysis._unknown_exports(scope)):
            actuals = tuple(_children(arguments))
            if name in {"sum", "real", "int", "dble"} and len(actuals) == 1:
                actual = actuals[0]
                if _kind(actual) == "Actual_Arg_Spec":
                    expected = "array" if name == "sum" else "a"
                    if str(actual.items[0]).lower() != expected:
                        raise DescriptorOutputUnsupported("descriptor output intrinsic keyword is unsupported")
                    actual = actual.items[1]
                value = _signature(check, routine, actual, depth + 1)
                if value[0] in {"real", "integer"}:
                    if name == "sum" and value[2] == 1:
                        return value[0], value[1], 0
                    if name != "sum" and value[2] == 0:
                        return ("integer" if name == "int" else "real"), (8 if name == "dble" else 4), 0
    raise DescriptorOutputUnsupported("descriptor output requires original fixed intrinsic numeric/character values")


def original_intrinsic_output(check, routine, node):
    """Check one reached original statement; never execute or move its items."""
    kind = _kind(node)
    if kind not in {"Write_Stmt", "Print_Stmt"}:
        return None
    if check.analysis._require_original(routine.qualified) is not routine:
        raise CompilationError("descriptor output requires its exact original routine")
    # This also rejects a lookalike statement carrying an original span/parent.
    check.analysis._selected_source(routine.qualified, (node,))
    if kind == "Print_Stmt":
        unit, fmt, outputs = None, node.items[0], node.items[1]
        unit_identity = "$preconnected::*"
    else:
        values, position, keyword_seen = {}, 0, False
        for control in node.items[0].items:
            key, value = control.items
            if key is None:
                if keyword_seen or position >= 2:
                    raise DescriptorOutputUnsupported("descriptor output WRITE controls are unsupported")
                key = ("unit", "fmt")[position]
                position += 1
            else:
                keyword_seen = True
            key = str(key).lower()
            if key not in {"unit", "fmt"} or key in values:
                raise DescriptorOutputUnsupported("descriptor output forbids status, asynchronous and control-transfer specifiers")
            values[key] = value
        if set(values) != {"unit", "fmt"}:
            raise DescriptorOutputUnsupported("descriptor output requires an explicit original unit and format")
        unit, fmt, outputs = values["unit"], values["fmt"], node.items[1]
        if str(unit) == "*":
            unit_identity = "$preconnected::*"
        else:
            reader = _DiagnosticReader(check.analysis, check.tick, check.resolution_tick)
            try:
                unit_identity = reader.unit(routine, unit)
            except DescriptorTerminationUnsupported as error:
                raise DescriptorOutputUnsupported("descriptor output requires an original preconnected unit") from error
            for scope in reader.scopes.values():
                check.scope_authority(scope)
    if str(fmt) != "*" and not (_kind(fmt) == "Char_Literal_Constant" and fmt.items[1] is None):
        raise DescriptorOutputUnsupported("descriptor output requires list-directed or literal character format")
    expressions = tuple(_children(outputs))
    check.resolution(routine, expressions)
    items = []
    for expression in expressions:
        signature = _signature(check, routine, expression)
        check.expression(routine, expression)
        items.append({"original_expression": str(expression), "type": signature[0],
                      "kind": signature[1], "rank": signature[2], "effect": "original readonly output evaluation"})
    return {"version": 1, "procedure": routine.qualified, "original_statement": str(node),
            "unit": unit_identity, "format": str(fmt), "outputs": items,
            "requires_original_position_and_guards": True, "requires_original_io_execution": True,
            "descriptor_evidence_only": True, "coherence_authorized": False,
            "payload_access_authorized": False, "entry_definition_authority": False,
            "execution_authorized": False, "motion_authorized": False}
