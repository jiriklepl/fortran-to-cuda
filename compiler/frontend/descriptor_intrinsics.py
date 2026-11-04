"""Typed effects of unchanged native intrinsics, independent of GPU lowering."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from operator import is_

from compiler.frontend.source_functions import _validate_resolution, _walk
from compiler.ir import CompilationError, SourceLocation

_MAPPING_COLUMNS = object()


def _source_mapping(value):
    from compiler.frontend.immutable_graph import FrozenDict

    # Only compiler-owned dictionaries with inert keys use the compact form.
    # Arbitrary Mapping implementations retain their original callback path.
    if type(value) in {dict, FrozenDict}:
        keys = tuple(dict.keys(value))
        if frozenset(map(id, map(type, keys))).issubset((id(str), id(int))):
            values = tuple(dict.values(value))
            return (_MAPPING_COLUMNS, keys, values, tuple(map(id, values)), type(value))
    return tuple(value.items())


def intrinsic_source_attachments(analysis):
    """Retain authoritative registry objects and their exact original entries."""
    from compiler.frontend.source_session import _ROOTS

    roots = []
    for name in _ROOTS:
        value = getattr(analysis, name, None)
        contents = (_source_mapping(value) if isinstance(value, Mapping)
                    else tuple(value) if isinstance(value, (tuple, list)) else None)
        roots.append((name, value, contents))
    return getattr(analysis, "_source_session", None), tuple(roots)


def same_intrinsic_source_attachments(analysis, state):
    session, roots = state
    if getattr(analysis, "_source_session", None) is not session:
        return False
    for name, value, contents in roots:
        if getattr(analysis, name, None) is not value:
            return False
        if contents is not None and contents and contents[0] is _MAPPING_COLUMNS:
            if type(value) is not contents[4] or dict.__len__(value) != len(contents[1]):
                return False
            if not frozenset(map(id, map(type, dict.keys(value)))).issubset((id(str), id(int))):
                return False
            try:
                # Identical ordered keys permit a built-in values iterator.
                # Keep the original ID column authoritative in both paths.
                values = (dict.values(value) if all(map(is_, dict.keys(value), contents[1]))
                          else map(dict.__getitem__.__get__(value), contents[1]))
                current = tuple(map(id, values))
            except KeyError:
                return False
            if current != contents[3]:
                return False
        elif contents is not None:
            if len(value) != len(contents):
                return False
            if isinstance(value, Mapping):
                if any(key not in value or value[key] is not item for key, item in contents):
                    return False
            elif any(left is not right for left, right in zip(value, contents, strict=True)):
                return False
    return True


def intrinsic_scope_attachments(scope, *, syntax=True):
    """Retain namespace fields and nested containers without evaluation hooks."""
    from compiler.frontend.descriptor_associations import _syntax_attachments

    attached = tuple((part.name, getattr(scope, part.name)) for part in fields(scope))
    containers, kinds, seen, pending = [], [], set(), [value for _name, value in attached]
    current = scope.kinds
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        kinds.append((current, current.parent, current.values, current.intrinsic_kinds))
        pending.extend((current.values, current.intrinsic_kinds))
        current = current.parent
    while pending:
        value = pending.pop()
        if id(value) in seen or not isinstance(value, (dict, list, tuple, set, frozenset)):
            continue
        seen.add(id(value))
        if isinstance(value, dict):
            items = tuple(value.items())
            containers.append((value, "mapping", items))
            pending.extend(item for _name, item in items)
        elif isinstance(value, (set, frozenset)):
            containers.append((value, "set", frozenset(value)))
        else:
            items = tuple(value)
            containers.append((value, "sequence", items))
            pending.extend(items)
    return (scope, attached, tuple(containers), tuple(kinds),
            _syntax_attachments(scope.node) if syntax else ())


def same_intrinsic_scope_attachments(state):
    """Cheap final check: no lookup, signature, hashing or metadata callbacks."""
    from compiler.frontend.descriptor_associations import _same_syntax

    scope, attached, containers, kinds, syntax = state
    if (any(getattr(scope, name) is not value for name, value in attached)
            or any(current.parent is not parent or current.values is not values
                   or current.intrinsic_kinds is not intrinsic for current, parent, values, intrinsic in kinds)
            or not _same_syntax(syntax)):
        return False
    for value, kind, saved in containers:
        if len(value) != len(saved):
            return False
        if kind == "mapping":
            if any(name not in value or value[name] is not item for name, item in saved):
                return False
        elif kind == "set":
            if value != saved:
                return False
        elif any(left is not right for left, right in zip(value, saved, strict=True)):
            return False
    return True


def _original_matrix_intrinsic(analysis, routine, expression, name):
    """Authenticate a native expression before inspecting its declaration types."""
    if (analysis._require_original(routine.qualified) is not routine
            or type(expression).__name__ != "Intrinsic_Function_Reference"
            or str(expression.items[0]).lower() != name):
        raise CompilationError("descriptor intrinsic requires exact original " + name.upper() + " syntax")
    position = next((index for index, node in enumerate(_walk(routine.execution))
                     if node is expression), None)
    if position is None:
        raise CompilationError("descriptor intrinsic requires its exact original expression")
    scope = analysis.source_scope_for(expression, routine.scope)
    _validate_resolution(analysis, scope, _walk(expression))
    current = scope
    while current is not None:
        if name in current.externals or name in current.procedure_arguments:
            raise CompilationError("descriptor " + name.upper() + " original intrinsic name is shadowed")
        current = current.parent
    if (analysis._binding(scope, name) is not None or analysis._candidates(scope, name)
            or analysis._unknown_exports(scope)):
        raise CompilationError("descriptor " + name.upper() + " original intrinsic resolution is unproved")
    return scope, position


def _matrix_arguments(expression, formals):
    arguments, keyword_seen = {}, False
    for index, argument in enumerate(getattr(expression.items[1], "items", ()) or ()):
        keyword, actual = None, argument
        if type(argument).__name__ == "Actual_Arg_Spec":
            keyword, actual = str(argument.items[0]).lower(), argument.items[1]
            keyword_seen = True
        elif keyword_seen:
            raise CompilationError("descriptor matrix intrinsic positional actual follows a keyword")
        formal = keyword if keyword is not None else formals[index] if index < len(formals) else None
        if formal not in formals or formal in arguments:
            raise CompilationError("descriptor matrix intrinsic original argument association is invalid")
        arguments[formal] = (index, keyword, actual)
    if set(arguments) != set(formals):
        raise CompilationError("descriptor matrix intrinsic requires all original arguments")
    return arguments


def _matrix_signature(analysis, routine, node, depth):
    # This local classifier adds no numerical-lowering or general expression
    # typing capability. Array expressions outside this finite native subset
    # remain unavailable; the complete consumer separately scans every actual.
    if depth > analysis.depth_limit:
        raise CompilationError("descriptor matrix intrinsic expression depth exhausted")
    if type(node).__name__ == "Parenthesis":
        return _matrix_signature(analysis, routine, node.items[1], depth + 1)
    if (type(node).__name__ == "Intrinsic_Function_Reference"
            and str(node.items[0]).lower() == "transpose"):
        _actuals, record = matrix_descriptor_effects(analysis, routine, node, _depth=depth + 1)
        result = record["result"]
        return result["type"], result["kind"], result["rank"]
    if type(node).__name__ != "Name":
        return None
    scope = analysis.source_scope_for(node, routine.scope)
    return analysis._signature(scope, node)


def matrix_descriptor_effects(analysis, routine, expression, *, _depth=0):
    """Classify unchanged native REAL MATMUL/TRANSPOSE descriptor reads only.

    Supported MATMUL operands have the same REAL(4/8) kind and ranks (2,2),
    (2,1), or (1,2). TRANSPOSE retains rank two and kind. Shape conformance,
    temporary creation, evaluation and floating point behavior remain with the
    original compiler and call. No source movement or GPU execution is granted.
    """
    name = str(expression.items[0]).lower() if getattr(expression, "items", ()) else ""
    if name not in {"matmul", "transpose"}:
        raise CompilationError("descriptor matrix intrinsic requires MATMUL or TRANSPOSE")
    _scope, position = _original_matrix_intrinsic(analysis, routine, expression, name)
    arguments = _matrix_arguments(expression, ("matrix",) if name == "transpose" else ("matrix_a", "matrix_b"))
    types = {formal: _matrix_signature(analysis, routine, item[2], _depth)
             for formal, item in arguments.items()}
    if any(signature is None or signature[0] != "real" or signature[1] not in {4, 8}
           or type(signature[2]) is not int or signature[2] not in {1, 2}
           for signature in types.values()):
        raise CompilationError("descriptor matrix intrinsic requires proved REAL(4/8) rank-one/two arrays")
    if name == "transpose":
        result = types["matrix"]
        if result[2] != 2:
            raise CompilationError("descriptor TRANSPOSE requires a rank-two matrix")
    else:
        left, right = types["matrix_a"], types["matrix_b"]
        if left[1] != right[1] or (left[2], right[2]) == (1, 1):
            raise CompilationError("descriptor MATMUL requires matching REAL kinds and at least one rank-two operand")
        result = "real", left[1], left[2] + right[2] - 2
    actuals = tuple(item[2] for item in sorted(arguments.values()))
    return actuals, {
        "schema_version": 1, "procedure": routine.qualified, "intrinsic": name,
        "source_expression": str(expression), "source_position": position,
        "arguments": [{"position": index, "formal": formal, "keyword": keyword,
                       "source_actual": str(actual), "type": types[formal][0],
                       "kind": types[formal][1], "rank": types[formal][2], "effect": "read"}
                      for formal, (index, keyword, actual) in sorted(arguments.items(), key=lambda item: item[1][0])],
        "result": {"type": result[0], "kind": result[1], "rank": result[2]},
        "requires_original_execution": True, "all_actual_effects_require_original_source_traversal": True,
        "argument_evaluation": "unchanged original expression order, position and control",
        "shape_conformance": "unchanged original native expression; no descriptor evaluation or bound inference",
        "descriptor_evidence_only": True, "storage_writes": False, "allocation_or_escape": False,
        "execution_authorized": False, "capture_authorized": False,
        "entry_definition_authority": False, "gpu_legality_established": False,
        "numerical_lowering_authorized": False, "motion_authorized": False,
    }


def norm2_descriptor_effects(analysis, routine, expression):
    """Classify exact original NORM2 reads; caller must traverse all actuals.

    This is not a retained token or a source publication boundary. The complete
    descriptor consumer authenticates and publishes its entire original scan.
    """
    if (analysis._require_original(routine.qualified) is not routine
            or type(expression).__name__ != "Intrinsic_Function_Reference"
            or str(expression.items[0]).lower() != "norm2"):
        raise CompilationError("descriptor intrinsic requires exact original NORM2 syntax")
    position = next((i for i, node in enumerate(_walk(routine.execution)) if node is expression), None)
    if position is None:
        raise CompilationError("descriptor intrinsic requires its exact original expression")
    scope = analysis.source_scope_for(expression, routine.scope)
    _validate_resolution(analysis, scope, _walk(expression))
    current = scope
    while current is not None:
        if "norm2" in current.externals or "norm2" in current.procedure_arguments:
            raise CompilationError("descriptor NORM2 original intrinsic name is shadowed")
        current = current.parent
    if (analysis._binding(scope, "norm2") is not None or analysis._candidates(scope, "norm2")
            or analysis._unknown_exports(scope)):
        raise CompilationError("descriptor NORM2 original intrinsic resolution is unproved")
    original = tuple(getattr(expression.items[1], "items", ()) or ())
    formals, arguments, keyword_seen = ("array", "dim"), {}, False
    for index, argument in enumerate(original):
        keyword = None
        actual = argument
        if type(argument).__name__ == "Actual_Arg_Spec":
            keyword, actual = str(argument.items[0]).lower(), argument.items[1]
            keyword_seen = True
        elif keyword_seen:
            raise CompilationError("descriptor NORM2 positional actual follows a keyword")
        formal = keyword if keyword is not None else formals[index] if index < len(formals) else None
        if formal not in formals or formal in arguments:
            raise CompilationError("descriptor NORM2 original argument association is invalid")
        arguments[formal] = (index, keyword, actual)
    if "array" not in arguments or len(arguments) not in {1, 2}:
        raise CompilationError("descriptor NORM2 requires ARRAY and optional scalar DIM")
    array = arguments["array"][2]
    signature = analysis._signature(scope, array)
    if (signature is None or signature[0] != "real" or signature[1] not in {4, 8}
            or type(signature[2]) is not int or not 1 <= signature[2] <= 4):
        raise CompilationError("descriptor NORM2 requires a proved REAL(4/8) array signature")
    types = {"array": signature}
    if "dim" in arguments:
        dim = arguments["dim"][2]
        dimension = analysis._signature(scope, dim)
        if dimension not in {("integer", 4, 0), ("integer", 8, 0)}:
            raise CompilationError("descriptor NORM2 DIM requires a proved integer scalar signature")
        types["dim"] = dimension
        # Constant source expressions can be rejected without reading payload.
        # A variable DIM remains evaluated only by the original Fortran call.
        try:
            constant = scope.kinds.integer(dim, SourceLocation(str(scope.path)))
        except CompilationError:
            constant = None
        if constant is not None and not 1 <= constant <= signature[2]:
            raise CompilationError("descriptor NORM2 constant DIM is outside original ARRAY rank")
    actuals = tuple(item[2] for item in sorted(arguments.values()))
    record = {"schema_version": 1, "procedure": routine.qualified, "intrinsic": "norm2",
              "source_expression": str(expression), "source_position": position,
              "arguments": [{"position": index, "formal": formal, "keyword": keyword,
                             "source_actual": str(actual), "type": types[formal][0],
                             "kind": types[formal][1], "rank": types[formal][2], "effect": "read"}
                            for formal, (index, keyword, actual) in sorted(arguments.items(), key=lambda x: x[1][0])],
              "result": {"type": "real", "kind": signature[1],
                         "rank": signature[2] - 1 if "dim" in arguments else 0},
              "requires_original_execution": True, "all_actual_effects_require_original_source_traversal": True,
              "argument_evaluation": "unchanged original expression order, position and control",
              "descriptor_evidence_only": True, "storage_writes": False, "allocation_or_escape": False,
              "execution_authorized": False, "capture_authorized": False,
              "entry_definition_authority": False, "gpu_legality_established": False,
              "numerical_lowering_authorized": False, "motion_authorized": False}
    return actuals, record
