"""Source-authenticated predicates executed by the original Fortran program.

The finite contract below classifies native input reads. It does not lower an
intrinsic, evaluate an actual, authorize a device worker or model exception
observers. The surrounding source traversal still proves every actual's effects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.frontend.native_environment import _resolve_intrinsic_export
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError

NATIVE_PREDICATE_VERSION = 1
_IS_NAN = "$intrinsic::ieee_arithmetic::ieee_is_nan"
_EXPORTS = {"ieee_arithmetic": {"ieee_is_nan": _IS_NAN}}
_SCAN_LIMIT = 65536


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _walk(node):
    stack, visits = [iter((node,))], 0
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        visits += 1
        if visits > _SCAN_LIMIT:
            raise CompilationError("native predicate original-source scan budget exhausted")
        if isinstance(item, (tuple, list)):
            stack.append(iter(item))
        elif hasattr(item, "items") or hasattr(item, "content"):
            yield item
            stack.append(iter(_children(item)))


def _hash(record):
    return sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def native_intrinsic_export(analysis, scope, name):
    """Resolve only this module's reviewed exports, including original renames."""
    return _resolve_intrinsic_export(analysis, scope, name, _EXPORTS)


def _origin(analysis, procedure):
    seen = set()
    while getattr(analysis, "_descriptor_source", None) is not None:
        if id(analysis) in seen:
            raise CompilationError("native predicate projection source is cyclic")
        seen.add(id(analysis))
        analysis = analysis._descriptor_source
    return analysis, analysis._require_original(procedure)


def _source_identity(analysis, routine):
    return _hash({"schema_version": NATIVE_PREDICATE_VERSION, "sources": analysis.inputs.identity(),
                  "procedure": routine.qualified, "signature": analysis._routine_signature(routine),
                  "source": str(routine.scope.node)})


def _associate(analysis, routine, expression):
    if (_kind(expression) not in {"Part_Ref", "Function_Reference", "Intrinsic_Function_Reference",
                                 "Structure_Constructor"}
            or len(expression.items) != 2 or _kind(expression.items[0]) not in {"Name", "Type_Name"}):
        raise CompilationError("native predicate requires an original direct function expression")
    scope = analysis.source_scope_for(expression, routine.scope)
    intrinsic = native_intrinsic_export(analysis, scope, expression.items[0])
    if intrinsic != _IS_NAN:
        raise CompilationError("native predicate intrinsic resolution is unproved")
    arguments = tuple(_children(expression.items[1]))
    if len(arguments) != 1:
        raise CompilationError("native predicate requires exactly one original X actual")
    actual, keyword = arguments[0], None
    if _kind(actual) in {"Actual_Arg_Spec", "Component_Spec"}:
        keyword, actual = str(actual.items[0]).lower(), actual.items[1]
        if keyword != "x":
            raise CompilationError("native predicate argument association is unproved")
    signature = analysis._signature(scope, actual)
    # Reuse the existing bounded signature proof: original REAL names include
    # whole arrays; scalar real arithmetic/literals do not evaluate their data.
    # Indexed/section or general array expressions require a separate signature
    # proof and deliberately remain unavailable in this initial contract.
    if (signature is None or len(signature) != 3 or signature[0] != "real" or signature[1] not in {4, 8}
            or type(signature[2]) is not int or not 0 <= signature[2] <= 4):
        raise CompilationError("native predicate requires a proved REAL(4/8) scalar or array X signature")
    return intrinsic, actual, keyword, signature


def _record(analysis, routine, expression):
    intrinsic, actual, keyword, signature = _associate(analysis, routine, expression)
    return intrinsic, actual, {
        "source_expression": str(expression),
        "arguments": [{"position": 1, "formal": "x", "keyword": keyword,
                       "source_actual": str(actual), "type": signature[0], "kind": signature[1],
                       "rank": signature[2], "effect": "read"}],
        "result": {"type": "logical", "kind": "original default logical", "rank": signature[2]},
        "native_only": True, "requires_original_execution": True,
        "argument_evaluation": "unchanged original expression position, ordinal and surrounding guards",
        "actual_effects_require_source_traversal": True,
        "storage_writes": False, "allocation_or_escape": False,
        "gpu_legality_established": False, "numerical_lowering_authorized": False,
        "exception_observers_authorized": False, "internal_cuts_authorized": False,
        "exception_behavior": "retained by unchanged original Fortran execution; no generated reevaluation",
    }


@dataclass(frozen=True)
class NativePredicateProof:
    procedure: str
    source_identity: str
    identity: str
    intrinsic: str
    actuals: tuple[object, ...] = field(repr=False, compare=False)
    _expression: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)

    def public(self):
        return {"schema_version": NATIVE_PREDICATE_VERSION, "proof_role": "native_readonly_predicate",
                "proof_identity": self.identity, "source_identity": self.source_identity,
                "procedure": self.procedure, "intrinsic": self.intrinsic, **_thaw(self._record)}

    def validate(self, analysis, procedure, original_expression):
        origin, routine = _origin(analysis, procedure)
        if (procedure != self.procedure or original_expression is not self._expression
                or getattr(origin, "_native_predicates", {}).get(self.identity) is not self
                or _source_identity(origin, routine) != self.source_identity
                or not any(node is original_expression for node in _walk(routine.execution))):
            raise CompilationError("native predicate lacks registered original source authority")
        intrinsic, actual, record = _record(origin, routine, original_expression)
        if intrinsic != self.intrinsic or actual is not self.actuals[0] or record != _thaw(self._record):
            raise CompilationError("native predicate original actual authority changed")
        return self


def prove_native_predicate(analysis, procedure, original_expression):
    """Prove native classification only; callers traverse ``actuals`` in place."""
    origin, routine = _origin(analysis, procedure)
    position = next((index for index, node in enumerate(_walk(routine.execution))
                     if node is original_expression), None)
    if position is None:
        raise CompilationError("native predicate requires an exact original expression")
    intrinsic, actual, record = _record(origin, routine, original_expression)
    source = _source_identity(origin, routine)
    identity = _hash({"source": source, "position": position, "intrinsic": intrinsic, "record": record})
    registry = getattr(origin, "_native_predicates", None)
    if registry is None:
        registry = origin._native_predicates = {}
    existing = registry.get(identity)
    if existing is not None:
        return existing.validate(origin, procedure, original_expression)
    if len(registry) >= origin.operation_limit:
        raise CompilationError("native predicate proof budget exhausted")
    proof = NativePredicateProof(procedure, source, identity, intrinsic, (actual,), original_expression, _freeze(record))
    registry[identity] = proof
    return proof
