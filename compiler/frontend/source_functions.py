"""Read effects of unchanged, source-backed scalar function expressions.

This deliberately grants no numerical lowering or callback authority. Functions
keep their original Fortran implementation, argument evaluation and IEEE effects.
Only a complete, bounded, read-only source closure may supply these data effects.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.call_bindings import resolve_source_function
from compiler.frontend.source_objects import _kind_state
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.frontend.summary_cache import SummaryCache
from compiler.ir import CompilationError

SOURCE_FUNCTION_VERSION = 1


def _walk(node):
    stack, visits = [iter((node,))], 0
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        visits += 1
        if visits > 65536:
            raise CompilationError("native source function original-source scan budget exhausted")
        if isinstance(item, (tuple, list)):
            stack.append(iter(item))
        elif isinstance(item, Base):
            # Name/Type_Name leaves have .string rather than .items. Include
            # them so resolution authentication cannot silently skip names.
            yield item
            stack.append(iter(getattr(item, "content", getattr(item, "items", ())) or ()))


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _resolution(scope):
    return _hash({"imports": scope.imports, "wildcards": scope.wildcards,
                  "exclusions": {key: sorted(value) for key, value in scope.wildcard_exclusions.items()},
                  "ambiguous": sorted(scope.ambiguous_imports), "access": scope.access,
                  "default_public": scope.default_public, "generics": scope.generics,
                  "procedures": scope.procedures, "externals": sorted(scope.externals),
                  "procedure_arguments": sorted(scope.procedure_arguments),
                  "intrinsics": scope.intrinsic_procedures, "intrinsic_wildcards": sorted(scope.intrinsic_wildcards),
                  "intrinsic_exclusions": {key: sorted(value) for key, value in scope.intrinsic_exclusions.items()},
                  "bindings": {key: value.public() for key, value in scope.bindings.items()}}), _kind_state(scope)


def register_source_function_authority(analysis):
    scopes = {id(scope): scope for scope in analysis.modules.values()}
    scopes.update((id(routine.scope), routine.scope) for routine in analysis.numerical_helpers.values())
    analysis._source_function_scopes = {key: (scope, scope.node, scope.parent, _resolution(scope))
                                        for key, scope in scopes.items()}
    analysis._source_function_used_scopes = set()


def resolution_changes(analysis):
    changes = []
    for key in sorted(analysis._source_function_used_scopes):
        scope, node, parent, state = analysis._source_function_scopes[key]
        current = _resolution(scope)
        if scope.node is not node or scope.parent is not parent or current != state:
            changes.append({"module": scope.module, "scope": getattr(scope, "qualified", scope.module),
                            "resolution": current})
    return changes


def _validate_resolution(analysis, scope, nodes):
    visited, checked = set(), set()

    def resolve(current, name, depth=0):
        if current is None or (id(current), name) in visited:
            return
        visited.add((id(current), name))
        if len(visited) > analysis.operation_limit or depth >= analysis.depth_limit:
            raise CompilationError("native source function resolution budget exhausted")
        if id(current) not in checked:
            original = analysis._source_function_scopes.get(id(current))
            if (original is None or original[0] is not current or original[1] is not current.node
                    or original[2] is not current.parent or original[3] != _resolution(current)):
                raise CompilationError("native source function original resolution authority changed")
            checked.add(id(current))
            analysis._source_function_used_scopes.add(id(current))
        if (name in current.bindings or name in current.procedures or name in current.procedure_arguments
                or name in current.intrinsic_procedures or name in current.generics):
            return
        sources = analysis.use_sources(current, name)
        for module, remote in sources or ():
            imported = analysis.modules.get(module)
            if imported is not None:
                resolve(imported, remote, depth + 1)
        resolve(current.parent, name, depth + 1)

    for node in nodes:
        if type(node).__name__ in {"Name", "Type_Name", "Intrinsic_Name"}:
            resolve(analysis.source_scope_for(node, scope), str(node).lower())


def _origin(analysis):
    seen = set()
    while getattr(analysis, "_descriptor_source", None) is not None:
        if id(analysis) in seen:
            raise CompilationError("native source function projection is cyclic")
        seen.add(id(analysis))
        analysis = analysis._descriptor_source
    return analysis


def _original(analysis, procedure):
    if procedure in analysis.routines:
        return analysis._require_original(procedure)
    analysis.inputs.verify()
    routine = analysis.numerical_helpers.get(procedure)
    role = analysis._numerical_roles.get(procedure)
    if (routine is None or routine.source_kind not in {"function", "internal_function"}
            or role is None or routine.scope.node is not role[0]
            or analysis._routine_signature(routine) != role[1] or str(routine.scope.node) != role[2]):
        raise CompilationError("native source function requires original lexical source authority")
    # Authenticate original module and span authority too. Functions remain
    # absent from public routine/structured-entry registration.
    if any(analysis.modules.get(name) is None or analysis.modules[name].node is not node or str(node) != text
           for name, (node, text) in analysis._source_module_roles.items()):
        raise CompilationError("native source function module authority changed")
    if any(getattr(node, "item", None) is not item or item.span != span
           or getattr(item, "fort_original_span", None) != original_span
           for node, item, span, original_span in analysis._source_provenance.values()):
        raise CompilationError("native source function span authority changed")
    return routine


def _source_identity(analysis, routine):
    return _hash({"version": SOURCE_FUNCTION_VERSION, "sources": analysis.inputs.identity(),
                  "procedure": routine.qualified, "signature": analysis._routine_signature(routine),
                  "source": str(routine.scope.node)})


def _contract(routine):
    statement = next(node for node in routine.scope.node.content if type(node).__name__ == "Function_Stmt")
    prefix = {str(item).lower() for item in getattr(statement.items[0], "items", ()) or ()}
    if "impure" in prefix or not prefix.intersection({"pure", "elemental"}):
        raise CompilationError("native source function requires PURE or implicitly pure ELEMENTAL source")
    result = str(statement.items[3].items[0] if statement.items[3] is not None else statement.items[1]).lower()
    binding = routine.scope.bindings.get(result)
    if (binding is None or binding.rank or binding.dtype not in {"real", "integer", "logical"}
            or binding.kind not in {4, 8}):
        raise CompilationError("native source function requires a fixed numeric or logical scalar result")
    forbidden = {"pointer", "allocatable", "optional", "value", "save", "volatile", "asynchronous"}
    if any(item.rank or item.attributes & forbidden
           or item.dtype not in {"real", "integer", "logical"} or item.kind not in {4, 8}
           for item in routine.scope.bindings.values()):
        raise CompilationError("native source function requires fixed private scalars and stable required arguments")
    return binding.public(), "pure" if "pure" in prefix else "implicitly pure elemental"


def _projection(analysis, routine):
    # Use the established private native-analysis isolation, including proof
    # registries, then add exactly this original helper to that private view.
    # No synthetic CALL, original entry role or persistent cache is created.
    from compiler.scopes.segments import _native_analysis

    projected = _native_analysis(analysis)
    projected.routines = dict(analysis.routines)
    projected.routines[routine.qualified] = routine
    projected.summaries, projected._closures, projected._native_sections = {}, {}, {}
    projected._summary_cache = SummaryCache(max_entries=0)
    projected._descriptor_source = _origin(analysis)
    return projected


@dataclass(frozen=True)
class NativeSourceFunctionProof:
    procedure: str
    callee: str
    identity: str
    source_identity: str
    callee_source_identity: str
    resolved: object = field(repr=False, compare=False)
    _expression: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)
    _effects: object = field(repr=False, compare=False)

    @property
    def effects(self):
        return _thaw(self._effects)

    def public(self):
        return {"schema_version": SOURCE_FUNCTION_VERSION, "proof_role": "native_readonly_source_function",
                "proof_identity": self.identity, "procedure": self.procedure, "callee": self.callee,
                "source_identity": self.source_identity, "callee_source_identity": self.callee_source_identity,
                **_thaw(self._record)}

    def validate(self, analysis, procedure, expression):
        origin = _origin(analysis)
        caller, callee = _original(origin, procedure), _original(origin, self.callee)
        _validate_resolution(origin, caller.scope, _walk(expression))
        _validate_resolution(origin, callee.scope, _walk(callee.execution))
        if (procedure != self.procedure or expression is not self._expression
                or origin._native_functions.get(self.identity) is not self
                or _source_identity(origin, caller) != self.source_identity
                or _source_identity(origin, callee) != self.callee_source_identity
                or not any(node is expression for node in _walk(caller.execution))):
            raise CompilationError("native source function lacks registered original expression authority")
        resolved = resolve_source_function(origin, origin.source_scope_for(expression, caller.scope), expression)
        if _hash(resolved.public()) != _hash(_thaw(self._record)["source_call"]):
            raise CompilationError("native source function original argument mapping changed")
        return self


def prove_native_source_function(analysis, procedure, expression, active, closure):
    origin = _origin(analysis)
    caller = _original(origin, procedure)
    position = next((index for index, node in enumerate(_walk(caller.execution)) if node is expression), None)
    if position is None:
        raise CompilationError("native source function requires an exact original expression")
    _validate_resolution(origin, caller.scope, _walk(expression))
    resolved = resolve_source_function(origin, origin.source_scope_for(expression, caller.scope), expression)
    callee = _original(origin, resolved.procedure)
    result, purity = _contract(callee)
    _validate_resolution(origin, callee.scope, _walk(callee.execution))
    syntax = [node for part in callee.scope.node.content
              if type(part).__name__ in {"Specification_Part", "Execution_Part"}
              for node in _walk(part)]
    if len(syntax) > origin.operation_limit:
        raise CompilationError("native source function syntax budget exhausted")
    projected = _projection(analysis, callee)
    child = projected.summarize(callee.qualified, (*active, procedure), _closure=closure)
    if not child["complete"]:
        raise CompilationError("native source function effects incomplete: " + callee.qualified + ": "
                               + "; ".join(child["reasons"]))
    completion = child.get("native_completion", {})
    if (not completion.get("available") or completion.get("has_openmp_in_closure") is not False
            or completion.get("has_opaque_calls_in_closure") is not False):
        raise CompilationError("native source function completion is unproved: " + callee.qualified)
    if (child.get("definition_changes") or child.get("persistent_state")
            or any(effect["kind"] not in {"read", "descriptor_read"} for effect in child["ordered_effects"])):
        raise CompilationError("native source function transitive effects are not read-only: " + callee.qualified)
    record = {"source_expression": str(expression), "source_position": position,
              "source_call": resolved.public(), "summary_identity": child["summary_identity"],
              "result": result, "purity": purity, "complete_transitive_effects": True,
              "native_only": True, "requires_original_execution": True,
              "argument_evaluation": "original expression position, keyword order and guards",
              "gpu_legality_established": False, "numerical_lowering_authorized": False,
              "native_callback_authorized": False, "exception_observers_authorized": False}
    source, target_source = _source_identity(origin, caller), _source_identity(origin, callee)
    identity = _hash({"source": source, "callee_source": target_source, "record": record})
    existing = origin._native_functions.get(identity)
    if existing is not None:
        return existing.validate(origin, procedure, expression), child
    if len(origin._native_functions) >= origin.operation_limit:
        raise CompilationError("native source function proof budget exhausted")
    proof = NativeSourceFunctionProof(procedure, callee.qualified, identity, source, target_source,
                                      resolved, expression, _freeze(record), _freeze(deepcopy(child["ordered_effects"])))
    origin._native_functions[identity] = proof
    return proof, child
