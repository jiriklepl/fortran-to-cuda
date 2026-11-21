"""Read effects of unchanged, source-backed scalar-result functions.

This deliberately grants no numerical lowering or callback authority. Functions
keep their original Fortran implementation, argument evaluation and IEEE effects.
Only a complete, bounded, read-only source closure may supply these data effects.
"""

from __future__ import annotations

import json
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.source_objects import _kind_state
from compiler.ir import CompilationError


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
    scopes.update((id(record.scope), record.scope) for record in analysis._associate_scopes.values()
                  if record.available)
    scopes.update((id(scope), scope) for scope in analysis._block_scopes.values())
    analysis._source_function_scopes = {key: (scope, scope.node, scope.parent, _resolution(scope),
                                            {name: (binding, binding.declaring_scope)
                                             for name, binding in scope.bindings.items()})
                                        for key, scope in scopes.items()}
    analysis._source_function_used_scopes = set()




def _validate_scope(analysis, scope, *, track=True):
    original = analysis._source_function_scopes.get(id(scope))
    if (original is None or original[0] is not scope or original[1] is not scope.node
            or original[2] is not scope.parent or original[3] != _resolution(scope)
            or any(scope.bindings.get(name) is not binding or binding.declaring_scope is not owner
                   for name, (binding, owner) in original[4].items())):
        raise CompilationError("native source function original resolution authority changed")
    if track:
        analysis._source_function_used_scopes.add(id(scope))


def _scope_roles(analysis):
    """The scope that each module, procedure and ASSOCIATE construct is registered with."""
    roles = {("module", name): scope for name, scope in analysis.modules.items()}
    roles.update((("routine", name), routine.scope) for name, routine in analysis.routines.items())
    roles.update((("numerical_helper", name), routine.scope)
                 for name, routine in analysis.numerical_helpers.items())
    roles.update((("associate", identity), record.scope)
                 for identity, record in analysis._associate_scopes.items() if record.available)
    return roles


def _validate_scope_roles(analysis, roles, consulted):
    # Authenticating a retained scope alone does not bind a registry that was
    # replaced by an equal-looking scope while this demand was running.
    current = _scope_roles(analysis)
    if any(current.get(role) is not scope for role, scope in roles.items() if id(scope) in consulted):
        raise CompilationError("native source function original resolution authority changed")




def _validate_resolution(analysis, scope, nodes, *, tick=None, names=(), _scope_validator=None):
    visited, checked = set(), set()

    def resolve(current, name, depth=0):
        if current is None or (id(current), name) in visited:
            return
        visited.add((id(current), name))
        if tick is not None:
            tick("lookup")
        if len(visited) > analysis.operation_limit or depth >= analysis.depth_limit:
            raise CompilationError("native source function resolution budget exhausted")
        if id(current) not in checked:
            if _scope_validator is None:
                _validate_scope(analysis, current)
            else:
                _scope_validator(current)
            checked.add(id(current))
        if (name in current.bindings or name in current.procedures or name in current.procedure_arguments
                or name in current.intrinsic_procedures or name in current.generics):
            return
        sources = analysis.use_sources(current, name)
        for module, remote in sources or ():
            imported = analysis.modules.get(module)
            if imported is not None:
                resolve(imported, remote, depth + 1)
        resolve(current.parent, name, depth + 1)

    # Explicit names are semantic lookup obligations such as registering a
    # visible module allocation. They never stand in for executable source or
    # grant expression authority; the same original namespace checks apply.
    for name in names:
        resolve(scope, name)
    for node in nodes:
        if tick is not None:
            tick("syntax")
        if type(node).__name__ in {"Name", "Type_Name", "Intrinsic_Name"}:
            resolve(analysis.source_scope_for(node, scope), str(node).lower())




def _original(analysis, procedure):
    if procedure in analysis.routines:
        return analysis._require_original(procedure)
    analysis.verify_source()
    syntax = analysis._source_syntax.checkpoint()
    routine = analysis.numerical_helpers.get(procedure)
    role = analysis._numerical_roles.get(procedure)
    if (routine is None or routine.source_kind not in {"function", "internal_function"}
            or role is None or routine.scope.node is not role[0]
            or analysis._routine_signature(routine) != role[1] or not syntax.unchanged(routine.scope.node)):
        raise CompilationError("native source function requires original lexical source authority")
    # Authenticate original module and span authority too. Functions remain
    # absent from public routine/structured-entry registration.
    if any(analysis.modules.get(name) is None or analysis.modules[name].node is not node or not syntax.unchanged(node)
           for name, (node, _text) in analysis._source_module_roles.items()):
        raise CompilationError("native source function module authority changed")
    if any(getattr(node, "item", None) is not item or item.span != span
           or getattr(item, "fort_original_span", None) != original_span
           for node, item, span, original_span in analysis._source_provenance.values()):
        raise CompilationError("native source function span authority changed")
    return routine


















