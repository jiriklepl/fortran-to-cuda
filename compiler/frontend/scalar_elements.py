"""Bounded source property for unchanged native scalar-element associations.

This proves neither effects nor completion. The backing storage remains native,
and the original CALL and scalar dummy retain their Fortran association.
"""

import re

from fparser.two.utils import Base

from compiler.frontend.call_bindings import _arrange
from compiler.frontend.source_functions import _original, _validate_resolution, _validate_scope, _walk
from compiler.ir import CompilationError

SOURCE_SCALAR_ELEMENT_VERSION = 1
_SCAN_LIMIT = 65536
_UNSAFE = {"optional", "pointer", "target", "allocatable", "value", "volatile", "asynchronous"}


class ScalarElementUnsupported(CompilationError):
    """An original association must run after its owner closes."""


def check_scalar_element_closure(analysis, procedure, mappings, *, consume=None):
    """Check source references and scalar forwarding, without actual coordinates."""
    from compiler.frontend.source_effects import _part

    backing = {mapping.scalar_element.binding.root: mapping.scalar_element.binding for mapping in mappings
               if mapping.scalar_element is not None}
    aliases = tuple(mapping.formal_binding for mapping in mappings if mapping.scalar_element is not None)
    if any(mapping.scalar_element is None and mapping.binding is not None and mapping.binding.root in backing
           for mapping in mappings):
        raise ScalarElementUnsupported("scalar element backing table aliases another original actual")
    visits, operations, scopes = 0, 0, set()

    def tick(_role=None):
        nonlocal visits
        visits += 1
        if consume is not None:
            consume(1)
        if visits > _SCAN_LIMIT:
            raise CompilationError("scalar element association source scan budget exhausted")

    def scope_authority(scope):
        depth = 0
        while scope is not None and id(scope) not in scopes:
            tick()
            if depth > analysis.depth_limit:
                raise CompilationError("scalar element association scope depth budget exhausted")
            for field in ("bindings", "imports", "wildcards", "wildcard_exclusions", "ambiguous_imports",
                          "access", "generics", "procedures", "externals", "procedure_arguments",
                          "intrinsic_procedures", "intrinsic_wildcards", "intrinsic_exclusions"):
                for _ in getattr(scope, field):
                    tick()
            _validate_scope(analysis, scope)
            scopes.add(id(scope))
            scope, depth = scope.parent, depth + 1

    for binding in backing.values():
        scope_authority(binding.declaring_scope)
    for mapping in mappings:
        if mapping.scalar_element is not None:
            for _ in range(mapping.scalar_element.declaration_work):
                tick()
            _validate_resolution(analysis, mapping.binding.declaring_scope,
                                 _walk(mapping.scalar_element.declaration), tick=tick)
    pending, checked, records = [(procedure, aliases, ())], set(), []
    controls = {"If_Construct", "If_Stmt", "Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct",
                "Select_Case_Construct", "Case_Construct", "Associate_Construct", "Where_Construct",
                "Forall_Construct"}
    while pending:
        name, associated, ancestors = pending.pop(0)
        if name in ancestors or len(ancestors) > analysis.depth_limit:
            raise CompilationError("scalar element association call depth or recursion budget exhausted")
        key = name, tuple(sorted(id(binding) for binding in associated))
        if key in checked:
            continue
        checked.add(key)
        if len(checked) > analysis.procedure_limit:
            raise CompilationError("scalar element association procedure budget exhausted")
        routine = _original(analysis, name)
        scope_authority(routine.scope)
        alias_ids = {id(binding) for binding in associated}
        calls, functions = [], []
        stack = [(routine.execution, 0), (_part(routine.scope.node, "Specification_Part"), 0)]
        while stack:
            node, depth = stack.pop()
            tick()
            if isinstance(node, (tuple, list)):
                stack.extend((item, depth) for item in reversed(node))
                continue
            if not isinstance(node, Base):
                continue
            kind = type(node).__name__
            depth += kind in controls
            if depth > analysis.depth_limit:
                raise CompilationError("scalar element association control depth budget exhausted")
            scope = analysis.source_scope_for(node, routine.scope)
            if kind.endswith("_Stmt"):
                operations += 1
                if operations > analysis.operation_limit:
                    raise CompilationError("scalar element association operation budget exhausted")
            if kind == "Name":
                _validate_resolution(analysis, scope, (), tick=tick, names=(str(node).lower(),))
                binding = analysis._binding(scope, node)
                if binding is not None:
                    scope_authority(binding.declaring_scope)
                    if binding.root in backing:
                        if binding is not backing[binding.root]:
                            raise CompilationError("scalar element original backing Binding changed")
                        raise ScalarElementUnsupported("scalar element backing storage is referenced inside child closure")
            elif kind == "Comment" and str(node).lstrip().lower().startswith("!$omp"):
                # Clause operands are original references too, although the
                # parser stores them inside directive text instead of Names.
                for spelling in re.findall(r"[A-Za-z][A-Za-z0-9_]*", str(node)):
                    tick()
                    _validate_resolution(analysis, scope, (), tick=tick, names=(spelling.lower(),))
                    binding = analysis._binding(scope, spelling.lower())
                    if binding is not None:
                        scope_authority(binding.declaring_scope)
                        if binding.root in backing:
                            if binding is not backing[binding.root]:
                                raise CompilationError("scalar element original backing Binding changed")
                            raise ScalarElementUnsupported(
                                "scalar element backing storage is referenced inside child closure")
            elif kind in {"Assignment_Stmt", "Pointer_Assignment_Stmt"}:
                target = node.items[0]
                binding = analysis._binding(scope, target)
                if binding is not None and id(binding) in alias_ids:
                    raise ScalarElementUnsupported("scalar element association cannot be written inside child closure")
            elif kind in {"Read_Stmt", "Write_Stmt", "Print_Stmt", "Open_Stmt", "Close_Stmt", "Inquire_Stmt",
                          "Wait_Stmt", "Backspace_Stmt", "Rewind_Stmt", "Endfile_Stmt", "Flush_Stmt"}:
                raise ScalarElementUnsupported("scalar element association closure contains unsupported I/O")
            elif kind in {"Associate_Construct", "Forall_Construct"}:
                raise ScalarElementUnsupported("scalar element association closure requires direct scalar references")
            elif kind == "Call_Stmt":
                calls.append((node, scope, False))
            elif (kind in {"Function_Reference", "Structure_Constructor"}
                  or kind == "Part_Ref" and analysis._binding(scope, node.items[0]) is None):
                functions.append((node, scope, True))
            elif kind == "Intrinsic_Function_Reference":
                target = str(node.items[0]).lower()
                if analysis._binding(scope, target) or analysis._candidates(scope, target) or analysis._unknown_exports(scope):
                    raise ScalarElementUnsupported("scalar element closure requires an unshadowed intrinsic")
            children = getattr(node, "content", getattr(node, "items", ())) or ()
            stack.extend((item, depth) for item in reversed(children))

        for node, scope, function in (*calls, *functions):
            scope_authority(scope)
            target = str(node.items[0]).lower()
            candidates = analysis._candidates(scope, target)
            for _ in candidates:
                tick()
            pool = analysis.numerical_helpers if function else analysis.routines
            if not candidates or any(candidate not in pool or function and pool[candidate].source_kind not in {
                    "function", "internal_function"} for candidate in candidates):
                if not function:
                    from compiler.frontend.native_environment import intrinsic_export

                    if intrinsic_export(analysis, scope, target) in {
                            "$intrinsic::ieee_exceptions::ieee_get_halting_mode",
                            "$intrinsic::ieee_exceptions::ieee_set_halting_mode"}:
                        analysis.native_environment(name, node).validate(analysis, name, node)
                        continue
                raise ScalarElementUnsupported("scalar element association requires a source-backed child closure")
            original = tuple(getattr(node.items[1], "items", ()) or ())
            for _ in original:
                tick()
            # This witness grants no invocation/effect legality. Inspect every
            # available source candidate instead of resolving unrelated array
            # actuals through the native coordinate mapper or selecting an
            # overload from an unsupported-coordinate diagnostic.
            for candidate in candidates:
                tick()
                callee = _original(analysis, candidate)
                scope_authority(callee.scope)
                for _ in callee.arguments:
                    tick()
                if callee.source_kind == "external":
                    interface = analysis.external_interface(callee)
                    if not interface["available"] or any(type(item).__name__ == "Actual_Arg_Spec" for item in original):
                        raise ScalarElementUnsupported(
                            "scalar element closure requires a supported original external interface")
                try:
                    actuals, _arguments = _arrange(callee, original, function=function)
                except CompilationError as error:
                    # Authority and budgets were checked above. _arrange only
                    # classifies the original positional/keyword/omission
                    # structure; incompatible candidates cannot grant a role.
                    raise ScalarElementUnsupported(
                        "scalar element closure requires supported source-call argument structure") from error
                forwarded = []
                for formal_name, actual in zip(callee.arguments, actuals, strict=True):
                    tick()
                    formal = callee.scope.bindings[formal_name]
                    scope_authority(formal.declaring_scope)
                    if type(actual).__name__ != "Name":
                        continue
                    binding = analysis._binding(scope, actual)
                    if binding is not None:
                        scope_authority(binding.declaring_scope)
                    if binding is not None and id(binding) in alias_ids:
                        if (formal.rank or formal.intent != "in" or formal.signature() != binding.signature()
                                or formal.attributes & _UNSAFE):
                            raise ScalarElementUnsupported("scalar element forwarding requires unchanged scalar INTENT(IN)")
                        forwarded.append(formal)
                pending.append((candidate, tuple(forwarded), (*ancestors, name)))
        records.append({"procedure": name, "associated_formals": sorted(binding.root for binding in associated),
                        "source": str(routine.scope.path), "source_sha256": analysis.sources[str(routine.scope.path)]})
    return {"version": SOURCE_SCALAR_ELEMENT_VERSION, "procedures": records, "source_work": visits,
            "operations": operations, "backing_resources": sorted(backing),
            "no_other_backing_reference": True, "only_scalar_in_forwarding": True,
            "effect_authority": False, "definition_authority": False, "completion_authority": False}
