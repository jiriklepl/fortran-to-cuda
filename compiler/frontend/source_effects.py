"""Bounded source effects for native calls, independent of numerical lowering.

This analysis never executes bounds or rewrites original calls. Its sections are
conservative whole resources; logical array subscripts are retained as evidence,
not mistaken for physical transfer coordinates. Unknown effects are boundaries.
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from fparser.two.utils import walk

from compiler.frontend.lowering import _KindScope
from compiler.frontend.source_inputs import SourceInputs
from compiler.ir import CompilationError, SourceLocation
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES


def _kind(node):
    return type(node).__name__


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _part(node, name):
    return next((child for child in _children(node) if _kind(child) == name), None)


@dataclass
class Binding:
    name: str
    root: str
    dtype: str
    kind: int | None
    rank: int
    intent: str | None = None
    attributes: frozenset[str] = frozenset()
    lower_bounds: tuple[str, ...] = ()

    def signature(self):
        return self.dtype, self.kind, self.rank

    def public(self):
        return {"name": self.name, "resource": self.root, "type": self.dtype,
                "kind": self.kind, "rank": self.rank, "intent": self.intent,
                "attributes": sorted(self.attributes), "lower_bounds": self.lower_bounds}


@dataclass
class Scope:
    module: str
    path: Path
    node: object
    parent: Scope | None = None
    kinds: _KindScope | None = None
    bindings: dict[str, Binding] = field(default_factory=dict)
    imports: dict[str, tuple[str, str]] = field(default_factory=dict)
    wildcards: list[str] = field(default_factory=list)
    generics: dict[str, list[tuple[str, str]]] = field(default_factory=dict)
    unresolved_imports: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    default_public: bool = True
    access: dict[str, bool] = field(default_factory=dict)


@dataclass
class Routine:
    scope: Scope
    arguments: tuple[str, ...]
    execution: object
    qualified: str
    issues: list[str]


@dataclass
class _Closure:
    summaries: dict = field(default_factory=dict)
    operations: int = 0


class SourceEffects:
    def __init__(self, paths, *, contracts=None, depth=8, procedures=64, operations=256, analysis_sources=None):
        if min(depth, procedures, operations) < 1:
            raise CompilationError("source effect budgets must be positive")
        self.depth_limit, self.procedure_limit, self.operation_limit = depth, procedures, operations
        self.modules, self.routines, self.summaries = {}, {}, {}
        self._closures = {}
        self.inputs = SourceInputs(paths,analysis_sources)
        self.sources, self.operation_count = dict(self.inputs.sources), 0
        self.kind_expressions = []
        self.contracts = contracts or {}
        if not isinstance(self.contracts, dict):
            raise CompilationError("effect contracts must be an object")
        for filename in self.inputs.paths:
            tree = self.inputs.parse(filename)
            for node in walk(tree):
                if _kind(node) != "Module":
                    continue
                module = str(_part(node, "Module_Stmt").items[1]).lower()
                if module in self.modules:
                    raise CompilationError(f"duplicate source module {module}")
                self.modules[module] = Scope(module, filename, node)
        for scope in self.modules.values():
            scope.issues = self._specification(scope)
        for scope in self.modules.values():
            for node in _children(_part(scope.node, "Module_Subprogram_Part")):
                if _kind(node) != "Subroutine_Subprogram":
                    continue
                stmt = _part(node, "Subroutine_Stmt")
                qualified = scope.module + "::" + str(stmt.items[1]).lower()
                if qualified in self.routines:
                    raise CompilationError(f"duplicate source procedure {qualified}")
                child = Scope(scope.module, scope.path, node, scope)
                arguments = tuple(str(a).lower() for a in _children(stmt.items[2]))
                issues = scope.issues + self._specification(child, arguments, qualified)
                if _part(node, "Internal_Subprogram_Part") is not None:
                    issues.append("internal procedures need an explicit effect closure")
                self.routines[qualified] = Routine(child, arguments, _part(node, "Execution_Part"), qualified, issues)
        for scope in self.modules.values():
            self._import_kinds(scope)
        for routine in self.routines.values():
            self._import_kinds(routine.scope)
        for binding, scope, selector in self.kind_expressions:
            # An unresolved kind cannot select a generic overload.
            with suppress(CompilationError):
                binding.kind = scope.kinds.integer(selector, SourceLocation(str(scope.path)))

    def _import_kinds(self, scope, active=frozenset()):
        if id(scope) in active:
            return
        active = active | {id(scope)}
        imported = []

        def exports(module, visited=frozenset()):
            target = self.modules.get(module)
            if target is None or module in visited:
                return set()
            names = set(target.bindings) | set(target.imports)
            for parent in target.wildcards:
                names.update(exports(parent, visited | {module}))
            return {name for name in names if self._exported(target, name)}

        for local, (module, remote) in scope.imports.items():
            imported.append((local, module, remote))
        for module in scope.wildcards:
            target = self.modules.get(module)
            if target:
                imported += [(name, module, name) for name in sorted(exports(module))]
        for local, module, remote in imported:
            target = self.modules.get(module)
            if target is None or not self._exported(target,remote):
                continue
            binding = self._binding(target, remote)
            if binding is None or binding.dtype != "integer" or binding.rank or "parameter" not in binding.attributes:
                continue
            owner = self.modules.get(binding.root.split("::",1)[0])
            if owner is None:
                continue
            self._import_kinds(owner, active)
            try:
                value = owner.kinds.integer(binding.name, SourceLocation(str(owner.path)))
            except CompilationError:
                continue
            if local not in scope.bindings:
                scope.kinds.values[local] = value

    def _specification(self, scope, arguments=(), qualified=None):
        spec = _part(scope.node, "Specification_Part")
        scope.kinds = _KindScope(spec, scope.parent.kinds if scope.parent else None)
        issues = []
        for node in _children(spec):
            name = _kind(node)
            if name == "Use_Stmt":
                nature, _, module, only, symbols = node.items
                module = str(module).lower()
                if module in _KindScope.intrinsic_kinds and str(nature).lower() != "non_intrinsic":
                    continue
                if symbols is None or str(only).upper().replace(" ", "") != ",ONLY:":
                    scope.wildcards.append(module)
                    if symbols is not None:
                        issues.append("USE renaming without ONLY requires complete export resolution")
                else:
                    for item in symbols.items:
                        local, remote = ((str(item.items[1]), str(item.items[2]))
                                         if _kind(item) == "Rename" else (str(item), str(item)))
                        scope.imports[local.lower()] = module, remote.lower()
                if module not in self.modules:
                    scope.unresolved_imports.append(module)
            elif name == "Interface_Block":
                start = _part(node, "Interface_Stmt")
                generic = str(start.items[0]).lower() if start.items[0] is not None else None
                members = []
                for item in _children(node):
                    if _kind(item) == "Procedure_Stmt":
                        members += [(scope.module, str(p).lower()) for p in item.items[0].items]
                if generic and members:
                    scope.generics[generic] = members
                else:
                    issues.append("non-module generic interface is unavailable")
            elif name == "Access_Stmt":
                mode, names = node.items
                public = str(mode).lower() == "public"
                if names is None:
                    scope.default_public = public
                else:
                    scope.access.update((str(item).lower(),public) for item in names.items)
            elif name == "Type_Declaration_Stmt":
                dtype, attrs, entities = node.items
                if _kind(dtype) == "Intrinsic_Type_Spec":
                    base, selector = dtype.items
                    base = str(base).lower()
                    width = 8 if base == "double precision" else 4
                    if selector is not None:
                        try:
                            width = scope.kinds.integer(selector.items[1], SourceLocation(str(scope.path)))
                        except CompilationError:
                            width = None
                    if base == "double precision":
                        base = "real"
                else:
                    base, width = str(dtype).lower(), None
                attrs = tuple(_children(attrs))
                flags = frozenset(str(a).split("(")[0].lower() for a in attrs)
                dimension = next((a.items[1] for a in attrs if _kind(a) == "Dimension_Attr_Spec"), None)
                intent = next((str(a.items[1]).lower() for a in attrs if _kind(a) == "Intent_Attr_Spec"), None)
                for entity in entities.items:
                    variable, shape, _, initializer = entity.items
                    variable = str(variable).lower()
                    shape = shape if shape is not None else dimension
                    dimensions = tuple(_children(shape))
                    lowers = tuple(str(d.items[0]) if d.items[0] is not None else "1" for d in dimensions)
                    persistent = "save" in flags or (initializer is not None and "parameter" not in flags)
                    attributes = flags | ({"save"} if persistent else set())
                    root = (f"argument::{variable}" if variable in arguments else
                            f"{qualified}::{variable}" if qualified else f"{scope.module}::{variable}")
                    binding = Binding(variable, root, base, width, len(dimensions), intent,
                                      frozenset(attributes), lowers)
                    scope.bindings[variable] = binding
                    if _kind(dtype) == "Intrinsic_Type_Spec" and selector is not None:
                        self.kind_expressions.append((binding, scope, selector.items[1]))
            elif name not in {"Implicit_Part", "Comment", "Public_Stmt", "Private_Stmt", "Access_Stmt"}:
                issues.append(f"specification effect unavailable: {name}")
        return issues

    def _exported(self, scope, name):
        name = str(name).lower()
        if name in scope.access:
            return scope.access[name]
        binding = scope.bindings.get(name)
        if binding and "private" in binding.attributes:
            return False
        if binding and "public" in binding.attributes:
            return True
        return scope.default_public

    def _binding(self, scope, name, visited=frozenset()):
        name = str(name).lower()
        key = (scope.module, id(scope), name)
        if key in visited:
            return None
        if name in scope.bindings:
            return scope.bindings[name]
        found = []
        if name in scope.imports:
            module, remote = scope.imports[name]
            if module in self.modules:
                if not self._exported(self.modules[module], remote):
                    return None
                binding = self._binding(self.modules[module], remote, visited | {key})
                if binding:
                    found.append(binding)
            else:
                return None  # A local USE name shadows any parent binding.
        else:
            for module in scope.wildcards:
                if module in self.modules:
                    if not self._exported(self.modules[module], name):
                        continue
                    binding = self._binding(self.modules[module], name, visited | {key})
                    if binding:
                        found.append(binding)
                else:
                    return None  # Unknown exports might shadow this name.
        roots = {b.root: b for b in found}
        if roots:
            return next(iter(roots.values())) if len(roots) == 1 else None
        return self._binding(scope.parent, name, visited | {key}) if scope.parent else None

    def _candidates(self, scope, name, visited=frozenset()):
        name = str(name).lower()
        key = (scope.module, id(scope), name)
        if key in visited:
            return []
        if name in scope.bindings:
            return []
        if name in scope.generics:
            return [m + "::" + p for m, p in scope.generics[name]]
        if name in scope.imports:
            module, remote = scope.imports[name]
            if module in self.modules:
                if not self._exported(self.modules[module],remote):
                    return []
                return self._candidates(self.modules[module], remote, visited | {key})
            return [module + "::" + remote]
        own = scope.module + "::" + name
        if scope.parent is None and own in self.routines:
            return [own]
        found = []
        for module in scope.wildcards:
            if module not in self.modules:
                return []
            if not self._exported(self.modules[module],name):
                continue
            found += self._candidates(self.modules[module], name, visited | {key})
        if found:
            return sorted(set(found))
        return self._candidates(scope.parent, name, visited | {key}) if scope.parent else []

    def _actual_binding(self, scope, node):
        if _kind(node) == "Name":
            return self._binding(scope, node)
        return None

    def _actual_mapping_boundary(self, scope, node):
        # An element actual is scalar at the callee, but reads/writes storage
        # belonging to its array root at the caller. It needs coherence before
        # evaluating its address/value and an expression mapping when owner
        # captures acquire private names. Whole-variable mappings do neither.
        for part in walk(node):
            if _kind(part) == "Part_Ref":
                binding = self._binding(scope, part.items[0])
                if binding and binding.rank:
                    return "array-element/section actual requires in-place mapping and coherence: " + str(node)
        return None

    def _unknown_exports(self, scope):
        # An unavailable wildcard USE can supply a procedure with the same
        # spelling as an intrinsic. A known local argument says nothing about
        # which implementation of SUM/SIZE/etc. was invoked.
        while scope is not None:
            if any(module not in self.modules for module in scope.wildcards):
                return True
            scope = scope.parent
        return False

    def _signature(self, scope, node):
        binding = self._actual_binding(scope, node)
        if binding:
            return binding.signature()
        if _kind(node) in {"Int_Literal_Constant", "Real_Literal_Constant", "Logical_Literal_Constant"}:
            dtype = {"Int_Literal_Constant": "integer", "Real_Literal_Constant": "real",
                     "Logical_Literal_Constant": "logical"}[_kind(node)]
            width = 8 if dtype == "real" and "d" in node.items[0].lower() else 4
            if node.items[1] is not None:
                try:
                    width = scope.kinds.integer(node.items[1], SourceLocation(str(scope.path)))
                except CompilationError:
                    width = None
            return dtype, width, 0
        return None

    def whole_overwrites(self, routine):
        """Prove complete assignments, including unit-stride full-array sweeps.

        This is a must-write proof, separate from the conservative may-write
        envelope. Uncertain bounds, conditional holes, extra loop iterators and
        early exits supply no proof. SIZE-based sweeps require dummy lower bound
        one; LBOUND/UBOUND sweeps also support negative declared lower bounds.
        """
        scope = routine.scope

        def normalized(node):
            return str(node).lower().replace(" ", "")

        def inquiry(node, name, binding, axis):
            if _kind(node) != "Intrinsic_Function_Reference" or str(node.items[0]).lower() != name:
                return False
            arguments = _children(node.items[1])
            if len(arguments) != 2 or _kind(arguments[0]) != "Name":
                return False
            array = self._binding(scope, arguments[0])
            return array is not None and array.root == binding.root and normalized(arguments[1]) == str(axis)

        def sweep(control, binding, axis):
            lower, upper, *steps = control
            if steps and steps[0] is not None and normalized(steps[0]) != "1":
                return False
            declared = binding.lower_bounds[axis-1].replace(" ", "").lower()
            lo = inquiry(lower, "lbound", binding, axis) or normalized(lower) == declared
            hi = inquiry(upper, "ubound", binding, axis) or (
                declared == "1" and inquiry(upper, "size", binding, axis))
            return lo and hi

        def full_target(target, loops):
            if _kind(target) == "Name" and not loops:
                binding = self._binding(scope, target)
                return binding if binding and binding.rank else None
            if _kind(target) != "Part_Ref":
                return None
            binding = self._binding(scope, target.items[0])
            indices = tuple(_children(target.items[1]))
            if not binding or not binding.rank or len(indices) != binding.rank:
                return None
            used = set()
            for axis, index in enumerate(indices, 1):
                if _kind(index) == "Subscript_Triplet" and all(v is None for v in index.items):
                    continue
                if _kind(index) != "Name":
                    return None
                iterator = normalized(index)
                if iterator in used or iterator not in loops or not sweep(loops[iterator], binding, axis):
                    return None
                used.add(iterator)
            return binding if used == set(loops) else None

        def prove(nodes, loops=None):
            loops = {} if loops is None else loops
            nodes = [node for node in nodes if _kind(node) != "Comment"]
            # A RETURN/EXIT/call could skip a candidate assignment or mutate its
            # sweep bounds. Restrict this proof to complete assignment-only nests.
            if any(_kind(n) not in {"Assignment_Stmt", "Block_Nonlabel_Do_Construct", "Continue_Stmt"}
                   for n in nodes):
                return set()
            found = set()
            for node in nodes:
                if _kind(node) == "Assignment_Stmt":
                    target = full_target(node.items[0], loops)
                    if target:
                        found.add(target.root)
                elif _kind(node) == "Block_Nonlabel_Do_Construct":
                    children = node.content
                    control = next((c for c in _children(children[0]) if _kind(c) == "Loop_Control"), None)
                    if control is None or control.items[1] is None:
                        continue
                    iterator, bounds = control.items[1]
                    iterator = normalized(iterator)
                    if iterator in loops:
                        continue
                    # Any assignment to an active iterator or its bound storage
                    # defeats a static coverage proof.
                    writes = [n.items[0] for n in walk(node) if _kind(n) == "Assignment_Stmt"]
                    bound_names = {str(n).lower() for b in bounds if b is not None
                                   for n in walk(b) if _kind(n) == "Name"}
                    if any(_kind(w) == "Name" and str(w).lower() in bound_names | {iterator}
                           for w in writes):
                        continue
                    found.update(prove(children[1:-1], {**loops, iterator: bounds}))
            return found

        return sorted(prove(_children(routine.execution)))

    def summarize(self, requested, active=(), *, _closure=None):
        # Each requested proof owns its budget. Unrelated extraction offers and
        # rejected branches cannot exhaust it or leave apparently complete,
        # empty summaries behind. Cache a closure only in its original context.
        if _closure is None:
            closure = self._closures.get(requested)
            if closure is None:
                closure = _Closure()
                self.summarize(requested, active, _closure=closure)
                if len(self._closures) < self.procedure_limit:
                    self._closures[requested] = closure
            self.operation_count = closure.operations
            self.summaries.update(closure.summaries)
            return closure.summaries[requested]
        if requested not in self.routines:
            raise CompilationError(f"source procedure unavailable: {requested}")
        if requested in active or len(active) >= self.depth_limit:
            return {"procedure": requested, "complete": False, "cloneable": False, "operations": [],
                    "reasons": ["recursive call or bounded source closure exhausted"]}
        if requested in _closure.summaries:
            cached = _closure.summaries[requested]
            if len(active) + cached["closure_depth"] <= self.depth_limit:
                return cached
            return {"procedure": requested, "complete": False, "cloneable": False, "operations": [],
                    "reasons": ["bounded source closure depth exhausted"]}
        if len(_closure.summaries) >= self.procedure_limit:
            return {"procedure": requested, "complete": False, "cloneable": False, "operations": [],
                    "reasons": ["bounded source closure procedure budget exhausted"]}
        routine = self.routines[requested]
        reasons = list(routine.issues)
        operations = []
        summary = {"procedure": requested, "complete": False, "cloneable": False,
                   "arguments": [routine.scope.bindings[a].public() for a in routine.arguments
                                 if a in routine.scope.bindings], "operations": operations, "reasons": reasons,
                   "definition_diagnostics": [], "closure_depth": 1}
        _closure.summaries[requested] = summary
        undeclared = set(routine.arguments) - routine.scope.bindings.keys()
        if undeclared:
            reasons.append("undeclared dummy arguments: " + ", ".join(sorted(undeclared)))
        persistent = [b.root for b in routine.scope.bindings.values() if "save" in b.attributes]
        summary["persistent_state"] = persistent
        summary["definition_changes"] = [b.root for b in routine.scope.bindings.values()
                                         if b.intent == "out" and b.name in routine.arguments]

        def emit(operation):
            _closure.operations += 1
            if _closure.operations > self.operation_limit:
                if "source operation budget exhausted" not in reasons:
                    reasons.append("source operation budget exhausted")
                return False
            operations.append(operation)
            return True

        def effect(binding, action, guard, spelling):
            if binding is None:
                reasons.append(f"unresolved storage: {spelling}")
                return
            if "parameter" in binding.attributes:
                return
            if {"pointer", "allocatable"} & binding.attributes:
                # Stable allocatable borrowing needs capture/lifetime proof later.
                reasons.append(f"storage lifetime requires capture proof: {binding.root}")
            external = binding.name in routine.arguments or not binding.root.startswith(requested + "::") or "save" in binding.attributes
            if external:
                emit({"kind": action, "resource": binding.root, "rank": binding.rank,
                      "section": "whole", "guard": guard, "source_access": spelling})

        def expression(node, guard, metadata=False):
            if node is None or isinstance(node, (str, int)):
                return
            name = _kind(node)
            if name == "Name":
                binding = self._binding(routine.scope, node)
                effect(binding, "descriptor_read" if metadata else "read", guard, str(node))
            elif name == "Part_Ref":
                base, indices = node.items
                binding = self._binding(routine.scope, base)
                if binding is None or not binding.rank:
                    reasons.append(f"unknown function or indexed storage: {node}")
                else:
                    effect(binding, "read", guard, str(node))
                for index in _children(indices):
                    expression(index, guard)
            elif name == "Intrinsic_Function_Reference":
                function, args = node.items
                intrinsic = str(function).lower()
                # A declared/imported name may shadow the intrinsic.
                shadowed = self._binding(routine.scope, intrinsic) or self._candidates(routine.scope, intrinsic)
                if shadowed or self._unknown_exports(routine.scope) or intrinsic not in set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
                    "allocated", "sum", "product", "any", "all", "count", "minval", "maxval"}:
                    reasons.append(f"unresolved function effects: {function}")
                for index, argument in enumerate(_children(args)):
                    if _kind(argument) == "Actual_Arg_Spec":
                        argument = argument.items[1]
                    expression(argument, guard, metadata=index == 0 and intrinsic in ARRAY_INQUIRIES | MODEL_INQUIRIES | {"allocated"})
            elif name.endswith("Literal_Constant"):
                return
            elif name in {"Data_Ref", "Function_Reference", "Structure_Constructor"}:
                reasons.append(f"unresolved component/function effects: {node}")
            else:
                for child in _children(node):
                    expression(child, guard, metadata)

        def call(node, guard):
            target, actuals = node.items
            if _kind(target) != "Name":
                reasons.append(f"indirect call effects unavailable: {target}")
                return
            actuals = tuple(_children(actuals))
            if any(_kind(a) == "Actual_Arg_Spec" for a in actuals):
                reasons.append(f"keyword call binding requires explicit resolution: {target}")
                return
            actual_boundary = next((reason for actual in actuals
                                    if (reason := self._actual_mapping_boundary(routine.scope, actual))), None)
            if actual_boundary:
                reasons.append(actual_boundary)
                emit({"kind": "boundary", "call": str(node), "guard": guard, "reason": actual_boundary})
                return
            candidates = self._candidates(routine.scope, target)
            matches = []
            for candidate in candidates:
                if candidate in self.routines:
                    callee = self.routines[candidate]
                    if len(callee.arguments) != len(actuals):
                        continue
                    formals = [callee.scope.bindings.get(a) for a in callee.arguments]
                    supplied = [self._signature(routine.scope, a) for a in actuals]
                    if all(f is not None and s is not None and None not in (f.kind, s[1]) and f.signature() == s
                           for f, s in zip(formals, supplied, strict=True)):
                        matches.append(candidate)
                elif candidate in self.contracts:
                    matches.append(candidate)
            if len(matches) != 1:
                reasons.append(f"call effects unresolved or ambiguous: {target}")
                emit({"kind": "boundary", "call": str(node), "guard": guard, "reason": reasons[-1]})
                return
            chosen = matches[0]
            # Evaluate effects of actuals in place; no early evaluation or lowering.
            for actual in actuals:
                if self._actual_binding(routine.scope, actual) is None:
                    expression(actual, guard)
            mapping = {}
            if chosen in self.routines:
                callee = self.routines[chosen]
                for formal, actual in zip(callee.arguments, actuals, strict=True):
                    binding = self._actual_binding(routine.scope, actual)
                    if binding:
                        mapping[f"argument::{formal}"] = binding.root
                child = self.summarize(chosen, active + (requested,), _closure=_closure)
                summary["closure_depth"] = max(summary["closure_depth"], 1 + child.get("closure_depth", 1))
                if not child["complete"]:
                    reasons.append(f"callee effects incomplete: {chosen}")
                summary["definition_diagnostics"].extend(child.get("definition_diagnostics",[]))
                emit({"kind": "call", "procedure": chosen, "actual_arguments": [str(a) for a in actuals],
                      "resource_mapping": mapping, "guard": guard, "complete": child["complete"]})
            else:
                contract = self.contracts[chosen]
                if (not isinstance(contract, dict) or contract.get("lifetime") != "stable"
                        or contract.get("escapes") is not False or contract.get("ordering") != "serial"
                        or contract.get("complete") is not True or contract.get("descriptor_changes") is not False
                        or not isinstance(contract.get("identity"), str) or not contract["identity"]
                        or not isinstance(contract.get("effects"), list)):
                    raise CompilationError(f"invalid native effect contract: {chosen}")
                contracted = []
                for item in contract["effects"]:
                    if (not isinstance(item, dict) or item.get("kind") not in {"read", "write", "overwrite"}
                            or item.get("section") != "whole"):
                        raise CompilationError(f"invalid native contract resource effect: {chosen}")
                    if "argument" in item and "resource" not in item:
                        if type(item["argument"]) is not int or not 0 <= item["argument"] < len(actuals):
                            raise CompilationError(f"invalid native contract argument: {chosen}")
                        binding = self._actual_binding(routine.scope, actuals[item["argument"]])
                    elif "resource" in item and "argument" not in item:
                        # Hidden fields need explicit, source-available identity.
                        found = [b for module in self.modules.values() for b in module.bindings.values()
                                 if b.root == item["resource"]]
                        binding = found[0] if len(found) == 1 else None
                    else:
                        raise CompilationError(f"invalid native contract binding: {chosen}")
                    if binding is None:
                        reasons.append(f"contract argument is not stable whole storage: {chosen}")
                        continue
                    if {"pointer", "allocatable"} & binding.attributes:
                        reasons.append(f"contract resource requires capture/lifetime proof: {binding.root}")
                    contracted.append({**item, "resource": binding.root})
                emit({"kind": "native_contract", "procedure": chosen, "identity": contract["identity"],
                      "contract_sha256": sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest(),
                      "effects": contracted, "guard": guard})

        def statements(nodes, guard=()):
            for node in nodes:
                if _closure.operations > self.operation_limit:
                    if "source operation budget exhausted" not in reasons:
                        reasons.append("source operation budget exhausted")
                    return
                kind = _kind(node)
                if kind == "Assignment_Stmt":
                    target, _, value = node.items
                    expression(value, guard)
                    if _kind(target) == "Name":
                        binding = self._binding(routine.scope, target)
                        effect(binding, "overwrite", guard, str(target))
                    elif _kind(target) == "Part_Ref":
                        effect(self._binding(routine.scope, target.items[0]), "write", guard, str(target))
                        for index in _children(target.items[1]):
                            expression(index, guard)
                    else:
                        reasons.append(f"unsupported assignment target effects: {target}")
                elif kind == "Call_Stmt":
                    call(node, guard)
                elif kind == "If_Stmt":
                    condition, action = node.items
                    expression(condition, guard)
                    statements((action,), guard + (str(condition),))
                elif kind == "If_Construct":
                    conditions, current = [], []
                    branch = guard
                    for item in node.content:
                        label = _kind(item)
                        if label in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                            statements(current, branch)
                            current = []
                            branch = guard + tuple(f".not.({c})" for c in conditions)
                            if label in {"If_Then_Stmt", "Else_If_Stmt"}:
                                condition = item.items[0]
                                expression(condition, branch)
                                branch += (str(condition),)
                                conditions.append(str(condition))
                        else:
                            current.append(item)
                elif kind in {"Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct"}:
                    children = [child for child in node.content if _kind(child) != "Comment"]
                    header = children[0]
                    # Retain loop execution protection as evidence; it is not a
                    # predicate that an adapter may evaluate outside the loop.
                    control = next((item for item in _children(header) if _kind(item) == "Loop_Control"), None)
                    if control is None or control.items[1] is None:
                        reasons.append("unsupported loop control effects")
                    else:
                        iterator, bounds = control.items[1]
                        for bound in bounds:
                            expression(bound, guard)
                        # DO assigns its control variable even for a zero-trip
                        # loop. It can be a dummy or module variable, rather
                        # than a private local used only for addressing.
                        effect(self._binding(routine.scope, iterator), "write", guard, str(iterator))
                        statements(children[1:-1], guard + (str(header),))
                elif kind == "Comment":
                    continue
                elif kind in {"Continue_Stmt", "Return_Stmt"}:
                    emit({"kind": "control", "source": str(node), "guard": guard})
                elif kind in {"Allocate_Stmt", "Deallocate_Stmt", "Pointer_Assignment_Stmt", "Nullify_Stmt"}:
                    reasons.append(f"storage lifetime/association boundary: {kind}")
                    emit({"kind": "boundary", "source": str(node), "guard": guard, "reason": reasons[-1]})
                else:
                    reasons.append(f"native ordering/effects unavailable: {kind}")
                    emit({"kind": "boundary", "source": str(node), "guard": guard, "reason": reasons[-1]})

        statements(_children(routine.execution))
        if not any(operation["kind"] in {"call","native_contract"} for operation in operations):
            # A leaf read before any payload write cannot consume values supplied
            # by its caller through INTENT(OUT). Do not hide this event by changing
            # the normalized access intent. This is an initial diagnostic, not a
            # complete proof for reads after partial or conditional writes.
            written = set()
            for operation in operations:
                root = operation.get("resource")
                if (operation["kind"] == "read" and operation["rank"]
                        and root in summary["definition_changes"] and root not in written):
                    diagnostic = {"procedure":requested,"resource":root,"source_access":operation["source_access"],
                                  "reason":"INTENT(OUT) payload read before any source write"}
                    if diagnostic not in summary["definition_diagnostics"]:
                        summary["definition_diagnostics"].append(diagnostic)
                elif operation["kind"] in {"write","overwrite"}:
                    written.add(root)
        summary["reasons"] = list(dict.fromkeys(reasons))
        summary["complete"] = not reasons
        summary["guaranteed_whole_overwrites"] = self.whole_overwrites(routine) if summary["complete"] else []
        directives = [str(node) for node in walk(routine.scope.node)
                      if _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")]
        summary["openmp_directives"] = directives
        summary["cloneable"] = summary["complete"] and not persistent and not directives
        summary["section_precision"] = "whole-resource conservative effects; physical refinement pending"
        return summary

    def summarize_span(self, procedures):
        """Prove a candidate span with one budget for its distinct closure."""
        closure = _Closure()
        for procedure in procedures:
            summary = self.summarize(procedure, _closure=closure)
            if not summary["complete"]:
                raise CompilationError("span effect closure incomplete: " + "; ".join(summary["reasons"]))
        return {"procedures": len(closure.summaries), "operations": closure.operations,
                "depth": max((s["closure_depth"] for s in closure.summaries.values()), default=0)}

    def report(self, entry):
        entry = entry.lower()
        matches = [p for p in self.routines if p == entry or ("::" not in entry and p.split("::")[-1] == entry)]
        if len(matches) != 1:
            raise CompilationError(f"native effect entry {entry!r} is unavailable or ambiguous")
        summary = self.summarize(matches[0])
        closure = self._closures.get(matches[0])
        if closure is None:
            closure = _Closure()
            summary = self.summarize(matches[0], _closure=closure)
        self.inputs.verify()
        return {"schema_version": 1, "entry": matches[0], "complete": summary["complete"],
                "sources": self.sources, "procedures": list(closure.summaries.values()),
                "analysis_sources": self.inputs.public(),
                "budgets": {"depth": self.depth_limit, "procedures": self.procedure_limit,
                            "operations": self.operation_limit}, "summarized_operations": closure.operations,
                "automatic_scope_available": False, "effect_coordinate_system": "logical source evidence; whole-resource effects"}


def analyze_source_effects(paths, entry, *, contracts=None, **budgets):
    """Return public native-effect JSON facts without requiring GPU eligibility."""
    return SourceEffects(paths, contracts=contracts, **budgets).report(entry)
