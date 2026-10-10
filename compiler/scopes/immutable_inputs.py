"""Source-authenticated immutable array associations and reached element operands.

The original arrays stay in Fortran. These proofs authorize neither a device
allocation nor an invented scalar Binding. Element values are supplied only by
an original, reached, nonempty numerical region.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from hashlib import sha256

from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.component_bindings import references
from compiler.frontend.source_effects import Binding, _children, _kind, _part
from compiler.ir import CompilationError, SourceLocation

IMMUTABLE_INPUT_VERSION = 1
IMMUTABLE_SOURCE_VISIT_LIMIT = 16384
_FORBIDDEN = {"allocatable", "pointer", "optional", "target", "volatile", "asynchronous", "value"}


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _integer(scope, node):
    value = scope.kinds.integer(node, SourceLocation(str(scope.path)))
    if not -(2**31) <= value < 2**31:
        raise CompilationError("immutable input bounds exceed the default INTEGER ABI")
    return value


def _spelling(value):
    return "(-2147483647 - 1)" if value == -(2**31) else str(value)


def _bounds(binding):
    if binding.rank != 1 or len(binding.shape_nodes) != 1 or binding.declaring_scope is None:
        raise CompilationError("immutable input requires original fixed rank-one storage")
    axis, = binding.shape_nodes
    if _kind(axis) != "Explicit_Shape_Spec" or axis.items[1] is None:
        raise CompilationError("immutable input requires constant explicit bounds")
    scope = binding.declaring_scope
    lower = 1 if axis.items[0] is None else _integer(scope, axis.items[0])
    upper = _integer(scope, axis.items[1])
    if not 0 < upper - lower + 1 <= 256:
        raise CompilationError("immutable input exceeds the bounded nonempty element limit")
    return lower, upper


def _declared(analysis, binding):
    """Validate the original declaration without evaluating its initializer."""
    scope = binding.declaring_scope
    if scope is None or scope.bindings.get(binding.name) is not binding:
        raise CompilationError("immutable input needs its original declaring binding")
    owner = getattr(scope, "qualified", None) or scope.module
    routine = analysis.routines.get(owner) or analysis.numerical_helpers.get(owner)
    module = analysis.modules.get(owner)
    if routine is not None:
        analysis._require_original(routine.qualified)
        if routine.scope is not scope:
            raise CompilationError("immutable input declaration has foreign source authority")
    else:
        role = analysis._source_module_roles.get(owner)
        if module is not scope or role is None or scope.node is not role[0] or str(scope.node) != role[-1]:
            raise CompilationError("immutable input declaration has foreign source authority")
    for declaration in _children(_part(scope.node, "Specification_Part")):
        if _kind(declaration) != "Type_Declaration_Stmt":
            continue
        dtype, attributes, entities = declaration.items
        dimension = next((item.items[1] for item in _children(attributes)
                          if _kind(item) == "Dimension_Attr_Spec"), None)
        for entity in _children(entities):
            if str(entity.items[0]).lower() != binding.name:
                continue
            flags = {str(item).split("(")[0].lower() for item in _children(attributes)}
            if ("parameter" in flags) != ("parameter" in binding.attributes):
                raise CompilationError("immutable input PARAMETER attribute differs from original declaration")
            if _kind(dtype) != "Intrinsic_Type_Spec":
                raise CompilationError("immutable input requires an original intrinsic numeric declaration")
            base, selector = dtype.items
            base = str(base).lower()
            width = 8 if base == "double precision" else 4
            if base == "double precision":
                base = "real"
            if selector is not None:
                width = _integer(scope, selector.items[1])
            if (base, width) != (binding.dtype, binding.kind):
                raise CompilationError("immutable input type differs from its original declaration")
            shape = entity.items[1] if entity.items[1] is not None else dimension
            axes = tuple(_children(shape))
            if len(axes) != len(binding.shape_nodes) or any(a is not b for a, b in zip(axes, binding.shape_nodes, strict=True)):
                raise CompilationError("immutable input descriptor differs from its original declaration")
            if "parameter" in binding.attributes and entity.items[3] is None:
                raise CompilationError("immutable PARAMETER input lacks its original initializer")
            return str(declaration)
    raise CompilationError("immutable input original declaration is unavailable")


@dataclass(frozen=True)
class ImmutableArrayInput:
    procedure: str
    caller: str
    formal: Binding = field(compare=False, repr=False)
    actual: Binding = field(compare=False, repr=False)
    original_actual: Binding = field(compare=False, repr=False)
    call: object = field(compare=False, repr=False)
    formal_bounds: tuple[int, int]
    actual_bounds: tuple[int, int]
    identity: str
    _authority: str = field(repr=False)
    parent: ImmutableArrayInput | None = field(default=None, compare=False, repr=False)

    @property
    def resource(self):
        return self.formal.root

    def validate(self, analysis, routine=None):
        caller = analysis._require_original(self.caller)
        callee = analysis._require_original(self.procedure)
        if routine is not None and callee is not routine:
            raise CompilationError("immutable association belongs to another original procedure")
        if self.parent is not None:
            self.parent.validate(analysis, caller)
        if not any(node is self.call for node in walk(caller.execution)):
            raise CompilationError("immutable association requires its original call node")
        if callee.scope.bindings.get(self.formal.name) is not self.formal:
            raise CompilationError("immutable association requires its original formal binding")
        if (self.formal.name not in callee.arguments or self.formal.intent != "in"
                or self.formal.attributes & _FORBIDDEN or self.actual.attributes & _FORBIDDEN
                or "parameter" not in self.actual.attributes
                or self.actual.signature() != self.formal.signature()
                or self.formal.signature()[:2] not in {("real", 4), ("real", 8), ("integer", 4)}):
            raise CompilationError("immutable association requires required fixed numeric INTENT(IN) storage")
        if self.parent is not None and (self.original_actual is not self.parent.formal or self.actual is not self.parent.actual):
            raise CompilationError("immutable forwarding differs from its original parent association")
        if self.parent is None and self.actual is not self.original_actual:
            raise CompilationError("immutable input differs from its original PARAMETER actual")
        from compiler.frontend.call_bindings import resolve_source_call
        resolved = resolve_source_call(analysis, caller.scope, self.call)
        mapping = next((item for item in resolved.mappings if item.formal == self.formal.name), None)
        if (resolved.procedure != self.procedure or mapping is None or mapping.formal_binding is not self.formal
                or mapping.binding is not self.original_actual or mapping.section is not None
                or mapping.presence != "supplied" or _kind(mapping.actual) != "Name"):
            raise CompilationError("immutable input differs from its original whole-array argument association")
        authority = _association_authority(analysis, caller, callee, self.formal, self.actual, self.call, self.parent)
        if (_bounds(self.formal) != self.formal_bounds or _bounds(self.actual) != self.actual_bounds
                or authority != self._authority or self.identity != sha256(authority.encode()).hexdigest()):
            raise CompilationError("immutable input source authority changed")
        return self

    def public(self):
        return {"schema_version": IMMUTABLE_INPUT_VERSION, "identity": self.identity,
                "procedure": self.procedure, "caller": self.caller,
                "formal_resource": self.formal.root, "actual_resource": self.actual.root,
                "formal_bounds": list(self.formal_bounds), "actual_bounds": list(self.actual_bounds),
                "type": self.formal.dtype, "kind": self.formal.kind,
                "association": "original required fixed-shape INTENT(IN) argument",
                "storage": "original PARAMETER owner; no TARGET, temporary, address or device registration",
                "payload_evaluation": "only at the original reached nonempty numerical region",
                "parent_identity": self.parent.identity if self.parent is not None else None}


def _association_authority(analysis, caller, callee, formal, actual, call, parent):
    return json.dumps({"version": IMMUTABLE_INPUT_VERSION, "caller": caller.qualified,
                       "procedure": callee.qualified, "formal": formal.public(), "actual": actual.public(),
                       "formal_declaration": _declared(analysis, formal), "actual_declaration": _declared(analysis, actual),
                       "call": str(call), "parent": parent.identity if parent is not None else None,
                       "sources": {str(scope.path): analysis.sources[str(scope.path)]
                                   for scope in (caller.scope, callee.scope, actual.declaring_scope)}},
                      sort_keys=True, separators=(",", ":"))


def immutable_call_inputs(analysis, caller, call, resolved, inherited=None):
    """Prove whole PARAMETER actuals without reading any element or creating storage."""
    inherited = inherited or {}
    callee = analysis._require_original(resolved.procedure)
    analysis._require_original(caller.qualified)
    result = {}
    for mapping in resolved.mappings:
        formal, original = mapping.formal_binding, mapping.binding
        if not formal.rank or original is None or mapping.section is not None or _kind(mapping.actual) != "Name":
            continue
        parent = inherited.get(original.root)
        actual = parent.actual if parent is not None else original
        if "parameter" not in actual.attributes:
            continue
        if parent is not None:
            parent.validate(analysis, caller)
        lower, upper = _bounds(formal)
        first, last = _bounds(actual)
        if upper - lower != last - first:
            raise CompilationError("immutable actual and formal require identical fixed extents")
        authority = _association_authority(analysis, caller, callee, formal, actual, call, parent)
        proof = ImmutableArrayInput(callee.qualified, caller.qualified, formal, actual, original, call,
                                    (lower, upper), (first, last), sha256(authority.encode()).hexdigest(), authority, parent)
        proof.validate(analysis, callee)
        result[formal.root] = proof
    return result


@dataclass(frozen=True)
class ScalarElementCapture:
    name: str
    resource: str
    dtype: str
    kind: int
    formal_resource: str
    expression: str
    index: str
    index_guard: str
    nonempty: str
    dependencies: tuple[str, ...]
    association_identity: str
    identity: str
    activation_guards: tuple[str, ...] = ()
    original_references: tuple[object, ...] = field(default=(), compare=False, repr=False)

    def public(self):
        return {"schema_version": IMMUTABLE_INPUT_VERSION, "identity": self.identity, "name": self.name,
                "resource": self.resource, "type": self.dtype, "kind": self.kind,
                "formal_resource": self.formal_resource, "source_expression": self.expression,
                "index": self.index, "index_guard": self.index_guard, "nonempty": self.nonempty,
                "activation_guards": list(self.activation_guards),
                "activation_evaluation": "nested outer-to-inner IF; no Fortran short-circuit assumption",
                "empty_domain": "whole original perfect rectangular region; skip query/run and retain original empty-loop execution",
                "dependencies": list(self.dependencies), "association_identity": self.association_identity,
                "evaluation": "original reached region after guards and nonempty-domain check",
                "query": "typed zero placeholder; operand must not affect query metadata",
                "automatic_estimate": {"available": False, "reason": "immutable element preparation is uncalibrated"}}


def scalar_element_captures(analysis, routine, loops, associations, occupied):
    """Collect unconditional element uses in independent rectangular loop nests."""
    if not associations:
        return ()
    from compiler.scopes.regions import _iterator
    for proof in associations.values():
        if not isinstance(proof, ImmutableArrayInput):
            raise CompilationError("immutable numerical input requires compiler source authority")
        proof.validate(analysis, routine)
    found, visits, proof_visits, scalar_facts = {}, 0, 0, {}

    def scalar(value, *, index=False):
        nonlocal proof_visits
        if _kind(value) in {"Int_Literal_Constant", "Level_2_Unary_Expr", "Parenthesis"}:
            try:
                return _spelling(_integer(routine.scope, value)), ()
            except CompilationError:
                pass
        if _kind(value) != "Name":
            raise CompilationError("immutable element index and domain require invariant scalar INTEGER operands")
        binding = analysis._binding(routine.scope, value)
        if binding is None or binding.signature() != ("integer", 4, 0) or binding.attributes & _FORBIDDEN:
            raise CompilationError("immutable element index and domain require ordinary default INTEGER storage")
        if "parameter" in binding.attributes:
            _declared(analysis, binding)
            return _spelling(_integer(binding.declaring_scope, F.Name(binding.name))), ()
        formal = (binding.name in routine.arguments and routine.scope.bindings.get(binding.name) is binding
                  and binding.intent == "in")
        module = (not index and binding.declaring_scope is not None
                  and analysis.modules.get(binding.declaring_scope.module) is binding.declaring_scope)
        if not formal and not module:
            raise CompilationError("immutable element index and domain require unchanged required INTENT(IN) scalars")
        key = binding.root, index
        if key in scalar_facts:
            return scalar_facts[key]
        if module:
            from compiler.scopes.participation import _threadprivate
            _declared(analysis, binding)
            if binding.root in _threadprivate(analysis) or analysis.resource_identity_boundary(binding):
                raise CompilationError("immutable activation bound requires unaliased shared original module storage")
        inspected = (routine.execution,) if formal else loops
        for item in (item for source in inspected for item in walk(source)):
            proof_visits += 1
            if proof_visits > IMMUTABLE_SOURCE_VISIT_LIMIT:
                raise CompilationError("immutable operand invariance exceeds the source visit budget")
            if _kind(item) in {"Assignment_Stmt", "Pointer_Assignment_Stmt", "Nonlabel_Do_Stmt", "Call_Stmt",
                               "Function_Reference", "Part_Ref", "Structure_Constructor"}:
                if _kind(item) in {"Call_Stmt", "Function_Reference", "Part_Ref", "Structure_Constructor"}:
                    array = analysis._binding(routine.scope, item.items[0]) if _kind(item) == "Part_Ref" else None
                    if array is not None and array.rank:
                        continue
                    changed = module or any(other is binding for other, _ in references(analysis, routine.scope, item))
                elif _kind(item) == "Nonlabel_Do_Stmt":
                    control = item.items[1]
                    counted = control.items[1] if control is not None else None
                    changed = bool(counted and analysis._binding(routine.scope, counted[0]) is binding)
                else:
                    target = item.items[0]
                    base = target.items[0] if _kind(target) == "Part_Ref" else target
                    changed = analysis._binding(routine.scope, base) is binding
                if changed:
                    raise CompilationError("immutable element index or domain can change or escape through original work")
        scalar_facts[key] = str(value), (binding.root,)
        return scalar_facts[key]

    def visit(node, domain=(), dependencies=(), conditional=False):
        nonlocal visits
        visits += 1
        if visits > 4096:
            raise CompilationError("immutable element capture exceeds the bounded source visit budget")
        if isinstance(node, (tuple, list)):
            for child in node:
                visit(child, domain, dependencies, conditional)
            return
        kind = _kind(node)
        if kind == "Block_Nonlabel_Do_Construct":
            _, bounds = _iterator(node)
            if len(bounds) not in {2, 3}:
                raise CompilationError("immutable element activation requires a rectangular counted domain")
            first, deps1 = scalar(bounds[0])
            last, deps2 = scalar(bounds[1])
            step = 1 if len(bounds) == 2 else _integer(routine.scope, bounds[2])
            if step <= 0:
                raise CompilationError("immutable element activation requires a constant positive stride")
            for child in _children(node):
                if _kind(child) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"}:
                    visit(child, (*domain, f"({last}) >= ({first})"), (*dependencies, *deps1, *deps2), conditional)
            return
        if kind == "Part_Ref":
            binding = analysis._binding(routine.scope, node.items[0])
            proof = associations.get(binding.root) if binding is not None else None
            if proof is not None:
                indices = _children(node.items[1])
                if len(indices) != 1 or not domain or conditional:
                    raise CompilationError("immutable element payload requires an unconditional reached nonempty loop body")
                index, deps = scalar(indices[0], index=True)
                first, last = proof.formal_bounds
                expression = binding.name + "(" + index + ")"
                key = proof.identity, expression, domain
                if key not in found:
                    if len(found) >= 64:
                        raise CompilationError("immutable element capture exceeds the bounded operand limit")
                    identity = _digest({"version": IMMUTABLE_INPUT_VERSION, "association": proof.identity,
                                        "expression": expression, "domain": domain,
                                        "selected": [str(loop) for loop in loops]})
                    name = "fort_region_element_" + identity[:12]
                    if (name in occupied or analysis._binding(routine.scope, name)
                            or analysis._candidates(routine.scope, F.Name(name))):
                        raise CompilationError("immutable element scalar namespace conflicts with original source")
                    occupied.add(name)
                    found[key] = ScalarElementCapture(name, "immutable_element::" + identity, binding.dtype,
                        binding.kind, binding.root, expression, index,
                        f"({index}) >= ({_spelling(first)}) .and. ({index}) <= ({_spelling(last)})",
                        " .and. ".join("(" + item + ")" for item in domain),
                        tuple(sorted(set((*dependencies, *deps)))), proof.identity, identity, domain, (node,))
                else:
                    found[key] = replace(found[key], original_references=(*found[key].original_references, node))
                return
        if kind == "Name":
            binding = analysis._binding(routine.scope, node)
            if binding is not None and binding.root in associations:
                raise CompilationError("immutable array input requires explicit scalar element uses")
        nested = conditional or kind in {"If_Construct", "If_Stmt", "Where_Construct", "Where_Stmt"}
        if (kind in {"Call_Stmt", "Function_Reference", "Structure_Constructor"}
                and any(binding.root in associations for binding, _ in references(analysis, routine.scope, node))):
            raise CompilationError("immutable element helper arguments require a separate reached evaluation proof")
        for child in _children(node):
            visit(child, domain, dependencies, nested)

    for loop in loops:
        # A proof is relevant only when this region actually uses that array;
        # unrelated loops must not inherit activation restrictions.
        if any(binding.root in associations for binding, _ in references(analysis, routine.scope, loop)):
            visit(loop)
    if found:
        # The reached caller can bypass query/argument evaluation for an empty
        # domain only when that domain covers the *entire* numerical region.
        # A coefficient in one inner/sibling loop cannot suppress other work.
        if len(loops) != 1:
            raise CompilationError("immutable element activation requires one complete rectangular loop nest")
        current, depth = loops[0], 0
        while _kind(current) == "Block_Nonlabel_Do_Construct":
            depth += 1
            body = tuple(child for child in _children(current)
                         if _kind(child) not in {"Nonlabel_Do_Stmt", "End_Do_Stmt", "Comment"})
            nested = tuple(child for child in body if _kind(child) == "Block_Nonlabel_Do_Construct")
            if not nested:
                if any(_kind(item) == "Block_Nonlabel_Do_Construct"
                       for child in body for item in walk(child)):
                    raise CompilationError("immutable element activation requires a perfect original loop nest")
                break
            if len(body) != 1 or len(nested) != 1:
                raise CompilationError("immutable element activation cannot omit sibling or outer-body work")
            current = nested[0]
        chains = {item.activation_guards for item in found.values()}
        if len(chains) != 1 or any(len(chain) != depth for chain in chains):
            raise CompilationError("immutable element captures require one common whole-region activation chain")
    return tuple(found.values())
