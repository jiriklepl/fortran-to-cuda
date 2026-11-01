"""Readable local scalars after an exact unchanged original native prefix.

This is definition consumption at a reached source site, not an entry fact or
a numerical range proof. Descriptor inquiries remain descriptor-only reads.
The first slice deliberately rejects branch/loop-defined values and calls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(record):
    return json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _same(left, right):
    return isinstance(right, tuple) and len(left) == len(right) and all(
        a is b for a, b in zip(left, right, strict=True))


def _site(builder, nodes):
    from compiler.scopes.closed_selection import candidates

    sites = [(prefix, controls) for original, prefix, controls in candidates(builder)
             if _same(nodes, original)]
    if len(sites) != 1:
        raise CompilationError("scalar definition requires one exact nonrepeated original selection")
    return sites[0]


def _local(analysis, routine, binding):
    from compiler.frontend.native_demand import _Declarations, _owned_storage

    if (binding.rank or (binding.dtype, binding.kind) != ("integer", 4)
            or binding.declaring_scope is not routine.scope or binding.name in routine.arguments
            or binding.attributes & {"save", "pointer", "target", "optional", "volatile", "asynchronous",
                                     "allocatable", "parameter"}
            or analysis._binding(routine.scope, binding.name) is not binding
            or analysis.resource_identity_boundary(binding)
            or not _owned_storage(analysis, binding, _Declarations())):
        raise CompilationError("scalar prefix requires independent original local default INTEGER storage")


def _constant_call_input(builder, binding):
    """Consume original associations, never seed a child capture fact.

    A reusable companion has no value specialization. This first slice proves
    every source call from its same parent, including guarded calls, then
    checks that every emitted companion invocation belongs to that inventory.
    Other callers and mutable actuals require independent readability proof.
    The existing registered prefix token owns this nested source evidence.
    """
    from fparser.two.utils import walk

    from compiler.frontend.constant_actual import constant_scalar_actual
    from compiler.frontend.immutable_graph import original_type
    from compiler.scopes.module_owner import ModuleOwner
    from compiler.scopes.segments import statement_span

    member = getattr(builder, "_native_host_owner", None)
    if member is None:
        return None
    if (original_type(member) is not ModuleOwner or member.builder is not builder
            or member.routine is not builder.entry or member.parent is None
            or member.parent.builder.analysis is not builder.analysis):
        raise CompilationError("scalar call input requires its exact original companion owner")
    if binding.declaring_scope is not member.routine.scope or binding.name not in member.routine.arguments:
        return None
    if (binding.rank or binding.intent != "in" or binding.dtype not in {"integer", "real", "logical"}
            or binding.name not in member.routine.arguments
            or member.routine.scope.bindings.get(binding.name) is not binding
            or binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "allocatable", "value"}):
        raise CompilationError("scalar call input requires an ordinary read-only original formal")
    parent, analysis = member.parent, builder.analysis
    calls, accepted, original = [], set(), False
    visits = 0
    for node in walk(parent.routine.execution):
        visits += 1
        if visits > analysis.operation_limit:
            raise CompilationError("scalar call input source inventory budget exhausted")
        if _kind(node) != "Call_Stmt":
            continue
        try:
            resolved = parent.builder.execution_source_call(parent.routine, node)
        except CompilationError:
            # Such a call cannot appear in the retained inventory below.
            continue
        if resolved.procedure != member.routine.qualified:
            continue
        matches = [item for item in resolved.mappings if item.formal_binding is binding]
        if len(matches) != 1:
            raise CompilationError("scalar call input changed its exact original formal binding")
        mapping = matches[0]
        scope = analysis.source_scope_for(node, parent.routine.scope)
        if (mapping.presence != "supplied" or mapping.actual is None or mapping.section is not None
                or mapping.scalar_element is not None
                or not constant_scalar_actual(analysis, scope, mapping.actual)):
            raise CompilationError("scalar call input requires a source-constant actual at every original call")
        graph, selected = analysis._selected_source(parent.routine.qualified, (node,))
        if not selected:
            raise CompilationError("scalar call input lacks its original reached call selection")
        first, last = statement_span(node)
        accepted.add((parent.routine.qualified, first, last))
        original |= node is member.original_source_call
        calls.append({"selected_node_ids": list(selected), "first_line": first, "last_line": last,
                      "source_call": resolved.public()})
    if not calls or not original:
        raise CompilationError("scalar call input lacks its original companion admission call")
    for site in member.call_sites:
        if (site.get("caller"), site.get("first_line"), site.get("last_line")) not in accepted:
            raise CompilationError("scalar call input does not cover every emitted companion call")
    return {"kind": "original source-constant scalar call inputs", "version": 1,
            "caller": parent.routine.qualified, "procedure": member.routine.qualified,
            "source_inputs": analysis.inputs.identity(), "caller_graph": graph.identity,
            "formal": binding.public(), "calls": calls,
            "semantics": "original actual evaluation and association at each unchanged guarded CALL",
            "value_specialization_authorized": False, "evaluation_motion_authorized": False,
            "array_payload_definition_authorized": False, "lifetime_authorized": False,
            "entry_fact_authorized": False}


def _derive(builder, nodes, demanded, prefix, controls, effects):
    from compiler.frontend.resource_reachability import ResourceReachabilityProof
    from compiler.scopes.closed_selection import selection_context
    from compiler.scopes.immutable_inputs import _declared
    from compiler.scopes.regions import _upward_reads
    from compiler.scopes.segments import statement_span

    analysis, routine = builder.analysis, builder.entry
    actual_prefix, actual_controls = _site(builder, nodes)
    if not _same(prefix, actual_prefix) or not _same(controls, actual_controls):
        raise CompilationError("scalar definition original prefix or predicates changed")
    executable = tuple(node for node in prefix if _kind(node) != "Comment")
    if not executable or any(_kind(node) not in {"Assignment_Stmt", "If_Then_Stmt", "Else_If_Stmt"}
                             for node in executable):
        raise CompilationError("scalar definition requires straight original local preparation")
    if any(_kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp") for node in prefix):
        raise CompilationError("scalar prefix cannot cross original OpenMP participation")
    if not isinstance(effects, tuple) or len(effects) != len(executable):
        raise CompilationError("scalar prefix requires one exact effect proof per original statement")
    for node, effect in zip(executable, effects, strict=True):
        if type(effect) is not ResourceReachabilityProof:
            raise CompilationError("scalar prefix requires original registered effect authority")
        effect.validate(analysis, routine.qualified, (node,), demanded)
        if effect.barriers or effect.calls:
            raise CompilationError("scalar prefix requires complete effects without calls or barriers")
    # These effects establish absence of hidden writes; they grant no definition
    # by themselves. Only the exact straight assignments below define scalars.
    for access in (access for effect in effects for access in effect.accesses):
        if access.kind in {"write", "undefine", "association"}:
            if access.kind != "write" or access.storage.components:
                raise CompilationError("scalar prefix changes storage association or definition state")
            _local(analysis, routine, access.storage.binding)
    for binding in demanded:
        _local(analysis, routine, binding)
    defined, dependencies, assignments = set(), {}, []
    for node in executable:
        # Existing liveness accounts for exact RHS/header scalar reads. It is
        # not used to grant CALL intent, conditional or zero-trip definitions.
        for root in sorted(_upward_reads(analysis, routine, (node,)) - defined):
            binding = builder.visible_binding(routine, root)
            if "parameter" in binding.attributes:
                _declared(analysis, binding)
                authority = {"kind": "original PARAMETER declaration"}
            else:
                # A local named in caller facts must never stand in for a
                # missing dominating original definition.
                if binding.declaring_scope is routine.scope and binding.name not in routine.arguments:
                    raise CompilationError("scalar prefix reads a local before its original definition: " + root)
                authority = _constant_call_input(builder, binding)
                if authority is None:
                    fact = builder.capture_fact(binding)
                    if fact["initialized"] != "whole" or fact.get("lifetime", "invocation") != "invocation":
                        raise CompilationError("scalar prefix dependency lacks whole invocation definition: " + root)
                    authority = {"kind": "explicit checked whole-defined scalar entry fact", "fact": {
                        key: fact[key] for key in ("storage", "initialized", "allocation_changes", "escapes", "lifetime")
                        if key in fact}}
            dependencies[root] = {"resource": root, "descriptor": binding.public(), "authority": authority}
        if _kind(node) == "Assignment_Stmt":
            target = node.items[0]
            binding = analysis._binding(routine.scope, target) if _kind(target) == "Name" else None
            if binding is None:
                raise CompilationError("scalar prefix assignment requires an exact local scalar target")
            _local(analysis, routine, binding)
            defined.add(binding.root)
            first, last = statement_span(node)
            assignments.append({"resource": binding.root, "first_line": first, "last_line": last,
                                "source": str(node)})
    if any(binding.root not in defined for binding in demanded):
        raise CompilationError("consumer scalar lacks a dominating original prefix definition")
    graph, selected = analysis._selected_source(routine.qualified, nodes)
    return {"version": 2, "kind": "original_reached_scalar_prefix_definitions",
            "procedure": routine.qualified, "source_inputs": analysis.inputs.identity(),
            "source_graph": graph.identity, "selected_node_ids": list(selected),
            "selection": selection_context(builder, nodes, prefix, controls),
            "resources": [binding.public() for binding in demanded],
            "assignments": assignments, "dependencies": [dependencies[root] for root in sorted(dependencies)],
            "effects_identities": [effect.identity for effect in effects],
            "semantics": "original unchanged prefix and predicates execute before scalar capture at this selection",
            "entry_definition_authorized": False, "array_payload_definition_authorized": False,
            "finite_or_constant_authorized": False, "motion_authorized": False,
            "execution_authorized": False, "lifetime_authorized": False}


@dataclass(frozen=True, slots=True)
class ReachedScalarDefinitions:
    identity: str
    demanded: tuple = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _routine: object = field(repr=False, compare=False)
    _nodes: tuple = field(repr=False, compare=False)
    _prefix: tuple = field(repr=False, compare=False)
    _controls: tuple = field(repr=False, compare=False)
    _effects: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"identity": self.identity, **json.loads(self._record)}

    def validate(self, builder, nodes, demanded):
        if (type(self) is not ReachedScalarDefinitions or self._issuer is not _ISSUER
                or builder.analysis is not self._analysis or builder.entry is not self._routine
                or not _same(self._nodes, nodes) or not _same(self.demanded, demanded)
                or getattr(self._analysis, "_reached_scalar_definition_proofs", {}).get(self.identity) is not self
                or sha256(self._record.encode()).hexdigest() != self.identity):
            raise CompilationError("scalar definition requires exact registered original source authority")
        record = _derive(builder, nodes, demanded, self._prefix, self._controls, self._effects)
        if _encoded(record) != self._record:
            raise CompilationError("scalar definition prefix, dependencies or source changed")
        return self


def prove_reached_scalar_definitions(builder, nodes, demanded):
    from compiler.frontend.resource_reachability import prove_resource_reachability
    from compiler.frontend.storage_mutation_effects import source_registry_budget

    if (not isinstance(nodes, tuple) or not nodes or not isinstance(demanded, tuple)
            or not demanded or len({id(binding) for binding in demanded}) != len(demanded)):
        raise CompilationError("scalar definition requires exact original selection and distinct demands")
    registry = getattr(builder.analysis, "_reached_scalar_definition_proofs", None)
    if registry is None:
        registry = builder.analysis._reached_scalar_definition_proofs = {}
    for proof in registry.values():
        # Speculative builders share exact immutable source authority. Their
        # container addresses are not source identity; rederive all consumed
        # facts below instead of replacing an equal registered prefix token.
        if (proof._analysis is builder.analysis and proof._routine is builder.entry
                and _same(proof._nodes, nodes) and _same(proof.demanded, demanded)):
            return proof.validate(builder, nodes, demanded)
    prefix, controls = _site(builder, nodes)
    executable = tuple(node for node in prefix if _kind(node) != "Comment")
    if not executable:
        raise CompilationError("consumer scalar lacks an original preparation prefix")
    # A reached ELSEIF/ELSE path includes earlier predicate headers, but never
    # their skipped sibling bodies. Those headers are not one effect segment;
    # retain singleton source proofs in the exact authenticated prefix order.
    effects = tuple(prove_resource_reachability(builder.analysis, builder.entry.qualified, (node,), demanded)
                    for node in executable)
    record = _encoded(_derive(builder, nodes, demanded, prefix, controls, effects))
    if len(registry) >= source_registry_budget(builder.analysis):
        raise CompilationError("scalar definition registry budget exhausted")
    identity = sha256(record.encode()).hexdigest()
    proof = ReachedScalarDefinitions(identity, demanded, builder.analysis, builder.entry,
                                    nodes, prefix, controls, effects, record, _ISSUER)
    registry[identity] = proof
    return proof.validate(builder, nodes, demanded)
