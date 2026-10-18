"""Zero definition state at an authenticated original selection.

The unchanged native prefix executes once before ownership starts. Its payload
state is deliberately unknown: supplied invocation definitions are not carried
across it. Exact source reads may establish only their missing host definitions
when the original numerical operation is reached, under separate typed proof.
This token grants neither numerical legality, alias freedom nor a new lifetime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.source_effects import _children, _kind
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError

_ISSUER = object()


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _same(left, right):
    return isinstance(right, tuple) and len(left) == len(right) and all(
        a is b for a, b in zip(left, right, strict=True))


def _registry(analysis):
    if getattr(analysis, "_selection_activation_registry_owner", None) is not analysis:
        analysis._selection_activation_proofs = {}
        analysis._selection_activation_registry_owner = analysis
    return analysis._selection_activation_proofs


def selection_activation_demands(owner):
    """Exact entry-visible storage and bound dependencies before draft issuance.

    This is a conservative source inventory, never an effects or alias proof.
    It deliberately precedes the owner-bound interval draft so its builder
    context is immutable once the checked source plan is issued.
    """
    analysis, routine = owner.builder.analysis, owner.routine
    found = {}
    for scope in (routine.scope, *analysis.modules.values()):
        for binding in scope.bindings.values():
            if (binding.root in owner.builder.facts["captures"]
                    and analysis._binding(routine.scope, binding.name) is binding):
                found[binding.root] = binding
    for original in owner.closed_selection:
        for node in walk(original):
            if _kind(node) not in {"Name", "Data_Ref"}:
                continue
            binding = analysis._binding(routine.scope, node)
            if binding is not None:
                previous = found.get(binding.root)
                if previous is not None and previous is not binding:
                    raise CompilationError("selection activation source inventory has ambiguous storage roots")
                found[binding.root] = binding
            if len(found) > analysis.operation_limit:
                raise CompilationError("selection activation demand inventory budget exhausted")
    return tuple(found[root] for root in sorted(found))


@dataclass(frozen=True)
class SelectionActivationProof:
    procedure: str
    identity: str
    demanded: tuple
    initializations: tuple
    _analysis: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _prefix: tuple = field(repr=False, compare=False)
    _controls: tuple = field(repr=False, compare=False)
    _configuration: str = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"proof_identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis, procedure, originals, demanded):
        if (self._issuer is not _ISSUER or analysis is not self._analysis or procedure != self.procedure
                or not _same(self._selection, originals) or not _same(self.demanded, demanded)
                or getattr(analysis, "_selection_activation_registry_owner", None) is not analysis
                or getattr(analysis, "_selection_activation_proofs", {}).get(self.identity) is not self):
            raise CompilationError("selection activation lacks exact registered source and demand authority")
        graph, selected = analysis._selected_source(procedure, originals)
        record = _thaw(self._record)
        if graph.identity != record["structured_identity"] or list(selected) != record["selected_node_ids"]:
            raise CompilationError("selection activation original source changed")
        if (_hash(record) != self.identity
                or record["initializations"] != _thaw(_freeze([item.public() for item in self.initializations]))
                or record["demanded_bindings"] != [id(item) for item in self.demanded]
                or record["initialization_bindings"] != [id(item) for item in self.initializations]
                or record["prefix_source_ids"] != [id(item) for item in self._prefix]
                or record["control_source_ids"] != [id(item) for item in self._controls]
                or record["configuration_identity"] != self._configuration):
            raise CompilationError("selection activation registered payload changed")
        if self.initializations or any(state["initialized"] != "none" for state in record["states"]):
            raise CompilationError("selection activation cannot grant prefix definitions")
        analysis.verify_source()
        return self

    def validate_configuration(self, builder):
        from compiler.scopes.execution_plan import _configuration

        if (_configuration(builder) != self._configuration
                or _hash(builder.facts) != _thaw(self._record)["entry_facts_identity"]):
            raise CompilationError("selection activation configuration or entry facts changed")
        return self

    def validate_prefix(self, owner):
        self.validate(owner.builder.analysis, owner.routine.qualified, owner.closed_selection, self.demanded)
        self.validate_configuration(owner.builder)
        if (not _same(self._prefix, owner.closed_prefix) or not _same(self._controls, owner.closed_controls)
                or owner.emission.selection_prefix is not self):
            raise CompilationError("selection activation prefix or controls changed")
        return self

    def capture_state(self, root):
        state = next((item for item in _thaw(self._record)["states"] if item["resource"] == root), None)
        if state is None:
            raise CompilationError("selection activation has no independently admitted capture: " + root)
        return dict(state)

    def activation_sections(self, root):
        self.capture_state(root)
        return None


def prove_selection_activation(owner, demanded):
    """Bind an original selection without inspecting dormant prefix effects."""
    from compiler.scopes.closed_selection import interval_candidates
    from compiler.scopes.execution_plan import _configuration

    builder, analysis, routine = owner.builder, owner.builder.analysis, owner.routine
    graph, selected = analysis._selected_source(routine.qualified, owner.closed_selection)
    sites = [site for site in interval_candidates(builder) if _same(site.originals, owner.closed_selection)]
    if (len(sites) != 1 or not _same(sites[0].prefix, owner.closed_prefix)
            or not _same(sites[0].controls, owner.closed_controls)):
        raise CompilationError("selection activation requires its exact original prefix and controls")
    if sites[0].loops:
        raise CompilationError("selection activation repeated interval requires independent backedge definition state")
    if not isinstance(demanded, tuple) or not demanded or len({id(item) for item in demanded}) != len(demanded):
        raise CompilationError("selection activation requires distinct original demanded bindings")
    arrays = {}
    for binding in demanded:
        if not binding.rank or binding.root in arrays or binding.root not in builder.facts["captures"]:
            continue
        if analysis._binding(routine.scope, binding.name) is not binding:
            continue
        fact = builder.capture_fact(binding)
        if fact.get("lifetime", "invocation") != "invocation":
            raise CompilationError("selection activation cannot infer selected-position storage lifetime")
        reason = analysis.resource_identity_boundary(binding)
        if reason:
            raise CompilationError(reason)
        module = analysis.modules.get(binding.root.partition("::")[0])
        if module is not None:
            specification = next((node for node in _children(module.node)
                                  if _kind(node) == "Specification_Part"), None)
            if any(_kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")
                   for node in walk(specification)):
                raise CompilationError("selection activation module OpenMP ownership requires source proof")
        arrays[binding.root] = binding
    record = {"schema_version": 2, "procedure": routine.qualified, "structured_identity": graph.identity,
        "selected_node_ids": list(selected), "entry_facts_identity": _hash(builder.facts),
        "configuration_identity": _configuration(builder),
        "demanded_bindings": [id(binding) for binding in demanded],
        "prefix_source_ids": [id(node) for node in owner.closed_prefix],
        "control_source_ids": [id(node) for node in owner.closed_controls],
        "initializations": [], "initialization_bindings": [],
        "prefix": [str(node) for node in owner.closed_prefix], "controls": [str(node) for node in owner.closed_controls],
        "states": [{"resource": root, "initialized": "none",
                    "selected_position_lifetime_proven": False} for root in sorted(arrays)],
        "definition_contract": "unknown at selection; exact original upward reads admitted only when reached",
        "prefix_execution": "unchanged original native prefix and controls exactly once before any context",
        "numerical_authority": False, "new_lifetime_authority": False}
    identity = _hash(record)
    registry = _registry(analysis)
    if identity in registry:
        return registry[identity].validate(analysis, routine.qualified, owner.closed_selection, demanded)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("selection activation registry budget exhausted")
    result = SelectionActivationProof(routine.qualified, identity, demanded, (), analysis,
        owner.closed_selection, owner.closed_prefix, owner.closed_controls, _configuration(builder),
        _freeze(record), _ISSUER)
    registry[identity] = result
    return result.validate(analysis, routine.qualified, owner.closed_selection, demanded)
