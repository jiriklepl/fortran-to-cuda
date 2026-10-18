"""Definition state at an original selection, without changing its prefix.

The prefix executes natively once.  Registered native preservation and logical
definition proofs transfer independently supplied invocation facts to this
position.  This token grants neither numerical legality nor a new lifetime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.capture_coverage import reached_sections
from compiler.frontend.native_definitions import intersect_rectangles, merge_rectangles, prove_native_definition_state
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.native_sections import NativeAxis, NativeRectangle, NativeResourceSections
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


def _whole(boxes):
    return any(not box.iterations and all(axis.lower is None and axis.upper is None for axis in box.axes)
               for box in boxes)


def _physical_template(builder, binding):
    """Retain only original descriptor mappings admitted by capture facts.

    Explicit-shape dummies can describe only a prefix of their actual and are
    therefore excluded.  Dynamic original dummy bounds use its existing
    descriptor, never reevaluation of a specification expression.
    """
    analysis, routine = builder.analysis, builder.entry
    fact = builder.capture_fact(binding)
    if fact.get("lifetime", "invocation") != "invocation":
        raise CompilationError("selection activation requires independently admitted invocation storage: " + binding.root)
    reason = analysis.resource_identity_boundary(binding)
    if reason:
        raise CompilationError(reason)
    module = analysis.modules.get(binding.root.partition("::")[0])
    if module is not None:
        specification = next((node for node in _children(module.node) if _kind(node) == "Specification_Part"), None)
        if any(_kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")
               for node in walk(specification)):
            raise CompilationError("selection activation module OpenMP ownership requires source proof")
    if binding.name in routine.arguments and "allocatable" not in binding.attributes and any(
            _kind(axis) != "Assumed_Shape_Spec" for axis in binding.shape_nodes):
        raise CompilationError("selection activation physical sections require assumed-shape dummies")
    if "allocatable" in binding.attributes:
        return NativeResourceSections(binding.root, binding.rank, (None,) * binding.rank, (), (), (),
                                      descriptor_lower_bounds=True)
    lowers = []
    for node in binding.lower_bound_nodes:
        try:
            value = 1 if node is None else int("".join(str(node).split()))
        except ValueError:
            if binding.name in routine.arguments:
                return NativeResourceSections(binding.root, binding.rank, (None,) * binding.rank, (), (), (),
                                              descriptor_lower_bounds=True, descriptor_origin="original_dummy")
            raise CompilationError("selection activation requires original static lower bounds") from None
        if not -(2**31) <= value < 2**31:
            raise CompilationError("selection activation lower bound exceeds source INTEGER ABI")
        lowers.append(value)
    if len(lowers) != binding.rank:
        raise CompilationError("selection activation lacks original rank-matched lower bounds")
    return NativeResourceSections(binding.root, binding.rank, tuple(lowers), (), (), ())


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
    for original in (*owner.closed_prefix, *owner.closed_selection):
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
    _definitions: tuple = field(repr=False, compare=False)
    _preservations: tuple = field(repr=False, compare=False)
    _sections: tuple = field(repr=False, compare=False)
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
        if (_hash(record) != self.identity or record["physical_sections"] != [item.public() for item in self._sections]
                or record["initializations"] != _thaw(_freeze([item.public() for item in self.initializations]))
                or record["preservation_proofs"] != [item.identity for item in self._preservations]
                or record["definition_proofs"] != [item.identity for item in self._definitions]
                or record["demanded_bindings"] != [id(item) for item in self.demanded]
                or record["initialization_bindings"] != [id(item) for item in self.initializations]
                or record["prefix_source_ids"] != [id(item) for item in self._prefix]
                or record["control_source_ids"] != [id(item) for item in self._controls]
                or record["configuration_identity"] != self._configuration):
            raise CompilationError("selection activation registered payload changed")
        bindings = {binding.root: binding for binding in self.demanded}
        for resource in self._sections:
            if any(bindings.get(dependency.resource) is not dependency.binding for dependency in resource.dependencies):
                raise CompilationError("selection activation physical bounds lost original binding authority")
        analysis.verify_source()
        for proof in self._preservations:
            proof.validate(analysis, procedure, proof._selection, proof.demanded)
        for proof in self._definitions:
            proof.validate(analysis, procedure, proof._selection)
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
        return next((resource for resource in self._sections if resource.resource == root), None)


def prove_selection_activation(owner, demanded):
    """Transfer original native prefix effects before a closed interval scan."""
    from compiler.frontend.native_preservation import prove_native_preservation
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
    arrays, coverage, templates = {}, {}, {}
    # Only independently admitted original invocation captures enter the
    # transfer. Uncaptured resources may occur in effects but acquire no facts.
    for binding in demanded:
        if not binding.rank or binding.root in arrays:
            continue
        if analysis._binding(routine.scope, binding.name) is not binding:
            continue
        if binding.root not in builder.facts["captures"]:
            continue
        fact = builder.capture_fact(binding)
        if fact.get("lifetime", "invocation") != "invocation":
            raise CompilationError("selection activation cannot infer selected-position storage lifetime")
        arrays[binding.root] = binding
        if fact["initialized"] == "whole":
            coverage[binding.root] = (NativeRectangle((NativeAxis(),) * binding.rank, binding.name),)
        elif fact["initialized"] == "sections":
            template = reached_sections(binding, fact)
            coverage[binding.root], templates[binding.root] = template.overwrites, template
        else:
            coverage[binding.root] = ()
    for root in graph.nodes[graph.entry_id].details.get("definition_changes", ()):
        if root in coverage:
            coverage[root] = ()
    preservations, definitions, work = [], [], 0

    def preservation(originals, state):
        proof = prove_native_preservation(analysis, routine.qualified, originals, demanded)
        result = proof.transfer(state)
        if not result.complete:
            reason = "; ".join(item.reason for item in proof.reachability.barriers)
            raise CompilationError("selection activation native prefix lacks complete preservation: " + reason)
        preservations.append(proof)
        return dict(result.definitions)

    def sequence(nodes, state, initialized=frozenset(), depth=0):
        nonlocal work
        if depth > analysis.depth_limit:
            raise CompilationError("selection activation prefix control budget exhausted")
        state, initialized = dict(state), set(initialized)
        for item in grouped_original_sequence(nodes, {}):
            work += 1
            if work > analysis.operation_limit:
                raise CompilationError("selection activation prefix operation budget exhausted")
            if isinstance(item, NativeGroupCandidate):
                originals = item.original_nodes
                before = state
                state = preservation(originals, state)
                outputs = tuple(binding for root, binding in arrays.items() if not _whole(state.get(root, ())))
                if outputs:
                    completion = analysis.joined_completion(routine.qualified, originals)
                    proof = prove_native_definition_state(analysis, routine.qualified, originals, outputs=outputs,
                        completion=completion, immutable_inputs=getattr(builder, "immutable_inputs", {}))
                    resources = {row["resource"] for row in proof.public()["resources"]}
                    result = proof.transfer({root: boxes for root, boxes in before.items() if root in resources})
                    if not result.reads_covered:
                        raise CompilationError("selection activation native initialization reads uncovered sections")
                    definitions.append(proof)
                    # Transfer with entry coverage discharges original reads;
                    # it also retains entry rectangles, which must never be
                    # recredited after preservation has killed them. Only new
                    # unconditional writes from an empty state are candidates.
                    fresh = proof.transfer({}).candidate_definitions
                    fresh = preservation(originals, dict(fresh))
                    for root, boxes in fresh.items():
                        if boxes:
                            templates.setdefault(root, _physical_template(builder, arrays[root]))
                            state[root] = merge_rectangles(state.get(root, ()), boxes)
            elif _kind(item) == "If_Construct":
                # The complete original branch proof includes its conditions.
                # Killing a fact for either arm before the meet is conservative
                # and avoids moving a condition or inventing guard invariance.
                entry = preservation((item,), state)
                alternatives, body, has_else, has_header = [], [], False, False
                for child in _children(item):
                    kind = _kind(child)
                    if kind in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        if has_header:
                            alternatives.append(sequence(tuple(body), entry, frozenset(initialized), depth + 1))
                        body = []
                        has_header = kind != "End_If_Stmt"
                        has_else |= kind == "Else_Stmt"
                    else:
                        body.append(child)
                if not has_else:
                    alternatives.append((entry, initialized))
                if not alternatives:
                    alternatives.append((entry, initialized))
                state, initialized = alternatives[0]
                for other, other_initialized in alternatives[1:]:
                    state = {root: intersect_rectangles(state.get(root, ()), other.get(root, ())) for root in arrays}
                    initialized = set(initialized).intersection(other_initialized)
            elif _kind(item) != "Comment":
                state = preservation((item,), state)
                if _kind(item) == "Assignment_Stmt" and _kind(item.items[0]) == "Name":
                    binding = analysis._binding(routine.scope, item.items[0])
                    if (binding is not None and not binding.rank and binding.declaring_scope is routine.scope
                            and binding.name not in routine.arguments and not binding.attributes & {
                                "save", "pointer", "target", "optional", "volatile", "asynchronous"}):
                        initialized.add(id(binding))
        return state, initialized

    coverage, initialized = sequence(owner.closed_prefix, coverage)
    # An earlier local assignment cannot price a scalar which an original OUT
    # association may subsequently leave undefined. This conservative filter
    # does not infer any definition from a native call's ordinary writes.
    undefined = {root for proof in preservations for root, reasons in proof.invalidations
                 if "entry undefinition" in reasons}
    initializations = tuple(binding for binding in routine.scope.bindings.values()
                            if id(binding) in initialized and binding.root not in undefined)
    sections = []
    for root, boxes in coverage.items():
        if boxes and not _whole(boxes):
            template = templates.get(root)
            if template is None:
                raise CompilationError("selection activation partial definition lacks original descriptor mapping")
            # Registration is lazy. Its source-coordinate expressions must
            # remain valid until every possible registration in this selection.
            # This check grants no definition credit for selected operations.
            physical = replace(template, reads=(), writes=(), overwrites=boxes)
            if physical.dependencies:
                retained = preservation(owner.closed_selection, {root: boxes})
                if retained.get(root, ()) != boxes:
                    raise CompilationError("selection activation partial bounds change before lazy registration")
            sections.append(physical)
    record = {"schema_version": 1, "procedure": routine.qualified, "structured_identity": graph.identity,
        "selected_node_ids": list(selected), "entry_facts_identity": _hash(builder.facts),
        "configuration_identity": _configuration(builder),
        "demanded_bindings": [id(binding) for binding in demanded],
        "prefix_source_ids": [id(node) for node in owner.closed_prefix],
        "control_source_ids": [id(node) for node in owner.closed_controls],
        "initializations": [binding.public() for binding in initializations],
        "initialization_bindings": [id(binding) for binding in initializations],
        "prefix": [str(node) for node in owner.closed_prefix], "controls": [str(node) for node in owner.closed_controls],
        "preservation_proofs": [proof.identity for proof in preservations],
        "definition_proofs": [proof.identity for proof in definitions],
        "states": [{"resource": root, "initialized": "whole" if _whole(boxes) else "sections" if boxes else "none",
                    "selected_position_lifetime_proven": False} for root, boxes in coverage.items()],
        "physical_sections": [resource.public() for resource in sections],
        "prefix_execution": "unchanged original native prefix and controls exactly once",
        "numerical_authority": False, "new_lifetime_authority": False}
    identity = _hash(record)
    registry = _registry(analysis)
    if identity in registry:
        return registry[identity].validate(analysis, routine.qualified, owner.closed_selection, demanded)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("selection activation registry budget exhausted")
    result = SelectionActivationProof(routine.qualified, identity, demanded, initializations, analysis,
        owner.closed_selection, owner.closed_prefix, owner.closed_controls, _configuration(builder),
        tuple(definitions), tuple(preservations), tuple(sections), _freeze(record), _ISSUER)
    registry[identity] = result
    return result.validate(analysis, routine.qualified, owner.closed_selection, demanded)
