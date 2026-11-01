"""Original control-local descriptor regions in the existing storage epoch.

Every selected source node belongs to a checked descriptor region or an exact
native retirement subtree. A retirement may change allocation identity, but
reopening requires an independently complete no-escape lifecycle. None of this
grants payload effects, initialized values, or permission to execute a child.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields, replace
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.descriptor_windows import (
    DescriptorWindowProof,
    DescriptorWindowUnsupported,
    prove_descriptor_window,
)
from compiler.frontend.source_effects import _children, _kind, _part
from compiler.frontend.storage_mutation_effects import StorageMutationEffects, prove_storage_mutation_effects
from compiler.ir import CompilationError

_ISSUER = object()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _attachments(value):
    return tuple((part.name, getattr(value, part.name)) for part in fields(value))


def _pure_scalar_condition(owner, identity):
    """A small original predicate with no calls, payload or aliasable reads."""
    analysis, routine = owner.builder.analysis, owner.routine
    operators = {"+", "-", "*", "(", ")", "==", "/=", "<", ">", "<=", ">=", ".eq.", ".ne.",
                 ".lt.", ".gt.", ".le.", ".ge.", ".and.", ".or.", ".not.", ".eqv.", ".neqv."}
    expressions = {"Level_1_Expr", "Level_2_Expr", "Level_2_Unary_Expr", "Level_3_Expr", "Level_4_Expr",
                   "Level_5_Expr", "And_Operand", "Or_Operand", "Equiv_Operand", "Parenthesis",
                   "Add_Operand", "Mult_Operand"}

    def check(node):
        kind = _kind(node)
        if kind in {"Int_Literal_Constant", "Logical_Literal_Constant"}:
            return True
        if kind == "Name":
            scope = analysis.source_scope_for(node, routine.scope)
            binding = analysis._binding(scope, node)
            declaration = binding.declaring_scope if binding is not None else None
            while declaration is not None:
                if any(_kind(item) in {"Common_Stmt", "Equivalence_Stmt"}
                       for item in walk(_part(declaration.node, "Specification_Part"))):
                    return False
                declaration = declaration.parent
            return (binding is not None and binding.dtype in {"integer", "logical"} and not binding.rank
                    and not binding.attributes & {"pointer", "target", "allocatable", "optional",
                                                   "volatile", "asynchronous", "value"})
        return kind in expressions and all(
            child.lower() in operators if isinstance(child, str) else check(child) for child in _children(node))

    node = owner.graph.nodes[identity]
    originals = owner.graph.source_nodes(identity)
    return (node.kind == "operation" and node.details.get("evaluation") == "condition"
            and len(originals) == 1 and check(originals[0]))


def _coverage(owner, selected, components, cuts):
    """Authenticate every original descendant; no unproved gap at a join."""
    graph = owner.graph
    retained, barred = set(components), set(cuts)
    if retained & barred or len(retained) != len(components) or len(barred) != len(cuts):
        raise CompilationError("control-local epoch has duplicate or overlapping source roles")
    visited, used_components, used_cuts = set(), set(), set()
    pending = list(reversed(selected))
    while pending:
        identity = pending.pop()
        if identity in visited or len(visited) >= graph.node_limit:
            raise CompilationError("control-local epoch source coverage is duplicate or over budget")
        visited.add(identity)
        node = graph.nodes[identity]
        if identity in barred:
            used_cuts.add(identity)
            continue
        if identity in retained:
            if node.kind not in {"call", "operation", "boundary"}:
                raise CompilationError("control-local descriptor must retain one original leaf")
            used_components.add(identity)
            continue
        if node.kind == "sequence":
            pending.extend(reversed(node.children))
        elif node.kind == "branch":
            original, = graph.source_nodes(identity)
            if (_kind(original) != "If_Construct"
                    or any(not _pure_scalar_condition(owner, condition)
                           for condition, _body in node.alternatives if condition is not None)):
                raise CompilationError("control-local epoch requires complete original scalar predicates")
            pending.extend(reversed(tuple(child for condition, body in node.alternatives
                                          for child in (condition, body) if child is not None)))
        else:
            raise CompilationError("control-local epoch has an uncovered original source node")
    if used_components != retained or used_cuts != barred:
        raise CompilationError("control-local epoch role escapes its exact original interval")


def _record(owner, selected, parent, roots, modules, components, cuts):
    closure, resources = {}, None
    for _identity, proof in components:
        public = proof.public()
        if resources is None:
            resources = public["resources"]
        elif resources != public["resources"]:
            raise CompilationError("control-local descriptor regions disagree on original storage declarations")
        for procedure, graph in public["source_call_closure"].items():
            if procedure in closure and closure[procedure] != graph:
                raise CompilationError("control-local descriptor regions changed an original CALL closure")
            closure[procedure] = graph
    return {"version": 1, "kind": "original_control_local_descriptor_regions",
            "procedure": owner.routine.qualified, "structured_identity": owner.graph.identity,
            "selected_node_ids": list(selected), "parent_sequence": parent,
            "roots": list(roots), "module_roots": list(modules), "resources": resources or [],
            "source_call_closure": closure,
            "regions": [{"node_id": identity, "descriptor_identity": proof.identity}
                        for identity, proof in components],
            "retirement_subtrees": [{"node_id": identity, "first_line": patch.first, "last_line": patch.last,
                                     "original_source_sha256": sha256(patch.original_source.encode()).hexdigest(),
                                     "storage_projection_identity": lifecycle.identity,
                                     "reopening": "complete original no-escape lifecycle; fresh generation and NONE"}
                                    for identity, patch, lifecycle in cuts],
            "entry_definition_authority": False, "payload_access_authorized": False,
            "execution_authorized": False, "cross_cut_retention_authorized": False}


@dataclass(frozen=True, slots=True)
class ControlLocalDescriptorRegions:
    identity: str
    _owner: object = field(repr=False, compare=False)
    _analysis: object = field(repr=False, compare=False)
    _selected: tuple = field(repr=False, compare=False)
    _parent: str = field(repr=False, compare=False)
    _roots: tuple = field(repr=False, compare=False)
    _module_roots: tuple = field(repr=False, compare=False)
    _components: tuple = field(repr=False, compare=False)
    _cuts: tuple = field(repr=False, compare=False)
    _record: str = field(repr=False, compare=False)
    _issuer: object = field(repr=False, compare=False)

    def public(self):
        return {"proof_identity": self.identity, **json.loads(self._record)}

    @property
    def call_selection(self):
        return tuple(identity for identity, _proof in self._components
                     if self._owner.graph.nodes[identity].kind == "call")

    def covers(self, selected):
        from compiler.scopes.graph_epochs import _covered

        graph = self._owner.graph
        covered = _covered(graph, selected)
        excluded = _covered(graph, tuple(identity for identity, _patch, _proof in self._cuts))
        return bool(selected) and covered <= _covered(graph, self._selected) and not covered & excluded

    def validate(self, analysis, procedure, selected):
        owner, attached = self._owner, _attachments(self)
        epoch = getattr(owner, "position_epoch", None)
        if (self._issuer is not _ISSUER or type(self) is not ControlLocalDescriptorRegions
                or analysis is not self._analysis or owner.builder.analysis is not analysis
                or procedure != owner.routine.qualified or selected != self._selected
                or epoch is None or epoch._descriptor is not self
                or type(self._record) is not str or sha256(self._record.encode()).hexdigest() != self.identity):
            raise CompilationError("control-local descriptor requires its exact issuing epoch")
        graph, current = analysis._selected_source(procedure, selected)
        if graph is not owner.graph or current != selected:
            raise CompilationError("control-local descriptor changed its original selected graph")
        if (any(type(proof) is not DescriptorWindowProof for _identity, proof in self._components)
                or any(type(proof) is not StorageMutationEffects for _identity, _patch, proof in self._cuts)):
            raise CompilationError("control-local descriptor requires original issued region and lifecycle proofs")
        retained = tuple((proof, _attachments(proof)) for _identity, proof in self._components)
        lifecycle_attachments = tuple((proof, _attachments(proof)) for _identity, _patch, proof in self._cuts)
        descriptor_registry = analysis._descriptor_window_proofs
        lifecycle_registry = analysis._storage_mutation_effects

        def require_attachments():
            if (owner.position_epoch is not epoch or epoch._descriptor is not self
                    or any(getattr(self, name) is not value for name, value in attached)
                    or analysis._descriptor_window_proofs is not descriptor_registry
                    or analysis._storage_mutation_effects is not lifecycle_registry
                    or any(descriptor_registry.get(proof.identity) is not proof for proof, _saved in retained)
                    or any(lifecycle_registry.get(proof.identity) is not proof
                           for proof, _saved in lifecycle_attachments)
                    or any(any(getattr(proof, name) is not value for name, value in saved)
                           for proof, saved in (*retained, *lifecycle_attachments))):
                raise CompilationError("control-local descriptor attachment changed during validation")

        def checkpoint():
            require_attachments()
            _coverage(owner, selected, tuple(identity for identity, _proof in self._components),
                      tuple(identity for identity, _patch, _proof in self._cuts))
            for identity, patch, lifecycle in self._cuts:
                if (not any(current is patch for _first, _last, current in owner.patches)
                        or patch.lifecycle is not lifecycle or patch.conditional_guard is not None):
                    raise CompilationError("control-local epoch lost an original mandatory retirement")
                original = graph.source_nodes(identity)
                if (len(original) != len(patch.originals)
                        or any(a is not b for a, b in zip(original, patch.originals, strict=True))):
                    raise CompilationError("control-local retirement changed its original subtree")
            require_attachments()

        checkpoint()
        for identity, proof in self._components:
            if proof._roots != self._roots or proof._module_roots != self._module_roots:
                raise CompilationError("control-local descriptor lost complete canonical storage coverage")
            proof.validate(analysis, procedure, (identity,))
            checkpoint()
        for _identity, patch, lifecycle in self._cuts:
            patch.validate(owner, require_source=True)
            checkpoint()
            if not lifecycle._record["complete"] or lifecycle._record["escapes"]:
                raise CompilationError("control-local retirement cannot reopen an unproved escape or lifetime")
        if _encoded(_record(owner, selected, self._parent, self._roots, self._module_roots,
                            self._components, self._cuts)) != self._record:
            raise CompilationError("control-local descriptor changed its original region or cut record")
        checkpoint()
        return self

    def resource_state(self, root):
        self.validate(self._analysis, self._owner.routine.qualified, self._selected)
        if root not in (*self._roots, *self._module_roots):
            raise CompilationError("control-local descriptor resource is outside its original inventory")
        return next(state for state in self.public()["resources"] if state["resource"] == root)


def discover_control_local_epoch(owner, selected, cut, roots, modules, bindings):
    """Partition one original CALL-led run before proving its complete closure."""
    from compiler.scopes.graph_epochs import _ISSUER as epoch_issuer
    from compiler.scopes.graph_epochs import GraphStorageEpoch, _calls

    graph, analysis = owner.graph, owner.builder.analysis
    if (graph.nodes[selected[0]].kind != "call"
            or not any(graph.nodes[identity].kind == "branch" for identity in selected)):
        return None
    parents = [node for node in graph.nodes.values() if node.kind == "sequence" and selected[0] in node.children]
    if len(parents) != 1:
        raise CompilationError("control-local epoch requires its exact original immediate sequence")
    parent = parents[0]
    start = parent.children.index(selected[0])
    if tuple(parent.children[start:start+len(selected)+1]) != (*selected, cut):
        raise CompilationError("control-local epoch changed its contiguous original close frontier")
    components, cuts, visited, retained_proofs = [], [], set(), []
    builder, routine = owner.builder, owner.routine
    previous = getattr(owner, "position_epoch", None)

    def checkpoint(expected_epoch=previous):
        if (owner.builder is not builder or builder.analysis is not analysis or owner.graph is not graph
                or owner.routine is not routine or builder.entry is not routine
                or getattr(owner, "position_epoch", None) is not expected_epoch
                or any(any(getattr(proof, name) is not value for name, value in attached)
                       for proof, attached in retained_proofs)):
            raise CompilationError("control-local discovery changed an original source or retained proof attachment")

    def retain(proof):
        if type(proof) not in {DescriptorWindowProof, StorageMutationEffects}:
            raise CompilationError("control-local discovery requires an original issued proof")
        retained_proofs.append((proof, _attachments(proof)))
        checkpoint()
        return proof

    def retire(identity):
        lifecycle = retain(prove_storage_mutation_effects(analysis, owner.routine.qualified, selection=(identity,)))
        if type(lifecycle) is not StorageMutationEffects:
            raise CompilationError("control-local cut requires its original storage lifecycle proof")
        if not lifecycle._record["complete"] or lifecycle._record["escapes"]:
            raise DescriptorWindowUnsupported("control-local cut has an unproved escape or asynchronous lifetime")
        cuts.append((identity, lifecycle))

    def visit(identity):
        if identity in visited or len(visited) >= graph.node_limit:
            raise CompilationError("control-local discovery exhausted original source coverage")
        visited.add(identity)
        node = graph.nodes[identity]
        if node.kind == "sequence":
            for child in node.children:
                visit(child)
        elif node.kind == "branch":
            original, = graph.source_nodes(identity)
            if _kind(original) != "If_Construct":
                raise DescriptorWindowUnsupported("control-local first slice requires original multiline IF")
            if any(not _pure_scalar_condition(owner, condition)
                   for condition, _body in node.alternatives if condition is not None):
                retire(identity)
                return
            for condition, body in node.alternatives:
                if condition is not None:
                    visit(condition)
                visit(body)
        elif node.kind in {"call", "operation", "boundary"}:
            try:
                proof = retain(prove_descriptor_window(analysis, owner.routine.qualified, (identity,), roots,
                                                       module_roots=modules))
            except DescriptorWindowUnsupported:
                checkpoint()
                retire(identity)
            else:
                if type(proof) is not DescriptorWindowProof:
                    raise CompilationError("control-local region requires its original descriptor proof")
                components.append((identity, proof))
        else:
            retire(identity)

    for identity in selected:
        visit(identity)
    if not cuts or not any(graph.nodes[identity].kind == "call" for identity, _proof in components):
        return None
    _coverage(owner, selected, tuple(identity for identity, _proof in components),
              tuple(identity for identity, _proof in cuts))
    call_selection = tuple(identity for identity, _proof in components if graph.nodes[identity].kind == "call")
    calls = _calls(owner, call_selection)
    final_lifecycle = retain(prove_storage_mutation_effects(analysis, owner.routine.qualified, selection=(cut,)))
    if type(final_lifecycle) is not StorageMutationEffects:
        raise CompilationError("control-local final cut requires its original storage lifecycle proof")
    if not final_lifecycle._record["complete"] or final_lifecycle._record["escapes"]:
        raise DescriptorWindowUnsupported("control-local final cut has an unproved escape or asynchronous lifetime")
    checkpoint()
    before_patches, before_boundaries = len(owner.patches), len(owner.boundaries)
    try:
        retained = []
        for identity, lifecycle in cuts:
            owner.boundary(graph.source_nodes(identity), "checked control-local allocation retirement",
                           force_unconditional=True)
            first, last, patch = owner.patches[-1]
            patch = replace(patch, lifecycle=lifecycle)
            owner.patches[-1] = (first, last, patch)
            retained.append((identity, patch, lifecycle))
        owner.boundary(graph.source_nodes(cut), "checked selected-position allocation epoch retirement",
                       force_unconditional=True)
        first, last, boundary = owner.patches[-1]
        boundary = replace(boundary, lifecycle=final_lifecycle)
        owner.patches[-1] = (first, last, boundary)
        record = _encoded(_record(owner, selected, parent.id, roots, modules, tuple(components), tuple(retained)))
        descriptor = ControlLocalDescriptorRegions(sha256(record.encode()).hexdigest(), owner, analysis, selected,
            parent.id, roots, modules, tuple(components), tuple(retained), record, _ISSUER)
        epoch = GraphStorageEpoch("", selected, roots, modules, cut, owner, descriptor, boundary, calls,
                                  bindings, epoch_issuer)
        public = epoch.public()
        public.pop("identity")
        epoch = replace(epoch, identity=sha256(_encoded(public).encode()).hexdigest())
        owner.position_epoch = epoch
        epoch.validate()
        checkpoint(epoch)
        return epoch
    except BaseException:
        del owner.patches[before_patches:]
        del owner.boundaries[before_boundaries:]
        owner.position_epoch = previous
        raise
