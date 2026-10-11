"""Original interval descriptor proofs, independent of entry payload facts.

No projected/synthetic body or whole-procedure descriptor proof grants this
authority. Selected original source and its complete direct call closure are
checked; excluded prefixes/tails, reachability, aliases and definitions remain
separate obligations before any owner may open.
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import Base

from compiler.frontend.call_bindings import resolve_source_call
from compiler.frontend.component_bindings import component_access
from compiler.frontend.native_completion import NativeCompletionProof
from compiler.frontend.native_environment import NativeEnvironmentProof, NativeLocalStateProof
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.native_intrinsics import NativePredicateProof
from compiler.frontend.native_prefix import NativePrefixProof
from compiler.frontend.numerical_completion import NumericalCompletionProof
from compiler.frontend.source_functions import _validate_resolution, _validate_scope, _walk
from compiler.frontend.source_objects import _TypeBuilder
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError
from compiler.ir.intrinsics import ARRAY_INQUIRIES, INTRINSICS, MODEL_INQUIRIES

DESCRIPTOR_WINDOW_VERSION = 5
_UNSAFE = {"pointer", "optional", "volatile", "asynchronous", "value"}
_INTRINSICS = set(INTRINSICS) | ARRAY_INQUIRIES | MODEL_INQUIRIES | {
    "allocated", "present", "sum", "product", "any", "all", "count", "minval", "maxval", "dot_product"}
_EXPRESSION_CONTAINERS = {"Actual_Arg_Spec_List", "Section_Subscript_List", "Actual_Arg_Spec", "Parenthesis"}


class DescriptorWindowUnsupported(CompilationError):
    """An unchanged original operation may run only after its owner closes.

    Original-source/token validation and aggregate analysis-budget failures
    never use this signal. It grants no descriptor or execution authority.
    """


def _kind(node):
    return type(node).__name__


def _children(node):
    return node if isinstance(node, (tuple, list)) else getattr(node, "content", getattr(node, "items", ())) or ()


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class _Check:
    def __init__(self, analysis):
        self.analysis = analysis
        self.nodes = self.operations = 0
        self.resolution_units = self.resolution_work = 0
        self.resolution_syntax = self.resolution_lookups = 0
        self.active, self.checked = [], {}
        self.functions = {}
        self.predicates = {}
        self.prefix_scopes, self.prefixes = {}, {}
        self.joined_directives, self.joins = {}, {}
        # Register source-derived group tokens only in this private view. Its
        # original procedures, bindings and parser nodes retain their exact
        # authority; no default summary or caller-owned proof registry changes.
        from compiler.scopes.segments import _native_analysis

        self.join_analysis = _native_analysis(analysis)
        # Proof builders must register in this view, not follow a projection
        # backlink and mutate its original analysis. Every accepted procedure
        # still passes _require_original against the retained source roles.
        self.join_analysis.__dict__.pop("_descriptor_source", None)
        self.environments, self.environment_states = {}, {}
        self.object_builder = _TypeBuilder(analysis)
        self.objects, self.object_schemas, self.object_mappings = {}, {}, {}

    def fixed_object(self, binding):
        if id(binding) not in self.objects:
            schema = self.object_builder.binding(binding)
            if id(schema.node) not in self.object_schemas:
                self.object_schemas[id(schema.node)] = self.object_builder.schema(schema)
            self.objects[id(binding)] = {
                "original_binding": binding.public(),
                "type_definition": self.object_schemas[id(schema.node)],
                "descriptor_evidence_only": True,
                "native_continuation_authority": False, "numerical_abi": False}
        return self.objects[id(binding)]

    def joined_groups(self, routine, graph, selected):
        roots = tuple(node for identity in selected for node in graph.source_nodes(identity))
        present = set()
        for original in _walk(roots):
            self.tick()
            present.add(id(original))

        def discover(items, peeled):
            for item in grouped_original_sequence(items, peeled):
                self.tick()
                if isinstance(item, NativeGroupCandidate):
                    try:
                        self.join_analysis.joined_completion(routine.qualified, item.original_nodes)
                    except CompilationError:
                        # Missing authority remains an error when its
                        # original directive is visited; an independently
                        # registered native-prefix proof remains usable.
                        with suppress(CompilationError):
                            self.join_analysis.numerical_joined_completion(routine.qualified, item.original_nodes)
                elif _kind(item) in {"Execution_Part", "If_Construct", "Block_Nonlabel_Do_Construct",
                                     "Associate_Construct"}:
                    discover(peeled.get(id(item), _children(item)), peeled)

        discover(roots, {})
        proofs = (*self.join_analysis._joined_completions.values(),
                  *self.join_analysis._numerical_completions.values())
        for proof in proofs:
            if not isinstance(proof, (NativeCompletionProof, NumericalCompletionProof)) or proof.procedure != routine.qualified:
                continue
            if isinstance(proof, NativeCompletionProof):
                proof.validate(self.join_analysis, routine.qualified, proof.selected_node_ids)
            else:
                proof.validate(self.join_analysis)
            originals = tuple(node for identity in proof.selected_node_ids for node in graph.source_nodes(identity))
            if not all(id(node) in present for node in originals):
                continue
            for original in _walk(originals):
                self.tick()
                if _kind(original) == "Comment" and str(original).lstrip().lower().startswith("!$omp"):
                    self.joined_directives[(routine.qualified, id(original))] = proof

    def directive(self, routine, original):
        proof = self.joined_directives.get((routine.qualified, id(original)))
        if proof is not None:
            if isinstance(proof, NativeCompletionProof):
                proof.validate(self.join_analysis, routine.qualified, proof.selected_node_ids)
            else:
                proof.validate(self.join_analysis)
            self.joins[proof.identity] = {
                "proof_identity": proof.identity, "procedure": proof.procedure,
                "structured_identity": proof.structured_identity,
                "selected_node_ids": list(proof.selected_node_ids),
                "proof_role": ("native_joined_completion" if isinstance(proof, NativeCompletionProof)
                               else "numerical_source_completion"),
                "descriptor_evidence_only": True, "whole_original_group_scanned": True,
                "retains_original_team_and_directives": True,
                "internal_cuts_authorized": False, "execution_authorized": False,
                "entry_definition_authority": False, "gpu_legality_established": False}
            return
        self.prefix_directive(routine)

    def prefix_directive(self, routine):
        proof = self.prefix_scopes.get(routine.qualified)
        if proof is None:
            raise DescriptorWindowUnsupported("descriptor window original OpenMP storage requires registered whole-closure authority")
        proof.validate(self.analysis, proof.procedure)
        public = proof.public()
        self.prefixes[proof.identity] = {
            "proof_identity": proof.identity, "procedure": proof.procedure,
            "covered_original_procedures": [item["procedure"] for item in public["closure"]],
            "descriptor_evidence_only": True,
            "whole_original_closure_scanned": True,
            "native_prefix_execution_requirements": {
                "requires_original_native_execution": public["requires_original_native_execution"],
                "requires_serial_caller": public["requires_serial_caller"],
                "requires_no_active_context": public["requires_no_active_context"]},
            "execution_authorized": False, "allocation_freshness_authority": False,
            "entry_definition_authority": False, "gpu_legality_established": False}

    def tick(self):
        self.nodes += 1
        if self.nodes > 65536:
            raise CompilationError("descriptor window original-source traversal budget exhausted")

    def resolution_tick(self, role):
        self.resolution_work += 1
        if role == "syntax":
            self.resolution_syntax += 1
        else:
            self.resolution_lookups += 1
        self.tick()

    def resolution(self, routine, originals, *, names=()):
        # Each original semantic unit retains the lookup/depth cap. Both its
        # syntax and resolution transitions spend the existing overall source
        # scan budget; splitting units cannot hide aggregate unbounded work.
        self.resolution_units += 1
        _validate_resolution(self.analysis, routine.scope, _walk(originals), tick=self.resolution_tick, names=names)

    def declaration_resolution(self, routine):
        specification = next((part for part in _children(routine.scope.node)
                              if _kind(part) == "Specification_Part"), None)
        declarations = (declaration for part in _children(specification)
                        for declaration in (_children(part) if _kind(part) == "Implicit_Part" else (part,)))
        for declaration in declarations:
            self.tick()
            if _kind(declaration) == "Type_Declaration_Stmt":
                dtype, _attributes, entities = declaration.items
                selector = dtype.items[1] if _kind(dtype) == "Intrinsic_Type_Spec" else None
                if selector is not None:
                    self.resolution(routine, selector)
                for entity in _children(entities):
                    self.tick()
                    if _kind(entity) == "Entity_Decl" and entity.items[3] is not None:
                        self.resolution(routine, entity.items[3])
            elif _kind(declaration) == "Parameter_Stmt":
                for definition in _children(declaration.items[1]):
                    self.tick()
                    if _kind(definition) == "Named_Constant_Def":
                        self.resolution(routine, definition.items[1])

    def expression(self, routine, node, depth=0):
        self.tick()
        if not isinstance(node, (Base, tuple, list)):
            return
        if depth > self.analysis.depth_limit:
            raise DescriptorWindowUnsupported("descriptor window expression depth exhausted")
        kind = _kind(node)
        child_depth = depth if isinstance(node, (tuple, list)) or kind in _EXPRESSION_CONTAINERS else depth + 1
        scope = self.analysis.source_scope_for(node, routine.scope)
        if (kind in {"Proc_Component_Ref", "Coindexed_Named_Object"}
                or kind.startswith("Defined_") and kind.endswith("_Op")):
            raise DescriptorWindowUnsupported("descriptor window association/operator effects require separate authority")
        if kind == "Data_Ref":
            access = component_access(self.analysis, scope, node)
            self.fixed_object(access.root_binding)
            if access.binding.dtype not in {"real", "integer", "logical"}:
                raise DescriptorWindowUnsupported("descriptor window whole-object effects require separate authority")
            for index in access.indices:
                self.expression(routine, index, depth + 1)
            return
        if kind == "Actual_Arg_Spec":
            self.expression(routine, node.items[1], child_depth)
            return
        if kind == "Name":
            binding = self.analysis._binding(scope, node)
            if binding is None or binding.attributes & _UNSAFE or binding.dtype.startswith("type("):
                raise DescriptorWindowUnsupported("descriptor window original expression storage is unproved: " + str(node))
            return
        if kind in {"Part_Ref", "Function_Reference", "Intrinsic_Function_Reference", "Structure_Constructor"}:
            name, args = node.items
            binding = self.analysis._binding(scope, name)
            predicates = [proof for proof in self.analysis._native_predicates.values()
                          if isinstance(proof, NativePredicateProof) and proof.procedure == routine.qualified
                          and proof._expression is node]
            if predicates:
                if len(predicates) != 1:
                    raise CompilationError("descriptor window requires unique registered original native predicate proof")
                proof = predicates[0].validate(self.analysis, routine.qualified, node)
                self.predicates[proof.identity] = proof.public()
                for actual in proof.actuals:
                    self.expression(routine, actual, depth + 1)
                return
            if kind == "Part_Ref" and binding is not None and binding.rank:
                self.expression(routine, name, depth + 1)
                self.expression(routine, args, depth + 1)
                return
            candidates = self.analysis._candidates(scope, name)
            if candidates:
                proofs = [proof for proof in self.analysis._native_functions.values()
                          if proof.procedure == routine.qualified and proof._expression is node]
                if len(proofs) != 1:
                    raise DescriptorWindowUnsupported("descriptor window requires registered original readonly source-function proof")
                proof = proofs[0].validate(self.analysis, routine.qualified, node,
                                           resolution_tick=self.resolution_tick)
                self.functions[proof.identity] = proof.public()
                for mapping in proof.resolved.mappings:
                    self.expression(routine, mapping.actual, depth + 1)
                return
            if (kind != "Intrinsic_Function_Reference" or binding is not None
                    or self.analysis._unknown_exports(scope) or str(name).lower() not in _INTRINSICS):
                raise DescriptorWindowUnsupported("descriptor window unknown function/escape effects: " + str(name))
            self.expression(routine, args, depth + 1)
            return
        for child in _children(node):
            self.expression(routine, child, child_depth)

    def visit(self, routine, graph, identities, depth=0):
        if depth > self.analysis.depth_limit:
            raise CompilationError("descriptor window control depth exhausted")
        for identity in identities:
            self.tick()
            node = graph.nodes[identity]
            if node.kind == "sequence":
                self.visit(routine, graph, node.children, depth)
            elif node.kind == "loop":
                self.visit(routine, graph, node.children, depth + 1)
            elif node.kind == "branch":
                for condition, body in node.alternatives:
                    self.visit(routine, graph, (body,) if condition is None else (condition, body), depth + 1)
            elif node.kind == "operation":
                self.operations += 1
                if self.operations > self.analysis.operation_limit:
                    raise CompilationError("descriptor window source operation budget exhausted")
                role = node.details.get("evaluation")
                for original in graph.source_nodes(identity):
                    if role == "directive":
                        self.directive(routine, original)
                    elif role == "condition":
                        self.resolution(routine, original)
                        self.expression(routine, original)
                    elif role == "loop_header":
                        control = original.items[1]
                        if control is None or control.items[1] is None:
                            raise DescriptorWindowUnsupported("descriptor window requires original counted loop control")
                        self.resolution(routine, control.items[1])
                        self.expression(routine, control.items[1])
                    elif _kind(original) == "Assignment_Stmt":
                        target, _equals, value = original.items
                        self.resolution(routine, (target, value))
                        scope = self.analysis.source_scope_for(original, routine.scope)
                        binding = self.analysis._binding(scope, target) if _kind(target) == "Name" else None
                        if binding is not None and "allocatable" in binding.attributes:
                            raise DescriptorWindowUnsupported("descriptor window whole allocatable assignment may replace storage")
                        self.expression(routine, target)
                        self.expression(routine, value)
                    elif _kind(original) != "Continue_Stmt":
                        raise DescriptorWindowUnsupported("descriptor window operation lacks descriptor/escape authority")
            elif node.kind == "call":
                self.operations += 1
                if self.operations > self.analysis.operation_limit:
                    raise CompilationError("descriptor window source operation budget exhausted")
                originals = graph.source_nodes(identity)
                if len(originals) != 1 or node.details.get("contract"):
                    raise DescriptorWindowUnsupported("descriptor window requires original direct source calls")
                call = originals[0]
                self.resolution(routine, call)
                scope = self.analysis.source_scope_for(call, routine.scope)
                resolved = resolve_source_call(self.analysis, scope, call)
                for mapping in resolved.mappings:
                    if (mapping.presence != "supplied"
                            or mapping.formal_binding.attributes & _UNSAFE
                            or mapping.binding is not None and mapping.binding.attributes & _UNSAFE):
                        raise DescriptorWindowUnsupported("descriptor window call association/escape authority is unavailable")
                    if mapping.source_object is not None:
                        self.fixed_object(mapping.formal_binding)
                        self.fixed_object(mapping.binding)
                        self.object_mappings[mapping.source_object.identity] = mapping.source_object.public()
                    else:
                        self.expression(routine, mapping.actual)
                self.procedure(resolved.procedure)
            elif node.kind == "boundary":
                originals = graph.source_nodes(identity)
                candidates = [proof for proof in self.join_analysis._native_environments.values()
                              if isinstance(proof, NativeEnvironmentProof) and len(originals) == 1
                              and proof.procedure == routine.qualified
                              and proof._call is originals[0]]
                if len(candidates) != 1:
                    raise DescriptorWindowUnsupported("descriptor window unknown selected storage effects: boundary"
                                           + (": " + node.details["reason"] if node.details.get("reason") else ""))
                proof = candidates[0].validate(self.join_analysis, routine.qualified, originals[0])
                self.resolution(routine, originals[0])
                self.operations += 1
                if self.operations > self.analysis.operation_limit:
                    raise CompilationError("descriptor window source operation budget exhausted")
                self.environments[proof.identity] = {**proof.public(), "descriptor_evidence_only": True,
                                                     "execution_authorized": False}
            else:
                raise DescriptorWindowUnsupported("descriptor window unknown selected storage effects: " + node.kind
                                       + (": " + node.details["reason"] if node.details.get("reason") else ""))

    def procedure(self, name):
        if name in self.active:
            raise CompilationError("descriptor window recursive call closure is unavailable")
        if name in self.checked:
            return
        if len(self.active) >= self.analysis.depth_limit or len(self.checked) + len(self.active) >= self.analysis.procedure_limit:
            raise CompilationError("descriptor window direct call closure budget exhausted")
        routine = self.analysis._require_original(name)
        # Namespace authority is separate from executable references. The
        # complete original child is still visited below, including every true
        # declaration bound; USE/definition spellings and unrelated CONTAINS
        # bodies are not synthetic whole-procedure lookup units.
        _validate_scope(self.analysis, routine.scope)
        proofs = [proof for proof in getattr(self.analysis, "_native_prefix_proofs", {}).values()
                  if isinstance(proof, NativePrefixProof) and proof.procedure == name]
        if proofs:
            if len(proofs) != 1:
                raise CompilationError("descriptor window requires unique registered native-prefix authority")
            proof = proofs[0].validate(self.analysis, name)
            for item in proof.public()["closure"]:
                self.prefix_scopes[item["procedure"]] = proof
        graph = self.analysis.structure(name)
        if not graph.available:
            raise CompilationError("descriptor window child bounded original graph is unavailable")
        self.joined_groups(routine, graph, (graph.root_id,))
        self.declaration_resolution(routine)
        for binding in routine.scope.bindings.values():
            if binding.dtype.startswith("type("):
                self.fixed_object(binding)
            elif binding.dtype not in {"real", "integer", "logical"} or binding.attributes & {"pointer", "volatile", "asynchronous"}:
                raise DescriptorWindowUnsupported("descriptor window child specification effects require separate authority")
            states = [proof for proof in self.join_analysis._native_environment_states.values()
                      if isinstance(proof, NativeLocalStateProof) and proof.procedure == name
                      and proof.binding is binding]
            if states:
                if len(states) != 1:
                    raise CompilationError("descriptor window native environment state requires unique original authority")
                state = states[0].validate(self.join_analysis, name, binding)
                self.environment_states[state.identity] = {**state.public(), "descriptor_evidence_only": True,
                                                          "execution_authorized": False}
            else:
                for axis in binding.shape_nodes:
                    self.resolution(routine, _children(axis))
                    self.expression(routine, _children(axis))
        for argument in routine.arguments:
            binding = routine.scope.bindings.get(argument)
            if binding is None or binding.attributes & _UNSAFE:
                raise DescriptorWindowUnsupported("descriptor window child formal storage is unproved")
            if "allocatable" in binding.attributes and binding.intent not in {"in", "inout"}:
                raise DescriptorWindowUnsupported("descriptor window allocatable OUT/unspecified entry may replace storage")
        self.active.append(name)
        self.visit(routine, graph, (graph.root_id,))
        self.active.pop()
        self.checked[name] = graph.identity


def _module_binding(analysis, routine, root, check):
    """Authenticate original visible allocation storage, never its contents."""
    parts = root.split("::")
    module = analysis.modules.get(parts[0]) if len(parts) == 2 and parts[0] != "argument" else None
    binding = module.bindings.get(parts[1]) if module is not None else None
    original = getattr(analysis, "_source_module_bindings", {}).get(root)
    if (binding is None or binding.root != root or "allocatable" not in binding.attributes
            or original is None or binding is not original[0]
            or _hash(binding.public()) != sha256(original[1].encode()).hexdigest()
            or binding.attributes & (_UNSAFE | {"target", "parameter"})
            or binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}
            or not 1 <= binding.rank <= 4):
        raise CompilationError("descriptor window requires original numeric module allocations: " + root)
    specification = next((part for part in _children(module.node)
                          if _kind(part) == "Specification_Part"), None)
    if any(_kind(node) in {"Common_Stmt", "Equivalence_Stmt", "Codimension_Attr_Spec"}
           or _kind(node) == "Comment" and str(node).lstrip().lower().startswith("!$omp")
           for node in _walk(specification)):
        raise CompilationError("descriptor window module association or OpenMP storage requires separate authority")
    if any(_kind(node) == "Entity_Decl" and str(node.items[0]).lower() == binding.name
           and node.items[2] is not None for node in _walk(specification)):
        raise CompilationError("descriptor window coarray module allocation requires separate authority")
    names, visited = set(), set()

    def collect(scope):
        if scope is None or id(scope) in visited:
            return
        check.tick()
        _validate_scope(analysis, scope)
        visited.add(id(scope))
        for name in (*scope.bindings, *scope.imports):
            check.tick()
            names.add(name)
        for imported in scope.wildcards:
            collect(analysis.modules.get(imported))
        for imported, _remote in scope.imports.values():
            collect(analysis.modules.get(imported))
        collect(scope.parent)

    collect(routine.scope)
    for name in sorted(names):
        check.tick()
        if analysis._binding(routine.scope, name) is binding:
            # Even an otherwise unused requested root must authenticate the
            # exact original import/rename/reexport visibility used to bind it.
            # This is one registration unit, not a synthetic source statement.
            check.resolution(routine, (), names=(name,))
            return binding, name, module
    raise CompilationError("descriptor window module allocation is not originally visible: " + root)


def _derive(analysis, procedure, identities, roots, module_roots=()):
    routine = analysis._require_original(procedure)
    graph = analysis.structure(procedure)
    if not graph.available:
        raise CompilationError("descriptor window requires a bounded original structured graph")
    if (not isinstance(identities, tuple) or not identities or len(identities) > analysis.operation_limit
            or any(not isinstance(identity, str) for identity in identities)):
        raise CompilationError("descriptor window requires exact original structured node IDs")
    _graph, selected = analysis._selected_source(procedure, identities)
    _validate_scope(analysis, routine.scope)
    sequences = []
    for parent in graph.nodes.values():
        if parent.kind == "sequence" and selected[0] in parent.children:
            first = parent.children.index(selected[0])
            if parent.children[first:first + len(selected)] == selected:
                sequences.append(parent.id)
    headers = [node.id for node in graph.nodes.values() if node.kind == "loop"
               and len(selected) == 1 and selected[0] in node.children
               and graph.nodes[selected[0]].details.get("evaluation") == "loop_header"]
    conditions = [node.id for node in graph.nodes.values() if node.kind == "branch"
                  and len(selected) == 1 and any(condition == selected[0] for condition, _body in node.alternatives)
                  and graph.nodes[selected[0]].details.get("evaluation") == "condition"]
    if len(sequences) != 1 and len(headers) != 1 and len(conditions) != 1:
        raise CompilationError("descriptor window requires one contiguous original immediate sequence")
    if (not isinstance(roots, tuple) or not isinstance(module_roots, tuple)
            or not (roots or module_roots) or len(roots) + len(module_roots) > analysis.operation_limit
            or any(not isinstance(root, str) for root in (*roots, *module_roots))
            or len(set((*roots, *module_roots))) != len(roots) + len(module_roots)):
        raise CompilationError("descriptor window requires bounded unique original formal roots")
    bindings = {binding.root: binding for binding in routine.scope.bindings.values() if binding.name in routine.arguments}
    states = []
    check = _Check(analysis)
    for root in roots:
        binding = bindings.get(root)
        if (binding is None or "allocatable" not in binding.attributes or binding.attributes & _UNSAFE
                or binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}
                or not 1 <= binding.rank <= 4):
            raise CompilationError("descriptor window requires original numeric allocatable formals: " + root)
        states.append({"resource": root, "resource_kind": "original_allocatable_formal",
                       "original_binding": binding.public(),
                       "stable": True, "original_descriptor_required": True,
                       "descriptor_changes_within_selection": False, "escapes_within_selection": False,
                       "requires_prior_noescape_and_current_allocation_proof": True,
                       "requires_original_allocation_presence_and_alias_guards": True,
                       "execution_requires_runtime_guard": True,
                       "entry_definition_authority": False})
    for root in module_roots:
        binding, spelling, module = _module_binding(analysis, routine, root, check)
        states.append({"resource": root, "resource_kind": "original_module_allocatable",
                       "original_binding": binding.public(), "original_visible_spelling": spelling,
                       "declaration_source": str(module.path),
                       "declaration_source_sha256": analysis.sources[str(module.path)],
                       "stable": True, "original_descriptor_required": True,
                       "descriptor_changes_within_selection": False, "escapes_within_selection": False,
                       "requires_prior_noescape_and_current_allocation_proof": True,
                       "requires_original_allocation_presence_and_alias_guards": True,
                       "requires_original_runtime_bounds": True,
                       "execution_requires_runtime_guard": True, "entry_definition_authority": False})
    check.active.append(procedure)
    check.joined_groups(routine, graph, selected)
    check.visit(routine, graph, selected)
    return {"schema_version": DESCRIPTOR_WINDOW_VERSION, "procedure": procedure,
            "source_identity": analysis.inputs.identity(), "structured_identity": graph.identity,
            "analysis_identity": graph.authority_identity, "selected_node_ids": list(selected),
            "parent_sequence": sequences[0] if sequences else None,
            "parent_node": sequences[0] if sequences else headers[0] if headers else conditions[0],
            "selection_role": ("original contiguous sequence" if sequences else
                               "original counted header" if headers else "original condition"),
            "original_guard": list(graph.nodes[selected[0]].guard),
            "resources": states, "module_roots": list(module_roots), "source_call_closure": check.checked,
            "native_source_function_requirements": list(check.functions.values()),
            "native_predicate_requirements": list(check.predicates.values()),
            "native_prefix_descriptor_requirements": list(check.prefixes.values()),
            "native_joined_descriptor_requirements": list(check.joins.values()),
            "native_environment_descriptor_requirements": list(check.environments.values()),
            "native_environment_state_requirements": list(check.environment_states.values()),
            "fixed_source_object_requirements": list(check.objects.values()),
            "source_object_mappings": list(check.object_mappings.values()),
            "source_operations": check.operations, "source_traversal_entries": check.nodes,
            "resolution_validation": {"mode": "original_semantic_units", "units": check.resolution_units,
                                      "work_entries": check.resolution_work,
                                      "syntax_entries": check.resolution_syntax,
                                      "lookup_transitions": check.resolution_lookups,
                                      "per_unit_lookup_limit": analysis.operation_limit,
                                      "depth_limit": analysis.depth_limit,
                                      "procedure_limit": analysis.procedure_limit,
                                      "aggregate_source_scan_limit": 65536},
            "prefix_effects_proved": False, "entry_definition_authority": False,
            "reachability_authority": False, "execution_authorized": False, "gpu_legality_established": False,
            "requires_close_before_excluded_tail": True, "freshness_by_address_or_shape": False,
            "limits": ["explicit original numeric formals/module allocations only",
                       "module allocation authority is conditional and exact to this selection and checked calls",
                       "no deferred groups or whole derived-object effects",
                       "fixed object mappings prove original storage/type only, never native continuation or numeric ABI",
                       "OpenMP requires registered whole joined-group or native-prefix proof and full descriptor rescan",
                       "excluded prefix and tail have no lifetime or definition authority",
                       "registered readonly source-function proof required before query",
                       "selected closure replacement, escapes, unknown calls and effects reject"]}


@dataclass(frozen=True)
class DescriptorWindowProof:
    procedure: str
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _selected: tuple[str, ...] = field(repr=False, compare=False)
    _roots: tuple[str, ...] = field(repr=False, compare=False)
    _module_roots: tuple[str, ...] = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)

    def public(self):
        return {"proof_identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis, procedure, selected_node_ids):
        if (analysis is not self._analysis or procedure != self.procedure
                or not isinstance(selected_node_ids, tuple) or selected_node_ids != self._selected
                or getattr(analysis, "_descriptor_window_proofs", {}).get(self.identity) is not self):
            raise CompilationError("descriptor window lacks registered original interval authority")
        if _freeze(_derive(analysis, procedure, self._selected, self._roots, self._module_roots)) != self._record:
            raise CompilationError("descriptor window original source authority changed")
        return self

    @property
    def module_roots(self):
        return self._module_roots

    def resource_state(self, root):
        self.validate(self._analysis, self.procedure, self._selected)
        states = {state["resource"]: state for state in _thaw(self._record)["resources"]}
        if root not in states:
            raise CompilationError("descriptor window resource was not proved")
        return states[root]


def prove_descriptor_window(analysis, procedure, selected_node_ids, requested_roots=(), *, module_roots=()):
    """Register selected-span stability, never prefix freshness or definitions."""
    record = _derive(analysis, procedure, selected_node_ids, requested_roots, module_roots)
    identity = _hash(record)
    registry = getattr(analysis, "_descriptor_window_proofs", None)
    if registry is None:
        registry = analysis._descriptor_window_proofs = {}
    if identity in registry:
        return registry[identity].validate(analysis, procedure, selected_node_ids)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("descriptor window proof registry budget exhausted")
    proof = DescriptorWindowProof(procedure, identity, analysis, selected_node_ids, requested_roots,
                                  module_roots, _freeze(record))
    registry[identity] = proof
    return proof
