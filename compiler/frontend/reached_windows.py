"""One-shot source-position state proofs; no execution or owner reopening.

The contract selects contiguous original statements at entry or in the first
traversal of one serial counted loop. It reuses ordered native effects and
source-relative descriptor proofs, meets prefix branches, and declines unknown
closures. Supplied invocation facts require independent admission. Registration,
runtime alias checks, mandatory closure and numerical legality remain separate.
"""

from __future__ import annotations

import json
from contextlib import suppress
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.native_sections import NativeAxis, NativeRectangle
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError

REACHED_WINDOW_VERSION = 4
_EFFECTS = {"read", "write", "overwrite", "descriptor_read", "definition_change"}
_FORBIDDEN = {"pointer", "optional", "volatile", "asynchronous", "value"}


def _hash(value):
    try:
        return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError, RecursionError) as error:
        raise CompilationError("reached window requires finite source facts") from error


def _binding(analysis, routine, root):
    choices = {id(binding): binding for owner in (routine.scope, *analysis.modules.values())
               for binding in owner.bindings.values() if binding.root == root}
    if len(choices) != 1:
        raise CompilationError("reached window requires an original visible array binding: " + root)
    binding = next(iter(choices.values()))
    if (binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}
            or not 1 <= binding.rank <= 4 or binding.attributes & _FORBIDDEN):
        raise CompilationError("reached window binding association or type is unsupported: " + root)
    return binding


def _whole_view(effect):
    """Only full assumed-shape forwarding preserves a formal whole overwrite."""
    for view in effect.get("view_chain", ()):
        descriptor = view.get("formal_descriptor", {})
        if (view.get("storage") != "whole" or view.get("presence") != "supplied"
                or view.get("section") is not None or not descriptor.get("rank")
                or any(axis.get("kind") != "Assumed_Shape_Spec" for axis in descriptor.get("shape", ()))
                or len(descriptor.get("shape", ())) != descriptor.get("rank")):
            return False
    return True


def _coverage_kind(rectangles):
    if any(not box.iterations and all(axis.lower is None and axis.upper is None for axis in box.axes)
           for box in rectangles):
        return "whole"
    return "sections" if rectangles else "none"


def _whole_coverage(binding):
    # Geometry only: invocation facts retain their independent admission.
    return (NativeRectangle(tuple(NativeAxis() for _ in range(binding.rank)), binding.name),)


def _position(analysis, procedure, selection):
    """Authenticate an immediate sequence without expanding its excluded tail."""
    try:
        graph, selected = analysis._selected_source(procedure, selection)
    except CompilationError as error:
        raise CompilationError("reached window requires contiguous original source authority: " + str(error)) from error
    parents = []
    for node in graph.nodes.values():
        if node.kind != "sequence" or not selected or selected[0] not in node.children:
            continue
        first = node.children.index(selected[0])
        if tuple(node.children[first:first + len(selected)]) == selected:
            parents.append((node, first))
    if len(parents) != 1:
        raise CompilationError("reached window requires one contiguous original immediate sequence")
    parent, first = parents[0]
    root = graph.nodes[graph.root_id]
    prefix, controls = list(parent.children[:first]), []
    if parent.id != root.id:
        loops = [node for node in graph.nodes.values()
                 if node.kind == "loop" and node.children and node.children[-1] == parent.id
                 and node.id in root.children]
        if len(loops) != 1:
            raise CompilationError("reached window nested entry requires a first-traversal top-level counted loop")
        loop = loops[0]
        originals = graph.source_nodes(loop.id)
        if len(originals) != 1 or type(originals[0]).__name__ != "Block_Nonlabel_Do_Construct":
            raise CompilationError("reached window requires original serial counted DO syntax")
        contents = tuple(originals[0].content)
        syntax = [node for node in contents if type(node).__name__ != "Comment"]
        header = syntax[0] if syntax else None
        control = getattr(header, "items", (None, None))[1] if header is not None else None
        if (type(header).__name__ != "Nonlabel_Do_Stmt" or type(control).__name__ != "Loop_Control"
                or control.items[1] is None or type(syntax[-1]).__name__ != "End_Do_Stmt"
                or any(type(node).__name__ == "Comment" and str(node).lstrip().lower().startswith("!$")
                       for node in contents)):
            raise CompilationError("reached window requires an unchanged serial counted header and join")
        headers = loop.children[:-1]
        if (len(headers) != 1 or graph.nodes[headers[0]].details.get("evaluation") != "loop_header"):
            raise CompilationError("reached window cannot split an attached OpenMP construct")
        prefix = [*root.children[:root.children.index(loop.id)], *headers, *prefix]
        controls = [loop.id]
    return graph, selected, tuple(prefix), tuple(controls)


def _derive(analysis, procedure, selection, roots, facts):
    routine = analysis._require_original(procedure)
    graph = analysis.structure(procedure)
    if not graph.available:
        raise CompilationError("reached window requires a complete bounded original structure")
    if any(type(node).__name__ in {"Common_Stmt", "Equivalence_Stmt"} for node in walk(routine.scope.node)):
        raise CompilationError("reached window storage association requires separate alias authority")
    if (not isinstance(selection, tuple) or not selection
            or len(selection) > analysis.operation_limit or any(isinstance(node, str) for node in selection)):
        raise CompilationError("reached window requires an exact original statement tuple")
    graph, selected, prefix, controls = _position(analysis, procedure, selection)
    if any(type(node).__name__ == "Comment" and str(node).lstrip().lower().startswith("!$omp")
           for identity in (*prefix, *selected) for node in walk(graph.source_nodes(identity))):
        raise CompilationError("reached window direct OpenMP groups require separate whole-completion authority")
    if (not isinstance(roots, tuple) or not roots or len(roots) > analysis.operation_limit
            or any(not isinstance(root, str) for root in roots) or len(set(roots)) != len(roots)):
        raise CompilationError("reached window requires bounded unique canonical roots")
    _hash(facts)
    if (not isinstance(facts, dict) or type(facts.get("schema_version")) is not int or facts["schema_version"] != 1
            or facts.get("sources") != analysis.sources or not isinstance(facts.get("captures"), dict)
            or len(facts["captures"]) > analysis.operation_limit):
        raise CompilationError("reached window requires matching invocation-entry source facts")
    bindings, coverage, module_roots = {}, {}, []
    for root in roots:
        binding, fact = _binding(analysis, routine, root), facts["captures"].get(root)
        if (not isinstance(fact, dict) or fact.get("storage") != "stable" or fact.get("escapes") is not False
                or fact.get("allocation_changes") is not False or type(fact.get("initialized")) is not str
                or fact["initialized"] not in {"whole", "none"}):
            raise CompilationError("reached window lacks stable entry coverage/no-escape facts: " + root)
        if "allocatable" in binding.attributes and binding.name not in routine.arguments:
            defining = [owner for owner in analysis.modules.values()
                        if owner.bindings.get(binding.name) is binding]
            if len(defining) != 1:
                raise CompilationError("reached window local allocation requires separate selected-position lifetime authority: " + root)
            module_roots.append(root)
        bindings[root] = binding
        coverage[root] = _whole_coverage(binding) if fact["initialized"] == "whole" else ()
    descriptor_roots = tuple(root for root, binding in bindings.items()
                             if "allocatable" in binding.attributes and root not in module_roots)
    module_roots = tuple(module_roots)
    for root in graph.nodes[graph.entry_id].details.get("definition_changes", ()):
        if root in coverage:
            coverage[root] = ()
    demands, descriptor_demands, native_prefix_demands, definition_demands = [], [], [], []
    visits, effects = 0, 0

    def leaf(identity, state, *, before_owner=False):
        nonlocal effects
        selected_analysis, native_prefix, native_reads = analysis, None, set()
        # Native prefix effects are conditional on the original call finishing
        # before any owner exists. They never grant a managed capture or make
        # an incomplete ordinary summary complete in the original analysis.
        if before_owner and graph.nodes[identity].kind == "call":
            initial = analysis.segment_summary(procedure, (identity,), capture_locals=True)
            if not initial["complete"] or not initial["native_completion"]["available"]:
                from compiler.frontend.native_prefix import _fork, prove_native_prefix
                child = graph.nodes[identity].details.get("procedure")
                if child not in analysis.routines or graph.nodes[identity].details.get("contract"):
                    raise CompilationError("reached window native prefix needs a complete original source call")
                native_prefix = prove_native_prefix(analysis, child).validate(analysis, child)
                native_reads = set(native_prefix.readonly_module_resources)
                selected_analysis = _fork(analysis, native_reads)
                # Any descriptor consumer authenticates its child token in the
                # same private analysis, never in the original GPU authority.
                prove_native_prefix(selected_analysis, child)
                # Rebuild readonly scalar-expression tokens in this exact
                # analysis. The prefix proof's deeper detached fork cannot
                # lend its registered function tokens to a descriptor query.
                child_effects = selected_analysis.summarize(child)
                if not child_effects["complete"]:
                    raise CompilationError("reached window native prefix child proof is incomplete")
                native_prefix_demands.append(native_prefix.identity)
        descriptor = None
        if descriptor_roots or module_roots:
            from compiler.frontend.descriptor_windows import prove_descriptor_window
            descriptor = prove_descriptor_window(selected_analysis, procedure, (identity,), descriptor_roots,
                                                 module_roots=module_roots)
            descriptor_demands.append(descriptor.identity)
        summary = selected_analysis.segment_summary(procedure, (identity,), capture_locals=True,
                                                    descriptor_window=descriptor)
        if not summary["complete"] or not summary.get("effect_composition", {}).get("available"):
            raise CompilationError("reached window needs complete prefix/window effects: "
                                   + "; ".join(summary.get("reasons", ())))
        demands.append(summary["summary_identity"])
        outer = set(graph.nodes[identity].guard)
        scalar_writes = set()
        for effect in summary["ordered_effects"]:
            effects += 1
            if effects > analysis.operation_limit:
                raise CompilationError("reached window ordered effect budget exhausted")
            if effect["kind"] not in _EFFECTS:
                raise CompilationError("reached window native environment/control effect needs distinct authority")
            root = effect["resource"]
            if root not in state:
                if (before_owner and native_prefix is not None and root in native_reads
                        and effect["kind"] in {"read", "descriptor_read"}):
                    continue  # Unchanged native read; no device capture or coverage.
                array = effect.get("rank", 0) or any(
                    view.get("actual_descriptor", {}).get("rank", 0)
                    for view in effect.get("view_chain", ()) if view.get("actual_descriptor") is not None)
                if array:
                    raise CompilationError("reached window untracked array needs definition/alias authority: " + root)
                if effect["kind"] in {"write", "overwrite", "definition_change"}:
                    scalar_writes.add(root)
                continue  # Complete source evidence for distinct scalar state.

        # Exact assignment/affine-loop effects retain physical source sections
        # and ordering. Calls and unsupported native syntax keep the original
        # conservative whole-definition contract below.
        from compiler.frontend.native_definitions import (
            invalidate_rectangles,
            prove_native_definitions,
        )
        definition = None
        needs_sections = any(effect["resource"] in state
                             and _coverage_kind(state[effect["resource"]]) != "whole"
                             and effect["kind"] in {"read", "write", "overwrite"}
                             for effect in summary["ordered_effects"])
        if not before_owner and needs_sections and graph.nodes[identity].kind in {"operation", "loop"}:
            with suppress(CompilationError):
                definition = prove_native_definitions(selected_analysis, procedure, graph.source_nodes(identity),
                                                      descriptor_window=descriptor)
        if definition is not None:
            definition_demands.append(definition.identity)
            resources = {item["resource"] for item in definition.public()["resources"]}
            transferred = definition.transfer({root: state[root] for root in resources})
            if transferred.uncovered_reads:
                raise CompilationError("reached window section read follows incomplete definition: "
                                       + transferred.uncovered_reads[0].resource)
            state.update(transferred.definitions)
        else:
            for effect in summary["ordered_effects"]:
                root = effect["resource"]
                if root not in state:
                    continue
                if effect["kind"] == "definition_change":
                    state[root] = ()
                elif effect["kind"] == "read" and _coverage_kind(state[root]) != "whole":
                    raise CompilationError("reached window whole read follows incomplete definition: " + root)
                elif effect["kind"] == "overwrite":
                    guarded = any(frame["procedure"] != procedure or frame["condition"] not in outer
                                  for frame in effect.get("guard_frames", ()))
                    if not guarded and _whole_view(effect):
                        state[root] = _whole_coverage(bindings[root])
        for root in summary["guaranteed_whole_overwrites"]:
            if root in state:
                state[root] = _whole_coverage(bindings[root])
        # A later value of a bound does not describe the points written using
        # its earlier value. This applies even to arrays not accessed here.
        if scalar_writes:
            state = {root: invalidate_rectangles(boxes, tuple(sorted(scalar_writes)))
                     for root, boxes in state.items()}
        return state

    def transfer(identity, state, depth=0, *, before_owner=False):
        nonlocal visits
        visits += 1
        if visits > graph.node_limit or depth > analysis.depth_limit:
            raise CompilationError("reached window control traversal budget exhausted")
        node = graph.nodes[identity]
        if node.kind == "sequence":
            for child in node.children:
                state = transfer(child, state, depth + 1, before_owner=before_owner)
            return state
        if node.kind == "branch":
            paths = []
            for condition, body in node.alternatives:
                path = dict(state)
                if condition is not None:
                    path = transfer(condition, path, depth + 1, before_owner=before_owner)
                paths.append(transfer(body, path, depth + 1, before_owner=before_owner))
            if not node.alternatives or node.alternatives[-1][0] is not None:
                paths.append(dict(state))
            from compiler.frontend.native_definitions import intersect_rectangles
            joined = dict(paths[0])
            for path in paths[1:]:
                joined = {root: intersect_rectangles(boxes, path[root]) for root, boxes in joined.items()}
            return joined
        if node.kind == "boundary":
            raise CompilationError("reached window unknown effects or deferred native group lacks window authority")
        if node.kind not in {"operation", "call", "loop", "associate"}:
            raise CompilationError("reached window source node requires distinct control authority")
        return leaf(identity, state, before_owner=before_owner)

    for identity in prefix:
        coverage = transfer(identity, coverage, before_owner=True)
    at_entry = dict(coverage)
    if any(_coverage_kind(boxes) == "sections" for boxes in at_entry.values()):
        raise CompilationError("reached prefix partial definitions require source-position section registration")
    for identity in selected:
        coverage = transfer(identity, coverage)
    states = [{"resource": root, "initialized": _coverage_kind(at_entry[root]),
               "definition_coverage": _coverage_kind(at_entry[root]),
               "coverage_on_return": _coverage_kind(coverage[root]),
               "defined_sections_on_return": [box.public() for box in coverage[root]]
                   if _coverage_kind(coverage[root]) == "sections" else [],
               "storage": "original current binding",
               "escapes": False, "allocation_changes": False,
               "descriptor_authority": "registered source-relative prefix/window proofs; excluded tail requires close",
               "requires_original_allocation_presence_guard": "allocatable" in bindings[root].attributes,
               "resource_kind": "original_module_allocatable" if root in module_roots else "original_entry_capture",
               "requires_runtime_layout_and_alias_guards": True,
               "address_or_shape_freshness_inferred": False} for root in roots]
    return {"schema_version": REACHED_WINDOW_VERSION, "procedure": procedure,
            "structured_identity": graph.identity, "source_identity": analysis.inputs.identity(),
            "analysis_identity": graph.authority_identity, "selected_node_ids": list(selected),
            "prefix_node_ids": list(prefix), "first_traversal_controls": list(controls),
            "descriptor_window_identities": descriptor_demands, "entry_facts_identity": _hash(facts),
            "module_descriptor_resources": list(module_roots),
            "native_prefix_proof_identities": native_prefix_demands,
            "native_definition_proof_identities": definition_demands,
            "entry_facts_authority": "conditional on supplied invocation-entry facts; independent admission required",
            "source_demands": demands, "control_visits": visits, "ordered_effects": effects,
            "states": states, "lifecycle": "one invocation; dormant -> active -> spent; no reopening",
            "native_only_proof": True, "gpu_legality_established": False,
            "execution_authorized": False, "definition_facts_reset": False,
            "requires_original_serial_caller_guard": True,
            "mandatory_close": "before the first excluded operation; before any enclosing loop backedge",
            "activation": "first traversal only; no live context during the native prefix",
            "limits": ["one original immediate sequence at top level or first traversal of one top-level counted loop",
                       "exact original assignment/perfect affine-loop partial definitions within the selected window",
                       "partial prefix definitions require separate source-position registration",
                       "calls and unsupported native footprints retain conservative whole-definition requirements",
                       "direct OpenMP groups require a separate complete-group selection contract",
                       "every payload array dependency must be included; runtime nonalias guards remain required",
                       "allocation replacement within prefix/window and incomplete closures rejected",
                       "runtime registration/layout/alias guards and numerical legality required separately"]}


@dataclass(frozen=True)
class ReachedWindowProof:
    procedure: str
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _roots: tuple[str, ...] = field(repr=False, compare=False)
    _facts: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)

    def public(self):
        return {"proof_identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis, procedure, original_selection):
        if (analysis is not self._analysis or procedure != self.procedure
                or not isinstance(original_selection, tuple) or len(original_selection) != len(self._selection)
                or any(left is not right for left, right in zip(original_selection, self._selection, strict=True))
                or getattr(analysis, "_reached_window_proofs", {}).get(self.identity) is not self):
            raise CompilationError("reached window lacks registered exact original source authority")
        if _derive(analysis, procedure, self._selection, self._roots, _thaw(self._facts)) != _thaw(self._record):
            raise CompilationError("reached window source-position authority changed")
        return self

    def capture_state(self, canonical_root):
        self.validate(self._analysis, self.procedure, self._selection)
        state = next((state for state in _thaw(self._record)["states"] if state["resource"] == canonical_root), None)
        if state is None:
            raise CompilationError("reached window capture root was not proved")
        return state


def prove_reached_window(analysis, procedure, original_selection, requested_roots, entry_definition_facts):
    """Transfer supplied entry facts; their truth requires separate admission.

    Registration authenticates original source and transfer, not invocation
    state, execution legality, owner creation or runtime resources.
    """
    record = _derive(analysis, procedure, original_selection, requested_roots, entry_definition_facts)
    identity = _hash(record)
    registry = getattr(analysis, "_reached_window_proofs", None)
    if registry is None:
        registry = analysis._reached_window_proofs = {}
    if identity in registry:
        return registry[identity].validate(analysis, procedure, original_selection)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("reached window proof registry budget exhausted")
    proof = ReachedWindowProof(procedure, identity, analysis, original_selection, requested_roots,
                               _freeze(entry_definition_facts), _freeze(record))
    registry[identity] = proof
    return proof
