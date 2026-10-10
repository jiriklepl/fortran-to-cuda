"""One-shot source-position state proofs; no execution or owner reopening.

The first contract selects contiguous original top-level statements. It reuses
the structured tree and ordered native effects, meets all prefix branches, and
declines allocation replacement and incomplete closures. Supplied invocation
facts are assumptions requiring independent admission. Registration guards,
runtime alias checks and numerical legality remain separate obligations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError

REACHED_WINDOW_VERSION = 1
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
    original = tuple(getattr(routine.execution, "content", ()))
    starts = [index for index in range(len(original) - len(selection) + 1)
              if all(node is original[index + offset] for offset, node in enumerate(selection))]
    if len(starts) != 1:
        raise CompilationError("reached window requires contiguous original top-level statements; nested entry is unavailable")
    graph, selected = analysis._selected_source(procedure, selection)
    children = graph.nodes[graph.root_id].children
    if not selected or any(identity not in children for identity in selected):
        raise CompilationError("reached window cannot split an original control or native joined group")
    first = children.index(selected[0])
    if tuple(children[first:first + len(selected)]) != selected:
        raise CompilationError("reached window selection is not one complete contiguous source span")
    if any(type(node).__name__ == "Comment" and str(node).lstrip().lower().startswith("!$omp")
           for node in walk(original[:starts[0] + len(selection)])):
        raise CompilationError("reached window direct OpenMP groups require separate whole-completion authority")
    if (not isinstance(roots, tuple) or not roots or len(roots) > analysis.operation_limit
            or any(not isinstance(root, str) for root in roots) or len(set(roots)) != len(roots)):
        raise CompilationError("reached window requires bounded unique canonical roots")
    _hash(facts)
    if (not isinstance(facts, dict) or type(facts.get("schema_version")) is not int or facts["schema_version"] != 1
            or facts.get("sources") != analysis.sources or not isinstance(facts.get("captures"), dict)
            or len(facts["captures"]) > analysis.operation_limit):
        raise CompilationError("reached window requires matching invocation-entry source facts")
    bindings, coverage = {}, {}
    for root in roots:
        binding, fact = _binding(analysis, routine, root), facts["captures"].get(root)
        if (not isinstance(fact, dict) or fact.get("storage") != "stable" or fact.get("escapes") is not False
                or fact.get("allocation_changes") is not False or type(fact.get("initialized")) is not str
                or fact["initialized"] not in {"whole", "none"}):
            raise CompilationError("reached window lacks stable entry coverage/no-escape facts: " + root)
        if "allocatable" in binding.attributes and binding.name not in routine.arguments:
            raise CompilationError("reached window local/module allocation requires separate selected-position lifetime authority: " + root)
        bindings[root], coverage[root] = binding, fact["initialized"]
    # Existing descriptor proof is intentionally stricter than this window.
    # A later replacement therefore declines, rather than inventing a selected
    # lifetime proof or restoring entry facts from a matching address.
    if any("allocatable" in binding.attributes for binding in bindings.values()):
        descriptors = analysis.descriptor_stability(procedure)
        stable = {item["resource"] for item in descriptors["resources"] if item["stable"]}
        if any("allocatable" in binding.attributes and root not in stable for root, binding in bindings.items()):
            raise CompilationError("reached window allocation replacement or escape requires source-relative lifetime authority")
    for root in graph.nodes[graph.entry_id].details.get("definition_changes", ()):
        if root in coverage:
            coverage[root] = "none"
    demands, visits, effects = [], 0, 0

    def leaf(identity, state):
        nonlocal effects
        summary = analysis.segment_summary(procedure, (identity,), capture_locals=True)
        if not summary["complete"] or not summary.get("effect_composition", {}).get("available"):
            raise CompilationError("reached window needs complete prefix/window effects: "
                                   + "; ".join(summary.get("reasons", ())))
        demands.append(summary["summary_identity"])
        outer = set(graph.nodes[identity].guard)
        for effect in summary["ordered_effects"]:
            effects += 1
            if effects > analysis.operation_limit:
                raise CompilationError("reached window ordered effect budget exhausted")
            if effect["kind"] not in _EFFECTS:
                raise CompilationError("reached window native environment/control effect needs distinct authority")
            root = effect["resource"]
            if root not in state:
                array = effect.get("rank", 0) or any(
                    view.get("actual_descriptor", {}).get("rank", 0)
                    for view in effect.get("view_chain", ()) if view.get("actual_descriptor") is not None)
                if array:
                    raise CompilationError("reached window untracked array needs definition/alias authority: " + root)
                continue  # Complete source evidence for distinct scalar state.
            if effect["kind"] == "definition_change":
                state[root] = "none"
            elif effect["kind"] == "read" and state[root] != "whole":
                raise CompilationError("reached window whole read follows incomplete definition: " + root)
            elif effect["kind"] == "overwrite":
                guarded = any(frame["procedure"] != procedure or frame["condition"] not in outer
                              for frame in effect.get("guard_frames", ()))
                if not guarded and _whole_view(effect):
                    state[root] = "whole"
        for root in summary["guaranteed_whole_overwrites"]:
            if root in state:
                state[root] = "whole"
        return state

    def transfer(identity, state, depth=0):
        nonlocal visits
        visits += 1
        if visits > graph.node_limit or depth > analysis.depth_limit:
            raise CompilationError("reached window control traversal budget exhausted")
        node = graph.nodes[identity]
        if node.kind == "sequence":
            for child in node.children:
                state = transfer(child, state, depth + 1)
            return state
        if node.kind == "branch":
            paths = []
            for condition, body in node.alternatives:
                path = dict(state)
                if condition is not None:
                    path = transfer(condition, path, depth + 1)
                paths.append(transfer(body, path, depth + 1))
            if not node.alternatives or node.alternatives[-1][0] is not None:
                paths.append(dict(state))
            return {root: "whole" if all(path[root] == "whole" for path in paths) else "none" for root in state}
        if node.kind == "boundary":
            raise CompilationError("reached window unknown effects or deferred native group lacks window authority")
        if node.kind not in {"operation", "call", "loop", "associate"}:
            raise CompilationError("reached window source node requires distinct control authority")
        return leaf(identity, state)

    for identity in children[:first]:
        coverage = transfer(identity, coverage)
    at_entry = dict(coverage)
    for identity in selected:
        coverage = transfer(identity, coverage)
    states = [{"resource": root, "initialized": at_entry[root], "definition_coverage": at_entry[root],
               "coverage_on_return": coverage[root], "storage": "original current binding",
               "escapes": False, "allocation_changes": False,
               "descriptor_authority": "complete original effects; existing whole-procedure allocatable proof when needed",
               "requires_original_allocation_presence_guard": "allocatable" in bindings[root].attributes,
               "requires_runtime_layout_and_alias_guards": True,
               "address_or_shape_freshness_inferred": False} for root in roots]
    return {"schema_version": REACHED_WINDOW_VERSION, "procedure": procedure,
            "structured_identity": graph.identity, "source_identity": analysis.inputs.identity(),
            "analysis_identity": graph.authority_identity, "selected_node_ids": list(selected),
            "prefix_node_ids": list(children[:first]), "entry_facts_identity": _hash(facts),
            "entry_facts_authority": "conditional on supplied invocation-entry facts; independent admission required",
            "source_demands": demands, "control_visits": visits, "ordered_effects": effects,
            "states": states, "lifecycle": "one invocation; dormant -> active -> spent; no reopening",
            "native_only_proof": True, "gpu_legality_established": False,
            "execution_authorized": False, "definition_facts_reset": False,
            "requires_original_serial_caller_guard": True,
            "limits": ["top-level contiguous original selections only", "whole/none coverage only",
                       "direct OpenMP groups require a separate complete-group selection contract",
                       "every payload array dependency must be included; runtime nonalias guards remain required",
                       "allocation replacement and incomplete closures rejected",
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
