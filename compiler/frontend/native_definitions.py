"""Original ordered partial definitions, independent of storage admission.

This source adapter retains original IF paths and typed native rectangles. It
does not evaluate bounds, flatten conditional may-writes into definitions, or
authorize runtime execution. Transfer is conditional on separately admitted
entry rectangles and valid original native execution. Runtime coverage/layout,
allocation lifetime, aliases and numerical legality remain separate proofs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from types import MappingProxyType

from fparser.two.utils import walk

from compiler.frontend.native_sections import RECTANGLE_LIMIT, NativeRectangle, NativeSections
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError

NATIVE_DEFINITION_VERSION = 2


def _hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _box_key(box):
    return {key: value for key, value in box.public().items() if key != "source_access"}


def _dependencies(box):
    roots = set()

    def bound(value):
        if value is not None:
            roots.update(item.resource for item in value.dependencies if item.kind == "scalar_read")
            for child in value.children:
                bound(child)

    for axis in box.axes:
        bound(axis.lower)
        bound(axis.upper)
    for iteration in box.iterations:
        bound(iteration.lower)
        bound(iteration.upper)
    return roots


def _covers(definition, read):
    """One source-symbolic rectangle, never a bounding hull of several faces."""
    if len(definition.axes) != len(read.axes):
        return False
    # A repeated point write is absent when its original loop is empty. Such
    # a point cannot cover a later unconditional read merely by matching axes.
    guards = [item.public() for item in read.iterations]
    if any(item.public() not in guards for item in definition.iterations):
        return False
    for left, right in zip(definition.axes, read.axes, strict=True):
        if left.lower is None and left.upper is None:
            continue
        if left.public() == right.public():
            continue
        bounds = left.lower, left.upper, right.lower, right.upper
        if (all(item is not None and item.kind == "literal" for item in bounds)
                and left.lower.value <= right.lower.value <= right.upper.value <= left.upper.value):
            continue
        return False
    return True


def _unique(boxes):
    result = {}
    for box in boxes:
        result.setdefault(_hash(_box_key(box)), box)
    if len(result) > RECTANGLE_LIMIT:
        raise CompilationError("native definition rectangle budget exhausted")
    return tuple(result.values())


def _rectangles(boxes):
    if (not isinstance(boxes, tuple) or len(boxes) > RECTANGLE_LIMIT
            or any(not isinstance(box, NativeRectangle) or not 1 <= len(box.axes) <= 4 for box in boxes)
            or len({len(box.axes) for box in boxes}) > 1):
        raise CompilationError("native definition geometry requires bounded same-rank rectangle tuples")
    return boxes


def rectangle_covers(definition, read):
    """Conditional geometry only; neither rectangle gains source authority."""
    _rectangles((definition,))
    _rectangles((read,))
    return _covers(definition, read)


def rectangle_dependencies(box):
    """Original scalar bounds whose writes invalidate this symbolic box."""
    _rectangles((box,))
    return tuple(sorted(_dependencies(box)))


def merge_rectangles(*groups):
    """Deduplicate a bounded union without inferring its bounding hull."""
    if len(groups) > RECTANGLE_LIMIT:
        raise CompilationError("native definition geometry group budget exhausted")
    for boxes in groups:
        _rectangles(boxes)
    boxes = tuple(box for group in groups for box in group)
    if len({len(box.axes) for box in boxes}) > 1:
        raise CompilationError("native definition geometry requires same-rank rectangles")
    return _unique(boxes)


def intersect_rectangles(left, right):
    """Conservative all-path meet; partial overlaps are not split or widened."""
    _rectangles(left)
    _rectangles(right)
    if left and right and len(left[0].axes) != len(right[0].axes):
        raise CompilationError("native definition geometry requires same-rank rectangles")
    return _unique(box for boxes, others in ((left, right), (right, left)) for box in boxes
                   if any(_covers(other, box) for other in others))


def invalidate_rectangles(boxes, scalar_writes):
    """Apply to every tracked root, including roots not read by this leaf."""
    _rectangles(boxes)
    if (not isinstance(scalar_writes, tuple) or len(scalar_writes) > 256
            or any(not isinstance(root, str) or not root for root in scalar_writes)):
        raise CompilationError("native definition invalidation requires bounded original scalar roots")
    return tuple(box for box in boxes if not _dependencies(box).intersection(scalar_writes))


@dataclass(frozen=True)
class NativeDefinitionStep:
    node_id: str
    kind: str
    guard: tuple[str, ...]
    sections: NativeSections | None = None
    scalar_writes: tuple[str, ...] = ()
    children: tuple[NativeDefinitionStep, ...] = ()
    alternatives: tuple[tuple[NativeDefinitionStep | None, NativeDefinitionStep], ...] = ()

    def public(self):
        return {"node_id": self.node_id, "kind": self.kind, "guard": list(self.guard),
                "scalar_writes": list(self.scalar_writes),
                "sections": self.sections.public() if self.sections is not None else None,
                "children": [child.public() for child in self.children],
                "alternatives": [{"condition": condition.public() if condition is not None else None,
                                   "body": body.public()} for condition, body in self.alternatives],
                "must_definition_scope": "only when this original source leaf is reached"}


@dataclass(frozen=True)
class NativeDefinitionRead:
    node_id: str
    guard: tuple[str, ...]
    resource: str
    rectangle: NativeRectangle

    def public(self):
        return {"node_id": self.node_id, "guard": list(self.guard), "resource": self.resource,
                "rectangle": self.rectangle.public(),
                "requirement": "defined before the original reached read; no early bound evaluation"}


@dataclass(frozen=True)
class NativeDefinitionTransfer:
    definitions: object
    uncovered_reads: tuple[NativeDefinitionRead, ...]

    @property
    def reads_covered(self):
        return not self.uncovered_reads

    def public(self):
        return {"defined_sections": {root: [box.public() for box in boxes]
                                     for root, boxes in self.definitions.items()},
                "uncovered_reads": [read.public() for read in self.uncovered_reads],
                "reads_covered": self.reads_covered, "execution_authorized": False,
                "entry_rectangles_require_independent_admission": True}


def _derive(analysis, procedure, selection, include_entry, descriptor_window=None):
    routine = analysis._require_original(procedure)
    if (not isinstance(selection, tuple) or not selection or len(selection) > analysis.operation_limit
            or any(isinstance(item, str) for item in selection) or type(include_entry) is not bool):
        raise CompilationError("native definitions require bounded exact original statements")
    graph = analysis.structure(procedure)
    # Single-line IF has both condition and statement identities. Select the
    # exact original statement role, without borrowing its condition alias.
    identities = tuple(graph.node_id(original) for original in selection)
    graph, identities = analysis._selected_source(procedure, identities)
    if not graph.available or graph.deferred_native_descendants(identities):
        raise CompilationError("native definitions require an ordinary complete original structure")
    parents = [node for node in graph.nodes.values() if node.kind == "sequence"
               and identities[0] in node.children and tuple(node.children[
                   node.children.index(identities[0]):node.children.index(identities[0]) + len(identities)]) == identities]
    if len(parents) != 1:
        raise CompilationError("native definitions require contiguous original immediate statements")
    if include_entry and identities != graph.nodes[graph.root_id].children:
        raise CompilationError("native definition entry changes require the complete original body")
    descriptor_roots, module_roots = (), ()
    if descriptor_window is not None:
        analysis._validate_descriptor_window(procedure, identities, descriptor_window)
        states = descriptor_window.public()["resources"]
        descriptor_roots = tuple(state["resource"] for state in states
                                 if state["resource_kind"] == "original_allocatable_formal")
        module_roots = tuple(state["resource"] for state in states
                             if state["resource_kind"] == "original_module_allocatable")
    if any(type(node).__name__ in {"Common_Stmt", "Equivalence_Stmt"} for node in walk(routine.scope.node)):
        raise CompilationError("native definitions storage association requires separate authority")
    visits, leaves, rectangle_count = 0, 0, 0
    resources, demands, leaf_descriptors = {}, [], []

    def step(identity, depth=0):
        nonlocal visits, leaves, rectangle_count
        visits += 1
        if visits > graph.node_limit or depth > analysis.depth_limit:
            raise CompilationError("native definition control traversal budget exhausted")
        node = graph.nodes[identity]
        if node.kind == "sequence":
            return NativeDefinitionStep(identity, "sequence", node.guard,
                                        children=tuple(step(child, depth + 1) for child in node.children))
        if node.kind == "branch":
            return NativeDefinitionStep(identity, "branch", node.guard,
                alternatives=tuple((step(condition, depth + 1) if condition is not None else None,
                                    step(body, depth + 1)) for condition, body in node.alternatives))
        originals = graph.source_nodes(identity)
        if node.kind not in {"operation", "loop"} or node.details.get("evaluation") == "directive":
            raise CompilationError("native definitions require original assignment/IF/affine DO source only")
        leaves += 1
        if leaves > analysis.operation_limit:
            raise CompilationError("native definition leaf budget exhausted")
        if node.kind == "loop":
            # One assignment in a perfect nest avoids claiming per-iteration
            # RAW/loop-carried definitions from a flattened whole-loop union.
            current = originals[0]
            while type(current).__name__ == "Block_Nonlabel_Do_Construct":
                contents = tuple(item for item in current.content if type(item).__name__ != "Comment")
                if len(contents) != 3 or type(contents[-1]).__name__ != "End_Do_Stmt":
                    raise CompilationError("native definition affine DO requires one perfect-nest assignment")
                current = contents[1]
            if type(current).__name__ != "Assignment_Stmt":
                raise CompilationError("native definition affine DO requires one perfect-nest assignment")
        elif node.details.get("evaluation") != "condition" and any(
                type(item).__name__ not in {"Assignment_Stmt", "Continue_Stmt"} for item in originals):
            raise CompilationError("native definitions require original assignment or condition source")
        descriptor = None
        if descriptor_window is not None:
            from compiler.frontend.descriptor_windows import prove_descriptor_window
            # SourceEffects requires exact selection authority. Re-prove only
            # this original descendant, with the same source-authenticated
            # roots; the enclosing token is never used as a leaf alias.
            descriptor = prove_descriptor_window(analysis, procedure, (identity,), descriptor_roots,
                                                 module_roots=module_roots)
            leaf_descriptors.append({"node_id": identity, "proof_identity": descriptor.identity})
        summary = analysis.segment_summary(procedure, (identity,), capture_locals=True,
                                           descriptor_window=descriptor)
        if (not summary["complete"] or not summary.get("effect_composition", {}).get("available")
                or not summary["native_completion"]["available"]
                or any(effect["kind"] not in {"read", "write", "overwrite", "descriptor_read"}
                       for effect in summary["ordered_effects"])):
            raise CompilationError("native definitions require complete original ordered effects and completion")
        sections = analysis.native_sections_for_nodes(procedure, (identity,), capture_locals=True,
                                                     descriptor_window=descriptor)
        if not sections.available:
            raise CompilationError("native definition sections unavailable: " + str(sections.reason))
        scalar_writes, iterators = set(), set()
        for original in originals:
            for syntax in walk(original):
                target = None
                if type(syntax).__name__ == "Assignment_Stmt":
                    target = syntax.items[0]
                elif type(syntax).__name__ == "Nonlabel_Do_Stmt":
                    control = syntax.items[1]
                    if control is not None and control.items[1] is not None:
                        target = control.items[1][0]
                        binding = analysis._binding(analysis.source_scope_for(syntax, routine.scope), target)
                        if binding is not None:
                            iterators.add(binding.root)
                if target is not None and type(target).__name__ in {"Name", "Data_Ref"}:
                    binding = analysis._binding(analysis.source_scope_for(syntax, routine.scope), target)
                    if binding is not None and not binding.rank:
                        scalar_writes.add(binding.root)
        for resource in sections.resources:
            if any(item.resource in iterators for item in resource.dependencies):
                raise CompilationError("native definition loop bounds require invariant non-iterator values")
            descriptor = {key: value for key, value in resource.public().items()
                          if key not in {"reads", "writes", "overwrites", "dependencies"}}
            if resource.resource in resources and resources[resource.resource] != descriptor:
                raise CompilationError("native definition original resource coordinates changed")
            resources[resource.resource] = descriptor
            rectangle_count += sum(len(getattr(resource, label)) for label in ("reads", "writes", "overwrites"))
        if len(resources) > analysis.operation_limit or rectangle_count > 3 * RECTANGLE_LIMIT * analysis.operation_limit:
            raise CompilationError("native definition bounded resource/rectangle budget exhausted")
        demands.append(summary["summary_identity"])
        return NativeDefinitionStep(identity, "access", node.guard, sections, tuple(sorted(scalar_writes)))

    steps = tuple(step(identity) for identity in identities)
    kills = tuple(root for root in graph.nodes[graph.entry_id].details.get("definition_changes", ())
                  if root in resources) if include_entry else ()
    record = {"schema_version": NATIVE_DEFINITION_VERSION, "procedure": procedure,
              "structured_identity": graph.identity, "analysis_identity": graph.authority_identity,
              "source_identity": analysis.inputs.identity(), "selected_node_ids": list(identities),
              "include_entry": include_entry, "entry_definition_changes": list(kills),
              "descriptor_window_identity": descriptor_window.identity if descriptor_window is not None else None,
              "leaf_descriptor_windows": leaf_descriptors,
              "resources": [resources[root] for root in sorted(resources)],
              "steps": [item.public() for item in steps], "source_demands": demands,
              "control_visits": visits, "leaf_count": leaves, "rectangle_count": rectangle_count,
              "rectangle_limit": RECTANGLE_LIMIT, "execution_authorized": False,
              "gpu_legality_established": False, "global_initialization_authority": False,
              "allocation_freshness_authority": False, "whole_definition_inferred": False,
              "requires_original_reached_selection": True,
              "requires_runtime_layout_alias_and_coverage_checks": True,
              "bounds_evaluated": False,
              "limits": ["serial original assignments/IF and single-assignment perfect affine DO nests",
                         "calls, directives, allocation changes and unknown effects rejected",
                         "branch return guarantees meet every path including missing ELSE",
                         "symbolic single-box containment only; no bounding hull or rectangle union inference",
                         "runtime coordinates and entry definitions need separate admission"]}
    return steps, kills, record


@dataclass(frozen=True)
class NativeDefinitionProof:
    procedure: str
    identity: str
    steps: tuple[NativeDefinitionStep, ...]
    _analysis: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _include_entry: bool = field(repr=False, compare=False)
    _descriptor_window: object = field(repr=False, compare=False)
    _kills: tuple[str, ...] = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)

    def public(self):
        return {"proof_identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis, procedure, original_selection):
        if (analysis is not self._analysis or procedure != self.procedure
                or not isinstance(original_selection, tuple) or len(original_selection) != len(self._selection)
                or any(left is not right for left, right in zip(original_selection, self._selection, strict=True))
                or getattr(analysis, "_native_definition_proofs", {}).get(self.identity) is not self):
            raise CompilationError("native definitions lack registered exact original source authority")
        if _derive(analysis, procedure, original_selection, self._include_entry, self._descriptor_window)[2] != _thaw(self._record):
            raise CompilationError("native definition source authority changed")
        return self

    def transfer(self, entry_rectangles=None):
        """Conditional transfer; supplied entry definitions need own authority."""
        self.validate(self._analysis, self.procedure, self._selection)
        entry_rectangles = {} if entry_rectangles is None else entry_rectangles
        resources = {item["resource"]: item["rank"] for item in _thaw(self._record)["resources"]}
        if (not isinstance(entry_rectangles, (dict, MappingProxyType))
                or any(root not in resources or not isinstance(boxes, tuple) or len(boxes) > RECTANGLE_LIMIT
                       or any(not isinstance(box, NativeRectangle) or len(box.axes) != resources[root] for box in boxes)
                       for root, boxes in entry_rectangles.items())):
            raise CompilationError("native definition entry rectangles require bounded original-resource coordinates")
        state = {root: _unique(entry_rectangles.get(root, ())) for root in resources}
        for root in self._kills:
            state[root] = ()
        uncovered, visits = [], 0

        def visit(item, current):
            nonlocal visits
            visits += 1
            if visits > self._analysis.operation_limit * 4 + 2:
                raise CompilationError("native definition transfer traversal budget exhausted")
            if item.kind == "sequence":
                for child in item.children:
                    current = visit(child, current)
            elif item.kind == "branch":
                paths, remaining = [], dict(current)
                for condition, body in item.alternatives:
                    if condition is not None:
                        remaining = visit(condition, remaining)
                    paths.append(visit(body, dict(remaining)))
                if not item.alternatives or item.alternatives[-1][0] is not None:
                    paths.append(remaining)
                current = {root: _unique(box for path in paths for box in path[root]
                                        if all(any(_covers(other, box) for other in alternative[root])
                                               for alternative in paths)) for root in current}
            else:
                for resource in item.sections.resources:
                    for box in resource.reads:
                        if not any(_covers(defined, box) for defined in current[resource.resource]):
                            uncovered.append(NativeDefinitionRead(item.node_id, item.guard, resource.resource, box))
                            if len(uncovered) > self._analysis.operation_limit * RECTANGLE_LIMIT:
                                raise CompilationError("native definition read-obligation budget exhausted")
                if item.scalar_writes:
                    current = {root: invalidate_rectangles(boxes, item.scalar_writes)
                               for root, boxes in current.items()}
                for resource in item.sections.resources:
                    current[resource.resource] = _unique((*current[resource.resource], *resource.overwrites))
            return current

        for item in self.steps:
            state = visit(item, state)
        return NativeDefinitionTransfer(MappingProxyType(state), tuple(uncovered))


def prove_native_definitions(analysis, procedure, original_selection, *, include_entry=False, descriptor_window=None):
    """Register a source transfer, never facts about any current allocation."""
    steps, kills, record = _derive(analysis, procedure, original_selection, include_entry, descriptor_window)
    identity = _hash(record)
    registry = getattr(analysis, "_native_definition_proofs", None)
    if registry is None:
        registry = analysis._native_definition_proofs = {}
    if identity in registry:
        return registry[identity].validate(analysis, procedure, original_selection)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("native definition proof registry budget exhausted")
    proof = NativeDefinitionProof(procedure, identity, steps, analysis, original_selection, include_entry,
                                  descriptor_window, kills, _freeze(record))
    registry[identity] = proof
    return proof
