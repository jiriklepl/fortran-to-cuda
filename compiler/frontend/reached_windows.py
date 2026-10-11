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
from dataclasses import dataclass, field, replace
from hashlib import sha256

from fparser.two.utils import walk

from compiler.frontend.descriptor_windows import DescriptorWindowUnsupported
from compiler.frontend.native_sections import NativeAxis, NativeRectangle, NativeResourceSections
from compiler.frontend.preowner_transfers import PreownerUnsupported, derive_preowner_step
from compiler.frontend.source_functions import _validate_scope
from compiler.frontend.structured_effects import _freeze, _thaw
from compiler.ir import CompilationError

REACHED_WINDOW_VERSION = 12
_EFFECTS = {"read", "write", "overwrite", "descriptor_read", "definition_change"}
_FORBIDDEN = {"pointer", "optional", "volatile", "asynchronous", "value"}


class _NeedsNativeClose(CompilationError):
    """A source-authenticated operation lacks active-owner effect support."""


def _hash(value):
    try:
        return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (TypeError, ValueError, RecursionError) as error:
        raise CompilationError("reached window requires finite source facts") from error


def _single_if_action_source(analysis, routine, graph, identity, *, consume=None):
    """Bind a conditional edit to one intact original IF, never its inner span."""
    if consume is not None:
        consume(1)
    if graph.procedure != routine.qualified or identity not in graph.nodes:
        raise CompilationError("conditional prefix action lacks its original procedure authority")
    node, originals = graph.nodes[identity], graph.source_nodes(identity)
    if (node.kind != "branch" or len(originals) != 1 or type(originals[0]).__name__ != "If_Stmt"
            or len(node.alternatives) != 1):
        raise CompilationError("conditional prefix action requires one exact original single-line IF")
    original = originals[0]
    condition_id, body_id = node.alternatives[0]
    if condition_id is None or body_id not in graph.nodes:
        raise CompilationError("conditional prefix action lacks its original condition and body")
    body = graph.nodes[body_id]
    if body.kind != "sequence" or len(body.children) != 1:
        raise CompilationError("conditional prefix action requires one exact original action")
    action_id, = body.children
    condition, action = original.items
    if (condition_id not in graph.nodes or action_id not in graph.nodes
            or len(graph.source_nodes(condition_id)) != 1 or graph.source_nodes(condition_id)[0] is not condition
            or len(graph.source_nodes(action_id)) != 1 or graph.source_nodes(action_id)[0] is not action):
        raise CompilationError("conditional prefix action differs from its original condition or action authority")
    item = getattr(original, "item", None)
    span = getattr(item, "fort_original_span", getattr(item, "span", None))
    if (not isinstance(span, tuple) or len(span) != 2 or any(type(value) is not int for value in span)
            or not 1 <= span[0] <= span[1] or tuple(node.span) != span):
        raise CompilationError("conditional prefix action lacks its exact original edit span")
    if str(routine.scope.path) not in analysis.sources:
        raise CompilationError("conditional prefix action lacks its original source file")
    lines = routine.scope.path.read_text().splitlines(keepends=True)
    if span[1] > len(lines):
        raise CompilationError("conditional prefix action exceeds its original source span")
    source = lines[span[0] - 1:span[1]]
    unsafe = None
    if getattr(item, "label", None) is not None:
        unsafe = "single original IF action has a labeled edit span"
    for line in source:
        if consume is not None:
            consume(len(line))
        if line.lstrip().startswith("#"):
            unsafe = "single original IF action crosses a preprocessor edit boundary"
        elif ";" in line:
            # Reject even a quoted semicolon conservatively. Replacing a whole
            # physical line must never discard a neighboring original statement.
            unsafe = "single original IF action has a potentially shared semicolon edit span"
    return {"condition_node_id": condition_id, "action_node_id": action_id,
            "condition_source_sha256": _hash([str(condition)]),
            "action_source_sha256": _hash([str(action)]), "action_source_kind": type(action).__name__,
            "original_span_sha256": sha256("".join(source).encode()).hexdigest(),
            "condition_evaluated_once": True, "action_evaluated_once": True}, unsafe


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


def _plain_counted_numerical_candidate(analysis, routine, graph, identity, consume):
    """Route only an intact original serial counted DO to the provider.

    This syntax check grants no numerical, descriptor or definition authority.
    The provider still authenticates and prepares the complete original loop.
    Attached or nested directives require the separate joined-group route.
    """
    if graph.nodes[identity].kind != "loop":
        return False
    originals = graph.source_nodes(identity)
    if len(originals) != 1 or type(originals[0]).__name__ != "Block_Nonlabel_Do_Construct":
        return False
    pending, iterators, writes = [originals[0]], set(), []
    while pending:
        syntax = []
        for node in pending.pop().content:
            consume(1)
            if type(node).__name__ == "Comment":
                if str(node).lstrip().startswith("!$"):
                    return False
            else:
                syntax.append(node)
        if not syntax or type(syntax[0]).__name__ != "Nonlabel_Do_Stmt":
            return False
        control = syntax[0].items[1]
        if (type(control).__name__ != "Loop_Control" or control.items[1] is None
                or type(syntax[-1]).__name__ != "End_Do_Stmt"):
            return False
        iterators.add(str(control.items[1][0]).lower())
        for node in syntax[1:-1]:
            consume(1)
            if type(node).__name__ == "Block_Nonlabel_Do_Construct":
                pending.append(node)
            elif type(node).__name__ == "Assignment_Stmt":
                writes.append(node.items[0])
            else:
                return False
    array_write = False
    for target in writes:
        consume(1)
        if type(target).__name__ != "Part_Ref":
            # Scalar and whole-array native operations keep their old path.
            return False
        scope = analysis.source_scope_for(target, routine.scope)
        binding = analysis._binding(scope, target.items[0])
        if binding is None or not binding.rank:
            return False
        array_write = True
    for node in walk(originals[0]):
        consume(1)
        if type(node).__name__ == "Part_Ref":
            scope = analysis.source_scope_for(node, routine.scope)
            binding = analysis._binding(scope, node.items[0])
            if binding is not None and binding.rank:
                for index in getattr(node.items[1], "items", ()):
                    consume(1)
                    if not (type(index).__name__ == "Int_Literal_Constant"
                            or type(index).__name__ == "Name" and str(index).lower() in iterators):
                        # In particular, native scatter/gather corrections
                        # must not become a speculative lowering hard error.
                        return False
    return array_write


def _derive(analysis, procedure, selection, roots, facts, planned_close_before=None, *,
            numerical_provider=None, _mapped=None, _work=None, _stack=(), _children=None, _result=None,
            _numerical_regions=None):
    if numerical_provider is not None:
        from compiler.frontend.numerical_preparation import NumericalPreparationProvider

        if planned_close_before is None or not isinstance(numerical_provider, NumericalPreparationProvider):
            raise CompilationError("numerical effects require an explicit registered planned-window provider")
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
    if _mapped is None and any(type(node).__name__ == "Comment" and str(node).lstrip().lower().startswith("!$omp")
           for identity in (selected if planned_close_before is not None else (*prefix, *selected))
           for node in walk(graph.source_nodes(identity))):
        raise CompilationError("reached window direct OpenMP groups require separate whole-completion authority")
    # Ordinary window proofs retain their complete-interval meaning. Planned
    # owners additionally authenticate exact close-before-original constructs;
    # nothing after a reached close receives effects or descriptor authority.
    planned = planned_close_before is not None
    forced, forced_selections = {}, {}
    selected_graph_ids, pending = set(), list(selected)
    while pending:
        identity = pending.pop()
        if identity in selected_graph_ids:
            continue
        selected_graph_ids.add(identity)
        node = graph.nodes[identity]
        pending.extend(node.children)
        for condition, body in node.alternatives:
            pending.extend((body,) if condition is None else (condition, body))
    if planned:
        if not isinstance(planned_close_before, tuple) or len(planned_close_before) > analysis.operation_limit:
            raise CompilationError("planned reached closure requires bounded original cut tuples")
        for item in planned_close_before:
            if not isinstance(item, tuple) or len(item) != 2:
                raise CompilationError("planned reached closure requires an original selection and reason pair")
            originals, reason = item
            if (not isinstance(originals, tuple) or not originals or len(originals) > analysis.operation_limit
                    or not isinstance(reason, str)):
                raise CompilationError("planned reached closure requires a complete original construct")
            _original_graph, identities = analysis._selected_source(procedure, originals)
            if not identities or not set(identities) <= selected_graph_ids or identities[0] in forced:
                raise CompilationError("planned reached closure escapes or duplicates its exact selected source")
            identity = identities[0]
            if len(identities) == 1 and len(originals) == 1 and (
                    type(originals[0]).__name__ not in {"Call_Stmt", "Assignment_Stmt", "If_Construct", "If_Stmt",
                                                      "Block_Nonlabel_Do_Construct", "Associate_Construct"}
                    or graph.nodes[identity].details.get("evaluation") in {"condition", "loop_header", "directive"}):
                raise CompilationError("planned reached closure cannot cut an original control or joined-group interior")
            forced[identity] = reason
            forced_selections[identity] = identities, originals
    if (not isinstance(roots, tuple) or not roots or len(roots) > analysis.operation_limit
            or any(not isinstance(root, str) for root in roots) or len(set(roots)) != len(roots)):
        raise CompilationError("reached window requires bounded unique canonical roots")
    _hash(facts)
    if (not isinstance(facts, dict) or type(facts.get("schema_version")) is not int or facts["schema_version"] != 1
            or facts.get("sources") != analysis.sources or not isinstance(facts.get("captures"), dict)
            or len(facts["captures"]) > analysis.operation_limit):
        raise CompilationError("reached window requires matching invocation-entry source facts")
    work = _work if _work is not None else {"visits": 0, "effects": 0, "calls": 0, "coordinates": 0}
    if procedure in _stack or len(_stack) >= analysis.depth_limit:
        raise CompilationError("mapped reached call recursion/control depth exhausted")
    stack = (*_stack, procedure)
    mapped_children = []
    bindings, coverage, module_roots = {}, {}, []
    for root in roots:
        binding = _binding(analysis, routine, root)
        fact_root = _mapped["facts_roots"].get(root, root) if _mapped is not None else root
        fact = facts["captures"].get(fact_root)
        _validate_scope(analysis, binding.declaring_scope)
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
        coverage[root] = (_mapped["entry"].get(root, ()) if _mapped is not None else
                          _whole_coverage(binding) if fact["initialized"] == "whole" else ())
    descriptor_roots = tuple(root for root, binding in bindings.items()
                             if "allocatable" in binding.attributes and root not in module_roots)
    module_roots = tuple(module_roots)
    for root in graph.nodes[graph.entry_id].details.get("definition_changes", ()):
        if root in coverage:
            if _mapped is not None:
                work["effects"] += 1
                if work["effects"] > analysis.operation_limit:
                    raise CompilationError("mapped reached entry definition budget exhausted")
            coverage[root] = ()
    demands, descriptor_demands, native_prefix_demands, definition_demands = [], [], [], []
    numerical_demands = []
    numerical_regions = _numerical_regions if _numerical_regions is not None else []
    prefix_steps, prefix_disables = [], {}
    visits, effects, effect_work = 0, 0, 0
    active_ids, closes, changed_scalars = set(), {}, set()
    activation_templates = {}

    def coordinate_work(amount):
        from compiler.frontend.native_group_structure import NATIVE_GROUP_SCAN_LIMIT
        work["coordinates"] += amount
        if work["coordinates"] > NATIVE_GROUP_SCAN_LIMIT:
            raise CompilationError("mapped reached aggregate coordinate/source scan budget exhausted")

    if numerical_provider is not None:
        numerical_provider.validate(analysis, consume=coordinate_work)

    def dormant_condition_reads(identity, summary, state):
        """Unchanged host inputs, with neither capture nor definition authority."""
        from compiler.frontend.call_bindings import _fixed_array_bounds
        from compiler.frontend.source_functions import _validate_resolution, _walk

        node = graph.nodes[identity]
        if _mapped is not None or node.details.get("evaluation") != "condition":
            return set()
        roots = set()
        for effect in summary["ordered_effects"]:
            coordinate_work(1)
            if effect["resource"] not in state and (effect.get("rank", 0) or any(
                    view.get("actual_descriptor", {}).get("rank", 0)
                    for view in effect.get("view_chain", ()) if view.get("actual_descriptor") is not None)):
                roots.add(effect["resource"])
        if not roots:
            return set()
        registered = False
        for branch in graph.nodes.values():
            coordinate_work(1)
            for condition, _body in branch.alternatives:
                coordinate_work(1)
                registered |= condition == identity
        if not registered:
            raise CompilationError("dormant condition lacks exact original branch authority")
        # Complete scalar-function proofs remain responsible for every called
        # computation. A real side effect cannot become readonly metadata just
        # because its array reads happen in an original branch condition.
        probe = procedure + "::fort_effect_condition"
        if any(effect["kind"] not in {"read", "descriptor_read"}
               and not (effect["resource"] == probe and "fort_effect_condition" not in routine.scope.bindings)
               for effect in summary["ordered_effects"]):
            return set()
        originals = graph.source_nodes(identity)
        _validate_resolution(analysis, routine.scope, _walk(originals), tick=lambda _role: coordinate_work(1))
        resources = []
        for root in sorted(roots):
            choices = {}
            for candidate in (routine.scope, *analysis.modules.values()):
                coordinate_work(1)
                for binding in candidate.bindings.values():
                    coordinate_work(1)
                    if binding.root == root:
                        choices[id(binding)] = binding
            if len(choices) != 1:
                raise _NeedsNativeClose("dormant condition needs an original fixed metadata binding: " + root)
            binding = next(iter(choices.values()))
            owner = binding.declaring_scope
            if owner is None:
                raise CompilationError("dormant condition lacks original declaration authority: " + root)
            _validate_scope(analysis, owner)
            if owner.bindings.get(binding.name) is not binding:
                raise CompilationError("dormant condition original binding identity changed: " + root)
            reason = analysis.resource_identity_boundary(binding)
            if reason:
                raise CompilationError(reason)
            if (root.startswith("argument::") or binding.dtype not in {"real", "integer", "logical"}
                    or binding.kind not in {4, 8} or not 1 <= binding.rank <= 4
                    or binding.attributes & (_FORBIDDEN | {"allocatable", "target"})
                    or len(binding.shape_nodes) != binding.rank
                    or any(type(axis).__name__ != "Explicit_Shape_Spec" for axis in binding.shape_nodes)):
                raise _NeedsNativeClose("dormant condition requires fixed unaliased native metadata: " + root)
            # Authenticate the original declaration's expressions, including
            # imported kind/bound PARAMETER bindings. No host payload is read.
            declarations, seen = [], set()
            for start in (owner, routine.scope):
                current = start
                while current is not None and id(current) not in seen:
                    seen.add(id(current))
                    _validate_scope(analysis, current)
                    specification = next((part for part in getattr(current.node, "content", ())
                                          if type(part).__name__ == "Specification_Part"), None)
                    for original in _walk(specification):
                        coordinate_work(1)
                        kind = type(original).__name__
                        if kind in {"Common_Stmt", "Equivalence_Stmt"}:
                            raise _NeedsNativeClose("dormant condition metadata storage association is unproved: " + root)
                        if kind == "Comment" and str(original).lstrip().lower().startswith("!$omp"):
                            raise _NeedsNativeClose("dormant condition metadata OpenMP ownership is unproved: " + root)
                        if current is owner and kind == "Type_Declaration_Stmt":
                            dtype, _attributes, entities = original.items
                            if (any(type(entity).__name__ == "Entity_Decl" and str(entity.items[0]).lower() == binding.name
                                    for entity in getattr(entities, "items", ()))
                                    and type(dtype).__name__ == "Intrinsic_Type_Spec" and dtype.items[1] is not None):
                                declarations.append(dtype.items[1])
                    current = current.parent
            _validate_resolution(analysis, owner, _walk((*binding.shape_nodes, *declarations)),
                                 tick=lambda _role: coordinate_work(1))
            bounds = _fixed_array_bounds(analysis, binding)
            resources.append({**_thaw(_freeze(binding.public())), "source": str(owner.path),
                              "source_sha256": analysis.sources[str(owner.path)],
                              "original_bounds": [list(axis) for axis in bounds]})
        if resources:
            prefix_steps.append({"node_id": identity, "selected_node_ids": [identity],
                "guard": list(node.guard), "span": list(node.span),
                "original_source_kind": "original_native_condition_reads",
                "original_source_sha256": _hash([str(original) for original in originals]),
                "resources": resources, "evaluation": "unchanged original host condition before ownership",
                "capture_authority": False, "managed_definition_authority": False,
                "descriptor_lifetime_authority": False, "alias_authority": False, "requires_no_active_context": True,
                "allocation_freshness_authority": False, "gpu_legality_established": False})
        return set(roots)

    joined_selections = {}
    if _mapped is not None or planned:
        from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence

        def groups(originals, peeled):
            for item in grouped_original_sequence(originals, peeled):
                coordinate_work(1)
                if isinstance(item, NativeGroupCandidate):
                    _graph, identities = analysis._selected_source(procedure, item.original_nodes)
                    joined_selections[identities[0]] = identities
                elif type(item).__name__ in {"If_Construct", "Block_Nonlabel_Do_Construct", "Associate_Construct"}:
                    groups(peeled.get(id(item), getattr(item, "content", ())), peeled)

        groups(selection if _mapped is not None else (), {})
        if planned and _mapped is None:
            groups(tuple(original for identity in prefix for original in graph.source_nodes(identity)), {})
    for identity, (identities, originals) in forced_selections.items():
        grouped = joined_selections.get(identity)
        if len(identities) != 1 or len(originals) != 1:
            exact = tuple(original for node_id in identities for original in graph.source_nodes(node_id))
            if grouped != identities or len(exact) != len(originals) or any(
                    left is not right for left, right in zip(exact, originals, strict=True)):
                raise CompilationError("planned reached closure requires one exact complete original joined selection")
        elif any(identity in members for first, members in joined_selections.items() if first != identity):
            raise CompilationError("planned reached closure cannot cut an original joined-group interior")
    if planned and _mapped is None:
        owned_directives = {id(node) for members in joined_selections.values() for identity in members
                            for original in graph.source_nodes(identity) for node in walk(original)
                            if type(node).__name__ == "Comment"}
        if any(id(node) not in owned_directives for identity in prefix for original in graph.source_nodes(identity)
               for node in walk(original) if type(node).__name__ == "Comment"
               and str(node).lstrip().lower().startswith("!$omp")):
            raise CompilationError("reached window direct OpenMP prefix needs exact whole joined selection authority")

    def checkpoint():
        return (tuple(map(len, (demands, descriptor_demands, native_prefix_demands, definition_demands,
                                mapped_children, prefix_steps, numerical_demands, numerical_regions))),
                effects, set(changed_scalars))

    def discard(before):
        nonlocal effects
        lengths, effects, scalar_snapshot = before
        changed_scalars.clear()
        changed_scalars.update(scalar_snapshot)
        for items, length in zip((demands, descriptor_demands, native_prefix_demands, definition_demands,
                                  mapped_children, prefix_steps, numerical_demands, numerical_regions),
                                 lengths, strict=True):
            del items[length:]

    def may_close(error):
        # Only an explicit local support rejection can close. Source/token
        # authentication and aggregate work failures remain hard exceptions,
        # independently of the wording of their diagnostics.
        from compiler.frontend.native_definitions import NativeCallCoordinatesUnsupported, NativeDefinitionUnsupported
        from compiler.frontend.numerical_effects import NumericalEffectUnsupported
        if (work["effects"] > analysis.operation_limit or not isinstance(error, (
                DescriptorWindowUnsupported, NativeCallCoordinatesUnsupported, NativeDefinitionUnsupported,
                NumericalEffectUnsupported, _NeedsNativeClose))):
            raise error

    def may_disable(error):
        from compiler.frontend.native_definitions import NativeDefinitionUnsupported
        if not isinstance(error, (PreownerUnsupported, DescriptorWindowUnsupported, NativeDefinitionUnsupported,
                                  _NeedsNativeClose)):
            raise error
        if work["effects"] > analysis.operation_limit:
            raise error

    def descendants(identity):
        result, pending = set(), [identity]
        while pending:
            current = pending.pop()
            if current in result:
                continue
            result.add(current)
            node = graph.nodes[current]
            pending.extend(node.children)
            for condition, body in node.alternatives:
                pending.extend((body,) if condition is None else (condition, body))
        return result

    def selection_span(identities):
        if len(identities) == 1:
            # A single-line IF action has no independent original edit span.
            # Its provisional cut is bound to the intact IF by the branch
            # transfer; never invent an inner statement's source position.
            return list(graph.nodes[identities[0]].span)
        first, last = graph.nodes[identities[0]].span, graph.nodes[identities[-1]].span
        if len(first) != 2 or len(last) != 2:
            raise CompilationError("joined reached selection lacks exact original endpoint spans")
        return [first[0], last[1]]

    def close_before(identity, reason, joined_selection=None):
        from compiler.frontend.source_functions import _validate_resolution, _validate_scope, _walk

        identities = joined_selection or (identity,)
        originals = tuple(original for item in identities for original in graph.source_nodes(item))
        completion = None
        if joined_selection is not None or graph.native_group_for_selection(identities) is not None:
            if joined_selections.get(identity) != identities:
                raise CompilationError("planned close lacks exact complete original joined selection")
            retained = next((prepared for owner, selected, prepared, _effect in numerical_regions
                             if owner == procedure and len(selected) == len(originals)
                             and all(left is right for left, right in zip(selected, originals, strict=True))), None)
            if retained is None and numerical_provider is not None and _mapped is not None:
                from compiler.frontend.numerical_preparation import NumericalPreparationUnsupported

                with suppress(NumericalPreparationUnsupported):
                    retained = numerical_provider.prepare(analysis, procedure, originals, consume=coordinate_work)
            if retained is not None:
                retained.validate(analysis, procedure, originals, consume=coordinate_work)
                completion = retained.completion_proof
            else:
                completion = analysis.joined_completion(procedure, identities)
        elif (len(originals) != 1 or type(originals[0]).__name__ not in {
                "Call_Stmt", "Assignment_Stmt", "If_Construct", "If_Stmt", "Block_Nonlabel_Do_Construct",
                "Associate_Construct", "Allocate_Stmt", "Deallocate_Stmt"}):
            raise CompilationError("planned reached closure needs a complete original native construct: " + reason)
        _validate_scope(analysis, routine.scope)
        if type(originals[0]).__name__ == "Call_Stmt":
            # Bind its original callee/actual spellings without asking for any
            # payload or effects from the child that executes after closure.
            _validate_resolution(analysis, routine.scope, _walk(originals[0]))
            child = graph.nodes[identity].details.get("procedure")
            if child in analysis.routines:
                _validate_scope(analysis, analysis._require_original(child).scope)
        closes[identity] = {"node_id": identity, "selected_node_ids": list(identities),
                            "guard": list(graph.nodes[identity].guard),
                            "original_source_sha256": _hash([str(node) for node in originals]),
                            "original_source_kind": "complete_original_joined_group" if completion else type(originals[0]).__name__,
                            "completion_identity": completion.identity if completion else None,
                            "span": selection_span(identities), "reason": reason,
                            "execution": "publish and close before unchanged original source; native suffix once",
                            "closed_is_absorbing": True, "reopen": False}
        return None, True

    def leaf(identity, state, *, before_owner=False, joined_selection=None):
        nonlocal effects, effect_work
        if planned and before_owner:
            originals = graph.source_nodes(identity)
            if len(originals) == 1:
                step = derive_preowner_step(analysis, procedure, originals[0], bindings, state, facts,
                                            consume=coordinate_work)
                if step is not None:
                    count = len(step.events)
                    effects += count
                    effect_work += count
                    work["effects"] += count
                    if work["effects"] > analysis.operation_limit:
                        raise CompilationError("reached window ordered pre-owner effect budget exhausted")
                    prefix_steps.append({"node_id": identity, "guard": list(graph.nodes[identity].guard),
                                         "original_source_sha256": _hash(str(originals[0])),
                                         "original_source_kind": type(originals[0]).__name__,
                                         "span": list(graph.nodes[identity].span), **step.public()})
                    return dict(step.definitions)
        if planned and not before_owner and joined_selection is None and graph.nodes[identity].kind == "call":
            child_name = graph.nodes[identity].details.get("procedure")
            child_routine = analysis.routines.get(child_name)
            child_graph = analysis.structure(child_name) if child_routine is not None else None
            needs_mapped = child_graph is not None and (
                any(node.kind in {"loop", "branch"} for node in child_graph.nodes.values())
                or any(node.kind == "call" and node.details.get("procedure") in analysis.routines
                       for node in child_graph.nodes.values())
                or any(binding.rank and binding.intent == "out" for name, binding in child_routine.scope.bindings.items()
                       if name in child_routine.arguments)
                or any(_coverage_kind(boxes) != "whole" for boxes in state.values()))
            if needs_mapped and child_routine.source_kind == "module":
                from compiler.frontend.native_definitions import prove_native_call_coordinates
                from compiler.frontend.source_effects import _children as source_children

                original, = graph.source_nodes(identity)
                mapping = prove_native_call_coordinates(analysis, procedure, original, consume=coordinate_work)
                child = analysis._require_original(child_name)
                if any("save" in binding.attributes for binding in child.scope.bindings.values()):
                    raise _NeedsNativeClose("mapped reached child saved storage needs separate lifetime authority")
                work["calls"] += 1
                if work["calls"] > analysis.procedure_limit:
                    raise CompilationError("mapped reached call traversal budget exhausted")
                resource_mapping = mapping.resource_mapping
                if any(actual not in state for actual in resource_mapping.values()):
                    raise _NeedsNativeClose("mapped reached child actual lacks independent managed authority")
                # Hidden resources retain their original module identity, but
                # only when this child's original namespace exposes them.
                scopes, scope = [], child.scope
                while scope is not None:
                    scopes.append(scope)
                    scope = scope.parent
                spellings = {name for scope in scopes for name in (*scope.bindings, *scope.imports)}
                hidden = set()
                for root, binding in bindings.items():
                    if root in resource_mapping.values() or root.startswith("argument::"):
                        continue
                    for spelling in {binding.name, *spellings}:
                        coordinate_work(1)
                        if analysis._binding(child.scope, spelling) is binding:
                            hidden.add(root)
                            break
                entry_input = {root: state[root] for root in (*resource_mapping.values(), *hidden)}
                for root in mapping.entry_definition_changes:
                    entry_input[root] = ()
                # A prior whole definition subsumes later same-resource
                # partial writes. Keep that exact existing rectangle rather
                # than requiring unrelated caller controls in this child.
                # OUT kills above still remove every prior definition first.
                for root, boxes in entry_input.items():
                    for box in boxes:
                        coordinate_work(1)
                        if box.iterations:
                            continue
                        whole = True
                        for axis in box.axes:
                            coordinate_work(1)
                            if axis.lower is not None or axis.upper is not None:
                                whole = False
                                break
                        if whole:
                            entry_input[root] = (box,)
                            break
                entry = mapping.caller_to_child(entry_input, consume=coordinate_work)
                entry = {**{root: entry.get(root, ()) for root in resource_mapping},
                         **{root: state[root] for root in hidden}}
                facts_roots = {root: (_mapped["facts_roots"].get(actual, actual) if _mapped is not None else actual)
                               for root, actual in resource_mapping.items()}
                facts_roots.update({root: (_mapped["facts_roots"].get(root, root) if _mapped is not None else root)
                                   for root in hidden})
                originals = tuple(source_children(child.execution))
                output, nested = {}, []
                record = _derive(analysis, child_name, originals, tuple(entry), facts, (),
                                 numerical_provider=numerical_provider,
                                 _mapped={"entry": entry, "facts_roots": facts_roots},
                                 _work=work, _stack=stack, _children=nested, _result=output,
                                 _numerical_regions=numerical_regions)
                mapped_children.append(_MappedChildPlan(identity, original, child_name, originals, mapping,
                                                       _freeze(record), tuple(nested)))
                if output["active"] is None:
                    return None
                returned = mapping.child_to_caller(output["active"], consume=coordinate_work)
                for root in (*resource_mapping.values(), *hidden):
                    state[root] = returned.get(root, ())
                from compiler.frontend.native_definitions import invalidate_rectangles
                changed = []
                for root in record["scalar_changes"]:
                    if root.startswith("argument::"):
                        raise _NeedsNativeClose("mapped reached child writable scalar forwarding is unsupported")
                    if not root.startswith(child_name + "::"):
                        changed.append(root)
                changed = tuple(changed)
                changed_scalars.update(changed)
                if changed:
                    state = {root: invalidate_rectangles(boxes, changed) for root, boxes in state.items()}
                return state
        selected_analysis, native_prefix, native_reads = analysis, None, set()
        # Native prefix effects are conditional on the original call finishing
        # before any owner exists. They never grant a managed capture or make
        # an incomplete ordinary summary complete in the original analysis.
        if before_owner and graph.nodes[identity].kind == "call":
            if planned and _mapped is None:
                from compiler.frontend.native_prefix import reject_unsupported_native_prefix_call
                original, = graph.source_nodes(identity)
                reject_unsupported_native_prefix_call(analysis, procedure, original, consume=coordinate_work)
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
        leaf_selection = joined_selection or (identity,)
        # Exact numerical computation can establish physical effects without
        # making an unsupported ordinary native child summary complete. This
        # capability is opt-in, mapped-child only, and conditional on the
        # original preparation guards. Descriptor/lifetime authority remains
        # independent and is checked before any definition credit is used.
        if (planned and not before_owner and _mapped is not None and numerical_provider is not None
                and (joined_selection is not None
                     or _plain_counted_numerical_candidate(analysis, routine, graph, identity, coordinate_work))):
            from compiler.frontend.numerical_effects import prove_numerical_effects
            from compiler.frontend.numerical_preparation import NumericalPreparationUnsupported

            originals = tuple(original for item in leaf_selection for original in graph.source_nodes(item))
            try:
                prepared = numerical_provider.prepare(analysis, procedure, originals, consume=coordinate_work)
            except NumericalPreparationUnsupported:
                prepared = None  # The ordinary native proof may still suffice.
            if prepared is not None:
                descriptor = None
                if descriptor_roots or module_roots:
                    from compiler.frontend.descriptor_windows import prove_descriptor_window

                    descriptor = prove_descriptor_window(analysis, procedure, leaf_selection, descriptor_roots,
                                                         module_roots=module_roots)
                    descriptor_demands.append(descriptor.identity)
                definition = prove_numerical_effects(analysis, procedure, originals, prepared, consume=coordinate_work)
                # Retain the source completion even when the subsequent input
                # check closes; closing grants no native memory-hook authority.
                numerical_regions.append((procedure, originals, prepared, definition))
                public = definition.public()
                count = public["ordered_effect_count"]
                effects += count
                effect_work += count
                work["effects"] += count
                if work["effects"] > analysis.operation_limit:
                    raise CompilationError("reached window ordered numerical effect budget exhausted")
                resources = {item.resource for item in definition.sections.resources}
                if resources - state.keys():
                    raise _NeedsNativeClose("reached numerical group needs independent managed resource authority: "
                                            + sorted(resources - state.keys())[0])
                transferred = definition.transfer({root: state[root] for root in resources}, consume=coordinate_work)
                if transferred.uncovered_reads:
                    raise _NeedsNativeClose("reached numerical group read requires original group-entry definition: "
                                            + transferred.uncovered_reads[0].resource)
                numerical_demands.append({"selected_node_ids": list(leaf_selection),
                    "preparation_identity": prepared.identity, "effect_proof_identity": definition.identity,
                    "conditional_close_before_original": bool(prepared.preconditions
                        or prepared.extraction.requires_numerical_environment
                        or prepared.requires_original_team_environment_preflight),
                    "preconditions": [guard.public() for guard in prepared.preconditions],
                    "guard_false_execution": "publish and close before original group; execute original once",
                    "native_summary_upgraded": False, "execution_authorized": False})
                demands.extend(public["source_demands"])
                for step in definition.steps:
                    for resource in step.sections.resources:
                        activation_templates[resource.resource] = resource
                changed = tuple(public["scalar_changes"])
                changed_scalars.update(changed)
                if changed:
                    from compiler.frontend.native_definitions import invalidate_rectangles

                    state = {root: invalidate_rectangles(boxes, changed) for root, boxes in state.items()}
                state.update(transferred.definitions)
                return state
        completion = None
        if joined_selection is not None or graph.native_group_for_selection(leaf_selection) is not None:
            # This exact whole-team token grants completion only. Invalid
            # source, token and aggregate budget failures must remain hard.
            completion = selected_analysis.joined_completion(procedure, leaf_selection)
        descriptor = None
        if descriptor_roots or module_roots:
            from compiler.frontend.descriptor_windows import prove_descriptor_window
            descriptor = prove_descriptor_window(selected_analysis, procedure, leaf_selection, descriptor_roots,
                                                 module_roots=module_roots)
            descriptor_demands.append(descriptor.identity)
        if completion is not None and (before_owner or any(_coverage_kind(boxes) != "whole"
                                                         for boxes in state.values())):
            from compiler.frontend.native_definitions import prove_native_definitions
            originals = tuple(original for item in leaf_selection for original in graph.source_nodes(item))
            definition = prove_native_definitions(selected_analysis, procedure, originals,
                descriptor_window=descriptor, completion=completion, consume=coordinate_work)
            public = definition.public()
            count = public["ordered_effect_count"]
            effects += count
            effect_work += count
            work["effects"] += count
            if work["effects"] > analysis.operation_limit:
                raise CompilationError("reached window ordered joined effect budget exhausted")
            resources = {item["resource"] for item in public["resources"]}
            if resources - state.keys():
                raise _NeedsNativeClose("reached joined group untracked array needs definition/alias authority: "
                                        + sorted(resources - state.keys())[0])
            if before_owner and any(resource.reads and facts["captures"][resource.resource].get("lifetime") == "selected_window"
                                    and facts["captures"][resource.resource]["initialized"] == "whole"
                                    for step in definition.steps for resource in step.sections.resources):
                raise PreownerUnsupported("joined-prefix read cannot use selected-window whole assertion before activation")
            transferred = definition.transfer({root: state[root] for root in resources}, consume=coordinate_work)
            if transferred.uncovered_reads:
                raise _NeedsNativeClose("reached joined group read requires original group-entry definition: "
                                        + transferred.uncovered_reads[0].resource)
            definition_demands.append(definition.identity)
            demands.extend(public["source_demands"])
            for step in definition.steps:
                for resource in step.sections.resources:
                    activation_templates[resource.resource] = resource
            state.update(transferred.definitions)
            if before_owner:
                prefix_steps.append({"node_id": identity, "selected_node_ids": list(leaf_selection),
                    "guard": list(graph.nodes[identity].guard), "original_source_kind": "complete_original_joined_group",
                    "span": selection_span(leaf_selection),
                    "original_source_sha256": _hash([str(node) for node in originals]),
                    "definition_proof_identity": definition.identity, "completion_identity": completion.identity,
                    "payload_provenance": "exact unconditional native must-writes after original final join",
                    "no_cross_unit_definition_credit": True})
            return state
        if completion is not None and graph.native_group_for_selection(leaf_selection) is not None:
            _graph, _ids, _group, units = selected_analysis.native_group_summaries(procedure, leaf_selection, completion)
            summary = {"complete": all(unit["complete"] for unit in units),
                       "effect_composition": {"available": all(unit.get("effect_composition", {}).get("available")
                                                                  for unit in units)},
                       "reasons": [reason for unit in units for reason in unit.get("reasons", ())],
                       "ordered_effects": [effect for unit in units for effect in unit["ordered_effects"]],
                       "guaranteed_whole_overwrites": [],
                       "summary_identity": _hash([completion.identity, *[unit["summary_identity"] for unit in units]])}
        else:
            summary = selected_analysis.segment_summary(procedure, leaf_selection, capture_locals=True,
                                                        descriptor_window=descriptor)
        if not summary["complete"] or not summary.get("effect_composition", {}).get("available"):
            # Source effects report unsupported source operations as an
            # incomplete summary. Reauthenticate every source namespace used
            # by that query before treating its incompleteness as support-only.
            for key in selected_analysis._source_function_used_scopes:
                _validate_scope(selected_analysis, selected_analysis._source_function_scopes[key][0])
            error = CompilationError if before_owner else _NeedsNativeClose
            raise error("reached window needs complete prefix/window effects: "
                        + "; ".join(summary.get("reasons", ())))
        demands.append(summary["summary_identity"])
        condition_reads = dormant_condition_reads(identity, summary, state) if before_owner else set()
        outer = set(graph.nodes[identity].guard)
        scalar_writes = set()
        for effect in summary["ordered_effects"]:
            effect_work += 1
            work["effects"] += 1
            effects += 1
            if work["effects"] > analysis.operation_limit:
                raise CompilationError("reached window ordered effect budget exhausted")
            if effect["kind"] not in _EFFECTS:
                raise CompilationError("reached window native environment/control effect needs distinct authority")
            root = effect["resource"]
            if root not in state:
                if root in condition_reads and effect["kind"] in {"read", "descriptor_read"}:
                    continue  # Original dormant host input; no managed definition.
                if (before_owner and native_prefix is not None and root in native_reads
                        and effect["kind"] in {"read", "descriptor_read"}):
                    continue  # Unchanged native read; no device capture or coverage.
                array = effect.get("rank", 0) or any(
                    view.get("actual_descriptor", {}).get("rank", 0)
                    for view in effect.get("view_chain", ()) if view.get("actual_descriptor") is not None)
                if array:
                    raise _NeedsNativeClose("reached window untracked array needs definition/alias authority: " + root)
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
                raise _NeedsNativeClose("reached window section read follows incomplete definition: "
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
                    raise _NeedsNativeClose("reached window whole read follows incomplete definition: " + root)
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
            changed_scalars.update(scalar_writes)
            state = {root: invalidate_rectangles(boxes, tuple(sorted(scalar_writes)))
                     for root, boxes in state.items()}
        return state

    def transfer(identity, state, depth=0, *, before_owner=False):
        nonlocal visits
        visits += 1
        work["visits"] += 1
        if work["visits"] > graph.node_limit or depth > analysis.depth_limit:
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

    def planned_transfer(identity, state, depth=0):
        nonlocal visits
        active, has_closed_path = state
        # CLOSED never inspects an original descriptor, constructs a query or
        # requests effects for the native suffix. It cannot reopen at a join.
        if active is None:
            return state
        visits += 1
        work["visits"] += 1
        if work["visits"] > graph.node_limit or depth > analysis.depth_limit:
            raise CompilationError("reached window control traversal budget exhausted")
        if identity in forced:
            return close_before(identity, forced[identity])
        node = graph.nodes[identity]
        if node.kind == "sequence":
            return planned_sequence(node.children, state, depth + 1)
        if node.kind == "branch":
            paths = []
            branch_before = checkpoint()
            previous_active, previous_closes = set(active_ids), dict(closes)
            false_path = dict(active)
            for condition, body in node.alternatives:
                path = dict(false_path)
                if condition is not None:
                    visits += 1
                    work["visits"] += 1
                    if work["visits"] > graph.node_limit or depth + 1 > analysis.depth_limit:
                        raise CompilationError("reached window control traversal budget exhausted")
                    before = checkpoint()
                    try:
                        path = leaf(condition, path)
                    except CompilationError as error:
                        may_close(error)
                        discard(before)
                        discard(branch_before)
                        active_ids.clear()
                        active_ids.update(previous_active)
                        closes.clear()
                        closes.update(previous_closes)
                        return close_before(identity, "original condition requires native closure: " + str(error))
                    false_path = dict(path)
                    active_ids.add(condition)
                paths.append(planned_transfer(body, (path, False), depth + 1))
            if not node.alternatives or node.alternatives[-1][0] is not None:
                paths.append((false_path, False))
            # A single-line IF has no distinct source span for an inner cut.
            # Close before its entire original condition/action, unconditionally.
            originals = graph.source_nodes(identity)
            if len(originals) == 1 and type(originals[0]).__name__ == "If_Stmt" and any(path[1] for path in paths):
                discard(branch_before)
                active_ids.clear()
                active_ids.update(previous_active)
                closes.clear()
                closes.update(previous_closes)
                return close_before(identity, "single original IF action requires whole-construct native closure")
            active_paths = [path[0] for path in paths if path[0] is not None]
            if not active_paths:
                return None, True
            from compiler.frontend.native_definitions import intersect_rectangles
            joined = dict(active_paths[0])
            for path in active_paths[1:]:
                joined = {root: intersect_rectangles(boxes, path[root]) for root, boxes in joined.items()}
            return joined, has_closed_path or any(path[1] for path in paths)
        before = checkpoint()
        try:
            if node.kind == "boundary" and graph.native_group_for_selection((identity,)) is None:
                raise _NeedsNativeClose("unknown reached original effects: " + node.details.get("reason", "boundary"))
            if node.kind not in {"operation", "call", "loop", "associate", "boundary"}:
                raise CompilationError("reached source node requires distinct control authority")
            result = leaf(identity, dict(active))
        except CompilationError as error:
            may_close(error)
            discard(before)
            return close_before(identity, str(error))
        active_ids.update(descendants(identity))
        child = next((item for item in reversed(mapped_children) if item.node_id == identity), None)
        child_closed = child is not None and child.public()["plan"]["planned_closures"]["closed_path_possible"]
        return result, has_closed_path or child_closed

    def planned_sequence(identities, state, depth=0):
        nonlocal visits
        index = 0
        while index < len(identities):
            identity = identities[index]
            joined = joined_selections.get(identity)
            if joined is not None and state[0] is not None:
                if tuple(identities[index:index + len(joined)]) != joined:
                    raise CompilationError("mapped reached selection splits an original joined group")
                visits += len(joined)
                work["visits"] += len(joined)
                if work["visits"] > graph.node_limit or depth > analysis.depth_limit:
                    raise CompilationError("mapped reached joined control traversal budget exhausted")
                before = checkpoint()
                try:
                    if identity in forced:
                        state = close_before(identity, forced[identity], joined)
                        index += len(joined)
                        continue
                    result = leaf(identity, dict(state[0]), joined_selection=joined)
                except CompilationError as error:
                    may_close(error)
                    discard(before)
                    state = close_before(identity, str(error), joined)
                    index += len(joined)
                    continue
                active_ids.update(item for child in joined for item in descendants(child))
                state = result, state[1]
                index += len(joined)
                continue
            state = planned_transfer(identity, state, depth)
            index += 1
        return state

    def disable_prefix(identity, reason, joined_selection=None):
        from compiler.frontend.source_functions import _validate_resolution, _walk

        identities = joined_selection or (identity,)
        originals = tuple(original for item in identities for original in graph.source_nodes(item))
        completion = None
        if joined_selection is not None:
            if joined_selections.get(identity) != identities:
                raise CompilationError("pre-owner disable lacks exact complete joined selection")
            completion = analysis.joined_completion(procedure, identities)
        elif (len(originals) != 1 or type(originals[0]).__name__ not in {
                "Call_Stmt", "Assignment_Stmt", "If_Construct", "If_Stmt", "Block_Nonlabel_Do_Construct",
                "Associate_Construct", "Allocate_Stmt", "Deallocate_Stmt", "Cycle_Stmt", "Exit_Stmt",
                "Return_Stmt", "Write_Stmt", "Print_Stmt", "Read_Stmt", "Stop_Stmt", "Error_Stop_Stmt"}):
            raise CompilationError("pre-owner disable requires a complete original structured statement: " + reason)
        _validate_scope(analysis, routine.scope)
        _validate_resolution(analysis, routine.scope, _walk(originals), tick=lambda _role: coordinate_work(1))
        prefix_disables[identity] = {
            "node_id": identity, "selected_node_ids": list(identities), "guard": list(graph.nodes[identity].guard),
            "original_source_sha256": _hash([str(node) for node in originals]),
            "original_source_kind": "complete_original_joined_group" if completion else type(originals[0]).__name__,
            "completion_identity": completion.identity if completion else None,
            "span": selection_span(identities), "reason": reason,
            "execution": "disable eligibility before unchanged original source; native remainder once",
            "disabled_is_absorbing": True, "context_exists": False, "reopen": False}
        return None, True

    def prefix_sequence(identities, state, depth=0):
        index = 0
        while index < len(identities):
            identity = identities[index]
            joined = joined_selections.get(identity)
            if joined is None or state[0] is None:
                state = prefix_transfer(identity, state, depth)
                index += 1
                continue
            if tuple(identities[index:index + len(joined)]) != joined:
                raise CompilationError("pre-owner joined selection is not an intact original sequence")
            work["visits"] += len(joined)
            if work["visits"] > graph.node_limit or depth > analysis.depth_limit:
                raise CompilationError("reached prefix joined control traversal budget exhausted")
            before = checkpoint()
            try:
                result = leaf(identity, dict(state[0]), before_owner=True, joined_selection=joined)
            except CompilationError as error:
                may_disable(error)
                discard(before)
                state = disable_prefix(identity, str(error), joined)
            else:
                state = result, state[1]
            index += len(joined)
        return state

    def prefix_transfer(identity, state, depth=0):
        nonlocal visits
        eligible, disabled = state
        if eligible is None:
            return state
        visits += 1
        work["visits"] += 1
        if work["visits"] > graph.node_limit or depth > analysis.depth_limit:
            raise CompilationError("reached window prefix control traversal budget exhausted")
        node = graph.nodes[identity]
        if node.kind == "sequence":
            return prefix_sequence(node.children, state, depth + 1)
        if node.kind == "branch":
            before = checkpoint()
            previous_disables = dict(prefix_disables)
            paths, false_path = [], dict(eligible)
            for condition, body in node.alternatives:
                path = dict(false_path)
                if condition is not None:
                    visits += 1
                    work["visits"] += 1
                    if work["visits"] > graph.node_limit or depth + 1 > analysis.depth_limit:
                        raise CompilationError("reached window prefix control traversal budget exhausted")
                    try:
                        path = leaf(condition, path, before_owner=True)
                    except CompilationError as error:
                        may_disable(error)
                        discard(before)
                        prefix_disables.clear()
                        prefix_disables.update(previous_disables)
                        return disable_prefix(identity, "original prefix condition is unsupported: " + str(error))
                    false_path = dict(path)
                paths.append(prefix_transfer(body, (path, False), depth + 1))
            if not node.alternatives or node.alternatives[-1][0] is not None:
                paths.append((false_path, False))
            originals = graph.source_nodes(identity)
            if len(originals) == 1 and type(originals[0]).__name__ == "If_Stmt" and any(path[1] for path in paths):
                conditional, unsafe = _single_if_action_source(analysis, routine, graph, identity,
                                                               consume=coordinate_work)
                action_id = conditional["action_node_id"]
                action_point = prefix_disables.get(action_id)
                if (action_point is None or action_id in previous_disables
                        or set(prefix_disables) - set(previous_disables) != {action_id}
                        or action_point["selected_node_ids"] != [action_id]
                        or action_point["original_source_sha256"] != conditional["action_source_sha256"]
                        or action_point["original_source_kind"] != conditional["action_source_kind"]):
                    raise CompilationError("conditional prefix disable differs from its exact original action")
                if unsafe is not None:
                    discard(before)
                    prefix_disables.clear()
                    prefix_disables.update(previous_disables)
                    return disable_prefix(identity, unsafe + ": " + action_point["reason"])
                del prefix_disables[action_id]
                prefix_disables[identity] = {**action_point, "node_id": identity, "selected_node_ids": [identity],
                    "guard": list(node.guard), "original_source_sha256": _hash([str(originals[0])]),
                    "original_source_kind": "If_Stmt", "span": selection_span((identity,)),
                    "conditional_action": conditional,
                    "execution": "original condition once; disable eligibility inside its true branch before original action once"}
            eligible_paths = [path[0] for path in paths if path[0] is not None]
            if not eligible_paths:
                return None, True
            from compiler.frontend.native_definitions import intersect_rectangles
            joined = dict(eligible_paths[0])
            for path in eligible_paths[1:]:
                joined = {root: intersect_rectangles(boxes, path[root]) for root, boxes in joined.items()}
            return joined, disabled or any(path[1] for path in paths)
        before = checkpoint()
        try:
            originals = graph.source_nodes(identity)
            allocation = len(originals) == 1 and type(originals[0]).__name__ in {"Allocate_Stmt", "Deallocate_Stmt"}
            if node.kind == "boundary" and not allocation:
                raise PreownerUnsupported("original prefix operation needs native-only eligibility: "
                                          + node.details.get("reason", "boundary"))
            if node.kind == "call" and node.details.get("procedure") not in analysis.routines:
                raise PreownerUnsupported("original prefix call has no complete source-backed effects")
            if node.kind not in {"operation", "call", "loop", "associate"} and not allocation:
                raise CompilationError("pre-owner source node requires distinct control authority")
            result = leaf(identity, dict(eligible), before_owner=True)
        except CompilationError as error:
            may_disable(error)
            discard(before)
            return disable_prefix(identity, str(error))
        return result, disabled

    prefix_disabled, activation_possible = False, True
    if planned and _mapped is None:
        state = coverage, False
        state = prefix_sequence(prefix, state)
        eligible, prefix_disabled = state
        activation_possible = eligible is not None
        coverage = eligible or {root: () for root in roots}
    else:
        for identity in prefix:
            coverage = transfer(identity, coverage, before_owner=True)
    at_entry = dict(coverage)
    if not planned and _mapped is None and any(_coverage_kind(boxes) == "sections" for boxes in at_entry.values()):
        raise CompilationError("reached prefix partial definitions require source-position section registration")
    entry_templates = dict(activation_templates)
    has_closed_path = False
    if planned:
        state = planned_sequence(selected, (coverage if activation_possible else None, False))
        active_coverage, has_closed_path = state
        has_closed_path = has_closed_path or any(item["conditional_close_before_original"] for item in numerical_demands)
        coverage = active_coverage or {root: () for root in roots}
        if set(forced) - set(closes):
            raise CompilationError("planned reached closure does not match an active complete source planning unit")
    else:
        for identity in selected:
            coverage = transfer(identity, coverage)
    activation_resources = {}
    for root, boxes in at_entry.items():
        if _coverage_kind(boxes) != "sections":
            continue
        template = entry_templates.get(root)
        if template is None:
            if _mapped is not None:
                continue  # An original caller registers before dummy rebasing.
            raise CompilationError("partial activation lacks typed original descriptor provenance: " + root)
        resource = replace(template, reads=(), writes=(), overwrites=boxes)
        if any(dependency.kind == "scalar_read" and dependency.resource in changed_scalars
               for dependency in resource.dependencies):
            raise CompilationError("partial activation bound changes before possible lazy registration: " + root)
        activation_resources[root] = resource
    if _children is not None:
        _children.extend(mapped_children)
    if _result is not None:
        _result.update(active=active_coverage if planned else coverage, closed=has_closed_path,
                       activation_sections=activation_resources)
    unmanaged_path = has_closed_path or prefix_disabled
    states = [{"resource": root, "initialized": _coverage_kind(at_entry[root]),
               "definition_coverage": _coverage_kind(at_entry[root]),
               "source_sections_on_activation": (activation_resources[root].public()
                                                  if root in activation_resources else None),
               "coverage_on_return": "unknown_after_close" if unmanaged_path else _coverage_kind(coverage[root]),
               "defined_sections_on_return": [box.public() for box in coverage[root]]
                   if not unmanaged_path and _coverage_kind(coverage[root]) == "sections" else [],
               "active_path_coverage_on_return": _coverage_kind(coverage[root]) if planned else None,
               "active_path_sections_on_return": [box.public() for box in coverage[root]]
                   if planned and _coverage_kind(coverage[root]) == "sections" else [],
               "active_path_return_is_conditional": unmanaged_path,
               "storage": "original current binding",
               "escapes": None if unmanaged_path else False,
               "allocation_changes": None if unmanaged_path else False,
               "active_interval_storage": {"escapes": False, "allocation_changes": False,
                                           "conditional_on_owner_remaining_active": planned},
               "closed_path_storage_authority": "unavailable" if unmanaged_path else None,
               "descriptor_authority": ("registered active-interval proofs only; no closed native suffix authority"
                                        if unmanaged_path else
                                        "registered source-relative prefix/window proofs; excluded tail requires close"),
               "requires_original_allocation_presence_guard": "allocatable" in bindings[root].attributes,
               "resource_kind": "original_module_allocatable" if root in module_roots else "original_entry_capture",
               "requires_runtime_layout_and_alias_guards": True,
               "address_or_shape_freshness_inferred": False} for root in roots]
    return {"schema_version": REACHED_WINDOW_VERSION, "procedure": procedure,
            "structured_identity": graph.identity, "source_identity": analysis.inputs.identity(),
            "analysis_identity": graph.authority_identity, "selected_node_ids": list(selected),
            "prefix_node_ids": list(prefix), "first_traversal_controls": list(controls),
            "prefix_eligibility": {"enabled": planned and _mapped is None,
                                   "activation_path_possible": activation_possible,
                                   "disabled_path_possible": prefix_disabled,
                                   "disable_points": list(prefix_disables.values()), "steps": prefix_steps,
                                   "disabled_is_absorbing": True, "reopen": False,
                                   "activation_coverage_is_conditional": prefix_disabled,
                                   "new_descriptor_event_destinations": "selected_window initialized none only",
                                   "new_copy_reads": "current whole coverage from invocation facts or source-defined selected_window none",
                                   "selected_window_whole_assertion_is_prefix_read_authority": False,
                                   "payload_from_allocation_presence": False},
            "descriptor_window_identities": descriptor_demands, "entry_facts_identity": _hash(facts),
            "module_descriptor_resources": list(module_roots),
            "native_prefix_proof_identities": native_prefix_demands,
            "native_definition_proof_identities": definition_demands,
            "numerical_preparation_provider": numerical_provider.public() if numerical_provider is not None else None,
            "numerical_effect_demands": numerical_demands,
            "entry_facts_authority": "independent original fact scopes preserved; selected-window whole is not new prefix-copy read authority; independent admission required",
            "source_demands": demands, "control_visits": visits, "ordered_effects": effects,
            "ordered_effect_work": effect_work,
            "mapped_work": dict(work),
            "scalar_changes": sorted(changed_scalars),
            "states": states, "lifecycle": "one invocation; dormant -> active -> spent; no reopening",
            "planned_closures": {"enabled": planned, "close_points": list(closes.values()),
                                 "active_node_ids": sorted(active_ids), "closed_path_possible": has_closed_path,
                                 "active_path_may_return": active_coverage is not None if planned else True,
                                 "conditional_return_is_unconditional_definition_summary": False,
                                 "mapped_child_plans": [child.public() for child in mapped_children]},
            "native_only_proof": not numerical_demands and all(
                child.public()["plan"]["native_only_proof"] for child in mapped_children),
            "gpu_legality_established": False,
            "execution_authorized": False, "definition_facts_reset": False,
            "requires_original_serial_caller_guard": True,
            "mandatory_close": "before the first excluded operation; before any enclosing loop backedge",
            "activation": "first traversal only; no live context during the native prefix",
            "limits": ["one original immediate sequence at top level or first traversal of one top-level counted loop",
                       "exact original assignment/perfect affine-loop partial definitions within the selected window",
                       "exact joined-prefix sections require typed original-layout first registration",
                       "calls and unsupported native footprints retain conservative whole-definition requirements",
                       "direct OpenMP groups require a separate complete-group selection contract",
                       "every payload array dependency must be included; runtime nonalias guards remain required",
                       "new dormant descriptor events require selected_window none and exact original transfer; active replacement closes",
                       "unmanaged dormant condition inputs require fixed unaliased original metadata within the existing 256-element bound",
                       "runtime registration/layout/alias guards and numerical legality required separately"]}


@dataclass(frozen=True)
class _MappedChildPlan:
    node_id: str
    original_call: object = field(repr=False, compare=False)
    procedure: str
    selection: tuple = field(repr=False, compare=False)
    coordinates: object = field(repr=False, compare=False)
    record: object = field(repr=False, compare=False)
    children: tuple = field(repr=False, compare=False)

    def public(self):
        return {"schema_version": 1, "caller_node_id": self.node_id, "procedure": self.procedure,
                "coordinate_proof": self.coordinates.public(), "plan": _thaw(self.record),
                "return_contract": "conditional on shared owning context remaining enabled",
                "unconditional_definition_summary": False}


@dataclass(frozen=True)
class ReachedWindowProof:
    procedure: str
    identity: str
    _analysis: object = field(repr=False, compare=False)
    _selection: tuple = field(repr=False, compare=False)
    _roots: tuple[str, ...] = field(repr=False, compare=False)
    _facts: object = field(repr=False, compare=False)
    _record: object = field(repr=False, compare=False)
    _planned_close_before: object = field(repr=False, compare=False)
    _children: tuple = field(repr=False, compare=False)
    _activation_resources: tuple = field(repr=False, compare=False)
    _numerical_provider: object = field(default=None, repr=False, compare=False)
    _numerical_regions: tuple = field(default=(), repr=False, compare=False)

    def public(self):
        return {"proof_identity": self.identity, **_thaw(self._record)}

    def validate(self, analysis, procedure, original_selection):
        if (analysis is not self._analysis or procedure != self.procedure
                or not isinstance(original_selection, tuple) or len(original_selection) != len(self._selection)
                or any(left is not right for left, right in zip(original_selection, self._selection, strict=True))
                or getattr(analysis, "_reached_window_proofs", {}).get(self.identity) is not self):
            raise CompilationError("reached window lacks registered exact original source authority")
        if _derive(analysis, procedure, self._selection, self._roots, _thaw(self._facts),
                   self._planned_close_before, numerical_provider=self._numerical_provider) != _thaw(self._record):
            raise CompilationError("reached window source-position authority changed")
        return self

    def numerical_region(self, procedure, original_selection):
        """Return exact typed capabilities, never authority reconstructed from JSON."""
        self.validate(self._analysis, self.procedure, self._selection)
        for owner, selected, prepared, effects in self._numerical_regions:
            if (owner == procedure and isinstance(original_selection, tuple)
                    and len(selected) == len(original_selection)
                    and all(left is right for left, right in zip(selected, original_selection, strict=True))):
                prepared.validate(self._analysis, procedure, original_selection)
                effects.validate(self._analysis, procedure, original_selection, prepared)
                return prepared, effects
        return None

    def capture_state(self, canonical_root):
        self.validate(self._analysis, self.procedure, self._selection)
        state = next((state for state in _thaw(self._record)["states"] if state["resource"] == canonical_root), None)
        if state is None:
            raise CompilationError("reached window capture root was not proved")
        return state

    def activation_sections(self, canonical_root):
        """Authenticated typed coverage, never thawed public geometry.

        Whole/none use the existing registration contract. Partial coverage is
        expressed in original logical coordinates and needs checked conversion
        against the guarded original descriptor before first registration.
        """
        state = self.capture_state(canonical_root)
        resource = next((item for root, item in self._activation_resources if root == canonical_root), None)
        if state["initialized"] != "sections":
            if resource is not None:
                raise CompilationError("whole/empty activation unexpectedly carries partial source authority")
            return None
        if (not isinstance(resource, NativeResourceSections) or resource.resource != canonical_root
                or resource.reads or resource.writes or resource.public() != state["source_sections_on_activation"]):
            raise CompilationError("partial activation lacks exact typed source-section authority")
        return resource

    def child_plan(self, path, original_call):
        """Authenticate a nested original call without inventing child facts."""
        self.validate(self._analysis, self.procedure, self._selection)
        children = self._children
        selected = None
        for identity in path:
            selected = next((child for child in children if child.node_id == identity), None)
            if selected is None:
                raise CompilationError("mapped child call path lacks registered reached authority")
            children = selected.children
        if selected is None or selected.original_call is not original_call:
            raise CompilationError("mapped child requires its exact original call and path")
        selected.coordinates.validate(self._analysis, selected.coordinates.caller, original_call)
        return selected


def _prove(analysis, procedure, original_selection, requested_roots, entry_definition_facts, close_before,
           numerical_provider=None):
    children, result, numerical_regions = [], {}, []
    record = _derive(analysis, procedure, original_selection, requested_roots, entry_definition_facts, close_before,
                     numerical_provider=numerical_provider, _children=children, _result=result,
                     _numerical_regions=numerical_regions)
    identity = _hash(record)
    registry = getattr(analysis, "_reached_window_proofs", None)
    if registry is None:
        registry = analysis._reached_window_proofs = {}
    if identity in registry:
        return registry[identity].validate(analysis, procedure, original_selection)
    if len(registry) >= analysis.procedure_limit:
        raise CompilationError("reached window proof registry budget exhausted")
    proof = ReachedWindowProof(procedure, identity, analysis, original_selection, requested_roots,
                               _freeze(entry_definition_facts), _freeze(record), close_before, tuple(children),
                               tuple(result["activation_sections"].items()), numerical_provider, tuple(numerical_regions))
    registry[identity] = proof
    return proof


def prove_reached_window(analysis, procedure, original_selection, requested_roots, entry_definition_facts):
    """Transfer supplied entry facts; their truth requires separate admission.

    Registration authenticates original source and transfer, not invocation
    state, execution legality, owner creation or runtime resources.
    """
    return _prove(analysis, procedure, original_selection, requested_roots, entry_definition_facts, None)


def plan_reached_window(analysis, procedure, original_selection, requested_roots, entry_definition_facts,
                        *, close_before=(), numerical_provider=None):
    """Plan conditional active intervals and exact native closure points.

    Eligible prefixes retain source-proven definitions; reaching unsupported
    original prefix source disables that invocation before activation. Each
    still-active unit is checked normally; after an authenticated close only
    original native code may execute. Ordinary complete summaries stay strict.
    """
    return _prove(analysis, procedure, original_selection, requested_roots, entry_definition_facts, close_before,
                  numerical_provider)
