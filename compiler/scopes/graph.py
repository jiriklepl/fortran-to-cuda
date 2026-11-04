"""Graph execution of an original mixed CPU/CUDA caller.

Discovery admits independently bounded numerical/native tasks on demand. Source
windows, their candidate enumeration, and their aggregate outline budgets do not
define resource lifetime. The original invocation owns coherent runtime epochs, and
existing source/effect/definition issuers still authorize every worker access.

Checked ready-task scheduling groups numerical work around total integer native
work in the same control region. Source branches, counted controls, optional
arguments, and opaque continuations retain their original execution points; no
completed source operation is replayed after a boundary.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from fparser.two.utils import walk

from compiler.analysis.task_dependencies import build_task_graph, structured_task
from compiler.frontend.source_effects import _children, _kind, _part
from compiler.ir import CompilationError
from compiler.ir.task_graph import BranchRegion, ExecutionTarget, LoopRegion, ResourceAccess, Task, TaskEffects
from compiler.scheduling.task_graph import schedule_tasks, verify_schedule
from compiler.scopes.lexical import LexicalOwner, check_structured_entry
from compiler.scopes.segments import statement_span


def _scope_effects(member, scope, arrays, scalars):
    """Project admitted effects, never reconstruct them from source INTENT."""
    actions = defaultdict(set)
    scalar_reads, scalar_writes, metadata = set(), set(), set()
    reserved = {root: binding for operation in scope.native for root, binding in operation.host_metadata.items()}

    def canonical(root):
        binding = arrays.get(root) or scalars.get(root) or reserved.get(root)
        return member.canonical(binding).root if binding is not None else root

    for call in scope.calls:
        effects, _definitions, _overwrites = member.builder.roots_for(call)
        for root, kinds in effects.items():
            actions[canonical(root)].update(kinds)
    for operation in scope.native:
        if operation.call_effects is not None:
            for binding, role, kinds in operation.call_effects.resources:
                if role != "constant":
                    actions[canonical(binding.root)].update(kinds & {"read", "write"})
        for root, kinds in operation.effects.items():
            actions[canonical(root)].update(kinds)
        for item in operation.summary.get("operations", ()):
            root = item.get("resource")
            if root is None:
                continue
            root = canonical(root)
            if item["kind"] == "descriptor_read":
                metadata.add(root)
            elif not item.get("rank", 0):
                if item["kind"] == "read":
                    scalar_reads.add(root)
                elif item["kind"] in {"write", "overwrite"}:
                    scalar_writes.add(root)
    # Query controls and captured scalars remain at their reached evaluation
    # point. Treating all captures as reads is conservative for future motion.
    scalar_reads.update(canonical(root) for root in scalars)
    return TaskEffects(
        reads=tuple(ResourceAccess(root) for root, kinds in sorted(actions.items()) if "read" in kinds),
        may_writes=tuple(ResourceAccess(root) for root, kinds in sorted(actions.items()) if "write" in kinds),
        scalar_reads=scalar_reads,
        scalar_writes=scalar_writes,
        metadata_reads=metadata,
        # Whole-overwrite hints are not sufficient to issue a new proof here.
        # Existing exact region/definition proofs remain in runtime emission.
    )


def _total_integer_source(member, originals):
    """Prove a small typed native subset total for a defined Fortran execution.

    Original effect/completion/storage proofs have already admitted this work.
    This additional motion check excludes floating-point state, division, calls,
    optional/pointer/volatile accesses, lifetime changes, and unstructured exits.
    Accesses keep the same original control path and use their authenticated
    native footprints; no work is speculated onto an untaken branch/iteration.
    """
    analysis, routine = member.builder.analysis, member.routine
    forbidden = {"optional", "pointer", "volatile", "asynchronous", "allocatable", "value"}
    scope = routine.scope
    while scope is not None:
        if any(
            _kind(node) in {"Common_Stmt", "Equivalence_Stmt"} for node in walk(_part(scope.node, "Specification_Part"))
        ):
            return False
        scope = scope.parent

    def binding(node):
        scope = analysis.source_scope_for(node, routine.scope)
        value = analysis._binding(scope, node)
        return (
            value
            if value is not None and value.dtype in {"integer", "logical"} and not (value.attributes & forbidden)
            else None
        )

    operators = {
        "+",
        "-",
        "*",
        "(",
        ")",
        "==",
        "/=",
        "<",
        ">",
        "<=",
        ">=",
        ".eq.",
        ".ne.",
        ".lt.",
        ".gt.",
        ".le.",
        ".ge.",
        ".and.",
        ".or.",
        ".not.",
        ".eqv.",
        ".neqv.",
    }

    def expression(node):
        kind = _kind(node)
        if kind in {"Int_Literal_Constant", "Logical_Literal_Constant"}:
            return True
        if kind == "Name":
            value = binding(node)
            return value is not None and not value.rank
        if kind == "Part_Ref":
            value = binding(node.items[0])
            indices = getattr(node.items[1], "items", ())
            return value is not None and value.rank == len(indices) and all(expression(index) for index in indices)
        if kind not in {
            "Level_1_Expr",
            "Level_2_Expr",
            "Level_2_Unary_Expr",
            "Level_3_Expr",
            "Level_4_Expr",
            "Level_5_Expr",
            "And_Operand",
            "Or_Operand",
            "Equiv_Operand",
            "Parenthesis",
            "Add_Operand",
            "Mult_Operand",
        }:
            return False
        return all(
            item.lower() in operators if isinstance(item, str) else expression(item)
            for item in getattr(node, "items", ())
        )

    def statement(node):
        kind = _kind(node)
        if kind == "Comment":
            return not str(node).lstrip().startswith("!$")
        if kind == "Assignment_Stmt":
            return expression(node.items[0]) and expression(node.items[2])
        if kind != "Block_Nonlabel_Do_Construct":
            return False
        content = tuple(node.content)
        header = next((item for item in content if _kind(item) == "Nonlabel_Do_Stmt"), None)
        control = header.items[1] if header is not None else None
        counted = control.items[1] if control is not None and _kind(control) == "Loop_Control" else None
        if not counted or _kind(content[-1]) != "End_Do_Stmt":
            return False
        iterator, bounds = counted
        if (
            binding(iterator) is None
            or not all(expression(bound) for bound in bounds)
            or len(bounds) > 2
            and (_kind(bounds[2]) != "Int_Literal_Constant" or int(str(bounds[2]).split("_")[0]) == 0)
        ):
            return False
        begin = content.index(header)
        return all(statement(item) for item in (*content[:begin], *content[begin + 1 : -1]))

    return bool(originals) and all(statement(node) for node in originals)


def _member_graph(member):
    from compiler.scopes.combined_team import CombinedNumericalTeam
    from compiler.scopes.joined_team import JoinedTeam

    source = member.graph
    selections = defaultdict(list)
    skipped = set()
    unit_ids = {}
    spans = {}
    bindings = {
        member.canonical(binding).root: binding
        for unit in member.units
        for binding in (*unit.arrays.values(), *unit.scalars.values())
    }
    gpu_storage = {
        member.canonical(binding).root: binding
        for unit in member.units
        if unit.leaves
        for binding in unit.arrays.values()
    }

    def alias_safe(effects):
        # Source movement happens before per-worker registration guards. Credit
        # distinct argument identities only when their declared numeric types
        # preclude overlapping storage in a defined Fortran invocation. Same-
        # type aliases need a separate dominating preflight and remain ordered.
        touched = effects.data_resources | effects.scalar_reads | effects.scalar_writes
        for root in touched:
            binding = bindings.get(root)
            if binding is None:
                return False
            if any(
                other != root and (binding.dtype, binding.kind) == (storage.dtype, storage.kind)
                for other, storage in gpu_storage.items()
            ):
                return False
        return True

    for index, unit in enumerate(member.units):
        originals = member.builder.inline.original_selection(unit.nodes)
        graph, selected = member.builder.analysis._selected_source(member.routine.qualified, originals)
        if graph is not source or not selected:
            raise CompilationError("graph unit lost its original source selection")
        task_id = f"{member.routine.qualified}:unit:{index}"
        unit_ids[str(index)] = task_id
        effects = _scope_effects(member, unit.scope, unit.arrays, unit.scalars)
        if (
            not unit.leaves
            and (not _total_integer_source(member, originals) or not alias_safe(effects))
            or unit.leaves
            and any(operation.kind != "guarded original numerical fallback" for operation in unit.scope.native)
            or isinstance(unit.scope, (CombinedNumericalTeam, JoinedTeam))
            or any(operation.preserve_original for operation in unit.scope.native)
        ):
            effects = replace(effects, ordered=True)
        task = Task(
            task_id,
            effects,
            frozenset({ExecutionTarget.CPU, ExecutionTarget.GPU}) if unit.leaves else frozenset({ExecutionTarget.CPU}),
            payload=unit,
        )
        selections[selected[0]].append(task)
        skipped.update(selected[1:])
        spans[task_id] = unit.scope.first, unit.scope.last
    native_ids = set()
    for index, external in enumerate(member.device_tasks):
        external.proof.validate(member.builder.analysis, member.routine, external.node)
        graph, selected = member.builder.analysis._selected_source(member.routine.qualified, (external.node,))
        if graph is not source or not selected:
            raise CompilationError("external device task lost original graph authority")
        task_id = f"{member.routine.qualified}:external:{index}"
        actions = {member.canonical(binding).root: set() for binding in external.proof.bindings}
        for resource, binding in zip(external.proof.public()["contract"]["resources"], external.proof.bindings, strict=True):
            actions[member.canonical(binding).root].update(effect["kind"] for effect in resource["effects"])
        # Exact physical boxes stay in the provider contract/transaction. Whole
        # conservative hazards do not grant whole-root definition credit.
        effects = TaskEffects(reads=tuple(ResourceAccess(root) for root, kinds in actions.items() if "read" in kinds),
            may_writes=tuple(ResourceAccess(root) for root, kinds in actions.items() if kinds & {"write", "overwrite"}),
            metadata_reads=frozenset(actions), ordered=True)
        selections[selected[0]].append(Task(task_id, effects, frozenset({ExecutionTarget.GPU}), payload=external))
        skipped.update(selected[1:])
        spans[task_id] = external.first, external.last
        native_ids.add(task_id)
    for index, source_task in enumerate(member.source_device_tasks):
        source_task.proof.validate(member.builder.analysis, member.routine, source_task.nodes)
        graph, selected = member.builder.analysis._selected_source(member.routine.qualified, source_task.nodes)
        if graph is not source or not selected:
            raise CompilationError("source device task lost its complete original graph selection")
        task_id = f"{member.routine.qualified}:source_device:{index}"
        from compiler.scopes.source_task_bindings import source_task_actions

        actions = source_task_actions(member, source_task)
        effects = TaskEffects(reads=tuple(ResourceAccess(root) for root in actions),
            may_writes=tuple(ResourceAccess(root) for root, written in actions.items() if written),
            scalar_reads=frozenset(binding.root for binding in source_task.proof.team.scalars),
            metadata_reads=frozenset(actions), ordered=True)
        targets = frozenset({ExecutionTarget.GPU})
        deferred = None
        if source_task.placement is not None:
            source_task.placement.validate(member)
            if source_task.placement.public()["task_id"] != task_id:
                raise CompilationError("source task placement changed its exact graph task")
            targets = frozenset({ExecutionTarget.CPU, ExecutionTarget.GPU})
            deferred = source_task.placement.deferred()
        selections[selected[0]].append(Task(task_id, effects, targets, payload=source_task,
                                           deferred_placement=deferred))
        skipped.update(selected[1:])
        spans[task_id] = source_task.first, source_task.last
        native_ids.add(task_id)
    for index, transfer in enumerate(member.storage_transfers):
        transfer.proof.validate(member.builder.analysis, member.routine, transfer.node)
        graph, selected = member.builder.analysis._selected_source(member.routine.qualified, (transfer.node,))
        if graph is not source or not selected:
            raise CompilationError("allocation exchange lost original graph authority")
        task_id = f"{member.routine.qualified}:storage:{index}"
        effects = TaskEffects(lifetime_resources=frozenset(member.canonical(binding).root
                                                         for binding in transfer.proof.bindings), ordered=True)
        selections[selected[0]].append(Task(task_id, effects, payload=transfer))
        skipped.update(selected[1:])
        spans[task_id] = transfer.first, transfer.last
        native_ids.add(task_id)
    for index, (nodes, scope) in enumerate(member.graph_native_tasks):
        originals = member.builder.inline.original_selection(nodes)
        graph, selected = member.builder.analysis._selected_source(member.routine.qualified, originals)
        if graph is not source or not selected:
            raise CompilationError("retained native graph task lost original source authority")
        task_id = f"{member.routine.qualified}:native:{index}"
        native_bindings = {root: binding for operation in scope.native for root, binding in operation.bindings.items()}
        bindings.update({member.canonical(binding).root: binding for binding in native_bindings.values()})
        effects = _scope_effects(member, scope, {}, native_bindings)
        # An original single-line IF action has exact graph authority but no
        # independently editable source span. Retain its original execution;
        # only physically editable statements can become motion candidates.
        editable = all(any(getattr(item, "item", None) for item in walk(node))
                       for node in (originals[0], originals[-1]))
        span = ((statement_span(originals[0])[0], statement_span(originals[-1])[1])
                if editable else None)
        if span is None or not _total_integer_source(member, originals) or not alias_safe(effects):
            effects = replace(effects, ordered=True)
        task = Task(task_id, effects, payload=originals)
        selections[selected[0]].append(task)
        skipped.update(selected[1:])
        if span is not None:
            spans[task_id] = span
        native_ids.add(task_id)
    used = set()

    def graph_for(tasks, name):
        return build_task_graph(tuple(tasks), name=name)

    def visit(identity):
        node = source.nodes[identity]
        if identity in selections:
            tasks = selections[identity]
            used.update(task.id for task in tasks)
            return graph_for(tasks, identity)
        if identity in skipped or node.kind == "entry":
            return graph_for((), identity)
        if node.kind == "sequence":
            return graph_for((task for child in node.children for task in visit(child).tasks), identity)
        if node.kind == "associate":
            tasks = tuple(task for child in node.children for task in visit(child).tasks)
            return graph_for((*tasks, Task(identity + ":end", TaskEffects(ordered=True))), identity)
        if node.kind == "branch":
            following = graph_for((), identity + ":otherwise")
            for ordinal in reversed(range(len(node.alternatives))):
                condition, body = node.alternatives[ordinal]
                body_graph = visit(body)
                if condition is None:
                    following = body_graph
                    continue
                predicate = source.source_nodes(condition)
                condition_graph = visit(condition)
                branch = structured_task(
                    f"{identity}:alternative:{ordinal}",
                    BranchRegion(predicate, body_graph, following),
                    header_effects=TaskEffects(ordered=True),
                )
                following = graph_for((*condition_graph.tasks, branch), f"{identity}:guard:{ordinal}")
            return following
        if node.kind == "loop":
            originals = source.source_nodes(identity)
            original = originals[0] if len(originals) == 1 else None
            header = next(
                (child for child in getattr(original, "content", ()) if _kind(child) == "Nonlabel_Do_Stmt"), None
            )
            control = header.items[1] if header is not None else None
            counted = control.items[1] if control is not None and _kind(control) == "Loop_Control" else None
            if counted and node.children:
                iterator, bounds = counted
                prefix = tuple(task for child in node.children[:-1] for task in visit(child).tasks)
                body_graph = visit(node.children[-1])
                lower, upper = bounds[:2]
                step = bounds[2] if len(bounds) > 2 else 1
                loop = structured_task(
                    identity,
                    LoopRegion(str(iterator), lower, upper, step, body_graph),
                    header_effects=TaskEffects(scalar_writes={str(iterator)}, ordered=True),
                )
                return graph_for((*prefix, loop), identity)
            # Unsupported control remains opaque and original. Admission could
            # only authorize the whole operation, never a speculative body.
        return graph_for((Task(identity, TaskEffects(opaque=True), payload=source.source_nodes(identity)),), identity)

    graph = visit(source.root_id)
    if used != set(unit_ids.values()) | native_ids:
        raise CompilationError("source graph does not cover every admitted execution unit exactly once")
    return graph, unit_ids, spans


def _motion_runs(graph, schedule, spans):
    """Return changed same-control-region runs in the already checked schedule."""
    placements = {placement.task_id: placement for placement in schedule.placements}
    run = []

    def finish():
        if len(run) < 2:
            return ()
        original = tuple(task.id for task in run)
        planned = tuple(placement.task_id for placement in schedule.placements if placement.task_id in original)
        return ((original, planned),) if original != planned else ()

    for task in graph.tasks:
        if task.structure is None and not task.effects.opaque and not task.effects.ordered and task.id in spans:
            run.append(task)
            continue
        yield from finish()
        run = []
        if isinstance(task.structure, BranchRegion):
            children = task.structure.then_graph, task.structure.else_graph
        elif isinstance(task.structure, LoopRegion):
            children = (task.structure.body,)
        else:
            children = ()
        for child, child_schedule in zip(children, placements[task.id].children, strict=True):
            yield from _motion_runs(child, child_schedule, spans)
    yield from finish()


def _apply_source_schedule(owner):
    """Move the actual generated/native fragments according to the sealed plan.

    Units retain original proof/trace provenance. Context/declaration insertions
    remain outside the moved sequence; original path/loop wrappers are never
    part of a movable run. No guessed source text is used as an effect proof.
    """
    builder = owner.builder
    motions = []
    for name, member in owner.members.items():
        graph, schedule = owner.task_graphs[name], owner.task_schedules[name]
        verify_schedule(graph, schedule)
        spans = owner.graph_task_spans[name]
        for original, planned in _motion_runs(graph, schedule, spans):
            ranges = [spans[task_id] for task_id in original]
            if any(left[1] >= right[0] for left, right in zip(ranges, ranges[1:], strict=False)):
                raise CompilationError("source motion requires non-overlapping original task spans")
            for left, right in zip(ranges, ranges[1:], strict=False):
                gap = member.lines[left[1] : right[0] - 1]
                if any(line.strip() and not line.lstrip().startswith("!") for line in gap):
                    raise CompilationError("source motion crosses an unrepresented original operation")
                if any(line.lstrip().startswith("!$") for line in gap):
                    raise CompilationError("source motion crosses an original directive")
            first, last = ranges[0][0], ranges[-1][1]
            edits = builder.edits.get(member.routine.scope.path, [])
            selected = []
            for edit in edits:
                lower, upper, _text = edit
                inside = first <= lower <= upper <= last or first < lower <= last and upper == lower - 1
                overlaps = lower <= last and first <= upper
                if overlaps and not inside:
                    raise CompilationError("source motion overlaps an enclosing original-source edit")
                if inside:
                    selected.append(edit)
            fragments = {}
            previous = first
            for task_id, (lower, upper) in zip(original, ranges, strict=True):
                lines = list(member.lines[lower - 1 : upper])
                grouped = defaultdict(list)
                for edit_first, edit_last, replacement in selected:
                    if (
                        lower <= edit_first <= edit_last <= upper
                        or lower < edit_first <= upper
                        and edit_last == edit_first - 1
                    ):
                        grouped[edit_first, edit_last].append(replacement)
                prior = upper + 1
                for (edit_first, edit_last), replacements in sorted(grouped.items(), reverse=True):
                    if edit_last >= prior:
                        raise CompilationError("source task contains overlapping generated edits")
                    lines[edit_first - lower : edit_last - lower + 1] = ["".join(replacements)]
                    prior = edit_first if edit_last >= edit_first else prior
                prefix = "".join(member.lines[previous - 1 : lower - 1])
                fragments[task_id] = prefix + "".join(lines)
                previous = upper + 1
            replacement = "".join(fragments[task_id] for task_id in planned)
            for edit in selected:
                edits.remove(edit)
            builder.add_edit(member.routine.scope.path, first, last, replacement)
            motions.append(
                {
                    "procedure": name,
                    "first_line": first,
                    "last_line": last,
                    "original_tasks": list(original),
                    "scheduled_tasks": list(planned),
                    "authority": "sealed source/effect proofs and verified task dependencies",
                    "native_motion_proof": "total INTEGER/LOGICAL source; same original control region",
                }
            )
    return motions


class GraphOwner(LexicalOwner):
    """One source invocation, with a checked hierarchical execution inventory."""

    def _scan_group(self, group, depth):
        epoch = getattr(self, "position_epoch", None)
        if epoch is None:
            return super()._scan_group(group, depth)
        originals = self.builder.inline.original_selection(group if isinstance(group, tuple) else (group,))
        boundaries = (epoch.boundaries() if hasattr(epoch, "boundaries") else
                      (getattr(epoch, "_boundary", None),))
        if any(boundary is not None and len(originals) == len(boundary.originals)
               and all(left is right for left, right in zip(originals, boundary.originals, strict=True))
               for boundary in boundaries):
            return  # The retained source boundary executes this original span once.
        previous = getattr(self.builder, "_graph_storage_epoch", None)
        self.builder._graph_storage_epoch = epoch if epoch.covers(self.builder.analysis, self.routine.qualified,
                                                                 originals) else None
        try:
            return super()._scan_group(group, depth)
        finally:
            self.builder._graph_storage_epoch = previous

    def validate_plan(self, nodes):
        graphs, schedules, unit_ids, spans = {}, {}, {}, {}
        for name, member in self.members.items():
            graph, identities, task_spans = _member_graph(member)
            required = {
                identities[str(index)]: ExecutionTarget.GPU if unit.leaves else ExecutionTarget.CPU
                for index, unit in enumerate(member.units)
            }
            schedule = schedule_tasks(graph, required_targets=required, target_switch_cost=0.25)
            verify_schedule(graph, schedule)
            graphs[name], schedules[name], unit_ids[name] = graph, schedule, identities
            spans[name] = task_spans
        self.task_graphs, self.task_schedules, self.graph_unit_ids, self.graph_task_spans = (
            graphs,
            schedules,
            unit_ids,
            spans,
        )
        from compiler.scopes.query_regions import discover_query_regions
        discover_query_regions(self)

    def _emit_lexical(self, nodes, checkpoint, *, native_cleanup=True):
        previous = getattr(self.builder, "_graph_storage_epoch", None)
        self.builder._graph_storage_epoch = getattr(self, "position_epoch", None)
        try:
            return self._emit_position_epoch(nodes, checkpoint, native_cleanup=native_cleanup)
        finally:
            self.builder._graph_storage_epoch = previous

    def _emit_position_epoch(self, nodes, checkpoint, *, native_cleanup=True):
        for name, graph in self.task_graphs.items():
            verify_schedule(graph, self.task_schedules[name])
        accepted = super()._emit_lexical(nodes, checkpoint, native_cleanup=native_cleanup)
        if accepted and not getattr(self.emission, "owned_graph_record_finished", False):
            self._finish_graph_record()
        return accepted

    def _finish_prepared_graph_record(self):
        """Resolve graph motions and diagnostics before owned image issuance."""
        if getattr(self.emission, "owned_graph_record_finished", False):
            raise CompilationError("owned emission graph record was already prepared")
        self._finish_graph_record()
        self.emission.owned_graph_record_finished = True

    def _finish_graph_record(self):
        record = self.builder.scopes[-1]
        external = [task.public(member) for member in self.members.values() for task in member.device_tasks]
        if external:
            record["external_device_tasks"] = external
            record["estimate_available"] = False
            record["planning_reason"] = "opaque external device provider is explicitly selected and unpriced"
        motions = _apply_source_schedule(self) if record.get("owner_variant") is not None else []
        if record.get("owner_variant") is not None:
            from compiler.scopes.query_regions import emit_query_regions
            emit_query_regions(self, record, motions)
        record["graph_execution"] = {
            "schema_version": 1,
            "source_order_executor": not motions,
            "source_motion": motions,
            "cost_units": "relative scheduling heuristic; no wall-time calibration",
            "source_windows": False,
            "resource_lifetime": "coherent epochs within one original invocation, including counted-loop backedges",
            "unknown_boundary": "publish and close; continue discovery; reopen with zero initialized coverage",
            "epochs": {"entry_coverage": ("none at selected-position allocation activation"
                                          if getattr(self, "position_epoch", None) is not None else
                                          "reviewed entry facts only before the first close"),
                       "reopened_coverage": "none; exact reached source reads admit only missing required values",
                       "handle_reuse": "validate current-context membership before descriptor registration"},
            "procedures": [
                {
                    "procedure": name,
                    "task_graph": graph.public(),
                    "schedule": self.task_schedules[name].public(),
                    "admitted_unit_tasks": dict(self.graph_unit_ids[name]),
                }
                for name, graph in self.task_graphs.items()
            ],
        }
        if record.get("owner_variant") is not None:
            expected = {task_id for identities in self.graph_unit_ids.values() for task_id in identities.values()}
            if getattr(self.emission, "graph_emitted_units", set()) != expected:
                raise CompilationError("source emission omitted a checked graph unit")
            expected_source = {patch.placement.identity for member in self.members.values()
                               for patch in member.source_device_tasks if patch.placement is not None}
            if getattr(self.emission, "graph_emitted_source_tasks", set()) != expected_source:
                raise CompilationError("source emission omitted a checked deferred source task")


def validate_scheduled_unit(member, unit):
    """Emission consumes the selected task, rather than rediscovering a window."""
    owner = member.owner
    if not isinstance(owner, GraphOwner):
        raise CompilationError("graph emission requires its owning checked source graph")
    name = member.routine.qualified
    graph = owner.task_graphs[name]
    ordinal = next((index for index, candidate in enumerate(member.units) if candidate is unit), None)
    if ordinal is None:
        raise CompilationError("graph worker is not an admitted source unit")
    task_id = owner.graph_unit_ids[name][str(ordinal)]

    def selected(current, schedule):
        for task, placement in zip(
            (next(item for item in current.tasks if item.id == placement.task_id) for placement in schedule.placements),
            schedule.placements,
            strict=True,
        ):
            if task.id == task_id:
                return task
            if isinstance(task.structure, BranchRegion):
                children = task.structure.then_graph, task.structure.else_graph
            elif isinstance(task.structure, LoopRegion):
                children = (task.structure.body,)
            else:
                children = ()
            for child, child_schedule in zip(children, placement.children, strict=True):
                found = selected(child, child_schedule)
                if found is not None:
                    return found
        return None

    task = selected(graph, owner.task_schedules[name])
    if task is None or task.payload is not unit:
        raise CompilationError("graph schedule lost its exact admitted worker")
    seen = getattr(owner.emission, "graph_emitted_units", None)
    if seen is None:
        seen = owner.emission.graph_emitted_units = set()
    if task_id in seen:
        raise CompilationError("source graph worker emitted more than once")
    seen.add(task_id)


def validate_scheduled_source_task(member, patch):
    """The renderer consumes the sealed graph choice, not a local rediscovery."""
    owner = member.owner
    if not isinstance(owner, GraphOwner) or patch.placement is None:
        raise CompilationError("source placement requires its owning checked source graph")
    patch.placement.validate(member)
    record = patch.placement.public()
    graph = owner.task_graphs[member.routine.qualified]
    schedule = owner.task_schedules[member.routine.qualified]
    verify_schedule(graph, schedule)

    def selected(current, planned):
        for placement in planned.placements:
            task = next(task for task in current.tasks if task.id == placement.task_id)
            if task.id == record["task_id"]:
                if (task.payload is not patch or task.deferred_placement is None
                        or task.deferred_placement.public() != patch.placement.deferred().public()
                        or placement.target is not None or placement.deferred_placement is None
                        or placement.deferred_placement.public() != task.deferred_placement.public()):
                    raise CompilationError("source emission contradicts its sealed deferred graph placement")
                return True
            children = ((task.structure.then_graph, task.structure.else_graph) if isinstance(task.structure, BranchRegion)
                        else (task.structure.body,) if isinstance(task.structure, LoopRegion) else ())
            for child, child_schedule in zip(children, placement.children, strict=True):
                if selected(child, child_schedule):
                    return True
        return False

    if not selected(graph, schedule):
        raise CompilationError("source graph lost its exact deferred source task")
    seen = getattr(owner.emission, "graph_emitted_source_tasks", None)
    if seen is None:
        seen = owner.emission.graph_emitted_source_tasks = set()
    if patch.placement.identity in seen:
        raise CompilationError("checked deferred source task emitted more than once")
    seen.add(patch.placement.identity)


def run_graph_owner(builder):
    """Compile an entire original caller without enumerating source windows."""
    reason = builder._missing_automatic_cost_reason()
    if reason:
        builder.boundaries.append(
            {"procedure": builder.entry.qualified, "reason": reason, "execution": "unchanged native source"}
        )
        return builder.finish(materialize_native_effects=False)
    try:
        check_structured_entry(builder.entry)
    except CompilationError as error:
        builder.boundaries.append(
            {"procedure": builder.entry.qualified, "reason": str(error), "execution": "unchanged native source"}
        )
        return builder.finish(materialize_native_effects=False)
    if builder.entry.source_kind != "module":
        builder.boundaries.append(
            {
                "procedure": builder.entry.qualified,
                "reason": "graph owner requires an original module procedure entry",
                "execution": "unchanged native source",
            }
        )
        return builder.finish(materialize_native_effects=False)
    selection = builder._native_ancestor_selection
    if selection is not None and selection.public().get("lifetime") == "supported_original_path":
        from compiler.scopes.supported_paths import run_supported_path
        return run_supported_path(builder)
    if selection is not None and selection.public().get("lifetime") == "owner_local_call_groups":
        from compiler.scopes.call_groups import run_owner_local_call_groups
        return run_owner_local_call_groups(builder)
    from compiler.scopes.hierarchical_regions import try_hierarchical_regions
    if try_hierarchical_regions(builder):
        return builder.finish(materialize_native_effects=False)
    from compiler.scopes.generation_transaction import GenerationTransaction

    candidate, owner = None, None
    attempt = None
    try:
        with GenerationTransaction(builder) as transaction:
            candidate = transaction.builder
            owner = GraphOwner(candidate)
            from compiler.scopes.conditional_return import prepare as prepare_return
            prepare_return(owner)
            from compiler.scopes.graph_epochs import discover_graph_storage_epoch
            if getattr(candidate, "_conditional_return_draft", None) is None:
                discover_graph_storage_epoch(owner)
            from compiler.scopes.native_ancestor import prepare
            nodes = prepare(candidate, _children(candidate.entry.execution))
            candidate.generated.update(candidate.inline.generated)
            candidate.numerical_ir.update(candidate.inline.ir)
            candidate.numerical_reasons.update({name: None for name in candidate.inline.regions})
            accepted = owner.run(nodes)
            attempt = {"procedure": candidate.entry.qualified, "admitted": bool(accepted),
                       "storage_epoch_search": getattr(owner, "position_epoch_search", None),
                       "boundaries": [dict(item) for member in owner.members.values() for item in member.boundaries],
                       "numerical_team_declines": [dict(item) for member in owner.members.values()
                                                   for item in member.numerical_team_declines]}
            if accepted:
                candidate.graph_owner_attempt = attempt
                transaction.adopt(candidate)
        if not accepted:
            builder.graph_owner_attempt = attempt
            from compiler.scopes.numerical_roots import admit_numerical_roots

            if builder._native_ancestor_selection is None and admit_numerical_roots(builder):
                return builder.finish(materialize_native_effects=False)
            builder.boundaries.append(
                {
                    "procedure": builder.entry.qualified,
                    "reason": "no admitted numerical tasks in graph owner",
                    "execution": "unchanged native source",
                }
            )
    except CompilationError as error:
        if owner is not None:
            builder.graph_owner_attempt = {"procedure": builder.entry.qualified, "admitted": False,
                "storage_epoch_search": getattr(owner, "position_epoch_search", None),
                "error": str(error), "boundaries": [dict(item) for member in owner.members.values()
                                                      for item in member.boundaries],
                **({"conditional_return_barrier": error.conditional_return_barrier}
                   if hasattr(error, "conditional_return_barrier") else {})}
        builder.boundaries.append(
            {
                "procedure": builder.entry.qualified,
                "reason": "graph owner unavailable: " + str(error),
                "execution": "unchanged native source; no partial publication",
            }
        )
    return builder.finish(materialize_native_effects=False)
