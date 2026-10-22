"""Closed owners for complete original numerical teams at their reached sites.

The surrounding procedure, branches and scalar prefix stay native. These owners
reuse the checked lexical plan and original-team emitter; they do not borrow
ownership across an unsupported native operation or assume a whole-entry effect
closure. A failed preparation closes before the original team executes once.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace

from fparser.two.utils import walk

from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.resource_reachability import prove_resource_reachability
from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.combined_team import CombinedNumericalTeam, is_combined_group
from compiler.scopes.lexical import LexicalOwner
from compiler.scopes.segments import fortran_lines, statement_span
from compiler.scopes.source import _name


class ClosedSelectionBoundary:
    """A scan fence only; its original group is never replaced by this object."""

    def __init__(self, originals):
        first, last = statement_span(originals[0])[0], statement_span(originals[-1])[1]
        self.item = SimpleNamespace(span=(first, last), fort_original_span=(first, last))

    def tofortran(self, tab="", isfix=None):
        return tab + "! compiler-owned closed selection scan boundary"

    def __str__(self):
        return self.tofortran()


def has_residual_candidates(builder, selections):
    covered = {id(node) for selection in selections for original in selection for node in walk(original)}
    for node in walk(builder.entry.execution):
        if id(node) in covered:
            continue
        if _kind(node) in {"Call_Stmt", "Block_Nonlabel_Do_Construct"}:
            return True
        if _kind(node) == "Assignment_Stmt":
            target = node.items[0]
            base = target.items[0] if _kind(target) == "Part_Ref" else target
            binding = builder.analysis._binding(builder.entry.scope, base)
            if binding is not None and binding.rank:
                return True
    return False


def selection_context(builder, originals, prefix, controls):
    """Source-only selection identity, shared by discovery and checked owners."""
    _, selected = builder.analysis._selected_source(builder.entry.qualified, tuple(originals))
    first, last = statement_span(originals[0])[0], statement_span(originals[-1])[1]
    text = "".join(builder.analysis.source_text(builder.entry.scope.path).splitlines(keepends=True)[first-1:last])
    return {
        "schema_version": 1, "procedure": builder.entry.qualified,
        "selected_node_ids": list(selected), "first_line": first, "last_line": last,
        "original_source_sha256": sha256(text.encode()).hexdigest(),
        "activation": "original complete team after unchanged native prefix and branch evaluation",
        "prefix": [str(node) for node in prefix], "enclosing_controls": [str(node) for node in controls],
        "fallback": "close before unchanged original whole team once; never replay executed work",
    }


def candidates(builder):
    """Enumerate original whole groups, retaining their actual branch prefix."""
    visits = 0

    def sequence(nodes, prefix=(), controls=(), depth=0):
        nonlocal visits
        if depth > builder.analysis.depth_limit:
            raise CompilationError("closed numerical selection control depth budget exhausted")
        before = list(prefix)
        for item in grouped_original_sequence(nodes, {}):
            visits += 1
            if builder.config.scope_execution != "graph" and visits > builder.analysis.operation_limit:
                raise CompilationError("closed numerical selection discovery budget exhausted")
            if isinstance(item, NativeGroupCandidate):
                original = item.original_nodes
                if is_combined_group(builder, original):
                    yield original, tuple(before), controls
                before.extend(original)
            elif _kind(item) == "If_Construct":
                body, guards, header = [], [], None
                for child in _children(item):
                    if _kind(child) in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        if header is not None:
                            yield from sequence(body, (*before, *guards), (*controls, header), depth + 1)
                        body = []
                        header = child if _kind(child) != "End_If_Stmt" else None
                        if _kind(child) in {"If_Then_Stmt", "Else_If_Stmt"}:
                            guards.append(child)
                    else:
                        body.append(child)
                before.append(item)
            else:
                # No ownership is opened inside a loop or other unproved
                # participation/lifetime construct by this discovery class.
                before.append(item)

    yield from sequence(_children(builder.entry.execution))


class OriginalSelectionOwner(LexicalOwner):
    """One source-position block shared by numerical and interval owners.

    This only supplies control and cleanup. Subclasses must establish source
    selection, participation, effects, definitions and completion before any
    block can be emitted.
    """

    def __init__(self, builder, originals, prefix=(), controls=(), *, bundle=None):
        super().__init__(builder)
        self.closed_selection = tuple(originals)
        self.closed_prefix = tuple(prefix)
        self.closed_controls = tuple(controls)
        self.closed_initializations = ()
        self.selection_bundle = bundle
        graph, selected = builder.analysis._selected_source(self.routine.qualified, self.closed_selection)
        if graph is not self.graph or not selected:
            raise CompilationError("original selection requires its exact source graph")
        first, last = statement_span(originals[0])[0], statement_span(originals[-1])[1]
        identity = f"{self.routine.qualified}:{first}:{last}"
        self.context = _name("fort_selection_context_", identity)
        self.enabled = _name("fort_selection_enabled_", identity)
        self.kind = _name("fort_selection_kind_", identity)
        from compiler.scopes.provenance import source_id
        self.trace_owner = source_id(builder, "owner", span=(first, last), lifetime="closed original selection")
        self.selection_context = selection_context(builder, originals, prefix, controls)

    def validate_plan(self, nodes):
        raise CompilationError("original selection control requires a concrete admission validator")

    def _validate_scalar_prefix(self, managed):
        """Retain the existing strict prefix contract, including its proof."""
        executable_prefix = tuple(node for node in self.closed_prefix if _kind(node) != "Comment")
        if executable_prefix:
            if any(_kind(node) not in {"Assignment_Stmt", "If_Then_Stmt", "Else_If_Stmt"}
                   for node in executable_prefix):
                raise CompilationError("closed selection prefix requires straight local scalar preparation")
            proof = prove_resource_reachability(self.builder.analysis, self.routine.qualified,
                                                executable_prefix, managed)
            if proof.barriers or proof.calls:
                raise CompilationError("closed selection prefix requires complete local scalar preparation")
            initialized = []
            for access in proof.accesses:
                if access.kind not in {"write", "undefine"}:
                    continue
                binding = access.storage.binding
                if (access.kind != "write" or binding.rank or access.storage.components
                        or binding.declaring_scope is not self.routine.scope
                        or binding.name in self.routine.arguments
                        or binding.attributes & {"save", "pointer", "target", "optional", "volatile", "asynchronous"}):
                    raise CompilationError("closed selection prefix changes storage or caller-visible definitions")
                if not any(binding is original for original in initialized):
                    initialized.append(binding)
            self.closed_initializations = tuple(initialized)
            # Retain the exact issued proof independently of portable emitted
            # semantics. The checked plan validates this identity at emission;
            # its immutable original prefix and bindings bind the fingerprint.
            self.emission.selection_prefix = proof
        self.builder.analysis._selected_source(self.routine.qualified, self.closed_selection)

    def emit_control(self, nodes):
        if self.execution_plan is None:
            raise CompilationError("original selection control requires its checked execution plan")
        self.execution_plan.validate()
        from compiler.offload.exact_window import control
        specification, entry = control(self) if self.exact_window is not None else ([], [])
        declarations = [f"integer({self.kind}) :: {self.context}", f"logical :: {self.enabled}",
                        *[f"integer({self.kind}) :: fort_buffer_{number}" for _binding, number in self.resources.values()]]
        epoch_initialization = []
        if self.builder.config.scope_execution == "graph":
            declarations += [f"logical :: {self.epoch_pristine}", f"logical :: {self.query_plan}"]
            # This independent original selection starts after native prefix
            # work; invocation entry coverage cannot seed its fresh context.
            epoch_initialization = [f"{self.epoch_pristine} = .false.",
                                    f"{self.query_plan} = .false."]
        before = ["block", f"use iso_c_binding, only: {self.kind} => c_int64_t",
                  "use fort_scoped_memory, only: fort_scope_serial_caller", *specification, *declarations,
                  f"{self.context} = 0",
                  *[f"fort_buffer_{number} = 0" for _binding, number in self.resources.values()],
                  f"{self.enabled} = fort_scope_serial_caller() /= 0", *epoch_initialization, *entry]
        first, last = self.selection_context["first_line"], self.selection_context["last_line"]
        self.builder.add_edit(self.routine.scope.path, first, first - 1,
                              "\n".join(fortran_lines(before)) + "\n", prepend=True)
        ending = []
        if self.exact_window is not None:
            from compiler.offload.selection_cost import finish
            ending = finish(self)
        after = [*self.close(disable=False, reason="closed original selection end", span=(last, last)),
                 *ending, "end block"]
        self.builder.add_edit(self.routine.scope.path, last + 1, last,
                              "\n".join(fortran_lines(after)) + "\n")


class ClosedNumericalOwner(OriginalSelectionOwner):
    """One reached original team, one checked plan and one local context."""

    def __init__(self, builder, originals, prefix=(), controls=(), *, bundle=None):
        if not is_combined_group(builder, tuple(originals)):
            raise CompilationError("closed numerical owner requires an exact whole original team")
        super().__init__(builder, originals, prefix, controls, bundle=bundle)

    def scan(self, nodes, depth=0):
        if len(nodes) != len(self.closed_selection) or any(
                left is not right for left, right in zip(nodes, self.closed_selection, strict=True)):
            raise CompilationError("closed numerical owner selection changed")
        self.admit(tuple(nodes))

    def validate_plan(self, nodes):
        if (len(self.units) != 1 or self.boundaries or len(self.members) != 1
                or not isinstance(self.units[0].scope, CombinedNumericalTeam)):
            reason = "; ".join(str(row.get("reason", "")) for row in self.boundaries)
            raise CompilationError("closed numerical selection requires one admitted complete original team"
                                   + (": " + reason if reason else ""))
        scope = self.units[0].scope
        scope._validate()
        if scope.native or scope.guarded:
            raise CompilationError("closed numerical selection cannot borrow native interruption effects")
        self._validate_scalar_prefix(tuple(self.units[0].arrays.values()))


@dataclass(frozen=True)
class OriginalIntervalSite:
    originals: tuple
    prefix: tuple
    controls: tuple
    loops: tuple
    repeated_suffix: tuple


_INTERVAL_EXITS = frozenset({
    "Return_Stmt", "Exit_Stmt", "Cycle_Stmt", "Goto_Stmt", "Computed_Goto_Stmt",
    "Arithmetic_If_Stmt", "Allocate_Stmt", "Deallocate_Stmt", "Pointer_Assignment_Stmt",
    "Nullify_Stmt", "Stop_Stmt", "Error_Stop_Stmt", "Entry_Stmt",
})


def interval_candidates(builder):
    """Bounded contiguous selections at original serial source positions.

    Complete joined groups stay indivisible. Candidate syntax grants no
    numerical, native-effect or lifetime authority; the owner checks those
    independently before publication.
    """
    from compiler.scopes.native_guard import descriptor_loop_body
    from compiler.scopes.window_discovery import MAX_ADJACENT_UNITS, MAX_ATTEMPTS

    visits = 0
    contexts = []

    def flatten(units):
        return tuple(node for unit in units for node in unit)

    def sequence(nodes, prefix=(), controls=(), loops=(), suffix=(), depth=0):
        nonlocal visits
        if depth > builder.analysis.depth_limit:
            raise CompilationError("original interval control depth budget exhausted")
        grouped = tuple(grouped_original_sequence(nodes, {}))
        units = tuple(item.original_nodes if isinstance(item, NativeGroupCandidate) else (item,)
                      for item in grouped)
        contexts.append((units, prefix, controls, loops, suffix, depth))
        # Inventory source positions before spending admission attempts. A
        # deep early branch must not hide a later producer/consumer pair.
        for index, item in enumerate(grouped):
            visits += 1
            if visits > builder.analysis.operation_limit:
                raise CompilationError("original interval source discovery budget exhausted")
            before = (*prefix, *flatten(units[:index]))
            after = (*flatten(units[index+1:]), *suffix)
            if isinstance(item, NativeGroupCandidate):
                continue
            if _kind(item) == "Block_Nonlabel_Do_Construct":
                try:
                    body = descriptor_loop_body(builder, item)
                except CompilationError:
                    continue
                header = next(node for node in _children(item) if _kind(node) == "Nonlabel_Do_Stmt")
                sequence(body, before, (*controls, header), (*loops, item), after, depth+1)
            elif _kind(item) == "If_Construct":
                body, guards, header = [], [], None
                for child in _children(item):
                    if _kind(child) in {"If_Then_Stmt", "Else_If_Stmt", "Else_Stmt", "End_If_Stmt"}:
                        if header is not None:
                            sequence(body, (*before, *guards), (*controls, header), loops, after, depth+1)
                        body = []
                        header = child if _kind(child) != "End_If_Stmt" else None
                        if _kind(child) in {"If_Then_Stmt", "Else_If_Stmt"}:
                            guards.append(child)
                    else:
                        body.append(child)

    sequence(_children(builder.entry.execution))
    # Reached loop bodies still precede their enclosing loop. Within that
    # lifetime, visit outer source sequences before their nested branches.
    # Try one maximal selection per position before overlapping shorter cuts.
    contexts.sort(key=lambda context: (-len(context[3]), context[5] - len(context[3])))
    choices, seen = [], set()
    for shortening in range(MAX_ADJACENT_UNITS):
        for units, prefix, controls, loops, suffix, _depth in contexts:
            active = [index for index, unit in enumerate(units)
                      if any(_kind(node) != "Comment" for node in unit)]
            for position, first in enumerate(active):
                width = min(MAX_ADJACENT_UNITS, len(active)-position) - shortening
                if width < 1:
                    continue
                last = active[position+width-1] + 1
                originals = flatten(units[first:last])
                if not any(_kind(node) in {"Call_Stmt", "Block_Nonlabel_Do_Construct"}
                           for original in originals for node in walk(original)):
                    continue
                key = tuple(map(id, originals))
                if key in seen:
                    continue
                seen.add(key)
                choices.append(OriginalIntervalSite(originals, (*prefix, *flatten(units[:first])), controls,
                                                    loops, (*flatten(units[last:]), *suffix) if loops else ()))
                if len(choices) >= MAX_ATTEMPTS:
                    return tuple(choices)
    return tuple(choices)


class ClosedIntervalOwner(OriginalSelectionOwner):
    """Checked multi-operation ownership, fresh at each original execution."""

    def __init__(self, builder, originals, prefix=(), controls=(), *, bundle=None):
        originals = tuple(originals)
        matches = [site for site in interval_candidates(builder)
                   if len(site.originals) == len(originals)
                   and all(left is right for left, right in zip(site.originals, originals, strict=True))]
        if len(matches) != 1:
            raise CompilationError("original interval requires an exact contiguous serial selection")
        self.interval_site = matches[0]
        if tuple(prefix) != self.interval_site.prefix or tuple(controls) != self.interval_site.controls:
            raise CompilationError("original interval source prefix or enclosing controls changed")
        if any(_kind(node) in _INTERVAL_EXITS for original in originals for node in walk(original)):
            raise CompilationError("original interval cannot contain lifetime changes or early exits")
        self._validate_edit_boundaries(builder, originals)
        super().__init__(builder, originals, prefix, controls, bundle=bundle)
        self.selection_activation_ready = False
        self.selection_context = {**self.selection_context, "schema_version": 2,
            "activation": "each original serial selection after unchanged prefix and controls",
            "lifetime": "one reached selection; close before every excluded suffix or backedge",
            "fallback": "publish and execute only the remaining original suffix once"}

    @staticmethod
    def _validate_edit_boundaries(builder, originals, *, routine=None):
        """Line insertions and replacements must own complete statements.

        Use original statement spans, as caller-observer instrumentation does;
        punctuation in a literal or comment cannot create a second statement.
        """
        from compiler.frontend.native_group_structure import NATIVE_GROUP_SCAN_LIMIT

        routine = builder.entry if routine is None else routine
        selected = {id(node) for original in originals for node in walk(original)}
        first, last = statement_span(originals[0])[0], statement_span(originals[-1])[1]
        statements = {}
        for count, node in enumerate(walk(routine.scope.node), 1):
            if count > NATIVE_GROUP_SCAN_LIMIT:
                raise CompilationError("original interval edit-boundary scan budget exhausted")
            item = getattr(node, "item", None)
            if item is None or _kind(node) == "Comment":
                continue
            span = getattr(item, "fort_original_span", item.span)
            if span is None or span[1] < first or span[0] > last:
                continue
            if id(node) not in selected and any(span[0] <= line <= span[1] for line in (first, last)):
                raise CompilationError("original interval edit boundary shares a source line with another statement")
            statements[id(item)] = span
        ordered = sorted(statements.values())
        if any(right[0] <= left[1] for left, right in zip(ordered, ordered[1:], strict=False)):
            raise CompilationError("original interval statements share a source edit line")

    def _bind_activation(self, nodes):
        if len(nodes) != len(self.closed_selection) or any(
                left is not right for left, right in zip(nodes, self.closed_selection, strict=True)):
            raise CompilationError("original interval selection changed before preparation")
        if self.selection_activation_ready:
            return

        # Retain the small established scalar contract when it is sufficient.
        # Otherwise start with unknown payload definitions after the unchanged
        # native prefix. Exact source reads are admitted only when their
        # numerical operation is reached; this token grants no new lifetime.
        from compiler.frontend.selection_activation import (
            prove_selection_activation,
            selection_activation_demands,
        )

        demanded = selection_activation_demands(self)
        try:
            self._validate_scalar_prefix(tuple(binding for binding in demanded if binding.rank))
        except CompilationError:
            proof = prove_selection_activation(self, demanded)
            self.builder = self.builder.bind_selection_activation(proof, self.closed_selection, demanded)
            self.closed_initializations = proof.initializations
            self.emission.selection_prefix = proof
        self.selection_activation_ready = True

    def run(self, nodes=None):
        nodes = self.closed_selection if nodes is None else tuple(nodes)
        # The ordinary numerical consumption pass and its checkpoint must be
        # owned by the already bound builder, never by the superseded parent.
        self._bind_activation(nodes)
        return super().run(nodes)

    def _prepare_lexical(self, nodes):
        self._bind_activation(nodes)
        from compiler.scopes.interval_draft import prepare_interval_draft

        prepare_interval_draft(self, self.closed_selection)
        prepared = self.builder.inline.prepare(self.closed_selection)
        self.builder.generated.update(self.builder.inline.generated)
        self.builder.numerical_ir.update(self.builder.inline.ir)
        self.builder.numerical_reasons.update({name: None for name in self.builder.inline.regions})
        return super()._prepare_lexical(prepared)

    def _validate_absorbing_boundaries(self):
        from compiler.scopes.lexical import LexicalBoundaryPatch

        for member in self.members.values():
            for suffix in member.closed_suffixes:
                suffix.validate(member)
            originals = (self.closed_selection if member is self else tuple(_children(member.routine.execution)))
            allowed = {id(node) for original in originals for node in walk(original)}
            patches = [(first, last, patch) for first, last, patch in member.patches
                       if isinstance(patch, LexicalBoundaryPatch)]
            if len(patches) != len(member.boundaries):
                raise CompilationError("original interval boundary inventory differs from its checked patches")
            for boundary, (first, last, patch) in zip(member.boundaries, patches, strict=True):
                patch.validate(member, require_source=True)
                if (any(id(node) not in allowed for node in patch.originals)
                        or (first, last) != (patch.first, patch.last)
                        or (first, last) != (boundary["first_line"], boundary["last_line"])
                        or patch.reason != boundary["reason"]
                        or patch.conditional_guard != boundary.get("conditional_close")
                        or boundary.get("reopen") is not False
                        or boundary.get("execution") != "publish and close when reached; original continuation once"):
                    raise CompilationError("original interval requires exact absorbing original boundaries")

    def validate_plan(self, nodes):
        originals = self.builder.inline.original_selection(tuple(nodes))
        if len(originals) != len(self.closed_selection) or any(
                left is not right for left, right in zip(originals, self.closed_selection, strict=True)):
            raise CompilationError("original interval prepared selection lost source authority")
        # Child companions rewrite their original bodies with the same
        # line-based editor as the root selection. Their CALL patches must
        # not absorb adjacent scalar preparation or native side effects.
        for member in self.members.values():
            if member is not self:
                self._validate_edit_boundaries(self.builder, tuple(_children(member.routine.execution)),
                                               routine=member.routine)
        self._validate_absorbing_boundaries()
        if not any(unit.leaves for member in self.members.values() for unit in member.units):
            raise CompilationError("original interval has no admitted numerical operation")
        from compiler.scopes.execution_plan import _invocations
        if _invocations(self, tuple(self.members.values())) < 2:
            raise CompilationError("original interval needs two admitted numerical invocations")
        managed = tuple({id(member.canonical(binding)): member.canonical(binding)
                         for member in self.members.values() for unit in member.units
                         for binding in unit.arrays.values()}.values())
        from compiler.frontend.selection_activation import SelectionActivationProof

        if type(self.emission.selection_prefix) is SelectionActivationProof:
            proof = self.emission.selection_prefix.validate_prefix(self)
            for binding in managed:
                if self.builder.visible_binding(self.routine, binding.root) is binding:
                    proof.capture_state(binding.root)
        else:
            self._validate_scalar_prefix(managed)
        # A fresh block does not prove fresh definitions after a previous
        # traversal. Until an independent reached-definition proof is supplied,
        # only unchanged local scalar work may surround a repeated selection.
        if self.interval_site.loops:
            suffix = tuple(node for node in self.interval_site.repeated_suffix if _kind(node) != "Comment")
            if suffix:
                if any(_kind(node) != "Assignment_Stmt" for node in suffix):
                    raise CompilationError("repeated original interval requires a local scalar suffix")
                proof = prove_resource_reachability(self.builder.analysis, self.routine.qualified, suffix, managed)
                if proof.barriers or proof.calls:
                    raise CompilationError("repeated original interval suffix has unproved source effects")
                for access in proof.accesses:
                    binding = access.storage.binding
                    if access.kind in {"write", "undefine"} and (
                            access.kind != "write" or binding.rank or access.storage.components
                            or binding.declaring_scope is not self.routine.scope
                            or binding.name in self.routine.arguments
                            or binding.attributes & {"save", "pointer", "target", "optional", "volatile", "asynchronous"}):
                        raise CompilationError("repeated original interval suffix changes nonlocal storage")
            from compiler.scopes.native_guard import descriptor_loop_body
            for loop in self.interval_site.loops:
                descriptor_loop_body(self.builder, loop)
        self.builder.analysis._selected_source(self.routine.qualified, self.closed_selection)



def discover_closed_selections(builder):
    """Use the same block control for original teams and checked intervals."""
    request = builder.config.exact_window or {}
    interval = request.get("coverage") == "checked_interval"
    if builder.config.scope_execution != ("reached" if interval else "bounded"):
        return ()
    if interval:
        from compiler.scopes.interval_ordering import order_intervals

        sites, builder.original_interval_discovery = order_intervals(builder, interval_candidates(builder))
        choices = tuple((site.originals, site.prefix, site.controls) for site in sites)
    else:
        choices = tuple(candidates(builder))
    if not choices:
        return ()
    if len(choices) > 32:
        raise CompilationError("closed numerical selection owner budget exhausted")
    from compiler.offload.exact_window import digest
    from compiler.scopes.generation_transaction import GenerationTransaction
    with GenerationTransaction(builder) as transaction:
        candidate, admitted = transaction.builder, []
        checkpoint = candidate.scope_checkpoint()
        # First retain every independently admitted exact original inventory.
        # Unsupported groups remain native; they cannot prevent another branch
        # from using a proved alternative or silently enter the priced bundle.
        for originals, prefix, controls in choices:
            before = candidate.scope_checkpoint()
            try:
                factory = ClosedIntervalOwner if interval else ClosedNumericalOwner
                owner = factory(candidate, originals, prefix, controls)
                if not owner._prepare_lexical(originals):
                    raise CompilationError("closed original selection has no numerical alternative")
                admitted.append(owner)
                if interval:
                    candidate = owner.builder
                    candidate.original_interval_discovery["selected"] = {
                        "first_line": statement_span(originals[0])[0], "last_line": statement_span(originals[-1])[1]}
                    break  # One nonoverlapping alternative per priced bundle.
            except CompilationError as error:
                candidate.restore_scope_checkpoint(before)
                candidate.boundaries.append({"first_line": statement_span(originals[0])[0],
                    "last_line": statement_span(originals[-1])[1], "reason": str(error),
                    "execution": "unchanged original selection" if interval else "unchanged original complete team",
                    "kind": "original_interval_decline" if interval else "closed_numerical_selection_decline"})
                if interval:
                    candidate.original_interval_discovery["attempts"].append(candidate.boundaries[-1])
        bundle = {"schema_version": 1, "entry": candidate.entry.qualified, "groups": []}
        for owner in admitted:
            # Pricing and build attestation bind the issued interval source
            # context. Owner lifecycle diagnostics are a separate record.
            context = owner.interval_draft.selection_context
            bundle["groups"].append({"selection_identity": digest(context), **{key: context[key] for key in
                                     ("first_line", "last_line", "original_source_sha256")}})
        bundle["identity"] = digest(bundle)
        for owner in admitted:
            owner.selection_bundle = bundle
            owner._emit_lexical(owner.closed_selection, candidate.scope_checkpoint(), native_cleanup=False)
            candidate.scopes[-1]["original_interval" if interval else "closed_selection"] = dict(owner.selection_context)
        native_capture = (request.get("coverage") in {"original_selection", "checked_interval"}
                          and request.get("calibration") == "native")
        static_native = (candidate.config.policy == "auto" and admitted
                         and all(owner.execution_plan.automatic_native_reason() is not None for owner in admitted))
        if native_capture or static_native:
            # All native observers were prepared while the admitted regions
            # remained available. Discard numerical artifacts once for the
            # complete bundle, preserving only observer edits and helpers.
            edits = {path: list(rows) for path, rows in candidate.edits.items()}
            observers = {name: text for name, text in candidate.outputs.items() if name.startswith("exact/")}
            candidate.restore_scope_checkpoint(checkpoint)
            candidate.edits = edits
            candidate.outputs.update(observers)
            for owner in admitted:
                owner.execution_plan.validate()
        transaction.adopt(candidate)
    # This class has one orchestration path. A decline never retries it through
    # the superseded synthetic owner with different proof/placement semantics.
    return tuple(tuple(originals) for originals, _prefix, _controls in choices)
