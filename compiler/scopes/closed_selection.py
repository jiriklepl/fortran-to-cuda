"""Closed owners for complete original numerical teams at their reached sites.

The surrounding procedure, branches and scalar prefix stay native. These owners
reuse the checked lexical plan and original-team emitter; they do not borrow
ownership across an unsupported native operation or assume a whole-entry effect
closure. A failed preparation closes before the original team executes once.
"""
from __future__ import annotations

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
            if visits > builder.analysis.operation_limit:
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


class ClosedNumericalOwner(LexicalOwner):
    """One reached original team, one checked plan and one local context."""

    def __init__(self, builder, originals, prefix=(), controls=(), *, bundle=None):
        super().__init__(builder)
        self.closed_selection = tuple(originals)
        self.closed_prefix = tuple(prefix)
        self.closed_controls = tuple(controls)
        self.closed_initializations = ()
        self.selection_bundle = bundle
        graph, selected = builder.analysis._selected_source(self.routine.qualified, self.closed_selection)
        if graph is not self.graph or not is_combined_group(builder, self.closed_selection):
            raise CompilationError("closed numerical owner requires an exact whole original team")
        first, last = statement_span(originals[0])[0], statement_span(originals[-1])[1]
        identity = f"{self.routine.qualified}:{first}:{last}"
        self.context = _name("fort_selection_context_", identity)
        self.enabled = _name("fort_selection_enabled_", identity)
        self.kind = _name("fort_selection_kind_", identity)
        from compiler.scopes.provenance import source_id
        self.trace_owner = source_id(builder, "owner", span=(first, last), lifetime="closed original selection")
        self.selection_context = selection_context(builder, originals, prefix, controls)

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
        managed = tuple(self.units[0].arrays.values())
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
        from compiler.offload.exact_window import control
        specification, entry = control(self) if self.exact_window is not None else ([], [])
        declarations = [f"integer({self.kind}) :: {self.context}", f"logical :: {self.enabled}",
                        *[f"integer({self.kind}) :: fort_buffer_{number}" for _binding, number in self.resources.values()]]
        before = ["block", f"use iso_c_binding, only: {self.kind} => c_int64_t",
                  "use fort_scoped_memory, only: fort_scope_serial_caller", *specification, *declarations,
                  f"{self.context} = 0",
                  *[f"fort_buffer_{number} = 0" for _binding, number in self.resources.values()],
                  f"{self.enabled} = fort_scope_serial_caller() /= 0", *entry]
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



def discover_closed_selections(builder):
    """Replace bounded synthetic ownership for complete original team sites."""
    if builder.config.scope_execution != "bounded":
        return ()
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
                owner = ClosedNumericalOwner(candidate, originals, prefix, controls)
                if not owner._prepare_lexical(originals):
                    raise CompilationError("closed original selection has no numerical alternative")
                admitted.append(owner)
            except CompilationError as error:
                candidate.restore_scope_checkpoint(before)
                candidate.boundaries.append({"first_line": statement_span(originals[0])[0],
                    "last_line": statement_span(originals[-1])[1], "reason": str(error),
                    "execution": "unchanged original complete team", "kind": "closed_numerical_selection_decline"})
        bundle = {"schema_version": 1, "entry": candidate.entry.qualified, "groups": []}
        for owner in admitted:
            context = owner.selection_context
            bundle["groups"].append({"selection_identity": digest(context), **{key: context[key] for key in
                                     ("first_line", "last_line", "original_source_sha256")}})
        bundle["identity"] = digest(bundle)
        for owner in admitted:
            owner.selection_bundle = bundle
            owner._emit_lexical(owner.closed_selection, candidate.scope_checkpoint(), native_cleanup=False)
            candidate.scopes[-1]["closed_selection"] = dict(owner.selection_context)
        request = candidate.config.exact_window or {}
        native_capture = request.get("coverage") == "original_selection" and request.get("calibration") == "native"
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
