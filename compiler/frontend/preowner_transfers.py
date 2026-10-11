"""Source-only transfers before a one-shot owner can acquire descriptors.

Original statements execute unchanged. This helper has no token registry and
grants neither active-window lifetime nor execution authority. Its caller owns
the registered source-position proof, conditional control and shared budgets.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

from compiler.frontend.native_sections import NativeAxis, NativeRectangle
from compiler.frontend.source_functions import _validate_resolution, _validate_scope, _walk
from compiler.ir import CompilationError, SourceLocation
from compiler.ir.integers import integer_literal

_UNSAFE = {"pointer", "target", "optional", "volatile", "asynchronous", "value"}


class PreownerUnsupported(CompilationError):
    """Authenticated original source must disable dormant eligibility."""


@dataclass(frozen=True)
class PreownerTransfer:
    definitions: object
    events: tuple

    def public(self):
        return {"ordered_events": [dict(item) for item in self.events],
                "original_execution_required": True, "execution_authorized": False,
                "descriptor_acquisition": "actual original descriptor after activation guards",
                "allocation_presence_proves_payload": False}


def derive_preowner_step(analysis, procedure, original, bindings, coverage, facts, *, consume):
    """Transfer one exact original allocation/copy, or return None for a leaf.

    Every syntax/resolution transition spends the owning derivation's source
    budget. Unsupported source uses a distinct signal; source identity, binding
    mutation and exhausted budgets remain ordinary hard CompilationError.
    """
    routine = analysis._require_original(procedure)
    kind = type(original).__name__
    if kind not in {"Assignment_Stmt", "Allocate_Stmt", "Deallocate_Stmt"}:
        return None
    scope = analysis.source_scope_for(original, routine.scope)
    if kind == "Assignment_Stmt":
        target = original.items[0]
        binding = analysis._binding(scope, target) if type(target).__name__ == "Name" else None
        if binding is None or "allocatable" not in binding.attributes:
            return None
    graph, selected = analysis._selected_source(procedure, (original,))
    if len(selected) != 1 or graph.source_nodes(selected[0]) != (original,):
        raise CompilationError("pre-owner transfer requires one exact original statement")
    _validate_scope(analysis, scope)
    _validate_resolution(analysis, scope, _walk(original), tick=lambda _role: consume(1))

    def array(node, *, allocated=False):
        if type(node).__name__ != "Name":
            raise PreownerUnsupported("pre-owner allocation/copy requires whole numeric names")
        binding = analysis._binding(scope, node)
        if binding is not None:
            _validate_scope(analysis, binding.declaring_scope)
        if (binding is None or bindings.get(binding.root) is not binding
                or binding.dtype not in {"real", "integer"} or binding.kind not in {4, 8}
                or not 1 <= binding.rank <= 4 or binding.attributes & _UNSAFE
                or allocated and "allocatable" not in binding.attributes):
            raise PreownerUnsupported("pre-owner allocation/copy requires independently tracked numeric storage")
        reason = analysis.resource_identity_boundary(binding)
        if reason:
            raise CompilationError(reason)
        if allocated:
            fact = facts["captures"][binding.root]
            if fact.get("lifetime", "invocation") != "selected_window" or fact["initialized"] != "none":
                raise PreownerUnsupported("pre-owner descriptor changes require selected-window storage with no entry payload assertion")
        return binding

    def bound(node, depth=0):
        consume(1)
        if depth > analysis.depth_limit:
            raise CompilationError("pre-owner allocation bound depth exhausted")
        node_kind = type(node).__name__
        if node_kind == "Int_Literal_Constant":
            # Keep the existing INTEGER ABI; never evaluate application state.
            if node.items[1] is not None and scope.kinds.integer(node.items[1], SourceLocation(str(scope.path))) != 4:
                raise PreownerUnsupported("pre-owner allocation literals require default INTEGER kind")
            integer_literal(str(node.items[0]), SourceLocation(str(scope.path)), source_token=True)
            return
        if node_kind == "Name":
            scalar = analysis._binding(scope, node)
            if scalar is not None:
                _validate_scope(analysis, scalar.declaring_scope)
            if (scalar is None or scalar.rank or scalar.dtype != "integer" or scalar.kind != 4
                    or scalar.attributes & (_UNSAFE | {"allocatable"})):
                raise PreownerUnsupported("pre-owner allocation bounds require plain INTEGER scalars")
            if "parameter" in scalar.attributes:
                integer_literal(str(scope.kinds.integer(node, SourceLocation(str(scope.path)))),
                                SourceLocation(str(scope.path)))
            return
        items = getattr(node, "items", ())
        if node_kind == "Parenthesis" or len(items) == 2 and str(items[0]) in {"+", "-"}:
            bound(items[1], depth + 1)
        elif len(items) == 3 and str(items[1]) in {"+", "-", "*"}:
            bound(items[0], depth + 1)
            bound(items[2], depth + 1)
        else:
            raise PreownerUnsupported("pre-owner allocation bound expression is unsupported")

    state = dict(coverage)
    events = []
    if kind == "Assignment_Stmt":
        destination = array(original.items[0], allocated=True)
        source = array(original.items[2])
        if (destination.rank != source.rank or destination.signature()[:2] != source.signature()[:2]
                or destination is source):
            raise PreownerUnsupported("pre-owner copy requires distinct equal-rank/kind numeric names")
        source_fact = facts["captures"][source.root]
        if source_fact.get("lifetime", "invocation") == "selected_window" and source_fact["initialized"] != "none":
            raise PreownerUnsupported("selected-window payload assertions cannot establish an earlier pre-owner copy read")
        if not any(not box.iterations and all(axis.lower is None and axis.upper is None for axis in box.axes)
                   for box in state[source.root]):
            raise PreownerUnsupported("pre-owner whole copy reads incompletely defined source payload")
        state[destination.root] = (NativeRectangle(tuple(NativeAxis() for _ in range(destination.rank)),
                                                   destination.name),)
        events.extend(({"kind": "read", "resource": source.root, "coverage": "whole",
                        "payload_provenance": ("ordered source definitions from none"
                                               if source_fact.get("lifetime") == "selected_window"
                                               else "independent invocation-entry facts and ordered source transfer")},
                       {"kind": "definition_change", "resource": destination.root},
                       {"kind": "overwrite", "resource": destination.root, "coverage": "whole",
                        "payload_provenance": "original successful whole numeric copy",
                        "descriptor": "may change; conforming destination may retain original bounds"}))
    elif kind == "Allocate_Stmt":
        specification, allocations, options = original.items
        objects = getattr(allocations, "items", ())
        if specification is not None or options is not None or len(objects) != 1:
            raise PreownerUnsupported("pre-owner ALLOCATE requires one plain explicit-shape object without options")
        allocation = objects[0]
        if type(allocation).__name__ != "Allocation":
            raise PreownerUnsupported("pre-owner ALLOCATE requires an original explicit shape")
        target, shape = allocation.items
        destination = array(target, allocated=True)
        axes = getattr(shape, "items", ())
        if len(axes) != destination.rank or any(type(axis).__name__ != "Allocate_Shape_Spec" for axis in axes):
            raise PreownerUnsupported("pre-owner ALLOCATE requires equal-rank explicit bounds")
        for axis in axes:
            for expression in axis.items:
                if expression is not None:
                    bound(expression)
        state[destination.root] = ()
        events.extend(({"kind": "descriptor_change", "resource": destination.root, "operation": "allocate"},
                       {"kind": "definition_change", "resource": destination.root, "coverage": "none"}))
    else:
        objects, options = original.items
        targets = getattr(objects, "items", ())
        if options is not None or len(targets) != 1:
            raise PreownerUnsupported("pre-owner DEALLOCATE requires one plain object without options")
        destination = array(targets[0], allocated=True)
        state[destination.root] = ()
        events.extend(({"kind": "descriptor_change", "resource": destination.root, "operation": "deallocate"},
                       {"kind": "definition_change", "resource": destination.root, "coverage": "none"}))
    events.append({"kind": "activation_requirement", "resource": destination.root,
                   "fact_lifetime": "selected_window", "fact_initialized": "none",
                   "fact_mutated": False, "active_descriptor_proof_required": True})
    return PreownerTransfer(MappingProxyType(state), tuple(MappingProxyType(item) for item in events))
