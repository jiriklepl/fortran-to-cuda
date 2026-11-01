"""Input-sized accounting for complete original descriptor proof walks.

The inventory is private to one consumer attempt. It counts only authenticated
reached source, never measurements, public proof JSON, or unrelated routines.
No effect, completion, execution or payload authority follows from a budget.
"""
from __future__ import annotations

from dataclasses import dataclass

from fparser.two.utils import Base

from compiler.ir import CompilationError

DEFAULT_SOURCE_SCAN_LIMIT = 65536
SOURCE_SYNTAX_PASSES = 8


def _children(node):
    return getattr(node, "content", getattr(node, "items", ())) or ()


def _syntax_inventory(roots):
    # Original syntax has already passed _require_original. Count its unique
    # parser objects, not visits to repeated leaves/containers. Parent pointers
    # are deliberately excluded. The traversal is bounded by this input graph.
    pending, seen, parser = list(roots), set(), {}
    while pending:
        node = pending.pop()
        if not isinstance(node, (Base, tuple, list)) or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, Base):
            parser[id(node)] = node
            pending.extend(_children(node))
        else:
            pending.extend(node)
    declarations = sum(type(node).__name__ in {
        "Type_Declaration_Stmt", "Entity_Decl", "Named_Constant_Def", "Data_Component_Def_Stmt"}
        for node in parser.values())
    return len(parser), declarations


def _semantic_inventory(graph, selected):
    pending, seen, units = list(selected), set(), 0
    while pending:
        identity = pending.pop()
        if identity in seen:
            continue
        if identity not in graph.nodes or len(seen) >= graph.node_limit:
            raise CompilationError("descriptor budget requires bounded exact original graph membership")
        seen.add(identity)
        node = graph.nodes[identity]
        units += node.kind in {"operation", "call", "boundary"}
        pending.extend(node.children)
        for condition, body in node.alternatives:
            if condition is not None:
                pending.append(condition)
            pending.append(body)
    return units


@dataclass(frozen=True, slots=True)
class _ReachedSource:
    routine: object
    graph: object
    selected: tuple
    syntax_nodes: int
    operation_units: int
    declaration_units: int
    lookup_limit: int

    @property
    def allowance(self):
        return (SOURCE_SYNTAX_PASSES * self.syntax_nodes
                + self.lookup_limit * (self.operation_units + self.declaration_units))

    def public(self):
        return {"procedure": self.routine.qualified, "source_graph": self.graph.identity,
                "selected_node_ids": list(self.selected), "unique_original_parser_nodes": self.syntax_nodes,
                "original_graph_operation_units": self.operation_units,
                "original_declaration_units": self.declaration_units, "per_unit_lookup_limit": self.lookup_limit,
                "allowance": self.allowance}


class DescriptorSourceBudget:
    """One exact reached inventory per original routine, without retagging."""

    def __init__(self, analysis):
        self.analysis = analysis
        self.lookup_limit = analysis.operation_limit
        self._reached = {}
        self._allowance = 0
        self._limit = DEFAULT_SOURCE_SCAN_LIMIT

    @property
    def limit(self):
        return self._limit

    def reached(self, routine, graph, selected):
        # No caller-provided count/digest or foreign projection can mint an
        # allowance. Authenticate original syntax, namespace and exact graph
        # selection before counting or reusing this consumer-local inventory.
        if not isinstance(getattr(routine, "qualified", None), str) or not isinstance(selected, tuple):
            raise CompilationError("descriptor budget requires exact original routine and selected IDs")
        original = self.analysis._require_original(routine.qualified)
        actual_graph, actual_selected = self.analysis._selected_source(routine.qualified, selected)
        if (original is not routine or actual_graph is not graph or tuple(selected) != actual_selected
                or self.analysis.operation_limit != self.lookup_limit):
            raise CompilationError("descriptor budget original resolution authority changed (routine, graph or analysis policy)")
        retained = self._reached.get(routine.qualified)
        if retained is not None:
            if retained.routine is not routine or retained.graph is not graph or retained.selected != actual_selected:
                raise CompilationError("descriptor budget cannot retag a reached original routine")
            return
        specification = next((part for part in _children(routine.scope.node)
                              if type(part).__name__ == "Specification_Part"), None)
        roots = [specification]
        roots.extend(node for identity in actual_selected for node in graph.source_nodes(identity))
        syntax, declarations = _syntax_inventory(roots)
        item = _ReachedSource(routine, graph, actual_selected, syntax,
            _semantic_inventory(graph, actual_selected), declarations, self.lookup_limit)
        self._reached[routine.qualified] = item
        self._allowance += item.allowance
        self._limit = max(DEFAULT_SOURCE_SCAN_LIMIT, self._allowance)

    def validate(self):
        actual = sum(item.allowance for item in self._reached.values())
        if self._allowance != actual or self._limit != max(DEFAULT_SOURCE_SCAN_LIMIT, actual):
            raise CompilationError("descriptor budget accounting changed its reached inventory")
        for item in tuple(self._reached.values()):
            self.reached(item.routine, item.graph, item.selected)

    def public(self):
        return {"version": 1, "formula": "max(65536, sum(8*S + per_unit_lookup_limit*U))",
                "syntax_passes": SOURCE_SYNTAX_PASSES,
                "syntax_inventory": "unique original parser nodes in selected owner source/full reached callee and specification",
                "semantic_inventory": "selected original graph operation/call/boundary nodes plus original declaration units",
                "reached": [item.public() for item in self._reached.values()], "limit": self.limit,
                "repeated_callee_credit": False, "unrelated_source_credit": False,
                "payload_access_authorized": False, "entry_definition_authority": False,
                "execution_authorized": False}
