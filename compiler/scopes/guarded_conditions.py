"""Edit geometry for reached publication at original ELSEIF predicates.

The original condition units retain all access and source authority. This
attachment only places their hooks inside the preceding false branch.
"""
from dataclasses import dataclass

from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError
from compiler.scopes.segments import fortran_lines, statement_span


def geometry(member, original):
    from compiler.scopes.closed_selection import ClosedIntervalOwner

    analysis = member.builder.analysis
    if analysis._require_original(member.routine.qualified) is not member.routine:
        raise CompilationError("guarded ELSEIF requires its exact original routine")
    graph, selected = analysis._selected_source(member.routine.qualified, (original,))
    if (_kind(original) != "If_Construct" or graph is not member.graph or len(selected) != 1
            or graph.nodes[selected[0]].kind != "branch"):
        raise CompilationError("guarded ELSEIF requires its exact original conditional graph")
    headers = tuple(node for node in original.content if _kind(node) in {"If_Then_Stmt", "Else_If_Stmt"})
    closing = original.content[-1]
    if (len(headers) < 2 or _kind(headers[0]) != "If_Then_Stmt"
            or any(_kind(header) != "Else_If_Stmt" for header in headers[1:])
            or _kind(closing) != "End_If_Stmt"):
        raise CompilationError("guarded ELSEIF requires one complete original IF chain")
    for node in (*headers, closing):
        first, last = statement_span(node)
        if (first != last or getattr(node.item, "label", None) is not None or getattr(node.item, "name", None)
                or len(node.items) > 1 and node.items[1] is not None):
            raise CompilationError("guarded ELSEIF requires unnamed single-line original headers and closing")
        ClosedIntervalOwner._validate_edit_boundaries(member.builder, (node,), routine=member.routine)
    return headers, closing


@dataclass(frozen=True)
class GuardedConditionChain:
    member: object
    original: object
    headers: tuple
    closing: object
    publications: tuple

    def validate(self, member):
        from compiler.frontend.immutable_graph import original_type

        if original_type(self) is not GuardedConditionChain or self.member is not member:
            raise CompilationError("guarded ELSEIF changed its original owning member")
        headers, closing = geometry(member, self.original)
        if (closing is not self.closing or len(headers) != len(self.headers)
                or any(a is not b for a, b in zip(headers, self.headers, strict=True))):
            raise CompilationError("guarded ELSEIF changed its exact original headers or closing")
        seen = set()
        for header, unit in self.publications:
            if (not any(header is item for item in headers) or id(header) in seen
                    or not any(unit is item for item in member.units)
                    or not unit.publication_only or unit.guarded_condition is not self
                    or unit.source_guard is not None or unit.native_source != ""
                    or unit.scope.calls or len(unit.scope.native) != 1 or unit.scope.native[0].nodes
                    or len(unit.nodes) != 1 or unit.nodes[0] is not header):
                raise CompilationError("guarded ELSEIF lost its exact original publication unit")
            seen.add(id(header))
        attached = tuple(unit for unit in member.units if unit.guarded_condition is self)
        if (len(attached) != len(self.publications)
                or any(not any(unit is saved for _header, saved in self.publications) for unit in attached)
                or tuple(id(header) for header, _unit in self.publications)
                != tuple(id(header) for header in headers if id(header) in seen)):
            raise CompilationError("guarded ELSEIF changed its complete ordered publication inventory")
        if not any(_kind(header) == "Else_If_Stmt" for header, _unit in self.publications):
            raise CompilationError("guarded ELSEIF lacks a reached later publication")
        return self

    def apply(self, member, edits):
        """Consume finished insertion text; retain every original branch body."""
        self.validate(member)
        path = member.routine.scope.path
        published = {id(header): unit for header, unit in self.publications}
        for header in self.headers[1:]:
            first, last = statement_span(header)
            prefix = ""
            unit = published.get(id(header))
            if unit is not None:
                saved = member.unit_emission(unit).source_edits
                if len(saved) != 1 or saved[0][:2] != (first, first-1):
                    raise CompilationError("guarded ELSEIF requires one exact finished publication insertion")
                prefix = saved[0][2]
                matches = [i for i, edit in enumerate(edits) if edit[:3] == (path, first, first-1)
                           and edit[3] is prefix and edit[4] is False]
                if len(matches) != 1:
                    raise CompilationError("guarded ELSEIF publication insertion is missing or duplicated")
                edits.pop(matches[0])
            if any(edit[0] == path and (edit[1] <= last and edit[2] >= first
                                      or edit[1:3] == (first, first-1)) for edit in edits):
                raise CompilationError("guarded ELSEIF header overlaps another emitted edit")
            text = "else\n" + prefix + "\n".join(fortran_lines([f"if ({header.items[0]}) then"])) + "\n"
            edits.append((path, first, last, text, False))
            if unit is not None:
                member.unit_emission(unit).source_edits = ((first, last, text),)
        _first, last = statement_span(self.closing)
        if any(edit[0] == path and edit[1] <= last and edit[2] >= last for edit in edits):
            raise CompilationError("guarded ELSEIF closing overlaps another emitted edit")
        # Apply every header before adding closures, so a nested chain's
        # closure can share the next outer header's insertion boundary.
        # Closures precede any next-statement hooks at that same boundary.
        return path, last+1, last, "endif\n" * (len(self.headers)-1), True


def chains(member, units):
    result = []
    for unit in units:
        chain = unit.guarded_condition
        if chain is not None and not any(chain is item for item in result):
            chain.validate(member)
            result.append(chain)
    return tuple(result)
