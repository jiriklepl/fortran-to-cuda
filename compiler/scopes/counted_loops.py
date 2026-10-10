"""Retain original serial counted control around reached operations."""

from dataclasses import dataclass

from fparser.two.utils import walk

from compiler.frontend.source_effects import _kind
from compiler.ir import CompilationError
from compiler.scopes.segments import directive, grouped_nodes, statement_span


@dataclass(frozen=True)
class CountedBody:
    nodes: tuple
    authority: dict


def has_coordinator_call(builder, node):
    """Classify original calls without granting effects or GPU legality.

    A loop containing an unresolved or non-PURE call is a host coordinator,
    not one numerical candidate. PURE helper-only loops still undergo normal
    transitive numerical extraction. Every reached child retains its budget.
    """
    from compiler.frontend.source_effects import _children, _part
    from compiler.scopes.regions import _helper

    for call in walk(node):
        if _kind(call) != "Call_Stmt":
            continue
        if _kind(call.items[0]) != "Name":
            return True
        helper = _helper(builder.analysis, builder.entry, call.items[0])
        if helper is None:
            return True
        statement = _part(helper.scope.node, "Subroutine_Stmt") or _part(helper.scope.node, "Function_Stmt")
        if statement is None or not any(str(prefix).lower() == "pure" for prefix in _children(statement.items[0])):
            return True
    return False


def reached_counted_body(builder, node):
    """Authenticate an unchanged header without evaluating any of its inputs.

    A prepared body may contain numerical facades, but its header and terminator
    must still be the exact original objects. The invocation owner is not reset
    at a backedge. Any reached boundary permanently disables that owner.
    """
    originals = builder.inline.original_selection(node)
    if len(originals) != 1 or _kind(originals[0]) != "Block_Nonlabel_Do_Construct":
        raise CompilationError("reached counted body requires one original DO construct")
    original = originals[0]
    original_content = tuple(original.content)
    header = next((item for item in original_content if _kind(item) == "Nonlabel_Do_Stmt"), None)
    end = original_content[-1] if original_content else None
    control = header.items[1] if header is not None else None
    counted = control.items[1] if control is not None and _kind(control) == "Loop_Control" else None
    if not counted or _kind(end) != "End_Do_Stmt":
        raise CompilationError("reached loop traversal requires original counted DO control")
    prefix = original_content[:original_content.index(header)]
    if any(directive(item) is not None for item in prefix):
        raise CompilationError("reached counted body cannot split an attached OpenMP construct")
    for item in walk(original):
        if (_kind(item) == "Comment" and str(item).lstrip().startswith("!$")
                and directive(item) is None):
            raise CompilationError("reached counted body contains unsupported conditional directives")

    def check_directives(items):
        for group in grouped_nodes(items):
            if isinstance(group, tuple):
                # The native group retains every original directive and join;
                # its separate completion proof controls any later admission.
                continue
            if directive(group) is not None:
                raise CompilationError("reached counted body requires complete original OpenMP groups")
            if _kind(group) in {"If_Construct", "Associate_Construct", "Block_Nonlabel_Do_Construct"}:
                check_directives(group.content)

    body = original_content[original_content.index(header)+1:-1]
    check_directives(body)
    from compiler.scopes.native_guard import descriptor_loop_body
    descriptor_loop_body(builder, original)
    content = tuple(node.content)
    if not any(item is header for item in content) or not content or content[-1] is not end:
        raise CompilationError("reached counted body lost its original header or terminator")
    first = next(index for index, item in enumerate(content) if item is header)
    graph = builder.analysis.structure(builder.entry.qualified)
    return CountedBody(content[first+1:-1], {
        "schema_version": 1, "structured_summary_identity": graph.identity,
        "header_identity": graph.node_id(header, role="header"),
        "first_line": statement_span(original)[0], "last_line": statement_span(original)[1],
        "header": str(header), "header_evaluation": "unchanged original counted DO",
        "owner_lifetime": "one invocation; no reset or inferred freshness at backedges",
        "reopen_after_boundary": False, "numerical_authority": False,
        "unsupported_reached_operation": "publish and disable before original continuation once",
    })
