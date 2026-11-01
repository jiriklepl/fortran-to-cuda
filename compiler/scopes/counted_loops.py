"""Retain original serial counted control around reached operations."""

import json
from dataclasses import dataclass
from hashlib import sha256

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


def reached_retirement_entry(member, originals):
    """An existing counted-header check, without evaluating or copying bounds.

    This is a structural emission record, not a new effect or lifetime issuer.
    The boundary retains its independent whole-original storage proof. Every
    use derives this record again from the exact source and current analysis.
    """
    if len(originals) != 1 or _kind(originals[0]) != "Block_Nonlabel_Do_Construct":
        raise CompilationError("reached retirement requires one original counted DO")
    original = originals[0]
    body = reached_counted_body(member.builder, original)
    header = next(item for item in original.content if _kind(item) == "Nonlabel_Do_Stmt")
    end = original.content[-1]
    control = header.items[1].items[1]
    iterator = control[0]
    analysis, procedure = member.builder.analysis, member.routine.qualified
    graph = analysis.structure(procedure)
    header_id = graph.node_id(header, role="header")
    proof = analysis.segment_summary(procedure, (header_id,), capture_locals=True)
    binding = analysis._binding(analysis.source_scope_for(header, member.routine.scope), iterator)
    if (binding is None or binding.rank or binding.dtype != "integer" or binding.kind not in {4, 8}
            or binding.attributes & {"pointer", "allocatable", "volatile", "asynchronous", "optional", "value"}
            or proof.get("native_function_requirements") or proof.get("native_predicate_requirements")
            or not proof.get("native_completion", {}).get("available")):
        raise CompilationError("reached retirement requires an ordinary original INTEGER iterator and call-free header")
    for operation in proof["operations"]:
        if (operation["kind"] in {"write", "overwrite"} and operation["resource"] != binding.root
                or operation["kind"] not in {"read", "write", "overwrite", "descriptor_read"}
                or operation["rank"] and operation["kind"] != "descriptor_read"):
            raise CompilationError("reached retirement header may write only its original iterator")
    for node in walk(header):
        kind = _kind(node)
        if kind in {"Part_Ref", "Function_Reference", "Structure_Constructor", "Data_Ref", "Proc_Component_Ref"}:
            raise CompilationError("reached retirement header requires scalar names and SIZE-only metadata")
        if kind == "Intrinsic_Function_Reference" and str(node.items[0]).lower() != "size":
            raise CompilationError("reached retirement header requires SIZE-only metadata")
        if kind == "Name":
            source = analysis._binding(analysis.source_scope_for(node, member.routine.scope), node)
            if source is not None and (source.attributes & {"pointer", "volatile", "asynchronous", "optional"}
                    or not source.rank and (source.dtype != "integer" or source.kind not in {4, 8}
                                           or "allocatable" in source.attributes)):
                raise CompilationError("reached retirement header requires ordinary original scalar/descriptor declarations")
    first, last = statement_span(original)
    hfirst, hlast = statement_span(header)
    end_first, _end_last = statement_span(end)
    executable = [item for item in body.nodes if _kind(item) != "Comment"]
    if (hfirst != first or hlast >= last or end_first <= hlast
            or any(statement_span(item)[0] <= hlast for item in executable)):
        raise CompilationError("reached retirement requires separate original header and body source lines")
    source = "".join(member.lines[first-1:last])
    header_source = "".join(member.lines[first-1:hlast])
    body_source = "".join(member.lines[hlast:last])
    if source != member.original(originals) or not header_source.endswith("\n"):
        raise CompilationError("reached retirement lost its exact original source split")
    selected_graph, selected = analysis._selected_source(procedure, originals)
    if selected_graph is not graph or len(selected) != 1:
        raise CompilationError("reached retirement requires its original whole-loop graph node")
    record = {
        "version": 1, "kind": "original_counted_body_retirement", "source_graph": graph.identity,
        "selected_node_ids": list(selected), "header_node_id": header_id,
        "header_summary_identity": proof["summary_identity"], "source_span": [first, last],
        "header_span": [hfirst, hlast], "header_source_sha256": sha256(header_source.encode()).hexdigest(),
        "body_source_sha256": sha256(body_source.encode()).hexdigest(),
        "flag": "fort_loop_retire_" + sha256((procedure + "|" + selected[0]).encode()).hexdigest()[:12],
        "context": member.context, "header_evaluation": "unchanged original counted DO exactly once",
        "retirement": "once before the first original body statement", "empty_loop_preserves_domain": True,
        "final_iterator": "unchanged original counted DO", "payload_access_authorized": False,
        "motion_authorized": False,
    }
    record["identity"] = sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return CountedBody(body.nodes, record)


def render_retirement_entry(member, originals, entry, close):
    """Keep the original DO and final iterator; close only on its first entry."""
    current = reached_retirement_entry(member, originals)
    if current.authority != entry.authority or any(a is not b for a, b in zip(current.nodes, entry.nodes, strict=True)):
        raise CompilationError("reached retirement changed its original counted entry")
    record = entry.authority
    first, last = record["source_span"]
    header_last = record["header_span"][1]
    flag = record["flag"]
    header = "".join(member.lines[first-1:header_last])
    body = "".join(member.lines[header_last:last])
    return ("block\nlogical :: " + flag + "\n" + flag + " = .true.\n" + header
            + "if (" + flag + ") then\n" + flag + " = .false.\n"
            + "\n".join(close) + "\nendif\n! fort original loop body " + flag + "\n"
            + body + "end block\n")
