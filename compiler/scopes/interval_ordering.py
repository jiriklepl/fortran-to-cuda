"""Prioritize possible array reuse before spending bounded admission attempts.

These source hints are deliberately not effect summaries. INTENT describes a
possible argument access, and a source loop is not numerical eligibility. No
runtime fact, branch predicate or payload is evaluated here. The selected owner
still proves effects, associations, completion and placement independently.
"""

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.native_group_structure import (
    NATIVE_GROUP_SCAN_LIMIT,
    NativeGroupCandidate,
    grouped_original_sequence,
)
from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.segments import statement_span


def order_intervals(builder, sites):
    """Prefer connected separate-output work, then its smallest source span.

    Keeping an intermediate for a consumer that writes another array is a
    useful first experiment. Partial scalar corrections can remain between
    those operations. In-place-only or unknown workloads retain source order.
    This orders the existing candidate set; it neither expands search budgets
    nor removes candidates with incomplete hints.
    """
    analysis = builder.analysis
    record = {"schema_version": 1,
              "authority": "source-only search order; no effect, legality or cost authority",
              "policy": "possible separate-output array reuse; shortest tied interval; original order otherwise",
              "candidate_count": len(sites), "candidates": [], "attempts": [], "selected": None}
    units, procedures, seen = {}, {}, set()

    def children(node):
        if id(node) not in seen:
            seen.add(id(node))
            if len(seen) > NATIVE_GROUP_SCAN_LIMIT:
                raise CompilationError("interval search hint source budget exhausted")
        return tuple(child for child in _children(node) if child is not None and not isinstance(child, (str, int)))

    def binding(node, routine):
        return analysis._binding(analysis.source_scope_for(node, routine.scope), node) if _kind(node) == "Name" else None

    def array(value):
        return value is not None and value.rank and value.dtype in {"real", "integer", "logical"}

    def references(node, routine):
        value = binding(node, routine)
        result = {value.root} if array(value) else set()
        for child in children(node):
            result.update(references(child, routine))
        return result

    def target(node, routine):
        base = node.items[0] if _kind(node) == "Part_Ref" else node
        value = binding(base, routine)
        return value if array(value) else None

    def bulk_body(routine):
        if routine.qualified not in procedures:
            pending = list(_children(routine.execution))
            found = False
            while pending:
                node = pending.pop()
                if _kind(node) == "Block_Nonlabel_Do_Construct":
                    found = True
                elif _kind(node) == "Assignment_Stmt" and _kind(node.items[0]) == "Name":
                    found |= target(node.items[0], routine) is not None
                pending.extend(children(node))
            procedures[routine.qualified] = found
        return procedures[routine.qualified]

    def footprint(originals):
        key = tuple(map(id, originals))
        if key in units:
            return units[key]
        reads, writes, unresolved = set(), set(), []
        bulk = False
        pending = list(originals)
        while pending:
            node = pending.pop()
            kind = _kind(node)
            if kind == "Block_Nonlabel_Do_Construct":
                bulk = True
            elif kind == "Assignment_Stmt":
                lhs, _operator, rhs = node.items
                value = target(lhs, builder.entry)
                reads.update(references(rhs, builder.entry))
                reads.update(references(lhs, builder.entry) - ({value.root} if value is not None else set()))
                if value is not None:
                    writes.add(value.root)
                    bulk |= _kind(lhs) == "Name"
            elif kind == "Call_Stmt":
                try:
                    call = resolve_original_call(analysis, analysis.source_scope_for(node, builder.entry.scope), node)
                except CompilationError as error:
                    unresolved.append({"source": str(node), "reason": str(error)})
                else:
                    bulk |= bulk_body(call.callee)
                    for argument in call.associations:
                        value = argument.binding
                        if not array(value):
                            continue
                        if argument.formal_binding.intent != "out":
                            reads.add(value.root)
                        if argument.formal_binding.intent != "in":
                            writes.add(value.root)
            pending.extend(children(node))
        result = {"first_line": statement_span(originals[0])[0], "last_line": statement_span(originals[-1])[1],
                  "possible_reads": sorted(reads), "possible_writes": sorted(writes),
                  "bulk_source_syntax": bulk, "unresolved_calls": unresolved}
        units[key] = result
        return result

    try:
        for ordinal, site in enumerate(sites):
            grouped = grouped_original_sequence(site.originals, {})
            rows = [footprint(item.original_nodes if isinstance(item, NativeGroupCandidate) else (item,))
                    for item in grouped if isinstance(item, NativeGroupCandidate) or _kind(item) != "Comment"]
            producers, edges = {}, []
            for consumer, row in enumerate(rows):
                if not row["bulk_source_syntax"] or not row["possible_writes"]:
                    continue
                for resource in set(row["possible_reads"]) - set(row["possible_writes"]):
                    if resource in producers:
                        edges.append({"producer": producers[resource], "consumer": consumer, "resource": resource})
                for resource in row["possible_writes"]:
                    producers[resource] = consumer
            record["candidates"].append({"original_ordinal": ordinal,
                "first_line": statement_span(site.originals[0])[0], "last_line": statement_span(site.originals[-1])[1],
                "units": rows, "possible_reuse": sorted(edges, key=lambda edge: (edge["consumer"], edge["resource"]))})
    except CompilationError as error:
        record.update(reason=str(error), ordering="unchanged original candidate order", source_nodes=len(seen))
        return tuple(sites), record
    ordered = sorted(record["candidates"], key=lambda row: (
        -len(row["possible_reuse"]), len(row["units"]) if row["possible_reuse"] else 0, row["original_ordinal"]))
    record.update(source_nodes=len(seen), ordered_ordinals=[row["original_ordinal"] for row in ordered])
    return tuple(sites[row["original_ordinal"]] for row in ordered), record
