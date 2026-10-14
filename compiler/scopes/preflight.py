"""Bounded, independent source diagnostics before ownership generation.

This is an immutable observation, not an execution plan or proof registry. It
uses the existing original-source graph, preparation and cost providers. A
failed operation cannot suppress the diagnostics for later source siblings.
Calls are traversed as separate local procedures; no whole-entry effect closure
is requested merely to produce a report.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from hashlib import sha256

from compiler.frontend.component_bindings import references
from compiler.frontend.native_group_structure import NativeGroupCandidate, grouped_original_sequence
from compiler.frontend.numerical_preparation import NumericalPreparationProvider, _Budget, _walk
from compiler.frontend.source_effects import _children, _kind
from compiler.ir import CompilationError
from compiler.offload.source_applicability import source_compute_applicability
from compiler.offload.source_compute import apply_source_compute_costs, fork_join_participation


def _digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ScopePreflight:
    """Detached diagnostic data: mutating public JSON never changes a proof."""

    _json: str

    def public(self):
        return json.loads(self._json)


def _status(status, reasons=(), **values):
    return {"status": status, "reasons": list(dict.fromkeys(str(reason) for reason in reasons if reason)), **values}


def _observe(observer, event, **values):
    if observer is not None:
        # An observer may save progress during a long diagnostic pass. It sees
        # detached data, never mutable compiler state, ASTs or proof tokens.
        observer(json.loads(json.dumps({"event": event, **values}, allow_nan=False)))


def _units(analysis, routine, graph):
    """Visit branches in lexical order and retain complete joined teams/loops."""
    peeled = {}

    def sequence(nodes):
        for item in grouped_original_sequence(nodes, peeled):
            if isinstance(item, NativeGroupCandidate):
                yield "joined_team", item.original_nodes, None
                continue
            if _kind(item) == "Comment" and not str(item).lstrip().lower().startswith("!$omp"):
                continue
            if _kind(item) in {"If_Construct", "If_Stmt"}:
                identity = graph.node_id(item)
                for condition, body in graph.nodes[identity].alternatives:
                    if condition is not None:
                        yield "condition", graph.source_nodes(condition), condition
                    yield from sequence(graph.source_nodes(body))
            elif _kind(item) == "Associate_Construct":
                identity = graph.node_id(item)
                node = graph.nodes[identity]
                if node.kind == "associate":
                    selector, body = node.children
                    yield "association", graph.source_nodes(selector), selector
                    yield from sequence(graph.source_nodes(body))
                else:
                    yield "boundary", (item,), identity
            else:
                yield ("loop" if _kind(item) in {"Block_Nonlabel_Do_Construct", "Block_Label_Do_Construct"}
                       else "call" if _kind(item) == "Call_Stmt" else "operation"), (item,), None

    yield from sequence(_children(routine.execution))


def _cost(prepared, analysis, config):
    completion = prepared.completion
    participation = (fork_join_participation(prepared.extraction.nodes)
                     if completion.get("has_openmp_in_closure") else
                     "serial" if completion.get("available") else "unknown")
    applicability = source_compute_applicability(analysis, prepared, native_participation=participation)
    priced = apply_source_compute_costs(prepared.offload, config.profile, participation,
                                       source_applicability=applicability)
    reasons = [unit.work_estimate_reason or "work estimate is unknown or conditional"
               for unit in priced.units if unit.work_per_iteration is None or unit.work_is_upper_bound]
    if prepared.function.requires_numerical_environment:
        reasons.append("numerical_environment_protocol_calibration_unavailable")
    if prepared.extraction.scalar_element_captures:
        reasons.append("immutable element preparation lacks compatible offline cost estimates")
    return _status("unavailable" if reasons else "available", reasons,
                   native_participation=participation,
                   source_applicability=applicability.public() if applicability else None,
                   unit_count=len(priced.units),
                   scope_transfer_estimate="not_requested", placement_authorized=False)


def preflight_scope(builder, *, follow_calls=True, observer=None):
    """Inspect independent original units without emission or owner selection.

    The supplied builder validates source-bound capture documents and options.
    Procedure-local resources deliberately remain unmapped: a call's actual to
    formal map is reported, but diagnostic observation cannot grant a borrowed
    lifetime, selected-window definition, or alias proof. All branches are
    reported; no configuration value or branch predicate is evaluated.
    ``observer`` receives detached procedure/operation progress events. These
    events are provisional until the returned snapshot reauthenticates inputs;
    an interrupted pass cannot advertise them as a complete preflight.
    """
    analysis = builder.analysis
    analysis.verify_source()
    facts_identity = _digest(builder.facts)
    provider = NumericalPreparationProvider(analysis, options=builder.options, config=builder.config,
                                           environment_root=builder.entry.qualified)
    operations, procedures, edges, blockers = [], [], [], []
    pending, seen = deque([(builder.entry.qualified, 0)]), set()
    truncated = False

    def blocker(row, category, reason, status="unavailable"):
        item = {"category": category, "status": status, "reason": str(reason)}
        if item not in row["blockers"]:
            row["blockers"].append(item)
            blockers.append({"operation": row["ordinal"], "procedure": row["procedure"],
                             "source": row["source"], "span": row["span"], **item})

    while pending and len(operations) < analysis.operation_limit:
        procedure, depth = pending.popleft()
        if procedure in seen:
            continue
        if len(seen) >= analysis.procedure_limit:
            truncated = True
            break
        seen.add(procedure)
        _observe(observer, "procedure_started", procedure=procedure, provisional=True)
        routine = analysis._require_original(procedure)
        graph = analysis.structure(procedure)
        procedures.append({"procedure": procedure, "source": str(routine.scope.path),
                           "structured_identity": graph.identity, "available": graph.available,
                           "reasons": list(graph.reasons)})
        if not graph.available:
            truncated |= any("budget" in reason or "exhausted" in reason for reason in graph.reasons)
            units = [("boundary", graph.source_nodes(graph.root_id), graph.root_id)]
        else:
            units = _units(analysis, routine, graph)
        for kind, selection, selected_id in units:
            if len(operations) >= analysis.operation_limit:
                truncated = True
                break
            # Selection uses the existing original graph. Diagnostics never
            # manufacture AST nodes or select a fragment of a joined team.
            if selected_id is None:
                _graph, identities = analysis._selected_source(procedure, selection)
            else:
                identities = (selected_id,)
            selected = [graph.nodes[identity] for identity in identities]
            spans = [node.span for node in selected if node.span]
            span = [min(item[0] for item in spans), max(item[-1] for item in spans)] if spans else []
            row = {"ordinal": len(operations), "procedure": procedure, "source": str(routine.scope.path),
                   "source_sha256": analysis.sources[str(routine.scope.path)], "span": span,
                   "selected_node_ids": list(identities), "kind": kind,
                   "guard": list(dict.fromkeys(guard for node in selected for guard in node.guard)),
                   "resources": [], "blockers": [],
                   "native": _status("not_requested"), "numerical": _status("not_applicable"),
                   "cost": _status("not_requested", ["no prepared numerical candidate"]),
                   "ownership": _status("not_requested", ["no mapped owner or definition interval admitted"]),
                   "completion": _status("not_requested"),
                   "transfers": {"status": "unknown", "upload_bytes": None, "download_bytes": None,
                                 "reason": "requires checked owner, runtime descriptors and section coherence"}}
            operations.append(row)
            _observe(observer, "operation_started", operation=row, provisional=True)
            for node in selected:
                if node.kind == "boundary":
                    blocker(row, "source", node.details.get("reason") or "original structure unavailable")

            # Use the existing bounded syntax scanner before reference/call
            # inventory. An oversized selection is reported, never inspected
            # by an unbounded fparser walk after its graph already declined.
            try:
                syntax = tuple(_walk(selection, _Budget()))
            except CompilationError as error:
                truncated = True
                blocker(row, "budget", error, "not_checked")
                _observe(observer, "operation_complete", operation=row, provisional=True)
                continue
            # Source references are an inventory only, never an effects proof.
            bindings = {}
            try:
                for original in selection:
                    for binding, _reference in references(analysis, routine.scope, original):
                        if binding.rank:
                            bindings[binding.root] = binding
            except CompilationError as error:
                blocker(row, "resources", error)
            for root, binding in sorted(bindings.items()):
                capture = _status("not_requested", ["callee formal requires original mapped call authority"])
                if procedure == builder.entry.qualified or not root.startswith("argument::"):
                    try:
                        fact = builder.capture_fact(binding)
                        capture = _status("available", lifetime=fact.get("lifetime", "invocation"),
                                          initialized=fact["initialized"])
                    except CompilationError as error:
                        capture = _status("unavailable", [str(error)])
                        blocker(row, "capture", error)
                row["resources"].append({"resource": root, "rank": binding.rank,
                                         "inventory_only": True, "capture_facts": capture})

            calls = [node for node in syntax if _kind(node) == "Call_Stmt"]
            for call in calls:
                try:
                    from compiler.frontend.call_bindings import resolve_source_call

                    resolved = resolve_source_call(analysis, analysis.source_scope_for(call, routine.scope), call)
                    edge = {"operation": row["ordinal"], "caller": procedure, **resolved.public()}
                    edge["body_inspection"] = "pending" if follow_calls else "not_requested"
                    if follow_calls and depth < analysis.depth_limit:
                        pending.append((resolved.procedure, depth + 1))
                    elif follow_calls:
                        edge["body_inspection"] = "depth_limit"
                        truncated = True
                        blocker(row, "budget", "preflight call depth exhausted", "not_checked")
                    edges.append(edge)
                except CompilationError as error:
                    blocker(row, "call", error)
            # Never force a transitive closure just to discover diagnostics.
            if calls:
                row["native"] = _status("not_requested", ["call effects require mapped child proofs; local bodies inspected separately"])
            elif graph.available:
                try:
                    if kind == "joined_team":
                        completion = analysis.joined_completion(procedure, selection)
                        row["completion"] = _status("available", proof=completion.public())
                        if graph.native_group_for_selection(selection) is not None:
                            _graph, _ids, _group, summaries = analysis.native_group_summaries(procedure, selection, completion)
                        else:
                            summaries = (analysis.segment_summary(procedure, identities, capture_locals=True),)
                    else:
                        summaries = (analysis.segment_summary(procedure, identities, capture_locals=True),)
                    reasons = [reason for summary in summaries for reason in summary.get("reasons", ())]
                    row["native"] = _status("unavailable" if reasons else "available", reasons,
                        effects=[effect for summary in summaries for effect in summary.get("ordered_effects", ())],
                        sections=[summary.get("native_sections", {}) for summary in summaries])
                    for reason in reasons:
                        blocker(row, "native", reason)
                    if kind != "joined_team":
                        completion = summaries[0].get("native_completion", {})
                        row["completion"] = _status("available" if completion.get("available") else "unavailable",
                                                    [] if completion.get("available") else [completion.get("reason")])
                except CompilationError as error:
                    row["native"] = _status("unavailable", [str(error)])
                    blocker(row, "native", error)

            if kind in {"loop", "joined_team"}:
                try:
                    prepared = provider.prepare(analysis, procedure, selection)
                    row["numerical"] = _status("available", preparation=prepared.public())
                    row["cost"] = _cost(prepared, analysis, builder.config)
                    for reason in row["cost"]["reasons"]:
                        blocker(row, "cost", reason)
                except CompilationError as error:
                    row["numerical"] = _status("unavailable", [str(error)])
                    blocker(row, "numerical", error)
            _observe(observer, "operation_complete", operation=row, provisional=True)

    if pending:
        truncated = True
    for edge in edges:
        if edge["body_inspection"] == "pending":
            edge["body_inspection"] = "inspected" if edge["procedure"] in seen else "budget_limit"
    # Reauthenticate inputs/policy after all independent demands. A stale or
    # mutated input aborts the entire snapshot; it is never just one blocker.
    provider.validate_configuration(options=builder.options, config=builder.config,
                                    environment_root=builder.entry.qualified)
    analysis.verify_source()
    if facts_identity != _digest(builder.facts):
        raise CompilationError("scope preflight capture facts changed during inspection")
    report = {"schema_version": 1, "entry": builder.entry.qualified,
              "role": "independent_source_diagnostics", "execution_authorized": False,
              "artifacts_emitted": False, "branch_predicates_evaluated": False,
              "order": "procedure discovery order; source order within each procedure; all branches retained",
              "sources": analysis.sources, "configuration_identity": provider.identity,
              "capture_facts_sha256": facts_identity, "procedures": procedures, "operations": operations,
              "calls": edges, "blockers": blockers, "truncated": truncated,
              "calibration": {"profile_supplied": builder.config.profile is not None,
                              "reason": builder.config.profile_reason,
                              "compatibility": "checked independently for each prepared source unit"},
              "limits": {"operations": analysis.operation_limit, "procedures": analysis.procedure_limit,
                         "call_depth": analysis.depth_limit},
              "scope_automatic_estimate": _status("not_requested", [
                  "independent numerical prices do not establish a resident-window estimate",
                  "host-only native ranges, IEEE coordination, immutable elements and indirect inspectors require owner cost checks"])}
    report["snapshot_identity"] = _digest(report)
    return ScopePreflight(json.dumps(report, sort_keys=True, allow_nan=False))


def preflight_source_scopes(paths, entry, *, facts, options, config, contracts=None,
                            numerical_sources=None, analysis_sources=None, summary_cache=None, follow_calls=True,
                            observer=None):
    """Public read-only counterpart to form_source_scopes for serial owners."""
    from compiler.scopes.source import ScopeBuilder

    _observe(observer, "analysis_started", entry=entry, provisional=True)
    builder = ScopeBuilder(paths, entry, facts=facts, options=options, config=config, contracts=contracts,
                           numerical_sources=numerical_sources, analysis_sources=analysis_sources,
                           summary_cache=summary_cache)
    return preflight_scope(builder, follow_calls=follow_calls, observer=observer)
