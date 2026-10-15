"""Observe an admitted original invocation without granting placement authority.

The timer surrounds the unchanged original CALL, so argument evaluation,
procedure automatic storage, original team completion and resident cleanup all
remain inside the observed interval. Separate native/resident artifacts execute
one original invocation each; no runtime selector or replay is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from fparser.two import Fortran2003 as F
from fparser.two.utils import walk

from compiler.frontend.source_effects import _children, _kind, _part
from compiler.ir import CompilationError
from compiler.offload.exact_window import digest, implementation_identity
from compiler.scopes.segments import directive, fortran_lines, grouped_nodes
from compiler.scopes.source import _span

PROTOCOL = "complete-lexical-call-clock-v1"
PREFIX = "FORT_SCOPE_MEASUREMENT "


@dataclass(frozen=True)
class CallSite:
    caller: object
    node: object
    text: str
    record: dict


@dataclass(frozen=True)
class OwnerMeasurement:
    record: dict
    sites: tuple[CallSite, ...]


def _check_caller_context(caller, selected):
    statement = _part(caller.scope.node, "Subroutine_Stmt")
    if statement is None or any(str(item).lower() in {"pure", "elemental"}
                                for item in _children(statement.items[0])):
        raise CompilationError("scope measurement requires an ordinary impure subroutine caller")

    def contains(node):
        return any(item is selected for item in walk(node))

    def check(nodes):
        # Reuse existing whole-team grouping, including optional END PARALLEL
        # DO. Completed groups before the call do not restrict its caller.
        for group in grouped_nodes(nodes):
            if isinstance(group, tuple):
                if any(contains(node) for node in group):
                    raise CompilationError("scope measurement does not instrument calls inside original directive groups")
                continue
            if directive(group) is not None:
                raise CompilationError("scope measurement requires a caller position without unsupported directive context")
            if group is selected:
                return True
            if not contains(group):
                continue
            if (_kind(group) in {"Forall_Construct", "Forall_Stmt"}
                    or any(_kind(item) == "Nonlabel_Do_Stmt" and "CONCURRENT" in str(item).upper()
                           for item in _children(group))):
                raise CompilationError("scope measurement does not instrument concurrent caller constructs")
            if check(_children(group)):
                return True
        return False

    # fparser can attach directives immediately before the first executable
    # statement to the specification's trailing Implicit_Part.
    prefix = [node for node in walk(_part(caller.scope.node, "Specification_Part"))
              if directive(node) is not None]
    if not check((*prefix, *_children(caller.execution))):
        raise CompilationError("scope measurement lost the original caller statement context")


def prepare(owner):
    """Retain only existing complete-owner admission and exact direct callers."""
    builder = owner.builder
    variant = builder.config.scope_measurement
    if variant is None:
        return None
    plan = owner.execution_plan.validate()
    if (owner.parent is not None or getattr(builder, "_reached_window_binding", None) is not None
            or hasattr(owner, "window_proof") or len(plan.members) != 1 or owner.boundaries):
        raise CompilationError("scope measurement requires one complete original lexical owner without ownership boundaries")
    analysis = builder.analysis
    sites = []
    for caller in sorted(analysis.routines.values(), key=lambda item: item.qualified):
        for node in walk(caller.execution, F.Call_Stmt):
            if _kind(node.items[0]) != "Name":
                continue
            scope = analysis.source_scope_for(node, caller.scope)
            candidates = analysis._candidates(scope, node.items[0])
            if owner.routine.qualified not in candidates:
                continue
            analysis._require_original(caller.qualified)
            if candidates != [owner.routine.qualified]:
                raise CompilationError("scope measurement requires an unambiguous original direct call")
            if caller.qualified == owner.routine.qualified:
                raise CompilationError("scope measurement does not wrap recursive owner calls")
            _check_caller_context(caller, node)
            item = getattr(node, "item", None)
            if (item is None or getattr(item, "label", None) is not None
                    or any(_kind(part) == "Alt_Return_Spec" for part in walk(node))):
                raise CompilationError("scope measurement requires a standalone unlabelled call without alternate returns")
            first, last = _span(node)
            # A shared source line cannot be replaced without also rewriting
            # neighbouring statements or the original single-line predicate.
            for other in walk(caller.execution):
                if other is node or getattr(other, "item", None) is None:
                    continue
                if _span(other) == (first, last):
                    raise CompilationError("scope measurement requires one standalone original call statement per span")
            path = str(caller.scope.path)
            text = "".join(analysis.source_text(path).splitlines(keepends=True)[first - 1:last])
            row = {"caller": caller.qualified, "source": path,
                   "source_sha256": analysis.sources[path], "first_line": first, "last_line": last,
                   "call_sha256": sha256(text.encode()).hexdigest()}
            row["identity"] = digest(row)
            sites.append(CallSite(caller, node, text, row))
    if not sites:
        raise CompilationError("scope measurement requires an authenticated standalone direct caller in the supplied source graph")
    public = plan.public()
    record = {
        "schema_version": 1, "protocol": PROTOCOL,
        "execution_identity": plan.execution_identity,
        **{key: public[key] for key in ("source_inputs_identity", "configuration_identity", "capture_facts_identity")},
        "runtime_identity": digest(builder.runtime),
        "compiler_identity": digest(implementation_identity()),
        "call_sites": [site.record for site in sites],
        "placement_authority": False, "automatic_estimate_available": False, "calibration_evidence": False,
        "interval": "before original call argument evaluation through return and finalization",
        "clock": "system_clock:int64", "observation_prefix": PREFIX,
        "limitations": [
            "No runtime workload key; observations are not automatic placement evidence.",
            "Resident variant may execute guarded native fallbacks; verify runtime traces separately.",
            "Only listed direct call sites in registered subroutines of the authenticated supplied source graph are observed.",
        ],
    }
    record["identity"] = digest(record)
    record["variant"] = variant
    return OwnerMeasurement(record, tuple(sites))


def _wrapper(builder, measurement, site):
    """Keep timer intrinsics in separate blocks from original call names."""
    suffix = site.record["identity"][:12]
    names = {key: "fort_measure_" + key + "_" + suffix for key in
             ("i64", "r64", "error", "start", "finish", "rate", "last_rate", "maximum", "last_maximum", "io")}
    scope = builder.analysis.source_scope_for(site.node, site.caller.scope)
    for name in names.values():
        if builder.analysis._binding(scope, name) or builder.analysis._candidates(scope, name):
            raise CompilationError("scope measurement conflicts with an original caller name: " + name)
    start, finish, rate = (names[key] for key in ("start", "finish", "rate"))
    maximum, last_rate, last_maximum = (names[key] for key in ("maximum", "last_rate", "last_maximum"))
    header = [
        "block",
        f"use, intrinsic :: iso_fortran_env, only: {names['i64']} => int64, &",
        f"    {names['r64']} => real64, {names['error']} => error_unit",
        f"integer({names['i64']}) :: {start}, {finish}, {rate}, {last_rate}, {maximum}, {last_maximum}",
        f"integer :: {names['io']}",
        "block", "intrinsic :: system_clock",
        f"call system_clock({start}, {rate}, {maximum})", "end block",
    ]
    prefix = [
        f"    '{PREFIX}{{\"identity\":\"' // &",
        f"    '{measurement.record['identity']}\",' // &",
        f"    '\"call_site\":\"{site.record['identity']}\",' // &",
        f"    '\"variant\":\"{measurement.record['variant']}\",' // &",
    ]
    footer = [
        "block", "intrinsic :: system_clock, real",
        f"call system_clock({finish}, {last_rate}, {last_maximum})",
        f"if ({rate} > 0 .and. {maximum} > 0 .and. {start} >= 0 .and. &",
        f"    {finish} >= {start} .and. {rate} == {last_rate} .and. {maximum} == {last_maximum}) then",
        f"write({names['error']}, '(a,es24.16,a,i0,a,i0,a,i0,a)', iostat={names['io']}) &",
        *prefix,
        "    '\"status\":\"complete\",\"seconds\":', &",
        f"    real({finish} - {start}, {names['r64']}) / real({rate}, {names['r64']}), &",
        f"    ',\"clock_start\":', {start}, ',\"clock_finish\":', {finish}, &",
        f"    ',\"clock_rate\":', {rate}, '}}'",
        "else",
        f"write({names['error']}, '(a)', iostat={names['io']}) &",
        *prefix,
        "    '\"status\":\"invalid_clock\"}'",
        "endif", "end block", "end block",
    ]
    # Retain original CALL bytes, including continuations and comments. No
    # argument, predicate, alias, or caller team directive is reconstructed.
    return ("\n".join(fortran_lines(header)) + "\n" + site.text
            + ("" if site.text.endswith("\n") else "\n") + "\n".join(fortran_lines(footer)) + "\n")


def emit(owner, measurement):
    """Add caller observation after final owner emission (or native rollback)."""
    if measurement is None:
        return
    owner.execution_plan.validate()
    builder = owner.builder
    replacements = []
    for site in measurement.sites:
        row = site.record
        path = site.caller.scope.path
        first, last = row["first_line"], row["last_line"]
        for edit_first, edit_last, _ in builder.edits.get(path, ()):
            if edit_last >= first and edit_first <= last or edit_last < edit_first and first < edit_first <= last:
                raise CompilationError("scope measurement call overlaps an existing compiler source edit")
        replacements.append((path, first, last, _wrapper(builder, measurement, site)))
    for path, first, last, replacement in replacements:
        builder.add_edit(path, first, last, replacement)
    builder.scopes[-1]["owner_measurement"] = measurement.record
    builder.scopes[-1]["estimate_available"] = False
    builder.scopes[-1]["planning_reason"] = "explicit measurement only; no automatic placement estimate"


def emit_native(owner, nodes, checkpoint, measurement):
    """Discard resident artifacts; leave original callee declarations and body."""
    builder = owner.builder
    interval_applicability = owner.execution_plan.interval_applicability().public()
    builder.restore_scope_checkpoint(checkpoint)
    plan = owner.execution_plan.validate()
    from compiler.scopes.segments import statement_span

    first, last = statement_span(nodes[0])[0], statement_span(nodes[-1])[1]
    builder.scopes.append({
        "owner": owner.context, "owner_variant": None, "path": str(owner.routine.scope.path),
        "first_line": first, "last_line": last, "gpu_leaves": [], "mode": builder.config.policy,
        "estimate_available": False,
        "source_authority": {"version": 1, "structured_summary_identity": owner.graph.identity},
        "checked_execution_plan": plan.public(), "boundaries": [],
        "interval_applicability": interval_applicability,
        "execution": "unchanged original native callee; authenticated caller timing only",
    })
    emit(owner, measurement)
    return True
