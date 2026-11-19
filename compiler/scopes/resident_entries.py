"""Numerical entries for a caller-owned residency domain.

This producer deliberately has no placement or lifetime policy. It exports
source-backed numerical regions through the same extraction used by checked
source scopes. The owner retains scalar control flow, native effects and
allocation epochs. Every exported entry still goes through ordinary numerical
lowering and dependence analysis before it can execute on a device.
"""
from fnmatch import fnmatchcase
from hashlib import sha256
from pathlib import Path

from fparser.two.Fortran2003 import Allocation, Call_Stmt, Intrinsic_Function_Reference, Name
from fparser.two.utils import walk

from compiler.frontend.call_bindings import resolve_original_call
from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.region_context import original_region_context
from compiler.scopes.regions import extract_region
from compiler.scopes.segments import directive, grouped_nodes, statement_span

_INQUIRIES = {"lbound", "ubound", "size", "shape", "allocated", "associated", "present", "kind", "rank",
              "storage_size", "is_contiguous"}
_CONSTRUCT_HEADERS = {"If_Then_Stmt", "Else_If_Stmt", "Associate_Stmt"}


def _residency_directive(node):
    return str(node).lstrip().lower().startswith("!$fort")


def _entry_groups(sequence, acknowledged=None):
    """Share an entry across adjacent serial nests in one control region.

    Calls, assignments, branch boundaries, OpenMP constructs and residency
    directives end the group; a nest with a directive inside it stays native.
    This only changes the numerical extraction boundary: ordinary dependence
    analysis still schedules every source pass and proves any later fusion.
    ``acknowledged`` collects the identities of the groups that directly follow
    a residency directive.
    """
    pending, follows_directive = [], False

    def mark(group):
        nonlocal follows_directive
        if follows_directive and acknowledged is not None:
            acknowledged.add(id(group[0] if isinstance(group, tuple) else group))
        follows_directive = False
        return group

    for group in grouped_nodes(sequence):
        if not isinstance(group, tuple) and _kind(group) == "Comment" and directive(group) is None:
            if _residency_directive(group):
                if pending:
                    yield mark(tuple(pending))
                    pending = []
                follows_directive = True
            continue
        if (not isinstance(group, tuple) and _kind(group) == "Block_Nonlabel_Do_Construct"
                and not any(directive(node) is not None or _residency_directive(node)
                            for node in walk(group) if _kind(node) == "Comment")):
            pending.append(group)
            continue
        if pending:
            yield mark(tuple(pending))
            pending = []
        yield mark(group)
    if pending:
        yield mark(tuple(pending))


def _resolve_procedures(analysis, procedures, *, allow_patterns=False):
    available = sorted(analysis.routines)
    if procedures is None:
        return available
    selected = []
    seen = set()
    for spec in procedures:
        pattern = spec.lower()
        is_pattern = allow_patterns or any(ch in pattern for ch in "*?[")
        if is_pattern:
            matches = [name for name in available if fnmatchcase(name, pattern)]
            if not matches:
                raise CompilationError("Resident procedure pattern matched no routines: " + spec)
            for name in matches:
                if name not in seen:
                    seen.add(name)
                    selected.append(name)
        else:
            if pattern not in analysis.routines:
                raise CompilationError("Resident procedure is unavailable: " + spec)
            if pattern not in seen:
                seen.add(pattern)
                selected.append(pattern)
    return selected


def export_resident_entries(paths, procedures, directory, *, analysis_sources=None, allow_patterns=False):
    """Export resident numerical entries for explicit procedures or glob patterns.

    Besides ``entries`` and ``rejected``, the result lists ``host_statements``:
    the statements of the selected procedures that stay host code and name
    arrays, each with its span, the arrays and whether a residency directive
    directly precedes it or sits inside it (``acknowledged``). They run beside
    the entries, so each one that touches an array an entry binds needs the
    host to hold current values.
    """
    analysis = SourceEffects(paths, analysis_sources=analysis_sources)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    resolved_procedures = _resolve_procedures(analysis, procedures, allow_patterns=allow_patterns)
    entries, rejected, grouping_rejections, host_statements = [], [], [], []
    for procedure in resolved_procedures:
        routine = analysis.routines[procedure]

        def emit(nodes, sequence):
            first = min(statement_span(node)[0] for node in nodes)
            last = max(statement_span(node)[1] for node in nodes)
            # Grouping can peel directive prefixes. Recover the actual
            # source nodes instead of supplying manufactured syntax.
            selected = tuple(node for node in sequence
                             if first <= statement_span(node)[0]
                             and statement_span(node)[1] <= last)
            loops = [node for node in nodes if _kind(node) == "Block_Nonlabel_Do_Construct"]
            try:
                context = original_region_context(routine, selected, analysis=analysis)
                extracted = extract_region(analysis, routine, selected,
                                           preceding=context.preceding, following=context.following,
                                           cyclic_following=context.cyclic_following)
            except CompilationError as error:
                # A larger interface can expose an unsupported association
                # even when its individual source nests remain exportable.
                if len(nodes) > 1 and len(loops) == len(nodes):
                    grouping_rejections.append({"procedure": procedure, "first_line": first,
                                                "last_line": last, "reason": str(error)})
                    for node in nodes:
                        emit((node,), sequence)
                    return
                rejected.append({"procedure": procedure, "first_line": first,
                                 "last_line": last, "reason": str(error)})
                return
            path = extracted.write(directory)
            entries.append({"procedure": procedure, "source": str(routine.scope.path),
                            "input_path": str(path.resolve()), **extracted.public(),
                            "source_loop_spans": [list(statement_span(node)) for node in loops]})

        def host(nodes, acknowledged):
            """Record host code that names arrays: it runs beside entries that may hold newer values."""
            arrays, calls = {}, set()
            for node in nodes:
                # Asking for an array's bounds or allocation, or allocating it, reads none of its values.
                inquired = {id(reference.items[1].items[0]) for reference in walk(node, Intrinsic_Function_Reference)
                            if str(reference.items[0]).lower() in _INQUIRIES and reference.items[1] is not None
                            and reference.items[1].items}
                inquired |= {id(allocation.items[0]) for allocation in walk(node, Allocation)}
                for name in walk(node, Name):
                    binding = analysis._binding(routine.scope, name)
                    if binding is not None and binding.rank and id(name) not in inquired:
                        arrays[binding.root] = binding.name
                for call in walk(node, Call_Stmt):
                    try:
                        calls.add(resolve_original_call(analysis, routine.scope, call).procedure)
                    except CompilationError:
                        calls.update(analysis._candidates(routine.scope, call.items[0]) or ("$unresolved",))
            if arrays:
                first = min(statement_span(node)[0] for node in nodes)
                last = max(statement_span(node)[1] for node in nodes)
                text = " ".join(" ".join(str(node).split()) for node in nodes)
                host_statements.append({"procedure": procedure, "source": str(routine.scope.path),
                                        "first_line": first, "last_line": last,
                                        "text": text if len(text) <= 240 else text[:237] + "...",
                                        # Stable while the statement and its procedure keep their text.
                                        "id": sha256((procedure + "\n" + text).encode()).hexdigest()[:16],
                                        "arrays": [{"name": name, "resource": root}
                                                   for root, name in sorted(arrays.items())],
                                        "calls": sorted(calls), "acknowledged": acknowledged,
                                        "only_calls": all(_kind(node) == "Call_Stmt" for node in nodes)})

        def scan(sequence):
            follows = set()
            for group in _entry_groups(sequence, follows):
                nodes = tuple(group) if isinstance(group, tuple) else (group,)
                carries = any(_residency_directive(item) for node in nodes for item in walk(node)
                              if _kind(item) == "Comment")
                if len(nodes) == 1 and _kind(nodes[0]) in {"If_Construct", "Associate_Construct", "Block_Construct"}:
                    # The construct's own statements (its conditions, selectors) are host code as well.
                    host(tuple(item for item in nodes[0].content if _kind(item) in _CONSTRUCT_HEADERS),
                         id(nodes[0]) in follows or carries)
                    scan(tuple(_children(nodes[0])))
                elif any(_kind(node) == "Block_Nonlabel_Do_Construct" for node in nodes) and not carries:
                    emit(nodes, sequence)
                else:
                    # A nest that carries a residency directive is host code that manages its own access.
                    host(nodes, id(nodes[0]) in follows or carries)

        scan(tuple(_children(routine.execution)))
    analysis.verify_source()
    return {"version": 2, "kind": "caller_owned_resident_entries", "source_inputs": analysis.sources,
            "entries": entries, "rejected": rejected, "grouping_rejections": grouping_rejections,
            "host_statements": host_statements,
            "lifetime_authority": "caller storage epoch; numerical export grants no native-effect authority"}


def pass_scalars_by_value(extracted, directory, *, resources=()):
    """Rewrite exported entries so chosen INTENT(IN) scalars are passed by VALUE.

    Lower-bound parameters always qualify: the caller evaluates them after it
    binds the reached allocation. ``resources`` names further scalars, as
    ``module::name``, that the caller guarantees are defined before any entry
    runs. VALUE is a language-level input contract; other scalars keep their
    guarded reference ABI. Returns the export with the rewritten paths.
    """
    import re
    from hashlib import sha256

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    resources = set(resources)
    entries = []
    for original in extracted["entries"]:
        values = [parameter["name"] for parameter in original["parameters"]
                  if parameter["lower_bound_dimension"] is not None or parameter["resource"] in resources]
        source = Path(original["input_path"])
        text = source.read_text()
        for name in values:
            pattern = (
                r"(?im)^([ \t]*(?:integer(?:\s*\([^)]+\))?|real(?:\s*\([^)]+\))?|double\s+precision)"
                r"(?:(?!\bvalue\b)[^\n])*?),\s*intent\s*\(\s*in\s*\)\s*::\s*("
                + re.escape(name)
                + r")\s*$"
            )
            if re.search(rf"(?im)^[ \t]*[^\n]*\bvalue\b[^\n]*::\s*{re.escape(name)}\s*$", text):
                continue
            text, count = re.subn(pattern, r"\1, value, intent(in) :: \2", text)
            if count != 1:
                raise CompilationError("Unexpected resident scalar declaration: " + name)
        path = directory / source.name
        path.write_text(text)
        entries.append({**original, "input_path": str(path), "value_inputs": values,
                        "value_contract": "lower bounds of the bound arrays and scalars the caller defines before any entry",
                        "original_numerical_input": {"path": str(source.resolve()),
                                                     "sha256": sha256(source.read_bytes()).hexdigest()}})
    return {**extracted, "entries": entries}


def guard_entries(extracted, directory, condition):
    """Make every exported entry do its work only where ``condition`` holds.

    The condition is wrapped around the body of each innermost loop, where the
    dependence analysis reads it as a fact about the loop bounds. It also joins
    the entry's runtime guards, so a caller that reaches the entry while the
    condition is false stops instead of silently skipping work. Use it for a
    property that the caller's activation implies and the loop bounds do not
    show, such as a minimum extent that keeps a halo copy from overlapping.
    Returns the export with the rewritten paths.
    """
    import re
    from hashlib import sha256

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for original in extracted["entries"]:
        source = Path(original["input_path"])
        lines = source.read_text().splitlines(keepends=True)
        open_loops, bodies = [], []
        for index, line in enumerate(lines):
            statement = line.strip().upper()
            if re.match(r"DO\b", statement):
                open_loops.append([index, True])
            elif re.match(r"END\s*DO\b", statement):
                first, innermost = open_loops.pop()
                if open_loops:
                    open_loops[-1][1] = False
                if innermost:
                    bodies.append((first + 1, index))
        if open_loops or not bodies:
            raise CompilationError("Cannot guard an entry without complete loops: " + original["entry"])
        for first, last in reversed(bodies):
            indent = lines[first][:len(lines[first]) - len(lines[first].lstrip())]
            lines[first:last] = [indent + "if (" + condition + ") then\n", *lines[first:last], indent + "endif\n"]
        path = directory / source.name
        path.write_text("".join(lines))
        entries.append({**original, "input_path": str(path),
                        "runtime_guards": [*original["runtime_guards"], condition],
                        "guard": {"condition": condition,
                                  "original": {"path": str(source.resolve()),
                                               "sha256": sha256(source.read_bytes()).hexdigest()}}})
    return {**extracted, "entries": entries}


def compile_resident_entry(source, entry, directory, *, host_threads=4, collective=False):
    """Compile one entry for forced device placement and write its sources.

    The directory receives the entry's sources, the runtime headers they
    include and ``compiler-report.json``. Returns that report; its ``scoped``
    member is what a caller needs to bind arguments and call the entry.
    """
    import json

    from compiler.driver.kernel_entry import (
        compile_kernel_entry,
        kernel_entry_outputs,
        kernel_entry_report,
        write_kernel_entry,
    )
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig

    try:
        config = OffloadConfig("sections", None, host_threads, collective, None, "direct")
    except ValueError as error:
        raise CompilationError(str(error)) from error
    options = CompilerOptions(opt_level=1, fallback="error", gpu_policy="sections", scope_transfers="direct")
    _function, plan, sources = compile_kernel_entry(Path(source).resolve(), entry, options=options,
                                                    offload_config=config, common_header="runtime.hpp")
    outputs = kernel_entry_outputs(sources, common_header="runtime.hpp")
    report = kernel_entry_report(entry, plan, sources, outputs)
    directory = write_kernel_entry(outputs, directory)
    (directory / "compiler-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compile_resident_entries(extracted, directory, *, host_threads=4):
    """Compile every exported entry; keep those whose regions all run on the device.

    An entry that the compiler rejects, or that would leave a region on the
    host, joins the export's rejections with its source span, so the caller
    keeps that span native. Writes and returns ``device-entries.json``.
    """
    import json

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    entries, rejected = [], list(extracted["rejected"])
    for region in extracted["entries"]:
        target = directory / region["source_identity"][:16]
        span = {"procedure": region["procedure"], "first_line": region["first_line"],
                "last_line": region["last_line"]}
        try:
            report = compile_resident_entry(region["input_path"], region["entry"], target, host_threads=host_threads)
        except CompilationError as error:
            rejected.append({**span, "reason": str(error)})
            continue
        unavailable = [item for item in report["scoped"]["region_execution"] if not item["gpu_available"]]
        if unavailable:
            rejected.append({**span, "reason": "Resident GPU entry contains native-only regions",
                             "regions": unavailable})
            continue
        entries.append({"region": region, "directory": str(target), "entry": report["scoped"]})
    record = {"version": 1, "kind": "resident_device_entries", "source_inputs": extracted["source_inputs"],
              "entries": entries, "rejected": rejected, "gpu_execution_established": False}
    (directory / "device-entries.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
