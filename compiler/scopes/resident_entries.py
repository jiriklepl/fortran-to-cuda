"""Numerical entries for a caller-owned residency domain.

This producer deliberately has no placement or lifetime policy. It exports
source-backed numerical regions through the same extraction used by checked
source scopes. The owner retains scalar control flow, native effects and
allocation epochs. Every exported entry still goes through ordinary numerical
lowering and dependence analysis before it can execute on a device.
"""
from fnmatch import fnmatchcase
from pathlib import Path

from fparser.two.utils import walk

from compiler.frontend.source_effects import SourceEffects, _children, _kind
from compiler.ir import CompilationError
from compiler.scopes.region_context import original_region_context
from compiler.scopes.regions import extract_region
from compiler.scopes.segments import directive, grouped_nodes, statement_span


def _entry_groups(sequence):
    """Share an entry across adjacent serial nests in one control region.

    Calls, assignments, branch boundaries and OpenMP constructs end the group.
    This only changes the numerical extraction boundary: ordinary dependence
    analysis still schedules every source pass and proves any later fusion.
    """
    pending = []
    for group in grouped_nodes(sequence):
        if not isinstance(group, tuple) and _kind(group) == "Comment" and directive(group) is None:
            continue
        if (not isinstance(group, tuple) and _kind(group) == "Block_Nonlabel_Do_Construct"
                and not any(directive(node) is not None for node in walk(group) if _kind(node) == "Comment")):
            pending.append(group)
            continue
        if pending:
            yield tuple(pending)
            pending = []
        yield group
    if pending:
        yield tuple(pending)


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
    """Export resident numerical entries for explicit procedures or glob patterns."""
    analysis = SourceEffects(paths, analysis_sources=analysis_sources)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    resolved_procedures = _resolve_procedures(analysis, procedures, allow_patterns=allow_patterns)
    entries, rejected, grouping_rejections = [], [], []
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

        def scan(sequence):
            for group in _entry_groups(sequence):
                nodes = tuple(group) if isinstance(group, tuple) else (group,)
                if len(nodes) == 1 and _kind(nodes[0]) in {"If_Construct", "Associate_Construct", "Block_Construct"}:
                    scan(tuple(_children(nodes[0])))
                elif any(_kind(node) == "Block_Nonlabel_Do_Construct" for node in nodes):
                    emit(nodes, sequence)

        scan(tuple(_children(routine.execution)))
    analysis.verify_source()
    return {"version": 2, "kind": "caller_owned_resident_entries", "source_inputs": analysis.sources,
            "entries": entries, "rejected": rejected, "grouping_rejections": grouping_rejections,
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
                        "value_contract": "reached descriptor origins and initialized primary-grid geometry",
                        "original_numerical_input": {"path": str(source.resolve()),
                                                     "sha256": sha256(source.read_bytes()).hexdigest()}})
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
    record = {"version": 1, "kind": "resident_rk_device_entries", "source_inputs": extracted["source_inputs"],
              "entries": entries, "rejected": rejected, "gpu_execution_established": False}
    (directory / "device-entries.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
