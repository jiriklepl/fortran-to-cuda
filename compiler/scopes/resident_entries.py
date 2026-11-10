"""Numerical entries for a caller-owned residency domain.

This producer deliberately has no placement or lifetime policy. It exports
source-backed numerical regions through the same extraction used by checked
source scopes. The owner retains scalar control flow, native effects and
allocation epochs. Every exported entry still goes through ordinary numerical
lowering and dependence analysis before it can execute on a device.
"""
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


def export_resident_entries(paths, procedures, directory, *, analysis_sources=None):
    analysis = SourceEffects(paths, analysis_sources=analysis_sources)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    entries, rejected, grouping_rejections = [], [], []
    for procedure in procedures:
        routine = analysis.routines.get(procedure.lower())
        if routine is None:
            raise CompilationError("Resident procedure is unavailable: " + procedure)

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
