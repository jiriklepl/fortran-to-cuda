"""Leave a proved host-only preparation prefix before lazy context creation.

The checked owner already admitted every operation and retains its dependencies.
This is an emission choice at the original statement, not new read, definition,
descriptor, placement, or lifetime authority.
"""

from compiler.frontend.source_effects import _children, _kind
from compiler.scopes.segments import Native


def _readonly_assignments(scope, definitions):
    if (scope.calls or not scope.native or len(scope.tree) != len(scope.native)
            or any(not isinstance(item, Native) for item in scope.tree)
            or definitions.required_whole or definitions.whole_on_return or definitions.changes):
        return False
    for operation in scope.native:
        if (operation.overwrites or operation.host_metadata or operation.native_environment
                or operation.preserve_original or operation.host_only_reads
                or not operation.summary.get("complete")
                or not operation.summary.get("native_completion", {}).get("available")
                or operation.summary.get("definition_changes")
                or any(_kind(node) != "Assignment_Stmt" for node in operation.nodes)
                or any(actions - {"read"} for actions in operation.effects.values())):
            return False
        sections = operation.coherence_sections
        if (sections is None or not sections.available
                or any(resource.writes or resource.overwrites for resource in sections.resources)):
            return False
        for effect in operation.summary.get("operations", ()):
            if effect["kind"] not in {"read", "write", "overwrite", "descriptor_read"}:
                return False
            if effect.get("rank", 0) and effect["kind"] not in {"read", "descriptor_read"}:
                return False
    return True


def fresh_native_prefix(member, unit):
    """Consume existing exact checked units; borrowed/live entries keep hooks."""
    owner = member.owner
    if (member is not owner or member.parent is not None or member.control_guard is not None
            or member.builder.config.scope_execution != "graph" or not owner.leaves
            or unit.leaves or unit.source_guard is not None or unit.native_source is not None
            or unit.publication_only or not unit.arrays
            or not _readonly_assignments(unit.scope, unit.definition_summary)):
        return None
    plan = owner.execution_plan
    # Include every checked child, with its canonical caller mapping. An array
    # used by any numerical unit remains managed, even if its use is guarded.
    gpu_roots = {current.canonical(binding).root for current in plan.members
                 for candidate in plan.units(current) if candidate.leaves
                 for binding in candidate.arrays.values()}
    roots = {member.canonical(binding).root for binding in unit.arrays.values()}
    if roots & gpu_roots:
        return None
    admitted = {}
    for candidate in plan.units(member):
        if (not candidate.leaves and candidate.source_guard is None and candidate.native_source is None
                and not candidate.publication_only
                and _readonly_assignments(candidate.scope, candidate.definition_summary)):
            for node in member.builder.inline.original_selection(candidate.nodes):
                admitted[id(node)] = candidate.scope
    from compiler.scopes.definitions import summarize

    for nodes, scope in member.graph_native_tasks:
        if _readonly_assignments(scope, summarize(scope)):
            for node in member.builder.inline.original_selection(nodes):
                admitted[id(node)] = scope
    originals = tuple(member.builder.inline.original_selection(unit.nodes))
    if len(originals) != 1 or _kind(originals[0]) != "Assignment_Stmt":
        return None
    # A prefix is a direct original sequence of already admitted scalar
    # assignments. No branch, loop, CALL, directive, or unproved source prefix
    # can be crossed. The runtime guard also preserves any active context.
    reached = False
    for node in _children(member.routine.execution):
        if _kind(node) == "Comment" and not str(node).lstrip().startswith("!$"):
            continue
        if id(node) not in admitted or _kind(node) != "Assignment_Stmt":
            return None
        lhs = node.items[0]
        scope = member.builder.analysis.source_scope_for(node, member.routine.scope)
        binding = member.builder.analysis._binding(scope, lhs) if _kind(lhs) == "Name" else None
        if (binding is None or binding.rank or binding.dtype not in {"integer", "real", "logical"}
                or binding.attributes & {"pointer", "optional", "volatile", "asynchronous", "allocatable"}):
            return None
        if node is originals[0]:
            reached = True
            break
    if not reached:
        return None
    return {"version": 1, "available": True, "resources": sorted(roots),
            "source_span": [unit.scope.first, unit.scope.last],
            "condition": "fresh original owning entry, context zero, no inherited query",
            "execution": "exact original assignment at its original source point; active context retains all hooks",
            "definitions_authorized": False, "placement_authorized": False,
            "estimates_available": False,
            "cost": "unmanaged original prefix before runtime context; not an owner cost estimate"}
